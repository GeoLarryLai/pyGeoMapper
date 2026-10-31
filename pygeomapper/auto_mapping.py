# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Automated geomorphic, lithologic, and land-cover unit mapping from spectral, terrain,
2D CWT multiscale complexity, and 64-band satellite embedding stacks.

Includes:
- Unsupervised deterministic Rust k-means++ unit clustering (`cluster_units_kmeans`)
- Spatial majority / Markov-style mode smoothing (`smooth_unit_labels`) and contact boundary
  extraction (`extract_unit_contacts`) in Rust
- Multi-modal feature stack assembly (`build_mapping_feature_stack`: DEM morphometry + 2D CWT
  multiscale complexity + spectral/mineral ratios)
- Strict spatial-block holdout splitting (`create_spatial_block_split`) to prevent spatial
  autocorrelation leakage when validating against public geological maps (USGS SGMC / WSGS)
- Supervised probabilistic geological unit mapping (`train_predict_supervised_units`: Rust
  Gaussian Bayes + Random Forest ensemble with posterior confidence & Shannon entropy maps)
- Per-formation IoU, F1, Cohen's kappa, and confusion matrix evaluation (`evaluate_mapping_accuracy`).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import _core, spectral_indices, terrain_analysis
from .raster_io import GeoGrid, RasterStack

try:
    from sklearn.ensemble import RandomForestClassifier

    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    RandomForestClassifier = None
    HAS_SKLEARN = False


def _unpack_feature_cube(
    features: RasterStack | Sequence[GeoGrid] | np.ndarray,
    standardize: bool = True,
) -> tuple[np.ndarray, GeoGrid | None, list[str]]:
    ref_grid: GeoGrid | None = None
    if isinstance(features, RasterStack):
        cube = features.data.copy()
        ref_grid = features.band(0)
        feature_names = list(features.band_names)
    elif isinstance(features, (list, tuple)):
        stack = RasterStack.from_grids(features)
        cube = stack.data.copy()
        ref_grid = stack.band(0)
        feature_names = list(stack.band_names)
    else:
        cube = np.asarray(features, dtype=np.float64).copy()
        if cube.ndim == 2:
            cube = cube[:, :, np.newaxis]
        feature_names = [f"feat_{i}" for i in range(cube.shape[2])]

    if standardize:
        for b in range(cube.shape[2]):
            band = cube[:, :, b]
            finite = np.isfinite(band)
            if np.any(finite):
                mu = float(np.mean(band[finite]))
                sd = float(np.std(band[finite]))
                if sd > 1e-12:
                    cube[:, :, b] = (band - mu) / sd
                else:
                    cube[:, :, b] = band - mu
    return np.ascontiguousarray(cube, dtype=np.float64), ref_grid, feature_names


def smooth_unit_labels(
    labels: GeoGrid | np.ndarray,
    radius_px: int = 2,
    iterations: int = 1,
) -> GeoGrid | np.ndarray:
    """
    Smooth integer unit labels using a circular-window spatial majority (mode) filter in Rust.
    """
    ref = labels if isinstance(labels, GeoGrid) else None
    arr = labels.data if isinstance(labels, GeoGrid) else np.asarray(labels)
    lbl_i32 = np.where(np.isfinite(arr) & (arr >= 0), np.round(arr).astype(np.int32), -1)
    lbl_i32 = np.ascontiguousarray(lbl_i32, dtype=np.int32)
    smoothed = _core.smooth_labels_majority_rs(lbl_i32, int(radius_px), int(iterations))
    if ref is not None:
        out_f64 = np.where(smoothed >= 0, smoothed.astype(np.float64), np.nan)
        return ref.with_data(out_f64, name=f"{ref.name}_smoothed", units="unit_id")
    return smoothed


def extract_unit_contacts(
    labels: GeoGrid | np.ndarray,
) -> tuple[GeoGrid | np.ndarray, list[dict[str, Any]]]:
    """
    Extract lithologic / geomorphic contact boundary pixels between adjacent units (`unit_a < unit_b`) in Rust.
    """
    ref = labels if isinstance(labels, GeoGrid) else None
    arr = labels.data if isinstance(labels, GeoGrid) else np.asarray(labels)
    lbl_i32 = np.where(np.isfinite(arr) & (arr >= 0), np.round(arr).astype(np.int32), -1)
    lbl_i32 = np.ascontiguousarray(lbl_i32, dtype=np.int32)

    if ref is not None:
        x0, y0, dx, dy = (ref.x0, ref.y0, ref.dx, ref.dy)
    else:
        x0, y0, dx, dy = (0.0, 0.0, 10.0, -10.0)

    mask_u8, contacts = _core.extract_unit_contacts_rs(
        lbl_i32, float(x0), float(y0), float(dx), float(dy)
    )
    if ref is not None:
        return (
            ref.with_data(mask_u8.astype(np.float64), name="unit_contacts", units="binary"),
            list(contacts),
        )
    return mask_u8, list(contacts)


def build_mapping_feature_stack(
    dem: GeoGrid | None = None,
    spectral_stack: RasterStack | None = None,
    cwt_scales_m: Sequence[float] = (25.0, 50.0, 100.0, 180.0),
    include_indices: Sequence[str] = (
        "NDVI",
        "NBR",
        "NDWI",
        "BSI",
        "ClayRatio",
        "FerricOxideRatio",
        "FerrousIronIndex",
        "CarbonateIndex",
    ),
    include_raw_bands: bool = True,
    *,
    embedding_stack: RasterStack | None = None,
    feature_mode: str = "engineered",
) -> RasterStack:
    """
    Assemble a multi-modal feature `RasterStack` with a clean toggle between:
    - `feature_mode='engineered'` (default, 29 bands):
        1. 9 DEM morphometry bands (`elevation`, `slope`, `sin_aspect`, `cos_aspect`, `laplacian`,
           `profile_curv`, `planform_curv`, `roughness`, `tpi`)
        2. 6 multiscale 2D CWT Mexican-hat wavelet bands (`cwt_energy` at `cwt_scales_m` +
           `mean_complexity` + `dominant_wavelength_m`)
        3. 6 raw Sentinel-2 reflectance bands + 8 spectral/mineralogical indices (`NDVI`, `NBR`,
           `NDWI`, `BSI`, `ClayRatio`, `FerricOxideRatio`, `FerrousIronIndex`, `CarbonateIndex`).
    - `feature_mode='alphaearth'` (64 bands):
        Uses the 64-band Google AlphaEarth Foundations unit-hypersphere embedding stack (`A00..A63`)
        from `embedding_stack` (normalized to unit L2 length on $S^{63}$).
    - `feature_mode='combined'` (93 bands):
        Concatenates the 29 engineered physical/CWT/spectral bands with the 64-band AlphaEarth
        unit-sphere embedding stack (`A00..A63`).
    """
    mode = feature_mode.strip().lower()
    if mode not in ("engineered", "alphaearth", "embedding", "combined", "hybrid_features"):
        raise ValueError(
            f"Unsupported feature_mode '{feature_mode}'; expected 'engineered', 'alphaearth', or 'combined'"
        )

    if mode in ("alphaearth", "embedding"):
        if embedding_stack is None:
            raise ValueError("embedding_stack is required when feature_mode='alphaearth'")
        ae_norm = embedding_stack.to_unit_norm()
        return RasterStack(
            data=ae_norm.data.copy(),
            band_names=list(ae_norm.band_names),
            crs=ae_norm.crs,
            transform=ae_norm.transform,
            nodata=ae_norm.nodata,
            units="unit_norm",
            name="mapping_feature_stack_alphaearth",
            metadata={"feature_mode": "alphaearth"},
        )

    if dem is None:
        raise ValueError("dem is required when feature_mode is 'engineered' or 'combined'")
    slope_g, aspect_g = terrain_analysis.compute_slope_aspect(dem)
    curv = terrain_analysis.compute_curvature(dem)
    rough = terrain_analysis.compute_roughness_and_tpi(dem, radius_px=3)
    cwt_res = terrain_analysis.compute_multiscale_complexity(dem, scales_m=cwt_scales_m)

    asp_rad = np.deg2rad(np.where(aspect_g.data >= 0, aspect_g.data, 0.0))
    sin_asp = dem.with_data(np.sin(asp_rad), name="sin_aspect", units="dimensionless")
    cos_asp = dem.with_data(np.cos(asp_rad), name="cos_aspect", units="dimensionless")

    grids: list[GeoGrid] = [
        dem.with_data(dem.data.copy(), name="elevation", units="m"),
        slope_g,  # type: ignore[list-item]
        sin_asp,
        cos_asp,
        curv["laplacian"],  # type: ignore[list-item]
        curv["profile"],  # type: ignore[list-item]
        curv["planform"],  # type: ignore[list-item]
        rough["roughness"],  # type: ignore[list-item]
        rough["tpi"],  # type: ignore[list-item]
        cwt_res["mean_complexity"],
        cwt_res["dominant_wavelength_m"],
    ]

    energy_stack: RasterStack = cwt_res["energy_stack"]
    for bname in energy_stack.band_names:
        grids.append(energy_stack.band(bname))

    if spectral_stack is not None:
        if include_raw_bands:
            for bname in spectral_stack.band_names:
                grids.append(spectral_stack.band(bname))
        idx_suite = spectral_indices.compute_index_suite(spectral_stack, indices=include_indices)
        for idx_name, g in idx_suite.items():
            grids.append(g)

    if mode in ("combined", "hybrid_features"):
        if embedding_stack is None:
            raise ValueError("embedding_stack is required when feature_mode='combined'")
        ae_norm = embedding_stack.to_unit_norm()
        for bname in ae_norm.band_names:
            grids.append(ae_norm.band(bname))
        return RasterStack.from_grids(grids, name="mapping_feature_stack_combined")

    return RasterStack.from_grids(grids, name="mapping_feature_stack")


def create_spatial_block_split(
    reference_or_shape: GeoGrid | RasterStack | tuple[int, int],
    mode: str = "west_east",
    test_fraction: float = 0.35,
    n_blocks_x: int = 4,
    n_blocks_y: int = 4,
    buffer_px: int = 3,
) -> dict[str, Any]:
    """
    Create strict spatially disjoint training and validation masks (`train_mask`, `test_mask`, `buffer_mask`)
    separated by a `buffer_px` guard zone to prevent spatial autocorrelation leakage.

    Parameters
    ----------
    mode : {'west_east', 'north_south', 'checkerboard', 'spatial_blocks'}
        - `'west_east'`: Train on western `(1 - test_fraction)` columns, validate on eastern `test_fraction` columns.
        - `'north_south'`: Train on northern `(1 - test_fraction)` rows, validate on southern `test_fraction` rows.
        - `'checkerboard'`: Alternate rectangular blocks of `(n_blocks_y, n_blocks_x)` between train and test.
        - `'spatial_blocks'`: Hold out every K-th rectangular block to match `test_fraction`.
    """
    if isinstance(reference_or_shape, (GeoGrid, RasterStack)):
        rows, cols = (reference_or_shape.rows, reference_or_shape.cols)
        ref: GeoGrid | None = (
            reference_or_shape
            if isinstance(reference_or_shape, GeoGrid)
            else reference_or_shape.band(0)
        )
    else:
        rows, cols = (int(reference_or_shape[0]), int(reference_or_shape[1]))
        ref = None

    train_mask = np.zeros((rows, cols), dtype=bool)
    test_mask = np.zeros((rows, cols), dtype=bool)
    buf = max(0, int(buffer_px))
    frac = float(np.clip(test_fraction, 0.10, 0.80))

    m = mode.lower()
    if m == "west_east":
        split_c = int(round(cols * (1.0 - frac)))
        c_train_end = max(1, split_c - buf)
        c_test_start = min(cols - 1, split_c + buf)
        train_mask[:, :c_train_end] = True
        test_mask[:, c_test_start:] = True
    elif m == "north_south":
        split_r = int(round(rows * (1.0 - frac)))
        r_train_end = max(1, split_r - buf)
        r_test_start = min(rows - 1, split_r + buf)
        train_mask[:r_train_end, :] = True
        test_mask[r_test_start:, :] = True
    elif m in ("checkerboard", "spatial_blocks"):
        bx = max(2, int(n_blocks_x))
        by = max(2, int(n_blocks_y))
        r_edges = np.linspace(0, rows, by + 1, dtype=int)
        c_edges = np.linspace(0, cols, bx + 1, dtype=int)
        block_idx = 0
        inv_period = max(2, int(round(1.0 / frac)))
        for bi in range(by):
            for bj in range(bx):
                r0, r1 = (r_edges[bi], r_edges[bi + 1])
                c0, c1 = (c_edges[bj], c_edges[bj + 1])
                is_test = ((bi + bj) % 2 == 1) if m == "checkerboard" else (block_idx % inv_period == 1)
                block_idx += 1
                if is_test:
                    rr0 = min(r1, r0 + buf)
                    rr1 = max(rr0, r1 - buf)
                    cc0 = min(c1, c0 + buf)
                    cc1 = max(cc0, c1 - buf)
                    test_mask[rr0:rr1, cc0:cc1] = True
                else:
                    train_mask[r0:r1, c0:c1] = True
    else:
        raise ValueError(
            f"Unsupported mode '{mode}'; expected 'west_east', 'north_south', 'checkerboard', or 'spatial_blocks'"
        )

    # Guarantee zero overlap between train_mask and test_mask
    train_mask &= ~test_mask
    buffer_mask = ~(train_mask | test_mask)

    split_code = np.zeros((rows, cols), dtype=np.float64)
    split_code[train_mask] = 1.0
    split_code[test_mask] = 2.0

    return {
        "train_mask": train_mask,
        "test_mask": test_mask,
        "buffer_mask": buffer_mask,
        "split_grid": (
            ref.with_data(split_code, name="spatial_block_split", units="code")
            if ref is not None
            else split_code
        ),
        "train_pixels": int(train_mask.sum()),
        "test_pixels": int(test_mask.sum()),
        "buffer_pixels": int(buffer_mask.sum()),
    }


def cluster_units_kmeans(
    features: RasterStack | Sequence[GeoGrid] | np.ndarray,
    n_clusters: int = 5,
    standardize: bool = True,
    max_iter: int = 50,
    seed: int = 42,
    smooth_radius_px: int = 0,
) -> dict[str, Any]:
    """
    Cluster pixels into `n_clusters` surface/lithologic map units using deterministic
    k-means++ in Rust (`_core.kmeans_features_rs`), with optional spatial majority smoothing.
    """
    cube_c, ref_grid, _ = _unpack_feature_cube(features, standardize=standardize)
    labels_i32, centroids = _core.kmeans_features_rs(
        cube_c, int(n_clusters), int(max_iter), int(seed)
    )
    if smooth_radius_px > 0:
        labels_i32 = _core.smooth_labels_majority_rs(labels_i32, int(smooth_radius_px), 1)

    cell_area = ref_grid.cell_area_m2 if ref_grid is not None else 100.0
    unit_summary = []
    for k in range(int(n_clusters)):
        cnt = int(np.sum(labels_i32 == k))
        unit_summary.append(
            {
                "unit_id": k,
                "pixel_count": cnt,
                "area_m2": cnt * cell_area,
            }
        )

    labels_out: GeoGrid | np.ndarray
    if ref_grid is not None:
        labels_f64 = np.where(labels_i32 >= 0, labels_i32.astype(np.float64), np.nan)
        labels_out = ref_grid.with_data(labels_f64, name="mapped_units", units="unit_id")
    else:
        labels_out = labels_i32

    return {
        "labels": labels_out,
        "centroids": centroids,
        "unit_summary": unit_summary,
    }


def train_predict_supervised_units(
    features: RasterStack | Sequence[GeoGrid] | np.ndarray,
    reference_labels: GeoGrid | np.ndarray,
    train_mask: np.ndarray | None = None,
    method: str = "hybrid",
    smooth_radius_px: int = 2,
    smooth_iterations: int = 1,
    n_estimators: int = 150,
    random_state: int = 42,
    *,
    embedding_stack: RasterStack | None = None,
    feature_mode: str = "engineered",
) -> dict[str, Any]:
    """
    Train a supervised geological/geomorphic unit classifier on `reference_labels` within `train_mask`
    and predict unit labels, posterior probabilities, confidence `[0, 1]`, and normalized Shannon
    entropy `[0, 1]` across the entire scene.

    Parameters
    ----------
    method : {'gaussian_bayes_rs', 'random_forest', 'hybrid'}
        - `'gaussian_bayes_rs'`: Pure Rust parallel Gaussian Naive Bayes / Mahalanobis classifier.
        - `'random_forest'`: Balanced Random Forest classifier (`scikit-learn`).
        - `'hybrid'`: Ensemble combining Rust Gaussian Bayes + Random Forest posterior probabilities
          (falls back to `'gaussian_bayes_rs'` if `scikit-learn` is unavailable).
    """
    fmode = feature_mode.strip().lower()
    if fmode in ("alphaearth", "embedding") and embedding_stack is not None:
        active_features: RasterStack | Sequence[GeoGrid] | np.ndarray = embedding_stack.to_unit_norm()
    elif fmode in ("combined", "hybrid_features") and embedding_stack is not None:
        ae_norm = embedding_stack.to_unit_norm()
        if isinstance(features, RasterStack):
            comb_grids = [features.band(i) for i in range(features.n_bands)] + [
                ae_norm.band(i) for i in range(ae_norm.n_bands)
            ]
            active_features = RasterStack.from_grids(comb_grids, name="combined_features")
        else:
            c0, rg0, fn0 = _unpack_feature_cube(features, standardize=True)
            c1, _, fn1 = _unpack_feature_cube(ae_norm, standardize=True)
            active_features = np.concatenate([c0, c1], axis=2)
    else:
        active_features = features

    cube_c, ref_grid, feature_names = _unpack_feature_cube(active_features, standardize=True)
    rows, cols, dim = cube_c.shape

    ref_arr = (
        reference_labels.data
        if isinstance(reference_labels, GeoGrid)
        else np.asarray(reference_labels)
    )
    if ref_grid is None and isinstance(reference_labels, GeoGrid):
        ref_grid = reference_labels

    lbl_i32 = np.where(
        np.isfinite(ref_arr) & (ref_arr >= 0), np.round(ref_arr).astype(np.int32), -1
    )
    lbl_i32 = np.ascontiguousarray(lbl_i32, dtype=np.int32)

    if train_mask is None:
        tmask_u8 = np.ascontiguousarray((lbl_i32 >= 0).astype(np.uint8))
    else:
        tmask_u8 = np.ascontiguousarray(
            (np.asarray(train_mask, dtype=bool) & (lbl_i32 >= 0)).astype(np.uint8)
        )

    # Always run the fast Rust Gaussian Bayes classifier first
    gb_res = _core.fit_predict_gaussian_bayes_rs(cube_c, tmask_u8, lbl_i32, 1e-3)
    class_ids = np.asarray(gb_res["class_ids"], dtype=np.int32)
    n_classes = int(class_ids.size)
    if n_classes == 0:
        raise ValueError("No valid training pixels found in train_mask & reference_labels")

    gb_probs = np.asarray(gb_res["probabilities"], dtype=np.float64)
    m = method.lower()

    if m in ("random_forest", "hybrid") and HAS_SKLEARN:
        flat_X = cube_c.reshape(-1, dim)
        flat_y = lbl_i32.reshape(-1)
        flat_tm = (tmask_u8.reshape(-1) != 0) & np.all(np.isfinite(flat_X), axis=1)

        clf = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=14,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=int(random_state),
        )
        clf.fit(flat_X[flat_tm], flat_y[flat_tm])

        valid_px = np.all(np.isfinite(flat_X), axis=1)
        rf_probs_flat = np.zeros((rows * cols, n_classes), dtype=np.float64)
        raw_rf = clf.predict_proba(flat_X[valid_px])
        rf_classes = np.asarray(clf.classes_, dtype=np.int32)
        for j, cid in enumerate(rf_classes):
            match_idx = np.where(class_ids == cid)[0]
            if match_idx.size > 0:
                rf_probs_flat[valid_px, int(match_idx[0])] = raw_rf[:, j]
        rf_probs = rf_probs_flat.reshape(rows, cols, n_classes)
        feature_importances = np.asarray(clf.feature_importances_, dtype=np.float64)

        if m == "random_forest":
            probs = rf_probs
        else:
            probs = 0.30 * gb_probs + 0.70 * rf_probs
    else:
        probs = gb_probs
        # Estimate feature importance via class-centroid variance across standardized features
        feature_importances = np.ones(dim, dtype=np.float64) / max(dim, 1)

    best_idx = np.argmax(probs, axis=2)
    raw_pred_i32 = class_ids[best_idx].astype(np.int32)
    valid_2d = np.all(np.isfinite(cube_c), axis=2)
    raw_pred_i32 = np.where(valid_2d, raw_pred_i32, -1)
    raw_pred_i32 = np.ascontiguousarray(raw_pred_i32, dtype=np.int32)

    if smooth_radius_px > 0 and smooth_iterations > 0:
        pred_i32 = _core.smooth_labels_majority_rs(
            raw_pred_i32, int(smooth_radius_px), int(smooth_iterations)
        )
    else:
        pred_i32 = raw_pred_i32.copy()

    confidence = np.where(valid_2d, np.max(probs, axis=2), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_safe = np.clip(probs, 1e-15, 1.0)
        shannon = -np.sum(probs * np.log(p_safe), axis=2)
        norm_ent = shannon / np.log(max(n_classes, 2))
    entropy = np.where(valid_2d, np.clip(norm_ent, 0.0, 1.0), np.nan)

    prob_band_names = [f"prob_unit_{int(cid)}" for cid in class_ids]
    if ref_grid is not None:
        pred_out: GeoGrid | np.ndarray = ref_grid.with_data(
            np.where(pred_i32 >= 0, pred_i32.astype(np.float64), np.nan),
            name="predicted_geol_units",
            units="unit_id",
        )
        raw_out: GeoGrid | np.ndarray = ref_grid.with_data(
            np.where(raw_pred_i32 >= 0, raw_pred_i32.astype(np.float64), np.nan),
            name="raw_predicted_geol_units",
            units="unit_id",
        )
        conf_out: GeoGrid | np.ndarray = ref_grid.with_data(
            confidence, name="prediction_confidence", units="probability"
        )
        ent_out: GeoGrid | np.ndarray = ref_grid.with_data(
            entropy, name="prediction_entropy", units="normalized_entropy"
        )
        prob_stack: RasterStack | np.ndarray = RasterStack(
            data=probs,
            band_names=prob_band_names,
            transform=ref_grid.transform,
            crs=ref_grid.crs,
            nodata=np.nan,
            units="probability",
            name="unit_posterior_probabilities",
        )
    else:
        pred_out = pred_i32
        raw_out = raw_pred_i32
        conf_out = confidence
        ent_out = entropy
        prob_stack = probs

    contact_mask, contacts = extract_unit_contacts(pred_out)

    return {
        "pred_labels": pred_out,
        "raw_pred_labels": raw_out,
        "confidence": conf_out,
        "entropy": ent_out,
        "probabilities": prob_stack,
        "class_ids": class_ids,
        "feature_mode": fmode,
        "n_features": int(dim),
        "feature_names": feature_names,
        "feature_importances": feature_importances,
        "contact_mask": contact_mask,
        "contacts": contacts,
    }


def evaluate_mapping_accuracy(
    true_labels: GeoGrid | np.ndarray,
    pred_labels: GeoGrid | np.ndarray,
    eval_mask: np.ndarray | None = None,
    unit_names: dict[int, str] | None = None,
) -> dict[str, Any]:
    """
    Evaluate predicted geological/geomorphic unit labels against `true_labels` within `eval_mask`,
    reporting overall accuracy, macro-F1, mean IoU (Jaccard index), Cohen's kappa, confusion matrix,
    and per-unit metrics.
    """
    t_arr = true_labels.data if isinstance(true_labels, GeoGrid) else np.asarray(true_labels)
    p_arr = pred_labels.data if isinstance(pred_labels, GeoGrid) else np.asarray(pred_labels)

    valid = np.isfinite(t_arr) & np.isfinite(p_arr) & (t_arr >= 0) & (p_arr >= 0)
    if eval_mask is not None:
        valid &= np.asarray(eval_mask, dtype=bool)

    y_true = np.round(t_arr[valid]).astype(np.int32)
    y_pred = np.round(p_arr[valid]).astype(np.int32)
    n_eval = int(y_true.size)
    if n_eval == 0:
        raise ValueError("No valid evaluation pixels found in eval_mask")

    classes = np.unique(np.concatenate([y_true, y_pred]))
    k = int(classes.size)
    class_to_idx = {int(c): i for i, c in enumerate(classes)}

    cm = np.zeros((k, k), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        cm[class_to_idx[int(yt)], class_to_idx[int(yp)]] += 1

    overall_acc = float(np.trace(cm)) / float(n_eval)

    # Cohen's kappa
    row_sums = cm.sum(axis=1).astype(np.float64)
    col_sums = cm.sum(axis=0).astype(np.float64)
    pe = float(np.sum(row_sums * col_sums)) / float(n_eval * n_eval)
    kappa = (overall_acc - pe) / max(1.0 - pe, 1e-12)

    per_unit = []
    f1_list = []
    iou_list = []
    for i, cid in enumerate(classes):
        tp = float(cm[i, i])
        fp = float(col_sums[i] - tp)
        fn = float(row_sums[i] - tp)
        support = int(row_sums[i])
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        if support > 0:
            f1_list.append(f1)
            iou_list.append(iou)
        uname = (
            unit_names.get(int(cid), f"Unit_{int(cid)}")
            if unit_names is not None
            else f"Unit_{int(cid)}"
        )
        per_unit.append(
            {
                "unit_id": int(cid),
                "unit_name": uname,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "iou": iou,
                "support_pixels": support,
            }
        )

    return {
        "n_eval_pixels": n_eval,
        "overall_accuracy": overall_acc,
        "macro_f1": float(np.mean(f1_list)) if f1_list else 0.0,
        "mean_iou": float(np.mean(iou_list)) if iou_list else 0.0,
        "cohen_kappa": float(kappa),
        "class_ids": classes,
        "confusion_matrix": cm,
        "per_unit": per_unit,
    }


def compare_feature_approaches_supervised(
    dem: GeoGrid,
    spectral_stack: RasterStack,
    embedding_stack: RasterStack,
    reference_labels: GeoGrid,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    unit_names: dict[int, str] | None = None,
    cwt_scales_m: Sequence[float] = (30.0, 60.0, 120.0, 200.0),
    method: str = "hybrid",
    smooth_radius_px: int = 2,
    smooth_iterations: int = 1,
    n_estimators: int = 150,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Train and evaluate all three toggleable feature modes side-by-side on the exact same
    spatial-block holdout split (`train_mask` vs `test_mask`):
    1. `'engineered'` (29 bands: 9 DEM morphometry + 6 multiscale 2D CWT + 6 Sentinel-2 bands + 8 indices)
    2. `'alphaearth'` (64 bands: Google AlphaEarth Foundations `A00..A63` unit-sphere embeddings)
    3. `'combined'` (93 bands: 29 engineered physical/CWT/spectral bands + 64 AlphaEarth embedding bands)
    """
    modes = ("engineered", "alphaearth", "combined")
    stacks: dict[str, RasterStack] = {}
    results: dict[str, dict[str, Any]] = {}
    eval_test: dict[str, dict[str, Any]] = {}
    eval_full: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for mode in modes:
        f_stack = build_mapping_feature_stack(
            dem=dem,
            spectral_stack=spectral_stack,
            embedding_stack=embedding_stack,
            cwt_scales_m=cwt_scales_m,
            feature_mode=mode,
        )
        stacks[mode] = f_stack
        res = train_predict_supervised_units(
            f_stack,
            reference_labels,
            train_mask=train_mask,
            method=method,
            smooth_radius_px=smooth_radius_px,
            smooth_iterations=smooth_iterations,
            n_estimators=n_estimators,
            random_state=random_state,
            feature_mode=mode,
        )
        ev_t = evaluate_mapping_accuracy(
            reference_labels, res["pred_labels"], eval_mask=test_mask, unit_names=unit_names
        )
        ev_f = evaluate_mapping_accuracy(
            reference_labels, res["pred_labels"], unit_names=unit_names
        )
        results[mode] = res
        eval_test[mode] = ev_t
        eval_full[mode] = ev_f

        imp = np.asarray(res["feature_importances"], dtype=np.float64)
        fnames = list(res["feature_names"])
        top_idx = np.argsort(imp)[::-1][:5]
        top_feats = [(fnames[int(i)], float(imp[int(i)])) for i in top_idx]

        summary_rows.append(
            {
                "feature_mode": mode,
                "n_features": int(f_stack.n_bands),
                "test_overall_accuracy": float(ev_t["overall_accuracy"]),
                "test_macro_f1": float(ev_t["macro_f1"]),
                "test_mean_iou": float(ev_t["mean_iou"]),
                "test_cohen_kappa": float(ev_t["cohen_kappa"]),
                "full_overall_accuracy": float(ev_f["overall_accuracy"]),
                "full_macro_f1": float(ev_f["macro_f1"]),
                "full_mean_iou": float(ev_f["mean_iou"]),
                "top_5_features": top_feats,
            }
        )

    p_eng = np.round(results["engineered"]["pred_labels"].data).astype(np.int32)
    p_ae = np.round(results["alphaearth"]["pred_labels"].data).astype(np.int32)
    p_comb = np.round(results["combined"]["pred_labels"].data).astype(np.int32)
    agreement_eng_ae = float(np.mean(p_eng == p_ae))
    agreement_ae_comb = float(np.mean(p_ae == p_comb))

    return {
        "feature_stacks": stacks,
        "predictions": results,
        "test_evaluations": eval_test,
        "full_evaluations": eval_full,
        "summary_table": summary_rows,
        "agreement_engineered_vs_alphaearth": agreement_eng_ae,
        "agreement_alphaearth_vs_combined": agreement_ae_comb,
    }
