# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Bi-temporal surface change detection in Rust + Python:
1. DEM of Difference (DoD: vertical elevation change, planar tilt detrending, Nuth & Kääb shift check,
   and erosion/deposition volumetric integration with uncertainty propagation).
2. Spectral & radar change indices (dNBR, dNDVI, dNDWI, Otsu thresholding, USGS severity classification,
   and 8-connected change patch morphometrics).
3. Multi-band embedding change (AlphaEarth Foundations 64-band cosine distance, cosine similarity,
   L2 distance, dominant latent dimension importance, and 1st principal change projection).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import _core, terrain_analysis
from .raster_io import GeoGrid, RasterStack


def compute_dod(
    dem_t1: GeoGrid | np.ndarray,
    dem_t2: GeoGrid | np.ndarray,
    detrend_stable_mask: np.ndarray | None = None,
    nodata: float = np.nan,
) -> dict[str, Any]:
    """
    Compute DEM of Difference (`DoD = dem_t2 - dem_t1`) in Rust, with optional planar tilt
    detrending and Nuth & Kääb (2011) co-registration shift check over stable terrain.

    Returns
    -------
    dict with keys:
        - `'dod'`: `GeoGrid` or `np.ndarray` of (detrended if requested) vertical change `z2 - z1`
        - `'raw_dod'`: `GeoGrid` or `np.ndarray` before planar detrending
        - `'planar_bias_coeffs'`: `(c0, cx, cy)` plane coefficients removed (0 if not detrended)
        - `'nuth_kaab_shift_m'`: `(dx_shift_m, dy_shift_m, dz_shift_m)` estimated on stable terrain
    """
    if isinstance(dem_t1, GeoGrid) and isinstance(dem_t2, GeoGrid):
        a1 = np.ascontiguousarray(dem_t1.data, dtype=np.float64)
        a2 = np.ascontiguousarray(dem_t2.data, dtype=np.float64)
        ref = dem_t1
        dx = abs(ref.dx)
        dy = abs(ref.dy)
    else:
        a1 = np.ascontiguousarray(np.asarray(dem_t1, dtype=np.float64))
        a2 = np.ascontiguousarray(np.asarray(dem_t2, dtype=np.float64))
        ref = dem_t1 if isinstance(dem_t1, GeoGrid) else None
        dx, dy = (10.0, 10.0)

    raw_arr = _core.compute_dod_rs(a1, a2, float(nodata))

    if detrend_stable_mask is not None:
        sm = np.ascontiguousarray(detrend_stable_mask.astype(np.uint8))
        det_arr, c0, cx, cy = _core.detrend_dod_planar_rs(raw_arr, sm, float(dx), float(dy))
        slope_g, aspect_g = terrain_analysis.compute_slope_aspect(a1, dx=dx, dy=dy)
        nk_shift = _core.estimate_nuth_kaab_shift_rs(
            det_arr,
            np.ascontiguousarray(slope_g, dtype=np.float64),
            np.ascontiguousarray(aspect_g, dtype=np.float64),
            sm,
        )
    else:
        det_arr = raw_arr.copy()
        c0, cx, cy = (0.0, 0.0, 0.0)
        nk_shift = (0.0, 0.0, 0.0)

    if ref is not None:
        dod_out: GeoGrid | np.ndarray = ref.with_data(det_arr, name="DoD", units="m")
        raw_out: GeoGrid | np.ndarray = ref.with_data(raw_arr, name="DoD_raw", units="m")
    else:
        dod_out = det_arr
        raw_out = raw_arr

    return {
        "dod": dod_out,
        "raw_dod": raw_out,
        "planar_bias_coeffs": (float(c0), float(cx), float(cy)),
        "nuth_kaab_shift_m": tuple(float(v) for v in nk_shift),
    }


def compute_volume_budget(
    dod: GeoGrid | np.ndarray,
    cell_area_m2: float | None = None,
    lod_threshold: float = 0.25,
    sigma_dz: float = 0.15,
) -> dict[str, Any]:
    """
    Integrate eroded and deposited area (m^2) and volume (m^3) above level-of-detection `lod_threshold` (m),
    propagating vertical error `sigma_dz` (m) via Rust.
    """
    if isinstance(dod, GeoGrid):
        arr = np.ascontiguousarray(dod.data, dtype=np.float64)
        area = float(cell_area_m2) if cell_area_m2 is not None else dod.cell_area_m2
    else:
        arr = np.ascontiguousarray(np.asarray(dod, dtype=np.float64))
        area = float(cell_area_m2) if cell_area_m2 is not None else 100.0

    return dict(_core.compute_volume_budget_rs(arr, area, float(lod_threshold), float(sigma_dz)))


def compute_index_change(
    index_t1: GeoGrid | np.ndarray,
    index_t2: GeoGrid | np.ndarray,
    convention: str = "t2_minus_t1",
    name: str | None = None,
) -> GeoGrid | np.ndarray:
    """
    Compute bi-temporal index difference.

    Parameters
    ----------
    convention : {'t2_minus_t1', 't1_minus_t2'}
        Use `'t2_minus_t1'` for standard differences (e.g. `dNDVI = NDVI_t2 - NDVI_t1`)
        or `'t1_minus_t2'` for disturbance indices where loss is positive (e.g. USGS `dNBR = NBR_pre - NBR_post`).
    """
    if isinstance(index_t1, GeoGrid):
        a1 = np.ascontiguousarray(index_t1.data, dtype=np.float64)
        ref: GeoGrid | None = index_t1
    else:
        a1 = np.ascontiguousarray(np.asarray(index_t1, dtype=np.float64))
        ref = None

    if isinstance(index_t2, GeoGrid):
        a2 = np.ascontiguousarray(index_t2.data, dtype=np.float64)
        if ref is None:
            ref = index_t2
    else:
        a2 = np.ascontiguousarray(np.asarray(index_t2, dtype=np.float64))

    if convention.lower() == "t1_minus_t2":
        diff = _core.compute_dod_rs(a2, a1, np.nan)
    elif convention.lower() == "t2_minus_t1":
        diff = _core.compute_dod_rs(a1, a2, np.nan)
    else:
        raise ValueError(f"Unknown convention '{convention}'; expected 't2_minus_t1' or 't1_minus_t2'")

    out_name = name or (f"d{ref.name}" if ref is not None else "index_change")
    if ref is not None:
        return ref.with_data(diff, name=out_name, units=ref.units)
    return diff


def otsu_threshold(
    values: GeoGrid | np.ndarray,
    n_bins: int = 256,
) -> float:
    """Compute Otsu threshold over finite pixels in Rust."""
    arr = values.data if isinstance(values, GeoGrid) else np.asarray(values, dtype=np.float64)
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return float(_core.otsu_threshold_rs(arr, int(n_bins)))


def classify_change_severity(
    values: GeoGrid | np.ndarray,
    thresholds: Sequence[float] = (0.10, 0.27, 0.44, 0.66),
    name: str = "severity_class",
) -> GeoGrid | np.ndarray:
    """
    Classify continuous change values into discrete severity classes `0..len(thresholds)` via Rust.
    Default thresholds `(0.10, 0.27, 0.44, 0.66)` correspond to USGS dNBR burn severity tiers.
    """
    ref = values if isinstance(values, GeoGrid) else None
    arr = values.data if isinstance(values, GeoGrid) else np.asarray(values, dtype=np.float64)
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    thr = np.ascontiguousarray(np.asarray(thresholds, dtype=np.float64))
    cls_u8 = _core.classify_change_severity_rs(arr, thr)
    cls_f64 = np.where(cls_u8 == 255, np.nan, cls_u8.astype(np.float64))
    if ref is not None:
        return ref.with_data(cls_f64, name=name, units="class")
    return cls_u8


def extract_change_patches(
    change_or_mask: GeoGrid | np.ndarray,
    magnitude: GeoGrid | np.ndarray | None = None,
    threshold: float | str | None = "otsu",
    min_pixels: int = 6,
) -> tuple[GeoGrid | np.ndarray, list[dict[str, Any]]]:
    """
    Extract 8-connected change patches and per-patch morphometric statistics in Rust.
    """
    ref = change_or_mask if isinstance(change_or_mask, GeoGrid) else None
    arr = (
        change_or_mask.data
        if isinstance(change_or_mask, GeoGrid)
        else np.asarray(change_or_mask)
    )

    if arr.dtype == bool or (threshold is None and np.all(np.isin(np.nan_to_num(arr), [0, 1]))):
        mask_u8 = np.ascontiguousarray((arr > 0).astype(np.uint8))
        mag_arr = (
            magnitude.data
            if isinstance(magnitude, GeoGrid)
            else (
                np.asarray(magnitude, dtype=np.float64)
                if magnitude is not None
                else arr.astype(np.float64)
            )
        )
    else:
        arr_f64 = np.ascontiguousarray(arr, dtype=np.float64)
        if threshold == "otsu":
            thr_val = otsu_threshold(arr_f64)
        elif isinstance(threshold, (int, float)):
            thr_val = float(threshold)
        else:
            thr_val = otsu_threshold(arr_f64)
        mask_u8 = np.ascontiguousarray(
            (np.isfinite(arr_f64) & (arr_f64 >= thr_val)).astype(np.uint8)
        )
        mag_arr = (
            magnitude.data
            if isinstance(magnitude, GeoGrid)
            else (
                np.asarray(magnitude, dtype=np.float64)
                if magnitude is not None
                else arr_f64
            )
        )

    mag_f64 = np.ascontiguousarray(mag_arr, dtype=np.float64)
    if ref is not None:
        x0, y0, dx, dy = (ref.x0, ref.y0, ref.dx, ref.dy)
    else:
        x0, y0, dx, dy = (0.0, 0.0, 10.0, -10.0)

    labels_i32, stats = _core.connected_change_patches_rs(
        mask_u8, mag_f64, float(x0), float(y0), float(dx), float(dy), int(min_pixels)
    )
    if ref is not None:
        return (
            ref.with_data(labels_i32.astype(np.float64), name="change_patches", units="patch_id"),
            list(stats),
        )
    return labels_i32, list(stats)


def compute_embedding_change(
    emb_t1: RasterStack | np.ndarray,
    emb_t2: RasterStack | np.ndarray,
    threshold: float | str = "otsu",
    min_patch_pixels: int = 6,
    nodata: float = np.nan,
) -> dict[str, Any]:
    """
    Compute pixel-wise cosine distance, cosine similarity, Euclidean L2 distance,
    per-dimension RMS change ranking, and 1st dominant change component across
    D-dimensional embeddings (e.g., 64-band Google DeepMind AlphaEarth Foundations) in Rust.
    """
    ref_stack = emb_t1 if isinstance(emb_t1, RasterStack) else None
    arr1 = emb_t1.data if isinstance(emb_t1, RasterStack) else np.asarray(emb_t1, dtype=np.float64)
    arr2 = emb_t2.data if isinstance(emb_t2, RasterStack) else np.asarray(emb_t2, dtype=np.float64)

    arr1 = np.ascontiguousarray(arr1, dtype=np.float64)
    arr2 = np.ascontiguousarray(arr2, dtype=np.float64)
    if arr1.shape != arr2.shape or arr1.ndim != 3:
        raise ValueError(
            f"Embedding arrays must have identical 3D shape (rows, cols, dim); got {arr1.shape} vs {arr2.shape}"
        )

    cos_dist, cos_sim, l2_dist = _core.embedding_change_metrics_rs(arr1, arr2, float(nodata))
    ranked_dims, dim_rms, dim_mean_diff, pc1_proj = _core.embedding_dimension_importance_rs(
        arr1, arr2, float(nodata)
    )

    if threshold == "otsu":
        thr_val = float(_core.otsu_threshold_rs(cos_dist, 256))
    else:
        thr_val = float(threshold)

    mask_u8 = np.ascontiguousarray(
        (np.isfinite(cos_dist) & (cos_dist >= thr_val)).astype(np.uint8)
    )

    if ref_stack is not None:
        ref_grid = ref_stack.band(0)
        cd_out: GeoGrid | np.ndarray = ref_grid.with_data(
            cos_dist, name="embedding_cosine_distance", units="cosine_distance"
        )
        cs_out: GeoGrid | np.ndarray = ref_grid.with_data(
            cos_sim, name="embedding_cosine_similarity", units="cosine_similarity"
        )
        l2_out: GeoGrid | np.ndarray = ref_grid.with_data(
            l2_dist, name="embedding_l2_distance", units="l2_distance"
        )
        pc1_out: GeoGrid | np.ndarray = ref_grid.with_data(
            pc1_proj, name="embedding_pc1_change", units="embedding_proj"
        )
        mask_out: GeoGrid | np.ndarray = ref_grid.with_data(
            mask_u8.astype(np.float64), name="embedding_change_mask", units="binary"
        )
        patch_labels, patches = extract_change_patches(
            mask_out, magnitude=cd_out, threshold=None, min_pixels=min_patch_pixels
        )
        dim_names = [ref_stack.band_names[int(d)] for d in ranked_dims]
    else:
        cd_out = cos_dist
        cs_out = cos_sim
        l2_out = l2_dist
        pc1_out = pc1_proj
        mask_out = mask_u8
        patch_labels, patches = extract_change_patches(
            mask_u8, magnitude=cos_dist, threshold=None, min_pixels=min_patch_pixels
        )
        dim_names = [f"A{int(d):02d}" for d in ranked_dims]

    return {
        "cosine_distance": cd_out,
        "cosine_similarity": cs_out,
        "l2_distance": l2_out,
        "pc1_change": pc1_out,
        "otsu_threshold": thr_val,
        "change_mask": mask_out,
        "patch_labels": patch_labels,
        "patches": patches,
        "top_changed_dims": np.asarray(ranked_dims, dtype=np.int64),
        "top_changed_band_names": dim_names,
        "dim_rms_change": np.asarray(dim_rms, dtype=np.float64),
        "dim_mean_diff": np.asarray(dim_mean_diff, dtype=np.float64),
    }


def compute_displacement_strain(
    displacement: GeoGrid | np.ndarray,
    dx: float = 1.0,
    dy: float = -1.0,
    nodata: float = np.nan,
) -> dict[str, Any]:
    """
    Compute 2D spatial displacement gradients (`du/dx`, `du/dy`), strain magnitude `|grad u|` (m/m),
    microstrain (`1e6 * |grad u|`), fault-scarp Laplacian curvature (`d2u/dx2 + d2u/dy2`, m^-1),
    and steepest-step azimuth (`[0, 360)` deg) in Rust (`_core.compute_displacement_strain_rs`).

    Parameters
    ----------
    displacement : GeoGrid or 2D np.ndarray
        Coseismic InSAR line-of-sight (LOS), vertical, or fault-parallel surface displacement field (m).
    dx, dy : float
        Pixel spacing in meters (ignored when `displacement` is a `GeoGrid`).
    nodata : float
        Sentinel nodata value (`np.nan` by default).

    Returns
    -------
    dict with keys:
        `'du_dx'`, `'du_dy'`, `'gradient_mag'`, `'microstrain'`, `'scarp_curvature'`,
        `'gradient_azimuth_deg'`, `'p99_strain_m_per_m'`, `'max_strain_m_per_m'`.
        Spatial rasters are returned as `GeoGrid` when `displacement` is a `GeoGrid`.
    """
    if isinstance(displacement, GeoGrid):
        arr = np.ascontiguousarray(displacement.data, dtype=np.float64)
        ref = displacement
        dx_val = float(ref.dx)
        dy_val = float(ref.dy)
    else:
        arr = np.ascontiguousarray(displacement, dtype=np.float64)
        ref = None
        dx_val = float(dx)
        dy_val = float(dy)

    raw = _core.compute_displacement_strain_rs(arr, dx_val, dy_val, float(nodata))
    gmag = np.asarray(raw["gradient_mag"], dtype=np.float64)
    finite_g = gmag[np.isfinite(gmag)]
    p99 = float(np.percentile(finite_g, 99.0)) if finite_g.size > 0 else 0.0
    gmax = float(np.max(finite_g)) if finite_g.size > 0 else 0.0

    def _wrap(a: np.ndarray, name: str) -> GeoGrid | np.ndarray:
        a64 = np.asarray(a, dtype=np.float64)
        return ref.with_data(a64, name=name) if ref is not None else a64

    return {
        "du_dx": _wrap(raw["du_dx"], "du_dx"),
        "du_dy": _wrap(raw["du_dy"], "du_dy"),
        "gradient_mag": _wrap(gmag, "displacement_strain_mag"),
        "microstrain": _wrap(raw["microstrain"], "microstrain"),
        "scarp_curvature": _wrap(raw["scarp_curvature"], "scarp_curvature"),
        "gradient_azimuth_deg": _wrap(raw["gradient_azimuth_deg"], "gradient_azimuth_deg"),
        "p99_strain_m_per_m": p99,
        "max_strain_m_per_m": gmax,
    }




def compute_los_from_unwrapped_phase(
    phase_rad: GeoGrid | np.ndarray,
    wavelength_m: float,
    *,
    phase_reference: float | None = None,
) -> GeoGrid | np.ndarray:
    """Convert already-unwrapped interferometric phase (radians) to LOS metres.

    ``u_los = (phase_rad - phase_reference) * wavelength_m / (4 * pi)``.
    ``phase_reference=None`` means zero; callers may supply a constant reference
    in radians, for example the median phase over independently selected stable
    terrain. No atmospheric correction or spatial ramp detrending is applied.

    For the UAVSAR ``SLC1 * conj(SLC2)`` convention, positive LOS means range
    increase (motion away from the radar). Other input phase conventions must
    be reconciled by the caller. One 2*pi cycle corresponds to wavelength_m/2
    metres; wrapped phase alone cannot recover the integer-cycle displacement.
    This converts a processed phase product, not raw SLCs into an interferogram.

    ``wavelength_m`` must be finite and positive, and the reference finite.
    NaN/nonfinite input cells (and a GeoGrid's nodata cells) become NaN. GeoGrid
    output retains georeferencing and metadata with units ``m``; ndarray input
    returns a float64 ndarray. Inputs are not modified. The result can be
    rewrapped with :func:`compute_insar_wrapped_phase` using the same wavelength.
    """
    wavelength = float(wavelength_m)
    if not np.isfinite(wavelength) or wavelength <= 0:
        raise ValueError("wavelength_m must be finite and positive")
    reference = 0.0 if phase_reference is None else float(phase_reference)
    if not np.isfinite(reference):
        raise ValueError("phase_reference must be finite or None")

    grid = phase_rad if isinstance(phase_rad, GeoGrid) else None
    arr = np.asarray(grid.data if grid is not None else phase_rad, dtype=np.float64)
    valid = grid.valid_mask() if grid is not None else np.isfinite(arr)
    los = np.where(valid, (arr - reference) * (wavelength / (4 * np.pi)), np.nan)
    if grid is not None:
        return grid.with_data(los, name="insar_los_displacement_m", units="m", nodata=np.nan)
    return los


def compute_insar_wrapped_phase(
    los_disp_m: GeoGrid | np.ndarray,
    wavelength_m: float = 0.05546576,
    nodata: float = np.nan,
) -> GeoGrid | np.ndarray:
    """
    Compute wrapped InSAR interferometric phase `dphi in [-pi, pi)` radians from a Line-of-Sight
    (LOS) displacement field `los_disp_m` (m) and radar wavelength `wavelength_m` (m) in Rust
    (`_core.compute_insar_wrapped_phase_rs`):

        dphi = wrap((4 * pi / wavelength_m) * u_los)

    For Sentinel-1 C-band SAR (`wavelength_m = 0.05546576` m = 5.55 cm), each `2*pi` color fringe
    cycle represents exactly `wavelength_m / 2 = 2.773` cm of ground-to-satellite range change.
    """
    ref = los_disp_m if isinstance(los_disp_m, GeoGrid) else None
    arr = np.ascontiguousarray(
        los_disp_m.data if isinstance(los_disp_m, GeoGrid) else los_disp_m,
        dtype=np.float64,
    )
    phase = _core.compute_insar_wrapped_phase_rs(arr, float(wavelength_m), float(nodata))
    phase_f64 = np.asarray(phase, dtype=np.float64)
    if ref is not None:
        return ref.with_data(phase_f64, name="insar_wrapped_phase_rad", units="rad")
    return phase_f64


def decompose_insar_gnss_3d(
    los_asc: GeoGrid | np.ndarray,
    los_desc: GeoGrid | np.ndarray,
    gnss_e: GeoGrid | np.ndarray | None = None,
    gnss_n: GeoGrid | np.ndarray | None = None,
    gnss_u: GeoGrid | np.ndarray | None = None,
    s_asc: tuple[float, float, float] = (-0.608, -0.112, 0.786),
    s_desc: tuple[float, float, float] = (0.618, -0.114, 0.778),
    gnss_weight_en: float = 0.25,
    gnss_weight_u: float = 0.10,
    nodata: float = np.nan,
) -> dict[str, Any]:
    """
    Invert Ascending (`los_asc`) and Descending (`los_desc`) InSAR Line-of-Sight (LOS) displacement
    fields together with continuous 3-component GNSS displacement priors (`gnss_e`, `gnss_n`, `gnss_u`)
    into a full 3D coseismic surface displacement vector field `(u_east, u_north, u_up)` and
    horizontal slip magnitude `horiz_mag = sqrt(u_east^2 + u_north^2)` in Rust (`_core.decompose_insar_3d_rs`).

    Governing Equations
    -------------------
    Each satellite radar track measures a 1D projection of the true 3D ground displacement vector
    `u = (u_E, u_N, u_U)^T` along its unit ground-to-satellite look vector `s = (s_E, s_N, s_U)`:

        u_LOS = (-sin(theta)*cos(alpha)) * u_E + (sin(theta)*sin(alpha)) * u_N + cos(theta) * u_U

    Because near-polar SAR satellites fly nearly North-South (`alpha approx -10 deg` or `-170 deg`),
    Ascending + Descending InSAR strongly resolves East-West (`u_E`) and Vertical (`u_U`) motion but is
    weakly sensitive to North-South (`u_N`) motion. Fusing interpolated GNSS station vectors `(u_E^GNSS,
    u_N^GNSS, u_U^GNSS)` via regularized weighted least squares `(A^T W A) u = A^T W d` recovers the
    complete 3D displacement field at full InSAR pixel resolution.
    """
    ref = los_asc if isinstance(los_asc, GeoGrid) else (los_desc if isinstance(los_desc, GeoGrid) else None)
    asc_arr = np.ascontiguousarray(
        los_asc.data if isinstance(los_asc, GeoGrid) else los_asc,
        dtype=np.float64,
    )
    desc_arr = np.ascontiguousarray(
        los_desc.data if isinstance(los_desc, GeoGrid) else los_desc,
        dtype=np.float64,
    )
    shape = asc_arr.shape
    ge_arr = _as_2d_f64(gnss_e, shape)
    gn_arr = _as_2d_f64(gnss_n, shape)
    gu_arr = _as_2d_f64(gnss_u, shape)

    res = _core.decompose_insar_3d_rs(
        asc_arr,
        desc_arr,
        ge_arr,
        gn_arr,
        gu_arr,
        [float(s_asc[0]), float(s_asc[1]), float(s_asc[2])],
        [float(s_desc[0]), float(s_desc[1]), float(s_desc[2])],
        float(gnss_weight_en),
        float(gnss_weight_u),
        float(nodata),
    )

    def _wrap(a: np.ndarray, name: str) -> GeoGrid | np.ndarray:
        a64 = np.asarray(a, dtype=np.float64)
        return ref.with_data(a64, name=name, units="m") if ref is not None else a64

    hm = np.asarray(res["horiz_mag"], dtype=np.float64)
    finite_hm = hm[np.isfinite(hm)]
    return {
        "u_east": _wrap(res["u_east"], "u_east_m"),
        "u_north": _wrap(res["u_north"], "u_north_m"),
        "u_up": _wrap(res["u_up"], "u_up_m"),
        "horiz_mag": _wrap(hm, "u_horiz_mag_m"),
        "max_horiz_slip_m": float(np.max(finite_hm)) if finite_hm.size > 0 else 0.0,
        "p95_horiz_slip_m": float(np.percentile(finite_hm, 95.0)) if finite_hm.size > 0 else 0.0,
        "s_asc": tuple(float(v) for v in s_asc),
        "s_desc": tuple(float(v) for v in s_desc),
    }


def compute_debris_flow_hazard_index(
    dnbr: GeoGrid | np.ndarray,
    slope_deg: GeoGrid | np.ndarray,
    d_ndmi: GeoGrid | np.ndarray | None = None,
    dnbr_scale: float = 0.65,
    slope_scale_deg: float = 32.0,
) -> GeoGrid | np.ndarray:
    """
    Compute a dimensionless `[0, 1]` post-wildfire steep-slope debris-flow hazard susceptibility
    index combining burn severity (`dNBR = NBR_pre - NBR_post`), topographic slope `sin(slope)`,
    and optional canopy/soil moisture loss (`-dNDMI`), following the post-fire debris-flow
    susceptibility factors of the M1 model of [Staley et al. (2017)](https://doi.org/10.1016/j.geomorph.2016.10.019).
    """
    ref = dnbr if isinstance(dnbr, GeoGrid) else (slope_deg if isinstance(slope_deg, GeoGrid) else None)
    b = np.asarray(dnbr.data if isinstance(dnbr, GeoGrid) else dnbr, dtype=np.float64)
    s = np.asarray(slope_deg.data if isinstance(slope_deg, GeoGrid) else slope_deg, dtype=np.float64)

    burn_term = np.clip(b / max(dnbr_scale, 1e-6), 0.0, 1.0)
    slope_term = np.clip(np.sin(np.deg2rad(np.clip(s, 0.0, 89.0))) / np.sin(np.deg2rad(slope_scale_deg)), 0.0, 1.0)
    if d_ndmi is not None:
        m = np.asarray(d_ndmi.data if isinstance(d_ndmi, GeoGrid) else d_ndmi, dtype=np.float64)
        dry_term = np.clip((-m) / 0.35, 0.0, 1.0)
        idx = np.clip(0.50 * burn_term * slope_term + 0.30 * burn_term + 0.20 * dry_term * slope_term, 0.0, 1.0)
    else:
        idx = np.clip(0.65 * burn_term * slope_term + 0.35 * burn_term, 0.0, 1.0)

    return ref.with_data(idx, name="post_fire_debris_flow_hazard") if ref is not None else idx


FINGERPRINT_AXES: tuple[str, ...] = (
    "Tectonic Strain (|∇u|)",
    "Mass-Wasting Volume (DoD)",
    "Flood Inundation (ΔNDWI)",
    "Wildfire Burn (ΔNBR)",
    "Agri/Urban Expansion (ΔNDBI)",
    "CWT Roughness Shift (ΔCWT)",
)


def _as_2d_f64(arr: GeoGrid | np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if arr is None:
        return np.zeros(shape, dtype=np.float64)
    a = np.asarray(arr.data if isinstance(arr, GeoGrid) else arr, dtype=np.float64)
    if a.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {a.shape}")
    return np.ascontiguousarray(a, dtype=np.float64)


def compute_multi_hazard_signature(
    dz_or_disp: GeoGrid | np.ndarray,
    *,
    strain_mag: GeoGrid | np.ndarray | None = None,
    d_ndvi: GeoGrid | np.ndarray | None = None,
    d_nbr: GeoGrid | np.ndarray | None = None,
    d_ndwi: GeoGrid | np.ndarray | None = None,
    d_ndmi: GeoGrid | np.ndarray | None = None,
    d_ndbi: GeoGrid | np.ndarray | None = None,
    d_bsi: GeoGrid | np.ndarray | None = None,
    d_cwt: GeoGrid | np.ndarray | None = None,
    emb_cos_dist: GeoGrid | np.ndarray | None = None,
    cell_area_m2: float | None = None,
    lod_m: float = 0.50,
    strain_threshold: float = 0.012,
    event_name: str = "study_site",
) -> dict[str, Any]:
    """
    Compute physical change metrics and a standardized 6-axis dimensionless `[0, 1]`
    process fingerprint in Rust (`_core.compute_multi_hazard_signature_rs`).

    Parameters
    ----------
    dz_or_disp : GeoGrid or 2D np.ndarray
        Vertical elevation change `DoD = z_post - z_pre` (m) or coseismic surface displacement `u` (m).
    strain_mag : GeoGrid or 2D np.ndarray, optional
        Spatial gradient magnitude `|grad u|` (m/m). If omitted, computed automatically from `dz_or_disp`.
    d_ndvi, d_nbr, d_ndwi, d_ndmi, d_ndbi, d_bsi, d_cwt, emb_cos_dist : GeoGrid or 2D np.ndarray, optional
        Co-registered spectral, moisture, built-up, 2D CWT complexity, and embedding distance rasters.
    cell_area_m2 : float, optional
        Pixel area in m^2 (inferred from `dz_or_disp` when it is a `GeoGrid`, else defaults to `400.0`).
    lod_m : float
        Minimum elevation change threshold (Level of Detection, m) for volumetric integration.
    strain_threshold : float
        Minimum strain magnitude `|grad u|` (m/m) to classify high-strain deformation zones.
    event_name : str
        Identifier label for the study case.
    """
    if isinstance(dz_or_disp, GeoGrid):
        dz_arr = np.ascontiguousarray(dz_or_disp.data, dtype=np.float64)
        dx_val = float(dz_or_disp.dx)
        dy_val = float(dz_or_disp.dy)
        c_area = float(abs(dx_val * dy_val)) if cell_area_m2 is None else float(cell_area_m2)
    else:
        dz_arr = np.ascontiguousarray(dz_or_disp, dtype=np.float64)
        c_area = 400.0 if cell_area_m2 is None else float(cell_area_m2)
        side = float(np.sqrt(max(c_area, 1e-6)))
        dx_val = side
        dy_val = -side

    shape = dz_arr.shape
    if strain_mag is None:
        st_res = _core.compute_displacement_strain_rs(dz_arr, dx_val, dy_val, float(np.nan))
        st_arr = np.ascontiguousarray(st_res["gradient_mag"], dtype=np.float64)
    else:
        st_arr = _as_2d_f64(strain_mag, shape)

    res = _core.compute_multi_hazard_signature_rs(
        dz_arr,
        st_arr,
        _as_2d_f64(d_ndvi, shape),
        _as_2d_f64(d_nbr, shape),
        _as_2d_f64(d_ndwi, shape),
        _as_2d_f64(d_ndmi, shape),
        _as_2d_f64(d_ndbi, shape),
        _as_2d_f64(d_bsi, shape),
        _as_2d_f64(d_cwt, shape),
        _as_2d_f64(emb_cos_dist, shape),
        c_area,
        float(lod_m),
        float(strain_threshold),
    )
    scores = np.asarray(res["fingerprint_scores"], dtype=np.float64)
    out = dict(res)
    out["event_name"] = event_name
    out["fingerprint_scores"] = scores
    out["fingerprint_axes"] = list(FINGERPRINT_AXES)
    out["fingerprint_dict"] = {axis: float(scores[i]) for i, axis in enumerate(FINGERPRINT_AXES)}
    return out


def compare_change_regimes(
    signatures: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """
    Assemble a multi-event regime comparison matrix, pairwise cosine/Euclidean regime separation
    matrix, and summary table across a list of `compute_multi_hazard_signature` dictionaries.
    """
    if len(signatures) == 0:
        raise ValueError("signatures must contain at least one event signature dict.")

    names = [str(s.get("event_name", f"event_{i}")) for i, s in enumerate(signatures)]
    dominant_regimes = [str(s["dominant_regime"]) for s in signatures]
    mat = np.vstack([np.asarray(s["fingerprint_scores"], dtype=np.float64) for s in signatures])

    # Pairwise cosine similarity and Euclidean distance between 6-axis fingerprint vectors
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    unit_mat = mat / np.maximum(norms, 1e-12)
    cosine_sim = unit_mat @ unit_mat.T
    cosine_dist = 1.0 - cosine_sim

    return {
        "event_names": names,
        "dominant_regimes": dominant_regimes,
        "fingerprint_axes": list(FINGERPRINT_AXES),
        "fingerprint_matrix": mat,
        "pairwise_cosine_similarity": cosine_sim,
        "pairwise_cosine_distance": cosine_dist,
        "records": list(signatures),
    }
