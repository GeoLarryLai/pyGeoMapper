// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArray3, PyUntypedArrayMethods,
};
use pygeomapper_core::{change, embedding, mapping, mesh3d, profile, spectral, terrain};
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction]
#[pyo3(signature = (dem, dx, dy, nodata=f64::NAN))]
fn compute_slope_aspect_rs<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    dx: f64,
    dy: f64,
    nodata: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let shape = dem.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let slice = dem.as_slice()?;
    let (slope, aspect) =
        py.allow_threads(|| terrain::compute_slope_aspect(slice, rows, cols, dx, dy, nodata));
    let s_arr = PyArray1::from_vec_bound(py, slope).reshape([rows, cols])?;
    let a_arr = PyArray1::from_vec_bound(py, aspect).reshape([rows, cols])?;
    Ok((s_arr, a_arr))
}

#[pyfunction]
#[pyo3(signature = (dem, dx, dy, azimuth_deg=315.0, altitude_deg=45.0, z_factor=1.0, nodata=f64::NAN))]
fn compute_hillshade_rs<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    dx: f64,
    dy: f64,
    azimuth_deg: f64,
    altitude_deg: f64,
    z_factor: f64,
    nodata: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = dem.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let slice = dem.as_slice()?;
    let hs = py.allow_threads(|| {
        terrain::compute_hillshade(
            slice,
            rows,
            cols,
            dx,
            dy,
            azimuth_deg,
            altitude_deg,
            z_factor,
            nodata,
        )
    });
    Ok(PyArray1::from_vec_bound(py, hs).reshape([rows, cols])?)
}

#[pyfunction]
#[pyo3(signature = (dem, dx, dy, nodata=f64::NAN))]
fn compute_curvature_rs<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    dx: f64,
    dy: f64,
    nodata: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let shape = dem.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let slice = dem.as_slice()?;
    let (lap, prof, plan) =
        py.allow_threads(|| terrain::compute_curvature(slice, rows, cols, dx, dy, nodata));
    Ok((
        PyArray1::from_vec_bound(py, lap).reshape([rows, cols])?,
        PyArray1::from_vec_bound(py, prof).reshape([rows, cols])?,
        PyArray1::from_vec_bound(py, plan).reshape([rows, cols])?,
    ))
}

#[pyfunction]
#[pyo3(signature = (dem, radius_px=3, nodata=f64::NAN))]
fn compute_roughness_tpi_rs<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    radius_px: usize,
    nodata: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let shape = dem.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let slice = dem.as_slice()?;
    let (rough, tpi) =
        py.allow_threads(|| terrain::compute_roughness_tpi(slice, rows, cols, radius_px, nodata));
    Ok((
        PyArray1::from_vec_bound(py, rough).reshape([rows, cols])?,
        PyArray1::from_vec_bound(py, tpi).reshape([rows, cols])?,
    ))
}

#[pyfunction]
#[pyo3(signature = (dem, dx, dy, scale_m, nodata=f64::NAN))]
fn compute_cwt2d_mexican_hat_rs<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    dx: f64,
    dy: f64,
    scale_m: f64,
    nodata: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let shape = dem.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let slice = dem.as_slice()?;
    let (cwt, energy) = py.allow_threads(|| {
        terrain::compute_cwt2d_mexican_hat(slice, rows, cols, dx, dy, scale_m, nodata)
    });
    Ok((
        PyArray1::from_vec_bound(py, cwt).reshape([rows, cols])?,
        PyArray1::from_vec_bound(py, energy).reshape([rows, cols])?,
    ))
}

#[pyfunction]
#[pyo3(signature = (dem, dx, dy, scales_m, nodata=f64::NAN))]
fn compute_multiscale_complexity_rs<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    dx: f64,
    dy: f64,
    scales_m: PyReadonlyArray1<'py, f64>,
    nodata: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = dem.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let s_dem = dem.as_slice()?;
    let s_scales = scales_m.as_slice()?;
    let k = s_scales.len();
    let res = py.allow_threads(|| {
        terrain::compute_multiscale_complexity(s_dem, rows, cols, dx, dy, s_scales, nodata)
    });
    let d = PyDict::new_bound(py);
    let cwt_cube: Bound<'py, PyArray3<f64>> =
        PyArray1::from_vec_bound(py, res.cwt_cube).reshape([rows, cols, k])?;
    let energy_cube: Bound<'py, PyArray3<f64>> =
        PyArray1::from_vec_bound(py, res.energy_cube).reshape([rows, cols, k])?;
    d.set_item("cwt_cube", cwt_cube)?;
    d.set_item("energy_cube", energy_cube)?;
    d.set_item(
        "mean_complexity",
        PyArray1::from_vec_bound(py, res.mean_complexity).reshape([rows, cols])?,
    )?;
    d.set_item(
        "max_energy",
        PyArray1::from_vec_bound(py, res.max_energy).reshape([rows, cols])?,
    )?;
    d.set_item(
        "dominant_scale_m",
        PyArray1::from_vec_bound(py, res.dominant_scale_m).reshape([rows, cols])?,
    )?;
    d.set_item(
        "dominant_wavelength_m",
        PyArray1::from_vec_bound(py, res.dominant_wavelength_m).reshape([rows, cols])?,
    )?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (band_a, band_b, nodata=f64::NAN))]
fn normalized_difference_rs<'py>(
    py: Python<'py>,
    band_a: PyReadonlyArray2<'py, f64>,
    band_b: PyReadonlyArray2<'py, f64>,
    nodata: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = band_a.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let sa = band_a.as_slice()?;
    let sb = band_b.as_slice()?;
    let out = py.allow_threads(|| spectral::normalized_difference(sa, sb, nodata));
    Ok(PyArray1::from_vec_bound(py, out).reshape([rows, cols])?)
}

#[pyfunction]
#[pyo3(signature = (band_a, band_b, nodata=f64::NAN))]
fn band_ratio_rs<'py>(
    py: Python<'py>,
    band_a: PyReadonlyArray2<'py, f64>,
    band_b: PyReadonlyArray2<'py, f64>,
    nodata: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = band_a.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let sa = band_a.as_slice()?;
    let sb = band_b.as_slice()?;
    let out = py.allow_threads(|| spectral::band_ratio(sa, sb, nodata));
    Ok(PyArray1::from_vec_bound(py, out).reshape([rows, cols])?)
}

#[pyfunction]
#[pyo3(signature = (swir1, red, nir, blue, nodata=f64::NAN))]
fn bare_soil_index_rs<'py>(
    py: Python<'py>,
    swir1: PyReadonlyArray2<'py, f64>,
    red: PyReadonlyArray2<'py, f64>,
    nir: PyReadonlyArray2<'py, f64>,
    blue: PyReadonlyArray2<'py, f64>,
    nodata: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = swir1.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let s1 = swir1.as_slice()?;
    let r = red.as_slice()?;
    let n = nir.as_slice()?;
    let b = blue.as_slice()?;
    let out = py.allow_threads(|| spectral::bare_soil_index(s1, r, n, b, nodata));
    Ok(PyArray1::from_vec_bound(py, out).reshape([rows, cols])?)
}

#[pyfunction]
#[pyo3(signature = (swir1, nir, green, red, nodata=f64::NAN))]
fn ferrous_iron_index_rs<'py>(
    py: Python<'py>,
    swir1: PyReadonlyArray2<'py, f64>,
    nir: PyReadonlyArray2<'py, f64>,
    green: PyReadonlyArray2<'py, f64>,
    red: PyReadonlyArray2<'py, f64>,
    nodata: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = swir1.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let s1 = swir1.as_slice()?;
    let n = nir.as_slice()?;
    let g = green.as_slice()?;
    let r = red.as_slice()?;
    let out = py.allow_threads(|| spectral::ferrous_iron_index(s1, n, g, r, nodata));
    Ok(PyArray1::from_vec_bound(py, out).reshape([rows, cols])?)
}

#[pyfunction]
#[pyo3(signature = (band1, band2, c1, c2, c0, d1, d2, d0, nodata=f64::NAN))]
#[allow(clippy::too_many_arguments)]
fn rational_two_band_index_rs<'py>(
    py: Python<'py>,
    band1: PyReadonlyArray2<'py, f64>,
    band2: PyReadonlyArray2<'py, f64>,
    c1: f64,
    c2: f64,
    c0: f64,
    d1: f64,
    d2: f64,
    d0: f64,
    nodata: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = band1.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let b1 = band1.as_slice()?;
    let b2 = band2.as_slice()?;
    let out = py.allow_threads(|| {
        spectral::rational_two_band_index(b1, b2, c1, c2, c0, d1, d2, d0, nodata)
    });
    Ok(PyArray1::from_vec_bound(py, out).reshape([rows, cols])?)
}

#[pyfunction]
#[pyo3(signature = (t1, t2, nodata=f64::NAN))]
fn compute_dod_rs<'py>(
    py: Python<'py>,
    t1: PyReadonlyArray2<'py, f64>,
    t2: PyReadonlyArray2<'py, f64>,
    nodata: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = t1.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let s1 = t1.as_slice()?;
    let s2 = t2.as_slice()?;
    let out = py.allow_threads(|| change::compute_dod(s1, s2, nodata));
    Ok(PyArray1::from_vec_bound(py, out).reshape([rows, cols])?)
}

#[pyfunction]
fn detrend_dod_planar_rs<'py>(
    py: Python<'py>,
    dod: PyReadonlyArray2<'py, f64>,
    stable_mask: PyReadonlyArray2<'py, u8>,
    dx: f64,
    dy: f64,
) -> PyResult<(Bound<'py, PyArray2<f64>>, f64, f64, f64)> {
    let shape = dod.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let sd = dod.as_slice()?;
    let sm = stable_mask.as_slice()?;
    let (out, c0, cx, cy) =
        py.allow_threads(|| change::detrend_dod_planar(sd, sm, rows, cols, dx, dy));
    Ok((
        PyArray1::from_vec_bound(py, out).reshape([rows, cols])?,
        c0,
        cx,
        cy,
    ))
}

#[pyfunction]
fn estimate_nuth_kaab_shift_rs<'py>(
    py: Python<'py>,
    dod: PyReadonlyArray2<'py, f64>,
    slope_deg: PyReadonlyArray2<'py, f64>,
    aspect_deg: PyReadonlyArray2<'py, f64>,
    stable_mask: PyReadonlyArray2<'py, u8>,
) -> PyResult<(f64, f64, f64)> {
    let sd = dod.as_slice()?;
    let ss = slope_deg.as_slice()?;
    let sa = aspect_deg.as_slice()?;
    let sm = stable_mask.as_slice()?;
    let res = py.allow_threads(|| change::estimate_nuth_kaab_shift(sd, ss, sa, sm));
    Ok(res)
}

#[pyfunction]
#[pyo3(signature = (dod, cell_area_m2, lod_threshold=0.0, sigma_dz=0.1))]
fn compute_volume_budget_rs<'py>(
    py: Python<'py>,
    dod: PyReadonlyArray2<'py, f64>,
    cell_area_m2: f64,
    lod_threshold: f64,
    sigma_dz: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let sd = dod.as_slice()?;
    let b = py.allow_threads(|| {
        change::compute_volume_budget(sd, cell_area_m2, lod_threshold, sigma_dz)
    });
    let d = PyDict::new_bound(py);
    d.set_item("eroded_pixels", b.eroded_pixels)?;
    d.set_item("deposited_pixels", b.deposited_pixels)?;
    d.set_item("stable_pixels", b.stable_pixels)?;
    d.set_item("eroded_area_m2", b.eroded_area_m2)?;
    d.set_item("deposited_area_m2", b.deposited_area_m2)?;
    d.set_item("eroded_volume_m3", b.eroded_volume_m3)?;
    d.set_item("deposited_volume_m3", b.deposited_volume_m3)?;
    d.set_item("net_volume_m3", b.net_volume_m3)?;
    d.set_item("volume_uncertainty_m3", b.volume_uncertainty_m3)?;
    d.set_item("mean_dz_m", b.mean_dz_m)?;
    d.set_item("max_erosion_m", b.max_erosion_m)?;
    d.set_item("max_deposition_m", b.max_deposition_m)?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (values, n_bins=256))]
fn otsu_threshold_rs<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    n_bins: usize,
) -> PyResult<f64> {
    let sv = values.as_slice()?;
    Ok(py.allow_threads(|| change::otsu_threshold(sv, n_bins)))
}

#[pyfunction]
fn classify_change_severity_rs<'py>(
    py: Python<'py>,
    values: PyReadonlyArray2<'py, f64>,
    thresholds: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let shape = values.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let sv = values.as_slice()?;
    let st = thresholds.as_slice()?;
    let out = py.allow_threads(|| change::classify_change_severity(sv, st));
    Ok(PyArray1::from_vec_bound(py, out).reshape([rows, cols])?)
}

#[pyfunction]
#[pyo3(signature = (mask, magnitude, x0=0.0, y0=0.0, dx=1.0, dy=-1.0, min_pixels=4))]
#[allow(clippy::too_many_arguments)]
fn connected_change_patches_rs<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray2<'py, u8>,
    magnitude: PyReadonlyArray2<'py, f64>,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    min_pixels: usize,
) -> PyResult<(Bound<'py, PyArray2<i32>>, Vec<Bound<'py, PyDict>>)> {
    let shape = mask.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let sm = mask.as_slice()?;
    let sg = magnitude.as_slice()?;
    let (labels, stats) = py.allow_threads(|| {
        change::connected_change_patches(sm, sg, rows, cols, x0, y0, dx, dy, min_pixels)
    });
    let lbl_arr = PyArray1::from_vec_bound(py, labels).reshape([rows, cols])?;
    let mut py_stats = Vec::with_capacity(stats.len());
    for st in stats {
        let d = PyDict::new_bound(py);
        d.set_item("patch_id", st.patch_id)?;
        d.set_item("pixel_count", st.pixel_count)?;
        d.set_item("area_m2", st.area_m2)?;
        d.set_item("mean_magnitude", st.mean_magnitude)?;
        d.set_item("max_magnitude", st.max_magnitude)?;
        d.set_item("integrated_volume_or_sum", st.integrated_volume_or_sum)?;
        d.set_item("centroid_x", st.centroid_x)?;
        d.set_item("centroid_y", st.centroid_y)?;
        d.set_item("bbox", (st.bbox_r0, st.bbox_c0, st.bbox_r1, st.bbox_c1))?;
        py_stats.push(d);
    }
    Ok((lbl_arr, py_stats))
}

#[pyfunction]
#[pyo3(signature = (emb_t1, emb_t2, nodata=f64::NAN))]
fn embedding_change_metrics_rs<'py>(
    py: Python<'py>,
    emb_t1: PyReadonlyArray3<'py, f64>,
    emb_t2: PyReadonlyArray3<'py, f64>,
    nodata: f64,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let shape = emb_t1.shape();
    let (rows, cols, dim) = (shape[0], shape[1], shape[2]);
    let n_pixels = rows * cols;
    let s1 = emb_t1.as_slice()?;
    let s2 = emb_t2.as_slice()?;
    let (cd, cs, l2) = py
        .allow_threads(|| embedding::embedding_change_metrics(s1, s2, n_pixels, dim, nodata));
    Ok((
        PyArray1::from_vec_bound(py, cd).reshape([rows, cols])?,
        PyArray1::from_vec_bound(py, cs).reshape([rows, cols])?,
        PyArray1::from_vec_bound(py, l2).reshape([rows, cols])?,
    ))
}

#[pyfunction]
#[pyo3(signature = (emb_t1, emb_t2, nodata=f64::NAN))]
fn embedding_dimension_importance_rs<'py>(
    py: Python<'py>,
    emb_t1: PyReadonlyArray3<'py, f64>,
    emb_t2: PyReadonlyArray3<'py, f64>,
    nodata: f64,
) -> PyResult<(
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let shape = emb_t1.shape();
    let (rows, cols, dim) = (shape[0], shape[1], shape[2]);
    let n_pixels = rows * cols;
    let s1 = emb_t1.as_slice()?;
    let s2 = emb_t2.as_slice()?;
    let (ranked, rms, mean_diff, pc1) = py.allow_threads(|| {
        embedding::embedding_dimension_importance(s1, s2, n_pixels, dim, nodata)
    });
    Ok((
        ranked.into_pyarray_bound(py),
        rms.into_pyarray_bound(py),
        mean_diff.into_pyarray_bound(py),
        PyArray1::from_vec_bound(py, pc1).reshape([rows, cols])?,
    ))
}

#[pyfunction]
#[pyo3(signature = (features, n_clusters=5, max_iter=50, seed=42))]
fn kmeans_features_rs<'py>(
    py: Python<'py>,
    features: PyReadonlyArray3<'py, f64>,
    n_clusters: usize,
    max_iter: usize,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray2<i32>>, Bound<'py, PyArray2<f64>>)> {
    let shape = features.shape();
    let (rows, cols, dim) = (shape[0], shape[1], shape[2]);
    let n_pixels = rows * cols;
    let sf = features.as_slice()?;
    let (labels, centroids) = py.allow_threads(|| {
        embedding::kmeans_features(sf, n_pixels, dim, n_clusters, max_iter, seed)
    });
    let k = n_clusters.max(1);
    Ok((
        PyArray1::from_vec_bound(py, labels).reshape([rows, cols])?,
        PyArray1::from_vec_bound(py, centroids).reshape([k, dim])?,
    ))
}

#[pyfunction]
#[pyo3(signature = (labels, radius_px=2, iterations=1))]
fn smooth_labels_majority_rs<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray2<'py, i32>,
    radius_px: usize,
    iterations: usize,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let shape = labels.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let sl = labels.as_slice()?;
    let out = py.allow_threads(|| {
        mapping::smooth_labels_majority(sl, rows, cols, radius_px, iterations)
    });
    Ok(PyArray1::from_vec_bound(py, out).reshape([rows, cols])?)
}

#[pyfunction]
#[pyo3(signature = (labels, x0=0.0, y0=0.0, dx=1.0, dy=-1.0))]
fn extract_unit_contacts_rs<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray2<'py, i32>,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Vec<Bound<'py, PyDict>>)> {
    let shape = labels.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let sl = labels.as_slice()?;
    let (mask, contacts) = py.allow_threads(|| {
        mapping::extract_unit_contacts(sl, rows, cols, x0, y0, dx, dy)
    });
    let mask_arr = PyArray1::from_vec_bound(py, mask).reshape([rows, cols])?;
    let mut py_contacts = Vec::with_capacity(contacts.len());
    for cp in contacts {
        let d = PyDict::new_bound(py);
        d.set_item("row", cp.row)?;
        d.set_item("col", cp.col)?;
        d.set_item("x", cp.x)?;
        d.set_item("y", cp.y)?;
        d.set_item("unit_a", cp.unit_a)?;
        d.set_item("unit_b", cp.unit_b)?;
        py_contacts.push(d);
    }
    Ok((mask_arr, py_contacts))
}

#[pyfunction]
#[pyo3(signature = (features, train_mask, train_labels, var_smoothing=1e-4))]
fn fit_predict_gaussian_bayes_rs<'py>(
    py: Python<'py>,
    features: PyReadonlyArray3<'py, f64>,
    train_mask: PyReadonlyArray2<'py, u8>,
    train_labels: PyReadonlyArray2<'py, i32>,
    var_smoothing: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = features.shape();
    let (rows, cols, dim) = (shape[0], shape[1], shape[2]);
    let n_pixels = rows * cols;
    let sf = features.as_slice()?;
    let sm = train_mask.as_slice()?;
    let sl = train_labels.as_slice()?;
    let res = py.allow_threads(|| {
        mapping::fit_predict_gaussian_bayes(sf, n_pixels, dim, sm, sl, var_smoothing)
    });
    let n_classes = res.class_ids.len().max(1);
    let d = PyDict::new_bound(py);
    d.set_item(
        "pred_labels",
        PyArray1::from_vec_bound(py, res.pred_labels).reshape([rows, cols])?,
    )?;
    if !res.probabilities.is_empty() {
        let probs: Bound<'py, PyArray3<f64>> =
            PyArray1::from_vec_bound(py, res.probabilities).reshape([rows, cols, n_classes])?;
        d.set_item("probabilities", probs)?;
    }
    d.set_item(
        "confidence",
        PyArray1::from_vec_bound(py, res.confidence).reshape([rows, cols])?,
    )?;
    d.set_item(
        "entropy",
        PyArray1::from_vec_bound(py, res.entropy).reshape([rows, cols])?,
    )?;
    d.set_item("class_ids", res.class_ids.into_pyarray_bound(py))?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (xs, ys, zs, robust_trim=true))]
fn estimate_bed_attitude_rs<'py>(
    py: Python<'py>,
    xs: PyReadonlyArray1<'py, f64>,
    ys: PyReadonlyArray1<'py, f64>,
    zs: PyReadonlyArray1<'py, f64>,
    robust_trim: bool,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let sx = xs.as_slice()?;
    let sy = ys.as_slice()?;
    let sz = zs.as_slice()?;
    let res = py.allow_threads(|| profile::estimate_bed_attitude(sx, sy, sz, robust_trim));
    if let Some(att) = res {
        let d = PyDict::new_bound(py);
        d.set_item("strike_deg", att.strike_deg)?;
        d.set_item("dip_deg", att.dip_deg)?;
        d.set_item("dip_dir_deg", att.dip_dir_deg)?;
        d.set_item("dz_dx", att.dz_dx)?;
        d.set_item("dz_dy", att.dz_dy)?;
        d.set_item("intercept_c", att.intercept_c)?;
        d.set_item("r_squared", att.r_squared)?;
        d.set_item("rmse_m", att.rmse_m)?;
        d.set_item("collinearity_ratio", att.collinearity_ratio)?;
        d.set_item("centroid_x", att.centroid_x)?;
        d.set_item("centroid_y", att.centroid_y)?;
        d.set_item("centroid_z", att.centroid_z)?;
        d.set_item("relief_m", att.relief_m)?;
        d.set_item("n_points", att.n_points)?;
        Ok(Some(d))
    } else {
        Ok(None)
    }
}

#[pyfunction]
#[pyo3(signature = (labels, dem, x0, y0, dx, dy, window_m=300.0, min_points=6, min_relief_m=8.0, min_r2=0.5, min_collinearity=0.05, nodata=f64::NAN))]
#[allow(clippy::too_many_arguments)]
fn estimate_contact_attitudes_grid_rs<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray2<'py, i32>,
    dem: PyReadonlyArray2<'py, f64>,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    window_m: f64,
    min_points: usize,
    min_relief_m: f64,
    min_r2: f64,
    min_collinearity: f64,
    nodata: f64,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let shape = labels.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let sl = labels.as_slice()?;
    let sd = dem.as_slice()?;
    let records = py.allow_threads(|| {
        profile::estimate_contact_attitudes_grid(
            sl,
            sd,
            rows,
            cols,
            x0,
            y0,
            dx,
            dy,
            window_m,
            min_points,
            min_relief_m,
            min_r2,
            min_collinearity,
            nodata,
        )
    });
    let mut out = Vec::with_capacity(records.len());
    for rec in records {
        let d = PyDict::new_bound(py);
        d.set_item("unit_a", rec.unit_a)?;
        d.set_item("unit_b", rec.unit_b)?;
        d.set_item("strike_deg", rec.attitude.strike_deg)?;
        d.set_item("dip_deg", rec.attitude.dip_deg)?;
        d.set_item("dip_dir_deg", rec.attitude.dip_dir_deg)?;
        d.set_item("dz_dx", rec.attitude.dz_dx)?;
        d.set_item("dz_dy", rec.attitude.dz_dy)?;
        d.set_item("intercept_c", rec.attitude.intercept_c)?;
        d.set_item("r_squared", rec.attitude.r_squared)?;
        d.set_item("rmse_m", rec.attitude.rmse_m)?;
        d.set_item("collinearity_ratio", rec.attitude.collinearity_ratio)?;
        d.set_item("centroid_x", rec.attitude.centroid_x)?;
        d.set_item("centroid_y", rec.attitude.centroid_y)?;
        d.set_item("centroid_z", rec.attitude.centroid_z)?;
        d.set_item("relief_m", rec.attitude.relief_m)?;
        d.set_item("n_points", rec.attitude.n_points)?;
        out.push(d);
    }
    Ok(out)
}

#[pyfunction]
fn compute_apparent_dip_deg_rs(
    dip_deg: f64,
    dip_dir_deg: f64,
    section_azimuth_deg: f64,
) -> f64 {
    profile::compute_apparent_dip_deg(dip_deg, dip_dir_deg, section_azimuth_deg)
}

#[pyfunction]
#[pyo3(signature = (grid, x0, y0, dx, dy, xa, ya, xb, yb, n_samples=200, swath_width_m=0.0, n_cross=11, nodata=f64::NAN))]
#[allow(clippy::too_many_arguments)]
fn sample_swath_profile_rs<'py>(
    py: Python<'py>,
    grid: PyReadonlyArray2<'py, f64>,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    xa: f64,
    ya: f64,
    xb: f64,
    yb: f64,
    n_samples: usize,
    swath_width_m: f64,
    n_cross: usize,
    nodata: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = grid.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let sg = grid.as_slice()?;
    let res = py.allow_threads(|| {
        profile::sample_swath_profile(
            sg,
            rows,
            cols,
            x0,
            y0,
            dx,
            dy,
            xa,
            ya,
            xb,
            yb,
            n_samples,
            swath_width_m,
            n_cross,
            nodata,
        )
    });
    let d = PyDict::new_bound(py);
    d.set_item("distance_m", res.distances_m.into_pyarray_bound(py))?;
    d.set_item("x", res.xs.into_pyarray_bound(py))?;
    d.set_item("y", res.ys.into_pyarray_bound(py))?;
    d.set_item("center", res.center_values.into_pyarray_bound(py))?;
    d.set_item("mean", res.swath_mean.into_pyarray_bound(py))?;
    d.set_item("min", res.swath_min.into_pyarray_bound(py))?;
    d.set_item("max", res.swath_max.into_pyarray_bound(py))?;
    d.set_item("std", res.swath_std.into_pyarray_bound(py))?;
    Ok(d)
}


fn mesh_to_pydict<'py>(py: Python<'py>, m: mesh3d::Mesh3DData) -> PyResult<Bound<'py, PyDict>> {
    let nv = m.n_vertices();
    let nt = m.n_triangles();
    let d = PyDict::new_bound(py);
    d.set_item("n_vertices", nv)?;
    d.set_item("n_triangles", nt)?;
    d.set_item(
        "vertices",
        PyArray1::from_vec_bound(py, m.vertices).reshape([nv, 3])?,
    )?;
    d.set_item(
        "normals",
        PyArray1::from_vec_bound(py, m.normals).reshape([nv, 3])?,
    )?;
    d.set_item(
        "triangles",
        PyArray1::from_vec_bound(py, m.triangles).reshape([nt, 3])?,
    )?;
    d.set_item(
        "vertex_colors_rgba",
        PyArray1::from_vec_bound(py, m.vertex_colors_rgba).reshape([nv, 4])?,
    )?;
    d.set_item("unit_ids", m.unit_ids.into_pyarray_bound(py))?;
    d.set_item("face_groups", m.face_groups.into_pyarray_bound(py))?;
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (
    dem,
    surface_units,
    structural_horizon,
    unit_zeta_thresholds,
    bedrock_unit_ids,
    x0,
    y0,
    dx,
    dy,
    z_bottom,
    z_top,
    nz=64,
    surficial_unit_id=-1,
    surficial_thickness_m=20.0,
    near_surface_lock_m=16.0,
    fault_params=None
))]
#[allow(clippy::too_many_arguments)]
fn build_3d_block_volume_rs<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    surface_units: PyReadonlyArray2<'py, i32>,
    structural_horizon: PyReadonlyArray2<'py, f64>,
    unit_zeta_thresholds: PyReadonlyArray1<'py, f64>,
    bedrock_unit_ids: PyReadonlyArray1<'py, i32>,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    z_bottom: f64,
    z_top: f64,
    nz: usize,
    surficial_unit_id: i32,
    surficial_thickness_m: f64,
    near_surface_lock_m: f64,
    fault_params: Option<(f64, f64, f64, f64, f64, f64, f64)>,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = dem.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let s_dem = dem.as_slice()?;
    let s_su = surface_units.as_slice()?;
    let s_sh = structural_horizon.as_slice()?;
    let s_thr = unit_zeta_thresholds.as_slice()?;
    let s_bid = bedrock_unit_ids.as_slice()?;
    let n_horizons = s_thr.len();

    let fault = fault_params.map(|(fx0, fy0, fz0, fstrike, fdip, fthrow, flistric)| {
        mesh3d::Fault3DParams {
            x0: fx0,
            y0: fy0,
            z0: fz0,
            strike_deg: fstrike,
            dip_deg: fdip,
            throw_m: fthrow,
            listric_curvature: flistric,
        }
    });

    let res = py.allow_threads(|| {
        mesh3d::build_3d_block_volume(
            s_dem,
            s_su,
            s_sh,
            s_thr,
            s_bid,
            surficial_unit_id,
            surficial_thickness_m,
            near_surface_lock_m,
            rows,
            cols,
            x0,
            y0,
            dx,
            dy,
            z_bottom,
            z_top,
            nz,
            fault,
        )
    });

    let nz_out = res.nz;
    let d = PyDict::new_bound(py);
    d.set_item("z_levels", res.z_levels.into_pyarray_bound(py))?;
    d.set_item(
        "voxel_units",
        PyArray1::from_vec_bound(py, res.voxel_units).reshape([rows, cols, nz_out])?,
    )?;
    d.set_item(
        "zeta_volume",
        PyArray1::from_vec_bound(py, res.zeta_volume).reshape([rows, cols, nz_out])?,
    )?;
    d.set_item(
        "fault_offset_volume",
        PyArray1::from_vec_bound(py, res.fault_offset_volume).reshape([rows, cols, nz_out])?,
    )?;
    d.set_item(
        "fault_surface_z",
        PyArray1::from_vec_bound(py, res.fault_surface_z).reshape([rows, cols])?,
    )?;
    if n_horizons > 0 {
        d.set_item(
            "horizon_elevations",
            PyArray1::from_vec_bound(py, res.horizon_elevations)
                .reshape([n_horizons, rows, cols])?,
        )?;
    }
    Ok(d)
}

#[pyfunction]
#[pyo3(signature = (
    dem,
    surface_units,
    surface_drape_rgba,
    voxel_units,
    z_levels,
    unit_palette_rgba,
    fault_surface_z,
    x0,
    y0,
    dx,
    dy,
    cutaway_bounds=None
))]
#[allow(clippy::too_many_arguments)]
fn build_block_diagram_meshes_rs<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    surface_units: PyReadonlyArray2<'py, i32>,
    surface_drape_rgba: PyReadonlyArray3<'py, u8>,
    voxel_units: PyReadonlyArray3<'py, i32>,
    z_levels: PyReadonlyArray1<'py, f64>,
    unit_palette_rgba: PyReadonlyArray2<'py, u8>,
    fault_surface_z: PyReadonlyArray2<'py, f64>,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    cutaway_bounds: Option<(usize, usize, usize, usize, usize)>,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = dem.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let s_dem = dem.as_slice()?;
    let s_su = surface_units.as_slice()?;
    let s_drape = surface_drape_rgba.as_slice()?;
    let s_vox = voxel_units.as_slice()?;
    let s_zl = z_levels.as_slice()?;
    let s_pal = unit_palette_rgba.as_slice()?;
    let s_flt = fault_surface_z.as_slice()?;

    let mesh = py.allow_threads(|| {
        mesh3d::build_block_diagram_meshes(
            s_dem,
            s_su,
            s_drape,
            s_vox,
            s_zl,
            s_pal,
            s_flt,
            rows,
            cols,
            x0,
            y0,
            dx,
            dy,
            cutaway_bounds,
        )
    });
    mesh_to_pydict(py, mesh)
}

#[pyfunction]
#[pyo3(signature = (dem, voxel_units, z_levels, unit_palette_rgba, x0, y0, dx, dy, xa, ya, xb, yb, n_samples=160))]
#[allow(clippy::too_many_arguments)]
fn slice_3d_block_fence_rs<'py>(
    py: Python<'py>,
    dem: PyReadonlyArray2<'py, f64>,
    voxel_units: PyReadonlyArray3<'py, i32>,
    z_levels: PyReadonlyArray1<'py, f64>,
    unit_palette_rgba: PyReadonlyArray2<'py, u8>,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    xa: f64,
    ya: f64,
    xb: f64,
    yb: f64,
    n_samples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = dem.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let s_dem = dem.as_slice()?;
    let s_vox = voxel_units.as_slice()?;
    let s_zl = z_levels.as_slice()?;
    let s_pal = unit_palette_rgba.as_slice()?;
    let nz = s_zl.len();
    let ns = n_samples.max(2);

    let res = py.allow_threads(|| {
        mesh3d::slice_3d_block_fence(
            s_dem,
            s_vox,
            s_zl,
            s_pal,
            rows,
            cols,
            x0,
            y0,
            dx,
            dy,
            xa,
            ya,
            xb,
            yb,
            ns,
        )
    });

    let d = PyDict::new_bound(py);
    d.set_item("distance_m", res.distances_m.into_pyarray_bound(py))?;
    d.set_item("x", res.xs.into_pyarray_bound(py))?;
    d.set_item("y", res.ys.into_pyarray_bound(py))?;
    d.set_item("z_topo_m", res.z_topo.into_pyarray_bound(py))?;
    d.set_item(
        "section_units",
        PyArray1::from_vec_bound(py, res.section_units).reshape([nz, ns])?,
    )?;
    d.set_item("mesh", mesh_to_pydict(py, res.mesh)?)?;
    Ok(d)
}

#[pyfunction]
fn export_block_glb_rs<'py>(
    py: Python<'py>,
    path: &str,
    vertices: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    triangles: PyReadonlyArray2<'py, u32>,
    colors_rgba: PyReadonlyArray2<'py, u8>,
) -> PyResult<usize> {
    let sv = vertices.as_slice()?;
    let sn = normals.as_slice()?;
    let st = triangles.as_slice()?;
    let sc = colors_rgba.as_slice()?;
    py.allow_threads(|| mesh3d::export_mesh_glb(path, sv, sn, st, sc))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
fn export_block_ply_rs<'py>(
    py: Python<'py>,
    path: &str,
    vertices: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    triangles: PyReadonlyArray2<'py, u32>,
    colors_rgba: PyReadonlyArray2<'py, u8>,
) -> PyResult<usize> {
    let sv = vertices.as_slice()?;
    let sn = normals.as_slice()?;
    let st = triangles.as_slice()?;
    let sc = colors_rgba.as_slice()?;
    py.allow_threads(|| mesh3d::export_mesh_ply(path, sv, sn, st, sc))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
fn export_block_obj_rs<'py>(
    py: Python<'py>,
    path: &str,
    vertices: PyReadonlyArray2<'py, f64>,
    normals: PyReadonlyArray2<'py, f64>,
    triangles: PyReadonlyArray2<'py, u32>,
    colors_rgba: PyReadonlyArray2<'py, u8>,
) -> PyResult<usize> {
    let sv = vertices.as_slice()?;
    let sn = normals.as_slice()?;
    let st = triangles.as_slice()?;
    let sc = colors_rgba.as_slice()?;
    py.allow_threads(|| mesh3d::export_mesh_obj(path, sv, sn, st, sc))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
fn compute_displacement_strain_rs<'py>(
    py: Python<'py>,
    disp: PyReadonlyArray2<'py, f64>,
    dx: f64,
    dy: f64,
    nodata: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = disp.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let s_disp = disp.as_slice()?;
    let res = py.allow_threads(|| {
        change::compute_displacement_strain(s_disp, rows, cols, dx, dy, nodata)
    });
    let d = PyDict::new_bound(py);
    d.set_item("du_dx", PyArray1::from_vec_bound(py, res.du_dx).reshape([rows, cols])?)?;
    d.set_item("du_dy", PyArray1::from_vec_bound(py, res.du_dy).reshape([rows, cols])?)?;
    d.set_item("gradient_mag", PyArray1::from_vec_bound(py, res.gradient_mag).reshape([rows, cols])?)?;
    d.set_item("microstrain", PyArray1::from_vec_bound(py, res.microstrain).reshape([rows, cols])?)?;
    d.set_item("scarp_curvature", PyArray1::from_vec_bound(py, res.scarp_curvature).reshape([rows, cols])?)?;
    d.set_item(
        "gradient_azimuth_deg",
        PyArray1::from_vec_bound(py, res.gradient_azimuth_deg).reshape([rows, cols])?,
    )?;
    Ok(d)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn compute_multi_hazard_signature_rs<'py>(
    py: Python<'py>,
    dz_or_disp: PyReadonlyArray2<'py, f64>,
    strain_mag: PyReadonlyArray2<'py, f64>,
    d_ndvi: PyReadonlyArray2<'py, f64>,
    d_nbr: PyReadonlyArray2<'py, f64>,
    d_ndwi: PyReadonlyArray2<'py, f64>,
    d_ndmi: PyReadonlyArray2<'py, f64>,
    d_ndbi: PyReadonlyArray2<'py, f64>,
    d_bsi: PyReadonlyArray2<'py, f64>,
    d_cwt: PyReadonlyArray2<'py, f64>,
    emb_cos_dist: PyReadonlyArray2<'py, f64>,
    cell_area_m2: f64,
    lod_m: f64,
    strain_threshold: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let s_dz = dz_or_disp.as_slice()?;
    let s_st = strain_mag.as_slice()?;
    let s_ndvi = d_ndvi.as_slice()?;
    let s_nbr = d_nbr.as_slice()?;
    let s_ndwi = d_ndwi.as_slice()?;
    let s_ndmi = d_ndmi.as_slice()?;
    let s_ndbi = d_ndbi.as_slice()?;
    let s_bsi = d_bsi.as_slice()?;
    let s_cwt = d_cwt.as_slice()?;
    let s_emb = emb_cos_dist.as_slice()?;

    let sig = py.allow_threads(|| {
        change::compute_multi_hazard_signature(
            s_dz,
            s_st,
            s_ndvi,
            s_nbr,
            s_ndwi,
            s_ndmi,
            s_ndbi,
            s_bsi,
            s_cwt,
            s_emb,
            cell_area_m2,
            lod_m,
            strain_threshold,
        )
    });

    let d = PyDict::new_bound(py);
    d.set_item("total_area_km2", sig.total_area_km2)?;
    d.set_item("max_abs_dz_or_disp_m", sig.max_abs_dz_or_disp_m)?;
    d.set_item("p95_abs_dz_or_disp_m", sig.p95_abs_dz_or_disp_m)?;
    d.set_item("eroded_volume_m3", sig.eroded_volume_m3)?;
    d.set_item("deposited_volume_m3", sig.deposited_volume_m3)?;
    d.set_item("active_elevation_area_km2", sig.active_elevation_area_km2)?;
    d.set_item("max_strain_m_per_m", sig.max_strain_m_per_m)?;
    d.set_item("p99_strain_m_per_m", sig.p99_strain_m_per_m)?;
    d.set_item("high_strain_area_km2", sig.high_strain_area_km2)?;
    d.set_item("mean_d_ndvi", sig.mean_d_ndvi)?;
    d.set_item("p05_d_ndvi", sig.p05_d_ndvi)?;
    d.set_item("p95_d_ndvi", sig.p95_d_ndvi)?;
    d.set_item("veg_loss_area_km2", sig.veg_loss_area_km2)?;
    d.set_item("veg_gain_area_km2", sig.veg_gain_area_km2)?;
    d.set_item("mean_d_nbr", sig.mean_d_nbr)?;
    d.set_item("p95_d_nbr", sig.p95_d_nbr)?;
    d.set_item("burned_area_km2", sig.burned_area_km2)?;
    d.set_item("mean_d_ndwi", sig.mean_d_ndwi)?;
    d.set_item("p95_d_ndwi", sig.p95_d_ndwi)?;
    d.set_item("inundated_area_km2", sig.inundated_area_km2)?;
    d.set_item("mean_d_ndmi", sig.mean_d_ndmi)?;
    d.set_item("p95_d_ndmi", sig.p95_d_ndmi)?;
    d.set_item("wetted_area_km2", sig.wetted_area_km2)?;
    d.set_item("mean_d_ndbi", sig.mean_d_ndbi)?;
    d.set_item("p95_d_ndbi", sig.p95_d_ndbi)?;
    d.set_item("built_expansion_area_km2", sig.built_expansion_area_km2)?;
    d.set_item("mean_d_bsi", sig.mean_d_bsi)?;
    d.set_item("p95_d_bsi", sig.p95_d_bsi)?;
    d.set_item("bare_soil_expansion_area_km2", sig.bare_soil_expansion_area_km2)?;
    d.set_item("mean_d_cwt", sig.mean_d_cwt)?;
    d.set_item("p95_abs_d_cwt", sig.p95_abs_d_cwt)?;
    d.set_item("mean_emb_cosine_dist", sig.mean_emb_cosine_dist)?;
    d.set_item("p95_emb_cosine_dist", sig.p95_emb_cosine_dist)?;
    d.set_item(
        "fingerprint_scores",
        sig.fingerprint_scores.to_vec().into_pyarray_bound(py),
    )?;
    d.set_item("dominant_regime", sig.dominant_regime)?;
    Ok(d)
}


#[pyfunction]
#[pyo3(signature = (los_disp_m, wavelength_m=0.05546576, nodata=f64::NAN))]
fn compute_insar_wrapped_phase_rs<'py>(
    py: Python<'py>,
    los_disp_m: PyReadonlyArray2<'py, f64>,
    wavelength_m: f64,
    nodata: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let shape = los_disp_m.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let s = los_disp_m.as_slice()?;
    let out = py.allow_threads(|| change::compute_insar_wrapped_phase(s, wavelength_m, nodata));
    Ok(PyArray1::from_vec_bound(py, out).reshape([rows, cols])?)
}

#[pyfunction]
#[pyo3(signature = (los_asc, los_desc, gnss_e, gnss_n, gnss_u, s_asc, s_desc, gnss_weight_en=0.25, gnss_weight_u=0.10, nodata=f64::NAN))]
#[allow(clippy::too_many_arguments)]
fn decompose_insar_3d_rs<'py>(
    py: Python<'py>,
    los_asc: PyReadonlyArray2<'py, f64>,
    los_desc: PyReadonlyArray2<'py, f64>,
    gnss_e: PyReadonlyArray2<'py, f64>,
    gnss_n: PyReadonlyArray2<'py, f64>,
    gnss_u: PyReadonlyArray2<'py, f64>,
    s_asc: [f64; 3],
    s_desc: [f64; 3],
    gnss_weight_en: f64,
    gnss_weight_u: f64,
    nodata: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = los_asc.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let sa = los_asc.as_slice()?;
    let sd = los_desc.as_slice()?;
    let ge = gnss_e.as_slice()?;
    let gn = gnss_n.as_slice()?;
    let gu = gnss_u.as_slice()?;

    let res = py.allow_threads(|| {
        change::decompose_insar_3d(
            sa,
            sd,
            ge,
            gn,
            gu,
            s_asc,
            s_desc,
            gnss_weight_en,
            gnss_weight_u,
            nodata,
        )
    });

    let d = PyDict::new_bound(py);
    d.set_item("u_east", PyArray1::from_vec_bound(py, res.u_east).reshape([rows, cols])?)?;
    d.set_item("u_north", PyArray1::from_vec_bound(py, res.u_north).reshape([rows, cols])?)?;
    d.set_item("u_up", PyArray1::from_vec_bound(py, res.u_up).reshape([rows, cols])?)?;
    d.set_item("horiz_mag", PyArray1::from_vec_bound(py, res.horiz_mag).reshape([rows, cols])?)?;
    Ok(d)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(compute_slope_aspect_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_hillshade_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_curvature_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_roughness_tpi_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cwt2d_mexican_hat_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_multiscale_complexity_rs, m)?)?;
    m.add_function(wrap_pyfunction!(normalized_difference_rs, m)?)?;
    m.add_function(wrap_pyfunction!(band_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bare_soil_index_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ferrous_iron_index_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rational_two_band_index_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_dod_rs, m)?)?;
    m.add_function(wrap_pyfunction!(detrend_dod_planar_rs, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_nuth_kaab_shift_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_volume_budget_rs, m)?)?;
    m.add_function(wrap_pyfunction!(otsu_threshold_rs, m)?)?;
    m.add_function(wrap_pyfunction!(classify_change_severity_rs, m)?)?;
    m.add_function(wrap_pyfunction!(connected_change_patches_rs, m)?)?;
    m.add_function(wrap_pyfunction!(embedding_change_metrics_rs, m)?)?;
    m.add_function(wrap_pyfunction!(embedding_dimension_importance_rs, m)?)?;
    m.add_function(wrap_pyfunction!(kmeans_features_rs, m)?)?;
    m.add_function(wrap_pyfunction!(smooth_labels_majority_rs, m)?)?;
    m.add_function(wrap_pyfunction!(extract_unit_contacts_rs, m)?)?;
    m.add_function(wrap_pyfunction!(fit_predict_gaussian_bayes_rs, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_bed_attitude_rs, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_contact_attitudes_grid_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_apparent_dip_deg_rs, m)?)?;
    m.add_function(wrap_pyfunction!(sample_swath_profile_rs, m)?)?;
    m.add_function(wrap_pyfunction!(build_3d_block_volume_rs, m)?)?;
    m.add_function(wrap_pyfunction!(build_block_diagram_meshes_rs, m)?)?;
    m.add_function(wrap_pyfunction!(slice_3d_block_fence_rs, m)?)?;
    m.add_function(wrap_pyfunction!(export_block_glb_rs, m)?)?;
    m.add_function(wrap_pyfunction!(export_block_ply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(export_block_obj_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_displacement_strain_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_multi_hazard_signature_rs, m)?)?;
    m.add_function(wrap_pyfunction!(compute_insar_wrapped_phase_rs, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_insar_3d_rs, m)?)?;
    Ok(())
}
