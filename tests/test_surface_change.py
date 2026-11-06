# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Surface-change, spectral and I/O tests for pyGeoMapper."""

from __future__ import annotations

from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np

import pygeomapper as pgm
from pygeomapper import _core


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLE_DIR = REPO_ROOT / "example_data"


def _make_synthetic_dem(rows: int = 80, cols: int = 90, dx: float = 10.0, dy: float = 10.0) -> pgm.GeoGrid:
    """Create a synthetic CRS-aware UTM GeoGrid with known analytical slope/plane."""
    transform = (500000.0, dx, 0.0, 5300000.0, 0.0, -dy)
    xs = np.arange(cols, dtype=np.float64) * dx
    ys = np.arange(rows, dtype=np.float64) * dy
    xx, yy = np.meshgrid(xs, ys)
    # Plane z = 100 + 0.10 * x - 0.05 * y
    z = 100.0 + 0.10 * xx - 0.05 * yy
    return pgm.GeoGrid(
        data=z,
        transform=transform,
        crs="EPSG:32610",
        nodata=np.nan,
        name="synthetic_dem",
        units="m",
    )


def test_rust_core_loaded():
    """Verify compiled PyO3 Rust extension is loaded and exposes all 20 kernels."""
    assert pgm.HAS_RUST_CORE is True
    expected_kernels = [
        "compute_slope_aspect_rs",
        "compute_hillshade_rs",
        "compute_curvature_rs",
        "compute_roughness_tpi_rs",
        "normalized_difference_rs",
        "band_ratio_rs",
        "bare_soil_index_rs",
        "ferrous_iron_index_rs",
        "rational_two_band_index_rs",
        "compute_dod_rs",
        "detrend_dod_planar_rs",
        "estimate_nuth_kaab_shift_rs",
        "compute_volume_budget_rs",
        "otsu_threshold_rs",
        "classify_change_severity_rs",
        "connected_change_patches_rs",
        "embedding_change_metrics_rs",
        "embedding_dimension_importance_rs",
        "kmeans_features_rs",
        "sample_swath_profile_rs",
    ]
    for fn_name in expected_kernels:
        assert hasattr(_core, fn_name), f"Missing Rust kernel: {fn_name}"


def test_geogrid_crs_and_geotiff_roundtrip():
    """Verify GeoGrid CRS, affine transform, bounds, and GeoTIFF I/O roundtrip."""
    dem = _make_synthetic_dem(60, 70, dx=12.5, dy=12.5)
    assert dem.crs == "EPSG:32610"
    assert np.isclose(dem.dx, 12.5)
    assert np.isclose(abs(dem.dy), 12.5)
    assert np.isclose(dem.cell_area_m2, 156.25)

    xmin, ymin, xmax, ymax = dem.bounds
    assert np.isclose(xmin, 500000.0)
    assert np.isclose(ymax, 5300000.0)
    assert np.isclose(xmax - xmin, 70 * 12.5)
    assert np.isclose(ymax - ymin, 60 * 12.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_tif = Path(tmpdir) / "test_dem.tif"
        pgm.raster_io.save_geotiff(dem, out_tif)
        reloaded = pgm.raster_io.load_geotiff(out_tif)
        assert reloaded.crs == "EPSG:32610"
        assert np.allclose(reloaded.transform, dem.transform)
        assert np.allclose(reloaded.data, dem.data)

        # Test align_to_reference on a shifted grid
        shifted_transform = (500025.0, 12.5, 0.0, 5299975.0, 0.0, -12.5)
        shifted = pgm.GeoGrid(
            data=dem.data.copy(),
            transform=shifted_transform,
            crs="EPSG:32610",
            name="shifted",
            units="m",
        )
        aligned = pgm.raster_io.align_to_reference(shifted, dem, resampling="bilinear")
        assert aligned.shape == dem.shape
        assert aligned.crs == dem.crs
        assert np.allclose(aligned.transform, dem.transform)


def test_terrain_analysis_rust_vs_analytical():
    """Verify Horn (1981) slope/aspect, hillshade, curvature, and roughness against analytical values."""
    dem = _make_synthetic_dem(60, 60, dx=10.0, dy=10.0)
    slope_grid, aspect_grid = pgm.terrain_analysis.compute_slope_aspect(dem)

    # Interior pixels of plane z = 100 + 0.10*x_east - 0.05*y_south
    expected_slope_deg = np.rad2deg(np.arctan(np.hypot(0.10, 0.05)))
    interior_slope = slope_grid.data[2:-2, 2:-2]
    assert np.allclose(interior_slope, expected_slope_deg, atol=1e-6)
    assert aspect_grid.crs == "EPSG:32610"

    # Curvature of a linear plane must be zero in the interior
    curv = pgm.terrain_analysis.compute_curvature(dem)
    assert np.allclose(curv["laplacian"].data[2:-2, 2:-2], 0.0, atol=1e-9)
    assert np.allclose(curv["profile"].data[2:-2, 2:-2], 0.0, atol=1e-9)
    assert np.allclose(curv["planform"].data[2:-2, 2:-2], 0.0, atol=1e-9)

    # Hillshade & multi-azimuth hillshade bounds [0, 1]
    hs = pgm.terrain_analysis.compute_hillshade(dem, multi_azimuth=False)
    mhs = pgm.terrain_analysis.compute_hillshade(dem, multi_azimuth=True)
    assert np.nanmin(hs.data) >= 0.0 and np.nanmax(hs.data) <= 1.0
    assert np.nanmin(mhs.data) >= 0.0 and np.nanmax(mhs.data) <= 1.0

    # TPI of a planar surface is 0 in the interior
    rt = pgm.terrain_analysis.compute_roughness_and_tpi(dem, radius_px=2)
    assert np.allclose(rt["tpi"].data[3:-3, 3:-3], 0.0, atol=1e-9)
    assert np.all(rt["roughness"].data[3:-3, 3:-3] > 0.0)


def test_spectral_indices_rust_vs_numpy():
    """Verify Rust spectral & mineral indices match direct NumPy formulas to machine precision."""
    rng = np.random.default_rng(42)
    rows, cols = 50, 60
    transform = (500000.0, 10.0, 0.0, 5300000.0, 0.0, -10.0)
    band_names = ["blue", "green", "red", "nir", "swir1", "swir2"]
    arr = rng.uniform(0.05, 0.65, size=(rows, cols, 6)).astype(np.float64)
    # Inject a NaN pixel to verify NaN propagation
    arr[10, 15, :] = np.nan

    stack = pgm.RasterStack(
        data=arr,
        band_names=band_names,
        transform=transform,
        crs="EPSG:32610",
        units="reflectance",
    )
    suite = pgm.spectral_indices.compute_index_suite(
        stack,
        indices=(
            "NDVI",
            "NBR",
            "NDWI",
            "NDMI",
            "NDBI",
            "BSI",
            "ClayRatio",
            "FerricOxideRatio",
            "FerrousIronIndex",
            "CarbonateIndex",
        ),
    )

    blue, green, red, nir, swir1, swir2 = [arr[:, :, i] for i in range(6)]
    np_ndvi = (nir - red) / (nir + red)
    np_nbr = (nir - swir2) / (nir + swir2)
    np_ndwi = (green - nir) / (green + nir)
    np_ndmi = (nir - swir1) / (nir + swir1)
    np_ndbi = (swir1 - nir) / (swir1 + nir)
    np_bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))
    np_clay = swir1 / swir2
    np_ferric = red / blue
    np_ferrous = (swir1 / nir) + (green / red)
    np_carb = (red - green) / (red + green)

    assert np.allclose(suite["NDVI"].data, np_ndvi, equal_nan=True)
    assert np.allclose(suite["NBR"].data, np_nbr, equal_nan=True)
    assert np.allclose(suite["NDWI"].data, np_ndwi, equal_nan=True)
    assert np.allclose(suite["NDMI"].data, np_ndmi, equal_nan=True)
    assert np.allclose(suite["NDBI"].data, np_ndbi, equal_nan=True)
    assert np.allclose(suite["BSI"].data, np_bsi, equal_nan=True)
    assert np.allclose(suite["ClayRatio"].data, np_clay, equal_nan=True)
    assert np.allclose(suite["FerricOxideRatio"].data, np_ferric, equal_nan=True)
    assert np.allclose(suite["FerrousIronIndex"].data, np_ferrous, equal_nan=True)
    assert np.allclose(suite["CarbonateIndex"].data, np_carb, equal_nan=True)
    assert np.isnan(suite["NDVI"].data[10, 15])
    assert suite["NDVI"].crs == "EPSG:32610"


def test_dod_detrend_and_volume_conservation():
    """Verify planar detrending, Nuth & Kääb shift, and exact volume conservation on synthetic DoD."""
    dem1 = _make_synthetic_dem(100, 100, dx=10.0, dy=10.0)
    z2 = dem1.data.copy()

    # Add a known 20x20 pixel erosion pit of -5.0 m and a 20x20 pixel deposition mound of +4.0 m
    # Cell area = 100 m^2 -> 400 pixels * 100 m^2 = 40,000 m^2 each
    # Expected eroded volume = 400 * 100 * 5.0 = 200,000 m^3
    # Expected deposited volume = 400 * 100 * 4.0 = 160,000 m^3
    # Expected net volume = -40,000 m^3
    z2[20:40, 20:40] -= 5.0
    z2[60:80, 60:80] += 4.0

    # Add a planar tilt bias across the whole scene
    xs = np.arange(100, dtype=np.float64) * 10.0
    ys = np.arange(100, dtype=np.float64) * 10.0
    xx, yy = np.meshgrid(xs, ys)
    tilt = 1.25 + 0.0005 * xx - 0.0003 * yy

    stable_mask = np.ones((100, 100), dtype=bool)
    stable_mask[20:40, 20:40] = False
    stable_mask[60:80, 60:80] = False

    dem2_tilted = dem1.with_data(z2 + tilt, name="dem2_tilted")
    dod_res = pgm.surface_change.compute_dod(
        dem1, dem2_tilted, detrend_stable_mask=stable_mask
    )

    c0, cx, cy = dod_res["planar_bias_coeffs"]
    assert np.isclose(c0, 1.25, atol=1e-6)
    assert np.isclose(cx, 0.0005, atol=1e-8)
    assert np.isclose(cy, -0.0003, atol=1e-8)

    budget = pgm.surface_change.compute_volume_budget(dod_res["dod"], lod_threshold=0.5, sigma_dz=1.0)
    assert np.isclose(budget["eroded_area_m2"], 40000.0)
    assert np.isclose(budget["deposited_area_m2"], 40000.0)
    assert np.isclose(budget["eroded_volume_m3"], 200000.0, rtol=1e-6)
    assert np.isclose(budget["deposited_volume_m3"], 160000.0, rtol=1e-6)
    assert np.isclose(budget["net_volume_m3"], -40000.0, rtol=1e-6)


def test_otsu_and_connected_patches():
    """Verify Otsu threshold on bimodal distribution and 8-connected patch extraction."""
    bimodal = np.concatenate([np.full(500, 0.10), np.full(500, 0.80)]).reshape(25, 40)
    grid = pgm.GeoGrid(
        data=bimodal,
        transform=(500000.0, 10.0, 0.0, 5300000.0, 0.0, -10.0),
        crs="EPSG:32610",
    )
    thr = pgm.surface_change.otsu_threshold(grid)
    assert 0.25 < thr < 0.65

    # Create two distinct patches above threshold
    patch_arr = np.zeros((30, 30), dtype=np.float64)
    patch_arr[3:8, 4:10] = 2.5   # 5x6 = 30 pixels -> 3000 m^2
    patch_arr[18:22, 20:25] = 4.0  # 4x5 = 20 pixels -> 2000 m^2
    patch_grid = pgm.GeoGrid(
        data=patch_arr,
        transform=(500000.0, 10.0, 0.0, 5300000.0, 0.0, -10.0),
        crs="EPSG:32610",
    )
    labels_grid, patches = pgm.surface_change.extract_change_patches(
        patch_grid, threshold=1.0, min_pixels=5
    )
    assert labels_grid.crs == "EPSG:32610"
    assert len(patches) == 2
    p0, p1 = patches
    assert p0["pixel_count"] == 30 and np.isclose(p0["area_m2"], 3000.0)
    assert p1["pixel_count"] == 20 and np.isclose(p1["area_m2"], 2000.0)


def test_alphaearth_embedding_change_bounds_and_accuracy():
    """Verify 64-band embedding cosine similarity/distance bounds, L2 distance, and top-k bands."""
    rows, cols, dim = 20, 25, 64
    rng = np.random.default_rng(7)
    raw1 = rng.standard_normal((rows, cols, dim))
    u1 = raw1 / np.linalg.norm(raw1, axis=2, keepdims=True)
    u2 = u1.copy()
    # Flip direction at (5, 5) -> cosine similarity = -1, cosine distance = 2.0, L2 = 2.0
    u2[5, 5, :] = -u1[5, 5, :]
    # Perturb band A17 across half the image so A17 is the #1 changed band
    u2[10:, :, 17] += 1.5

    transform = (500000.0, 10.0, 0.0, 5300000.0, 0.0, -10.0)
    band_names = [f"A{i:02d}" for i in range(dim)]
    s1 = pgm.RasterStack(data=u1, band_names=band_names, transform=transform, crs="EPSG:32610")
    s2 = pgm.RasterStack(data=u2, band_names=band_names, transform=transform, crs="EPSG:32610")

    res = pgm.surface_change.compute_embedding_change(s1, s2)
    cos_sim = res["cosine_similarity"].data
    cos_dist = res["cosine_distance"].data
    l2_dist = res["l2_distance"].data

    assert np.nanmin(cos_sim) >= -1.0 - 1e-12 and np.nanmax(cos_sim) <= 1.0 + 1e-12
    assert np.nanmin(cos_dist) >= 0.0 - 1e-12 and np.nanmax(cos_dist) <= 2.0 + 1e-12
    assert np.isclose(cos_sim[5, 5], -1.0, atol=1e-12)
    assert np.isclose(cos_dist[5, 5], 2.0, atol=1e-12)
    assert np.isclose(l2_dist[5, 5], 2.0, atol=1e-12)
    assert np.isclose(cos_dist[2, 2], 0.0, atol=1e-12)
    assert res["top_changed_band_names"][0] == "A17"


def test_kmeans_clustering_and_swath_profile():
    """Verify deterministic Rust k-means++ clustering and bilinear swath profile extraction."""
    dem = _make_synthetic_dem(50, 50, dx=10.0, dy=10.0)
    prof = pgm.cross_section.extract_swath_profile(
        dem,
        start_xy=(500050.0, 5299950.0),
        end_xy=(500450.0, 5299950.0),
        n_samples=41,
        swath_width_m=40.0,
        n_cross=5,
    )
    assert prof["distance_m"].shape == (41,)
    assert np.isclose(prof["distance_m"][-1], 400.0)
    # Along constant Northing y = 5299950, z increases linearly with Easting at slope +0.10 m/m:
    dz_dx = np.diff(prof["center"]) / np.diff(prof["distance_m"])
    assert np.allclose(dz_dx, 0.10, atol=1e-6)

    # K-means clustering on 2-cluster synthetic stack
    feat = np.zeros((40, 40, 3), dtype=np.float64)
    feat[:20, :, :] = -2.0
    feat[20:, :, :] = +2.0
    stack = pgm.RasterStack(
        data=feat,
        band_names=["f1", "f2", "f3"],
        transform=dem.transform,
        crs="EPSG:32610",
    )
    km = pgm.auto_mapping.cluster_units_kmeans(stack, n_clusters=2)
    assert len(km["unit_summary"]) == 2
    assert km["labels"].data[:20, :].std() == 0.0
    assert km["labels"].data[20:, :].std() == 0.0
    assert km["labels"].data[0, 0] != km["labels"].data[30, 0]


def test_real_oso_example_data_and_viewer_3d():
    """Integration test on real Oso Landslide GeoTIFFs and AlphaEarth embeddings."""
    dem_pre = pgm.raster_io.load_geotiff(EXAMPLE_DIR / "oso_dem_pre_cop30.tif")
    dem_post = pgm.raster_io.load_geotiff(EXAMPLE_DIR / "oso_dem_post_3dep.tif")
    s2_17 = pgm.raster_io.load_multiband_geotiff(EXAMPLE_DIR / "oso_s2_20170826.tif")
    s2_24 = pgm.raster_io.load_multiband_geotiff(EXAMPLE_DIR / "oso_s2_20240906.tif")
    ae_17 = pgm.raster_io.load_multiband_geotiff(
        EXAMPLE_DIR / "oso_alphaearth_2017.tif", dequantize_alphaearth=True
    )
    ae_24 = pgm.raster_io.load_multiband_geotiff(
        EXAMPLE_DIR / "oso_alphaearth_2024.tif", dequantize_alphaearth=True
    )

    assert dem_pre.crs == "EPSG:32610"
    assert dem_post.crs == "EPSG:32610"
    assert s2_17.shape == (200, 200, 6)
    assert s2_24.shape == (200, 200, 6)
    assert ae_17.shape == (200, 200, 64)
    assert ae_24.shape == (200, 200, 64)

    raw_dod = dem_post.data - dem_pre.data
    stable_mask = np.abs(raw_dod) < 3.5
    dod_res = pgm.surface_change.compute_dod(dem_pre, dem_post, detrend_stable_mask=stable_mask)
    budget = pgm.surface_change.compute_volume_budget(dod_res["dod"], lod_threshold=2.0)
    assert budget["eroded_volume_m3"] > 1.0e6
    assert budget["deposited_volume_m3"] > 5.0e6

    ae_change = pgm.surface_change.compute_embedding_change(ae_17, ae_24)
    assert len(ae_change["top_changed_band_names"]) == 64

    # Test 2.5D hillshade drape and 3D plot helper
    hs = pgm.terrain_analysis.compute_hillshade(dem_post, multi_azimuth=True)
    rgb = pgm.viewer_3d.drape_overlay_on_hillshade(hs, dod_res["dod"], vmin=-40, vmax=40)
    assert rgb.shape == (200, 200, 3)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    pgm.viewer_3d.plot_3d_terrain_drape(dem_post, ae_change["cosine_distance"], stride=8, ax=ax)
    plt.close(fig)


def test_notebook_1_executed_cleanly():
    """Saved notebook JSON only: every code cell has an execution count and no error outputs.
    This does NOT re-run the notebook; re-execute it (e.g. `jupyter nbconvert --execute --inplace`) before a release."""
    nb_path = REPO_ROOT / "notebooks" / "pyGeoMapper_1_spectral_and_surface_change.ipynb"
    assert nb_path.exists()
    nb = nbformat.read(nb_path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code_cells) >= 8
    for idx, cell in enumerate(code_cells):
        assert cell.get("execution_count") is not None, f"Code cell {idx} was not executed"
        for out in cell.get("outputs", []):
            assert out.get("output_type") != "error", f"Error in cell {idx}: {out}"
