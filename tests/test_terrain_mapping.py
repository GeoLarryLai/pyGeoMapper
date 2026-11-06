# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Terrain complexity, mapping and cross-section tests for pyGeoMapper:
- Rust 2D Mexican-hat Continuous Wavelet Transform (2D CWT) & multiscale complexity
- Spatial-block holdout leakage guard & supervised/unsupervised geological auto-mapping
- 3-point / least-squares bed attitude recovery (< 0.5 deg error) & shallow geological cross-sections
- Real Sheep Mountain Anticline dataset & executed Notebooks 2 and 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from pygeomapper import (
    auto_mapping,
    cross_section,
    raster_io,
    terrain_analysis,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "example_data"


def test_cwt2d_mexican_hat_wavelength_recovery_and_scale_invariance() -> None:
    """Verify synthetic 2D sinusoid wavelength recovery (< 2% error) and zero response on linear ramps."""
    dx = 5.0
    xs = np.arange(112, dtype=np.float64) * dx
    ys = np.arange(112, dtype=np.float64) * dx
    xx, yy = np.meshgrid(xs, ys)

    # 1. Pure linear ramp z = 0.08*x - 0.05*y + 100 should have near-zero CWT response in interior
    ramp = 100.0 + 0.08 * xx - 0.05 * yy
    cwt_ramp = terrain_analysis.compute_cwt2d_mexican_hat(ramp, scale_m=20.0, pixel_size_m=dx)
    interior_ramp = cwt_ramp["cwt"][20:-20, 20:-20]
    assert np.max(np.abs(interior_ramp)) < 1e-9

    # 2. 2D sinusoid with wavelength lambda = 120 m (characteristic Mexican-hat scale s = 27.01 m)
    target_lambda = 120.0
    syn = 20.0 * np.sin(2.0 * np.pi * xx / target_lambda) + 15.0 * np.cos(
        2.0 * np.pi * yy / target_lambda
    )
    scales_m = [12.0, 16.0, 21.0, 27.0, 35.0, 45.0, 60.0]
    ms = terrain_analysis.run_pytopocomplexity(syn, scales_m=scales_m, dx=dx, dy=dx)
    assert "Lai" in ms["citation"]
    core_wl = ms["dominant_wavelength_m"][24:-24, 24:-24]
    median_wl = float(np.nanmedian(core_wl))
    assert abs(median_wl - target_lambda) / target_lambda < 0.02


def test_spatial_block_holdout_leakage_guard() -> None:
    """Ensure train_mask and test_mask are strictly disjoint and separated by buffer_px."""
    shape = (100, 120)
    for mode in ("west_east", "north_south", "checkerboard", "spatial_blocks"):
        split = auto_mapping.create_spatial_block_split(
            shape, mode=mode, test_fraction=0.30, n_blocks_x=4, n_blocks_y=4, buffer_px=2
        )
        tm = split["train_mask"]
        vm = split["test_mask"]
        bm = split["buffer_mask"]
        assert tm.shape == shape and vm.shape == shape and bm.shape == shape
        # Strict disjointness
        assert int(np.sum(tm & vm)) == 0
        assert int(np.sum(tm)) > 0 and int(np.sum(vm)) > 0
        assert int(np.sum(bm)) > 0
        # Verify 4-connected immediate neighbors of test_mask do not touch train_mask when buffer_px >= 2
        dilated_test = binary_dilation(vm, iterations=1)
        assert int(np.sum(dilated_test & tm)) == 0


def test_supervised_and_unsupervised_auto_mapping_on_sheep_mtn() -> None:
    """Test feature stack assembly, spatial smoothing, contact extraction, and supervised mapping on Sheep Mountain."""
    dem = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_dem_3dep.tif")
    s2 = raster_io.load_multiband_geotiff(DATA_DIR / "sheep_mtn_s2.tif")
    geol = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_geol_units.tif")

    fstack = auto_mapping.build_mapping_feature_stack(
        dem, spectral_stack=s2, cwt_scales_m=[30.0, 60.0, 120.0]
    )
    assert fstack.n_bands >= 25

    split = auto_mapping.create_spatial_block_split(
        dem, mode="spatial_blocks", test_fraction=0.30, buffer_px=2
    )
    res = auto_mapping.train_predict_supervised_units(
        fstack,
        geol,
        train_mask=split["train_mask"],
        method="hybrid",
        smooth_radius_px=2,
    )

    conf = res["confidence"].data
    ent = res["entropy"].data
    assert np.nanmin(conf) >= 0.0 and np.nanmax(conf) <= 1.0 + 1e-9
    assert np.nanmin(ent) >= 0.0 and np.nanmax(ent) <= 1.0 + 1e-9
    assert len(res["contacts"]) > 100

    metrics_test = auto_mapping.evaluate_mapping_accuracy(
        geol, res["pred_labels"], eval_mask=split["test_mask"]
    )
    metrics_full = auto_mapping.evaluate_mapping_accuracy(geol, res["pred_labels"])
    if not auto_mapping.HAS_SKLEARN:
        pytest.skip("accuracy thresholds need the random forest (pip install scikit-learn)")
    assert metrics_test["overall_accuracy"] > 0.60
    assert metrics_full["overall_accuracy"] > 0.80
    assert metrics_test["cohen_kappa"] > 0.50


def test_3point_bed_attitude_recovery_within_half_degree() -> None:
    """Verify exact 3-point / least-squares strike & dip recovery within < 0.5 deg across all quadrants."""
    rng = np.random.default_rng(123)
    test_planes = [
        (35.0, 18.0),
        (145.0, 32.0),
        (250.0, 47.0),
        (325.0, 24.0),
    ]
    for true_strike, true_dip in test_planes:
        true_dip_dir = (true_strike + 90.0) % 360.0
        tan_d = np.tan(np.deg2rad(true_dip))
        a = -tan_d * np.sin(np.deg2rad(true_dip_dir))
        b = -tan_d * np.cos(np.deg2rad(true_dip_dir))

        xs = rng.uniform(-300.0, 300.0, size=12)
        ys = rng.uniform(-300.0, 300.0, size=12)
        zs = 1400.0 + a * xs + b * ys

        att = cross_section.estimate_bed_attitude(xs, ys, zs, robust_trim=True)
        assert att is not None
        assert abs(att["strike_deg"] - true_strike) < 0.5
        assert abs(att["dip_deg"] - true_dip) < 0.5
        assert abs(att["dip_dir_deg"] - true_dip_dir) < 0.5
        assert att["r_squared"] > 0.999

        # Apparent dip along dip direction equals +true_dip; along strike equals 0
        app_parallel = cross_section.compute_apparent_dip(
            att["dip_deg"], att["dip_dir_deg"], true_dip_dir
        )
        app_strike = cross_section.compute_apparent_dip(
            att["dip_deg"], att["dip_dir_deg"], true_strike
        )
        assert abs(app_parallel - true_dip) < 0.05
        assert abs(app_strike) < 0.05


def test_geological_cross_section_geometry_and_fault_offset() -> None:
    """Verify cross-section construction, subsurface unit grid, and fault offset on Sheep Mountain."""
    dem = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_dem_3dep.tif")
    geol = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_geol_units.tif")
    meta = json.loads((DATA_DIR / "sheep_mtn_units.json").read_text())
    unit_metadata = {u["unit_id"]: u for u in meta["units"]}

    attitudes = cross_section.estimate_contact_attitudes(
        geol, dem, window_m=400.0, min_points=5, min_relief_m=5.0, min_r2=0.45
    )
    assert len(attitudes) >= 20

    start_xy = (726900.0, 4942800.0)
    end_xy = (729700.0, 4944600.0)
    sec = cross_section.build_geological_cross_section(
        dem, geol, start_xy=start_xy, end_xy=end_xy, attitudes=attitudes, depth_m=350.0
    )
    assert len(sec["contact_crossings"]) >= 5
    assert sec["subsurface_units"].shape == (len(sec["z_levels_m"]), len(sec["distance_m"]))
    # Top row is above max(z_topo), so subsurface_units there must be all NaN
    assert np.all(np.isnan(sec["subsurface_units"][-1, :]))
    # Bottom row is below min(z_topo), so subsurface_units there must be all finite unit IDs
    assert np.all(np.isfinite(sec["subsurface_units"][0, :]))

    tip = cross_section.locate_fault_tip_along_section(dem, start_xy, end_xy)
    fault_3d = cross_section.reconstruct_planar_fault(
        dem,
        strike_deg=140.0,
        dip_deg=50.0,
        throw_m=float(np.clip(0.5 * tip["section_relief_m"], 80.0, 200.0)),
        name="Sheep Mountain basement thrust",
        tip_section=(start_xy, end_xy, tip["distance_m"]),
        z0=tip["z"],
        published={"strike_deg": 140.0, "dip_deg": 50.0},
        inferred={"throw_m": tip["section_relief_m"]},
    )
    fault_2d = cross_section.section_fault_params(fault_3d, start_xy, end_xy)
    assert fault_2d["dip_deg"] < 0  # SW dip toward A on A→A′
    assert abs(fault_2d["dip_deg"] + 50.0) < 5.0
    assert abs(fault_2d["distance_m"] - tip["distance_m"]) < 1.0

    sec_fault = cross_section.build_geological_cross_section(
        dem,
        geol,
        start_xy=start_xy,
        end_xy=end_xy,
        attitudes=attitudes,
        depth_m=350.0,
        faults=[fault_2d],
    )
    assert len(sec_fault["fault_traces"]) == 1
    assert sec_fault["fault_traces"][0]["dip_deg"] < 0
    fig, _ = cross_section.plot_geological_cross_section(sec_fault, unit_metadata=unit_metadata)
    plt.close(fig)


def test_cross_section_contacts_follow_fault_and_stay_in_order() -> None:
    """Projected contacts agree with the unit colors on both sides of a dipping fault (either dip direction),
    and horizons stay in stratigraphic order (older below younger)."""
    dem = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_dem_3dep.tif")
    geol = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_geol_units.tif")
    attitudes = cross_section.estimate_contact_attitudes(
        geol, dem, window_m=400.0, min_points=5, min_relief_m=5.0, min_r2=0.45
    )
    for dip in (65.0, -65.0):
        sec = cross_section.build_geological_cross_section(
            dem, geol, start_xy=(726900.0, 4942800.0), end_xy=(729700.0, 4944600.0), attitudes=attitudes,
            depth_m=350.0, faults=[{"distance_m": 1350.0, "dip_deg": dip, "throw_m": 55.0, "name": "F"}],
        )
        pcs = sec["projected_contacts"]
        upper_units = [max(pc["unit_a"], pc["unit_b"]) for pc in pcs]
        assert upper_units == sorted(upper_units), "horizons out of stratigraphic order"
        zl, units, z_topo = sec["z_levels_m"], sec["subsurface_units"], sec["z_topo_m"]
        step = float(zl[1] - zl[0])
        checked = agree = 0
        for pc in pcs:
            lower, upper = min(pc["unit_a"], pc["unit_b"]), max(pc["unit_a"], pc["unit_b"])
            for z_branch in pc["z_subsurface_branches_m"]:
                inside = np.isfinite(z_branch) & (z_branch < z_topo - 25.0) & (z_branch > zl[0] + 2.0 * step)
                for j in np.where(inside)[0]:
                    i_below = int(np.searchsorted(zl, z_branch[j] - 2.0 * step))
                    i_above = int(np.searchsorted(zl, z_branch[j] + 2.0 * step))
                    u_below, u_above = units[i_below, j], units[i_above, j]
                    if np.isfinite(u_below) and np.isfinite(u_above):
                        checked += 1
                        agree += int(u_below <= lower and u_above >= upper)
        assert checked > 200 and agree / checked > 0.95, (dip, checked, agree)


def test_notebooks_2_and_3_executed_cleanly() -> None:
    """Saved notebook JSON only: every code cell has an execution count, no error outputs, and the figures exist.
    This does NOT re-run the notebooks; re-execute them (e.g. `jupyter nbconvert --execute --inplace`) before a release."""
    for nb_name in (
        "pyGeoMapper_2_terrain_complexity_and_cwt.ipynb",
        "pyGeoMapper_3_auto_mapping_and_cross_section.ipynb",
    ):
        nb_path = REPO_ROOT / "notebooks" / nb_name
        assert nb_path.exists(), f"Missing {nb_name}"
        nb = json.loads(nb_path.read_text())
        for idx, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is not None, f"{nb_name} cell {idx} not executed"
                for out in cell.get("outputs", []):
                    assert out.get("output_type") != "error", f"{nb_name} cell {idx} had error"

    for fig_name in (
        "oso_cwt_change.png",
        "cwt_multiscale.png",
        "auto_mapping_validation.png",
        "cross_section.png",
    ):
        p_img = REPO_ROOT / "image" / fig_name
        assert p_img.exists() and p_img.stat().st_size > 20_000



def test_toggleable_feature_modes_and_alphaearth_comparison() -> None:
    """Verify toggleable feature_mode ('engineered' | 'alphaearth' | 'combined') and comparison."""
    dem = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_dem_3dep.tif")
    s2 = raster_io.load_multiband_geotiff(
        DATA_DIR / "sheep_mtn_s2.tif",
        band_names=["blue", "green", "red", "nir", "swir1", "swir2"],
    )
    ae = raster_io.load_multiband_geotiff(
        DATA_DIR / "sheep_mtn_alphaearth_2024.tif",
        dequantize_alphaearth=True,
    )
    geol = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_geol_units.tif")

    f_eng = auto_mapping.build_mapping_feature_stack(
        dem, spectral_stack=s2, cwt_scales_m=[30.0, 60.0, 120.0, 200.0], feature_mode="engineered"
    )
    f_ae = auto_mapping.build_mapping_feature_stack(
        dem, embedding_stack=ae, feature_mode="alphaearth"
    )
    f_comb = auto_mapping.build_mapping_feature_stack(
        dem,
        spectral_stack=s2,
        embedding_stack=ae,
        cwt_scales_m=[30.0, 60.0, 120.0, 200.0],
        feature_mode="combined",
    )
    assert f_eng.nbands == 29
    assert f_ae.nbands == 64
    assert f_comb.nbands == 93

    split = auto_mapping.create_spatial_block_split(
        dem, mode="west_east", test_fraction=0.35, buffer_px=2
    )
    comp = auto_mapping.compare_feature_approaches_supervised(
        dem=dem,
        spectral_stack=s2,
        embedding_stack=ae,
        reference_labels=geol,
        train_mask=split["train_mask"],
        test_mask=split["test_mask"],
        cwt_scales_m=[30.0, 60.0, 120.0, 200.0],
        method="hybrid",
        smooth_radius_px=2,
    )
    assert comp["test_evaluations"]["engineered"]["overall_accuracy"] > 0.70
    assert comp["test_evaluations"]["alphaearth"]["overall_accuracy"] > 0.85
    assert comp["test_evaluations"]["combined"]["overall_accuracy"] > 0.85
    assert comp["agreement_engineered_vs_alphaearth"] > 0.85
