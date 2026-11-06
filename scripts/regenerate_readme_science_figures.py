#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regenerate README science figures from the current 0.2 package (no git PNG blobs)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pygeomapper import (  # noqa: E402
    cross_section,
    plotting,
    raster_io,
    terrain_analysis,
    viewer_3d,
)

IMG = ROOT / "image"
DATA = ROOT / "example_data"
HAZ = DATA / "hazards"
IMG.mkdir(exist_ok=True)
ART = Path("/opt/cursor/artifacts")
ART.mkdir(parents=True, exist_ok=True)


def _copy_artifact(path: Path) -> None:
    import shutil

    dest = ART / path.name
    shutil.copy2(path, dest)
    print(f"artifact: {dest} ({dest.stat().st_size / 1e3:.0f} KB)")


def regenerate_cwt_multiscale() -> Path:
    sheep_dem = raster_io.load_geotiff(DATA / "sheep_mtn_dem_3dep.tif")
    oso_pre = raster_io.load_geotiff(DATA / "oso_dem_pre_cop30.tif")
    oso_post = raster_io.load_geotiff(DATA / "oso_dem_post_3dep.tif")
    sheep_hs = terrain_analysis.compute_hillshade(sheep_dem)
    sheep_cwt = terrain_analysis.compute_multiscale_complexity(
        sheep_dem, scales_m=[30.0, 60.0, 120.0, 220.0]
    )
    oso_comp = terrain_analysis.compare_complexity_change(
        oso_pre, oso_post, scales_m=[15.0, 30.0, 60.0, 120.0]
    )
    delta_c = oso_comp["delta_complexity"].data

    fig, axes = plt.subplots(2, 3, figsize=(16, 10.6), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.06, hspace=0.08, wspace=0.06)
    ext_s = [
        sheep_dem.bounds[0] / 1e3,
        sheep_dem.bounds[2] / 1e3,
        sheep_dem.bounds[1] / 1e3,
        sheep_dem.bounds[3] / 1e3,
    ]
    p_start, p_end = (726900.0, 4942800.0), (729700.0, 4944600.0)

    ax = axes[0, 0]
    ax.imshow(sheep_hs.data, cmap="gray", extent=ext_s, origin="upper")
    im0 = ax.imshow(sheep_dem.data, cmap="terrain", alpha=0.58, extent=ext_s, origin="upper")
    ax.plot(
        [p_start[0] / 1e3, p_end[0] / 1e3],
        [p_start[1] / 1e3, p_end[1] / 1e3],
        "r--",
        lw=2.0,
        label="Swath A–A'",
    )
    ax.set_title("(a) Sheep Mountain DEM + hillshade", fontweight="bold", pad=8)
    ax.set_xlabel("UTM Easting (km, Zone 12N)")
    ax.set_ylabel("UTM Northing (km)")
    ax.legend(loc="lower right", fontsize=8)
    plt.colorbar(im0, ax=ax, fraction=0.046, pad=0.04, label="Elevation (m)")

    ax = axes[0, 1]
    cwt_30 = sheep_cwt["cwt_stack"].band("cwt_30m").data
    vmax_w = float(np.nanpercentile(np.abs(cwt_30), 98))
    im1 = ax.imshow(cwt_30, cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w, extent=ext_s, origin="upper")
    ax.set_title("(b) CWT response W (s = 30 m): ridges + / valleys −", fontweight="bold", pad=8)
    ax.set_xlabel("UTM Easting (km, Zone 12N)")
    ax.set_ylabel("UTM Northing (km)")
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04, label="Wavelet Amplitude (m)")

    ax = axes[0, 2]
    im2 = ax.imshow(sheep_cwt["mean_complexity"].data, cmap="magma", extent=ext_s, origin="upper")
    ax.set_title("(c) Mean complexity (s = 30–220 m)", fontweight="bold", pad=8)
    ax.set_xlabel("UTM Easting (km, Zone 12N)")
    ax.set_ylabel("UTM Northing (km)")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="Mean |W(s)| / s")

    ax = axes[1, 0]
    im3 = ax.imshow(sheep_cwt["dominant_wavelength_m"].data, cmap="viridis", extent=ext_s, origin="upper")
    ax.set_title("(d) Dominant wavelength λ_dom", fontweight="bold", pad=8)
    ax.set_xlabel("UTM Easting (km, Zone 12N)")
    ax.set_ylabel("UTM Northing (km)")
    plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04, label="Wavelength λ (m)")

    ax = axes[1, 1]
    ext_o = [
        oso_post.bounds[0] / 1e3,
        oso_post.bounds[2] / 1e3,
        oso_post.bounds[1] / 1e3,
        oso_post.bounds[3] / 1e3,
    ]
    oso_hs = terrain_analysis.compute_hillshade(oso_post)
    ax.imshow(oso_hs.data, cmap="gray", extent=ext_o, origin="upper")
    vlim_o = float(np.nanpercentile(np.abs(delta_c), 98))
    im4 = ax.imshow(delta_c, cmap="PuOr_r", vmin=-vlim_o, vmax=vlim_o, alpha=0.78, extent=ext_o, origin="upper")
    ax.set_title("(e) Oso Δ complexity (post − pre)", fontweight="bold", pad=8)
    ax.set_xlabel("UTM Easting (km, Zone 10N)")
    ax.set_ylabel("UTM Northing (km)")
    plt.colorbar(im4, ax=ax, fraction=0.046, pad=0.04, label="ΔMean Complexity")

    ax = axes[1, 2]
    mprof = cross_section.extract_multilayer_profile(
        {
            "Elevation": sheep_dem,
            "Energy_30m": sheep_cwt["energy_stack"].band("energy_30m"),
            "Energy_120m": sheep_cwt["energy_stack"].band("energy_120m"),
        },
        start_xy=p_start,
        end_xy=p_end,
        n_samples=200,
        swath_width_m=120.0,
    )
    d_m = mprof["distance_m"]
    ax.plot(d_m, mprof["layers"]["Elevation"]["mean"], "k-", lw=2.0, label="Elevation (m)")
    ax.fill_between(
        d_m,
        mprof["layers"]["Elevation"]["min"],
        mprof["layers"]["Elevation"]["max"],
        color="gray",
        alpha=0.22,
    )
    ax.set_xlabel("Distance Along Profile A–A' (m)")
    ax.set_ylabel("Elevation (m NAVD88)", color="k")
    ax2 = ax.twinx()
    ax2.plot(d_m, mprof["layers"]["Energy_30m"]["mean"], color="#d95f02", lw=1.6, label="CWT Energy (s=30 m)")
    ax2.plot(
        d_m,
        mprof["layers"]["Energy_120m"]["mean"],
        color="#7570b3",
        lw=1.8,
        ls="--",
        label="CWT Energy (s=120 m)",
    )
    ax2.set_ylabel("Wavelet Energy |W(s)|/s", color="#7570b3")
    ax.set_title("(f) Swath A–A': elevation vs. CWT energy", fontweight="bold", pad=8)
    ax.grid(True, ls=":", alpha=0.45)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7.5)
    for a in axes.ravel()[:5]:
        plotting.use_plain_axes(a)

    fig.suptitle(
        "pyGeoMapper — Rust 2D Mexican-hat CWT and multiscale topographic complexity",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )
    out = IMG / "cwt_multiscale.png"
    plotting.save_figure(fig, out, max_width_px=1400, palette=True)
    return out


def _sheep_fault(dem):
    start_xy, end_xy = (726900.0, 4942800.0), (729700.0, 4944600.0)
    tip = cross_section.locate_fault_tip_along_section(dem, start_xy, end_xy)
    throw_m = float(np.clip(0.5 * tip["section_relief_m"], 80.0, 200.0))
    fault = cross_section.reconstruct_planar_fault(
        dem,
        strike_deg=140.0,
        dip_deg=50.0,
        throw_m=throw_m,
        name="Sheep Mountain basement thrust",
        tip_section=(start_xy, end_xy, tip["distance_m"]),
        z0=tip["z"],
        published={
            "strike_deg": 140.0,
            "dip_deg": 50.0,
            "dip_direction": "SW",
            "source": "Bellahsen; Stanton and Erslev 2004",
        },
        inferred={
            "tip_distance_m": tip["distance_m"],
            "tip_xy": (tip["x"], tip["y"]),
            "z0_m": tip["z"],
            "throw_m": throw_m,
            "note": "Tip = DEM steep-forelimb toe on A–A'; throw ≈ ½ crest–basin relief for shallow demo",
        },
    )
    return start_xy, end_xy, fault


def regenerate_cross_section() -> Path:
    dem = raster_io.load_geotiff(DATA / "sheep_mtn_dem_3dep.tif")
    geol = raster_io.load_geotiff(DATA / "sheep_mtn_geol_units.tif")
    meta = json.loads((DATA / "sheep_mtn_units.json").read_text())
    unit_metadata = {u["unit_id"]: u for u in meta["units"]}
    attitudes = cross_section.estimate_contact_attitudes(
        geol, dem, window_m=400.0, min_points=5, min_relief_m=5.0, min_r2=0.45
    )
    start_xy, end_xy, fault_3d = _sheep_fault(dem)
    fault_2d = cross_section.section_fault_params(fault_3d, start_xy, end_xy)
    print("cross-section fault published:", fault_3d["published"])
    print("cross-section fault inferred:", fault_3d["inferred"])
    print("section params:", fault_2d)

    sec_u = cross_section.build_geological_cross_section(
        dem, geol, start_xy=start_xy, end_xy=end_xy, attitudes=attitudes, depth_m=360.0, faults=None
    )
    sec_f = cross_section.build_geological_cross_section(
        dem, geol, start_xy=start_xy, end_xy=end_xy, attitudes=attitudes, depth_m=360.0, faults=[fault_2d]
    )
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.2), layout="constrained")
    cross_section.plot_geological_cross_section(
        sec_u,
        unit_metadata=unit_metadata,
        vertical_exaggeration=1.6,
        ax=axes[0],
        title="(a) Sheep Mountain Anticline cross-section A–A' (SW core to NE limb, 3-point attitudes)",
    )
    cross_section.plot_geological_cross_section(
        sec_f,
        unit_metadata=unit_metadata,
        vertical_exaggeration=1.6,
        ax=axes[1],
        title=(
            f"(b) Same section with SW-dipping ~50° basement thrust "
            f"(apparent {fault_2d['dip_deg']:.0f}°, throw = +{fault_2d['throw_m']:.0f} m inferred)"
        ),
    )
    out = IMG / "cross_section.png"
    plotting.save_figure(fig, out, max_width_px=1600)
    plt.close(fig)
    return out


def regenerate_3d_block() -> tuple[Path, Path]:
    dem = raster_io.load_geotiff(DATA / "sheep_mtn_dem_3dep.tif")
    geol = raster_io.load_geotiff(DATA / "sheep_mtn_geol_units.tif")
    units_meta = json.loads((DATA / "sheep_mtn_units.json").read_text())
    attitudes = cross_section.estimate_contact_attitudes(geol, dem, window_m=320.0)
    start_xy, end_xy, fault_spec = _sheep_fault(dem)

    block_plain = viewer_3d.build_3d_block_diagram(
        dem=dem,
        geol_labels=geol,
        units_metadata=units_meta,
        attitudes=attitudes,
        z_bottom=750.0,
        nz=64,
        fault_spec=fault_spec,
        include_fault=False,
        cutaway_fraction=(0.48, 0.48, 0.36),
    )
    block_f = viewer_3d.build_3d_block_diagram(
        dem=dem,
        geol_labels=geol,
        units_metadata=units_meta,
        attitudes=attitudes,
        z_bottom=750.0,
        nz=64,
        fault_spec=fault_spec,
        include_fault=True,
        cutaway_fraction=(0.48, 0.48, 0.36),
    )

    fig, _ = viewer_3d.plot_3d_section_comparison(
        block_plain,
        block_f,
        start_xy,
        end_xy,
        vertical_exaggeration=1.7,
        elevation_deg=22.0,
        n_samples=160,
        show_fault_plane=True,
        titles=(
            "(a) 3D section A–A′ (no fault)",
            f"(b) 3D section A–A′ + SW-dipping ~{fault_spec['dip_deg']:.0f}° basement thrust "
            f"(throw +{fault_spec['throw_m']:.0f} m inferred)",
        ),
    )
    out1 = IMG / "3d_block_diagram.png"
    plotting.save_figure(fig, out1, max_width_px=1500)
    plt.close(fig)

    fence_panels = viewer_3d.slice_3d_fence_diagram(block_f, transects=[(start_xy, end_xy)], n_samples=160)
    fence_panels[0]["name"] = "Fence A-A'"
    section_az = float(np.rad2deg(np.arctan2(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])) % 360.0)
    view_az = float((section_az + 180.0) % 360.0 - 180.0)
    fig = plt.figure(figsize=(16, 7.2), layout="constrained")
    ax1 = fig.add_subplot(121, projection="3d")
    viewer_3d.plot_3d_cutaway_block(
        block_f,
        vertical_exaggeration=1.7,
        azimuth_deg=-52.0,
        elevation_deg=30.0,
        face_stride=1,
        show_fault_plane=True,
        show_legend=True,
        ax=ax1,
        title="(a) 3D cutaway block (fault plane on)",
    )
    ax2 = fig.add_subplot(122, projection="3d")
    viewer_3d.plot_3d_fence_diagram(
        block_f,
        fence_panels=fence_panels,
        vertical_exaggeration=1.7,
        azimuth_deg=view_az,
        elevation_deg=22.0,
        show_fault_plane=True,
        show_wireframe_surface=True,
        show_legend=False,
        ax=ax2,
        title="(b) Single 3D A–A′ fence + reconstructed fault plane",
    )
    out2 = IMG / "3d_cutaway_and_fence.png"
    plotting.save_figure(fig, out2, max_width_px=1500)
    plt.close(fig)
    return out1, out2


def regenerate_case1a() -> Path:
    # Minimal helpers mirrored from notebook 5
    from scipy.ndimage import zoom

    def read_scaled(path, band_1based):
        with rasterio.open(path) as src:
            arr = src.read(band_1based).astype(np.float64)
            scale = float(src.tags(band_1based).get("scale_factor", "1"))
            nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        return arr * scale

    def hillshade_tiles(paths, cellsize=90.0, z_factor=1.0, step=1):
        tiles = []
        for p in paths:
            with rasterio.open(p) as src:
                z = src.read(1).astype(np.float64)
                if src.nodata is not None:
                    z = np.where(z == src.nodata, np.nan, z)
                tiles.append(z)
        dem = np.vstack(tiles)
        if step > 1:
            dem = dem[::step, ::step]
            cellsize = cellsize * step
        dem = np.where(np.isfinite(dem), dem, np.nanmedian(dem))
        gy, gx = np.gradient(dem * z_factor, cellsize)
        slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
        aspect = np.arctan2(-gx, gy)
        altitude = np.deg2rad(45.0)
        azimuth = np.deg2rad(315.0)
        hs = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)
        return np.clip(hs, 0, 1)

    def block_vectors(ue, un, extent, k, min_frac=0.5):
        rows, cols = ue.shape
        br, bc = rows // k, cols // k
        E = np.full((br, bc), np.nan)
        N = np.full((br, bc), np.nan)
        lon0, lon1, lat0, lat1 = extent
        resx = (lon1 - lon0) / cols
        resy = (lat1 - lat0) / rows
        xs, ys = [], []
        for i in range(br):
            for j in range(bc):
                eblk = ue[i * k : (i + 1) * k, j * k : (j + 1) * k]
                nblk = un[i * k : (i + 1) * k, j * k : (j + 1) * k]
                m = np.isfinite(eblk) & np.isfinite(nblk)
                if m.mean() >= min_frac:
                    E[i, j] = np.nanmedian(eblk[m])
                    N[i, j] = np.nanmedian(nblk[m])
                xs.append(lon0 + (j + 0.5) * k * resx)
                ys.append(lat1 - (i + 0.5) * k * resy)
        X = np.array(xs).reshape(br, bc)
        Y = np.array(ys).reshape(br, bc)
        return X, Y, E, N

    def draw_vectors(ax, X, Y, E, N, scale=28.0, ref=2.0, min_mag=0.3, width=0.008):
        m = np.isfinite(E) & np.isfinite(N) & (np.hypot(E, N) >= min_mag)
        ax.quiver(
            X[m],
            Y[m],
            E[m],
            N[m],
            angles="xy",
            scale_units="xy",
            scale=scale,
            width=width,
            color="k",
            zorder=5,
        )

    IFG_P, OFF_P = HAZ / "myanmar_licsar_106D_ifg.tif", HAZ / "myanmar_licsar_offsets.tif"
    with rasterio.open(IFG_P) as src:
        tg = src.tags()
    extent_mm = [float(tg["lon_min"]), float(tg["lon_max"]), float(tg["lat_min"]), float(tg["lat_max"])]
    mm_lon_min, mm_lon_max, mm_lat_min, mm_lat_max = extent_mm
    wrapped_real = read_scaled(IFG_P, 1)
    azi_106D = read_scaled(OFF_P, 1)
    rng_106D = read_scaled(OFF_P, 2)
    u_north_real = read_scaled(OFF_P, 5)
    u_east_real = read_scaled(OFF_P, 6)
    res_mm = (mm_lon_max - mm_lon_min) / wrapped_real.shape[1]
    X_mm, Y_mm, E_mm, N_mm = block_vectors(u_east_real, u_north_real, extent_mm, k=round(0.1 / res_mm), min_frac=0.5)
    CM_PER_CYCLE = 2.77
    phase_cm = np.mod(wrapped_real, 2 * np.pi) / (2 * np.pi) * CM_PER_CYCLE
    with rasterio.open(HAZ / "myanmar_eq_dem_north.tif") as src:
        dem_arcsec = round(src.res[0] * 3600)
    hs_mm = hillshade_tiles(
        [HAZ / "myanmar_eq_dem_north.tif", HAZ / "myanmar_eq_dem_south.tif"],
        cellsize=30.0 * dem_arcsec,
        z_factor=1.0,
        step=max(1, 3 // dem_arcsec),
    )
    # Match notebook hillshade shape to display grid if needed
    if hs_mm.shape != wrapped_real[:: max(1, round(0.002 / res_mm)), :: max(1, round(0.002 / res_mm))].shape:
        step_d = max(1, round(0.002 / res_mm))
        target = wrapped_real[::step_d, ::step_d].shape
        hs_mm = zoom(hs_mm, (target[0] / hs_mm.shape[0], target[1] / hs_mm.shape[1]), order=1)
    else:
        step_d = max(1, round(0.002 / res_mm))

    bwr_sar = LinearSegmentedColormap.from_list(
        "bwr_sar", ["#08306b", "#6baed6", "#ffffff", "#fb6a4a", "#67000d"], N=256
    )
    sar_rainbow = LinearSegmentedColormap.from_list(
        "sar_rainbow",
        ["#ff0000", "#ffff00", "#00ff00", "#00ffff", "#0000ff", "#ff00ff", "#ff0000"],
        N=256,
    )

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 12.8), sharey=True, layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.06, hspace=0.06, wspace=0.04)
    fig.suptitle(
        "Case 1A — 2025 Mw 7.7 Myanmar (Sagaing): REAL Sentinel-1 (COMET LiCSAR, us7000pn9s)",
        fontsize=10,
        fontweight="bold",
        color="#0f172a",
    )
    try:
        fig.get_layout_engine().set(rect=(0, 0.02, 1, 0.97))
    except Exception:
        pass
    panels = [
        (u_north_real, bwr_sar, -5, 5, "N-S displacement (meters)", "a  N–S displacement"),
        (azi_106D, bwr_sar, -5, 5, "Azi offset (meters)", "b  Azimuth offset (106D)"),
        (rng_106D, bwr_sar, -1, 1, "Range offset (meters)", "c  Range offset (106D)"),
        (phase_cm, sar_rainbow, 0, CM_PER_CYCLE, "LOS phase (range increase, cm)", "d  Wrapped phase (106D)"),
    ]
    geo_aspect = 1.0 / np.cos(np.deg2rad(0.5 * (mm_lat_min + mm_lat_max)))
    for ax, (arr, cmap, vmin, vmax, clabel, title) in zip(axes, panels):
        ax.imshow(hs_mm, cmap="gray", extent=extent_mm, origin="upper", vmin=0.0, vmax=1.0, aspect="auto")
        im = ax.imshow(
            np.ma.masked_invalid(arr[::step_d, ::step_d]),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=extent_mm,
            origin="upper",
            alpha=0.60,
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=6)
        ax.text(
            0.03,
            0.995,
            "REAL DATA",
            transform=ax.transAxes,
            fontsize=7.5,
            color="#065f46",
            fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, edgecolor="none"),
        )
        for cx, cy, cname in [(96.08, 21.98, "Mandalay"), (96.13, 19.76, "Naypyitaw")]:
            ax.scatter([cx], [cy], marker="s", s=30, facecolor="white", edgecolor="black", lw=1.3, zorder=6)
            ax.text(
                cx + 0.05,
                cy,
                cname,
                fontsize=8,
                va="center",
                zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.85, edgecolor="none"),
            )
        ax.set_xlim(mm_lon_min, mm_lon_max)
        ax.set_ylim(mm_lat_min, mm_lat_max)
        ax.set_aspect(geo_aspect)
        ax.set_xticks([96.0])
        ax.set_xticklabels(["96°E"])
        ax.set_yticks(np.arange(19, 24))
        ax.set_yticklabels([f"{v}°N" for v in range(19, 24)])
        fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02).set_label(clabel, fontsize=8.5)
    draw_vectors(axes[0], X_mm, Y_mm, E_mm, N_mm, scale=28.0, ref=2.0, min_mag=0.3, width=0.008)
    # north arrow / scale on panel a
    axes[0].annotate(
        "",
        xy=(mm_lon_min + 0.12, mm_lat_min + 0.55),
        xytext=(mm_lon_min + 0.12, mm_lat_min + 0.25),
        arrowprops=dict(arrowstyle="->", color="k", lw=1.4),
    )
    axes[0].text(mm_lon_min + 0.12, mm_lat_min + 0.62, "N", ha="center", fontsize=9, fontweight="bold")
    axes[0].plot([mm_lon_min + 0.25, mm_lon_min + 0.25 + 50 / 111.0], [mm_lat_min + 0.25, mm_lat_min + 0.25], "k-", lw=2)
    axes[0].text(mm_lon_min + 0.25 + 25 / 111.0, mm_lat_min + 0.32, "50 km", ha="center", fontsize=8)
    fig.text(
        0.5,
        0.005,
        "a: N–S + vectors (11 km) · b–c: 106D offsets · d: 106D ifg 2025-03-24→04-05 · α=0.60 GLO-30\n"
        "Contains modified Copernicus Sentinel-1 data (2025) processed by COMET LiCSAR (JASMIN). "
        "Copernicus DEM GLO-30. Blank = no coverage / masked by LiCSAR.",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#334155",
    )
    out = IMG / "case1a_myanmar_earthquake.jpg"
    plotting.save_figure(fig, out, max_width_px=1400, quality=82)
    return out


def main() -> None:
    t0 = time.perf_counter()
    outs = []
    print("=== CWT six-panel ===")
    outs.append(regenerate_cwt_multiscale())
    print("=== cross-section ===")
    outs.append(regenerate_cross_section())
    print("=== 3D A–A′ comparison ===")
    outs.extend(regenerate_3d_block())
    print("=== Case 1A LiCSAR 4-panel ===")
    outs.append(regenerate_case1a())
    for p in outs:
        print(f"saved {p} ({p.stat().st_size / 1e3:.0f} KB)")
        _copy_artifact(p)
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
