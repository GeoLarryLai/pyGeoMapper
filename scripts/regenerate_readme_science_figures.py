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


def _rgba_overlay_from_values(
    values: np.ndarray,
    cmap_name: str,
    vmin: float,
    vmax: float,
    *,
    zero_is_white: bool = False,
) -> np.ndarray:
    """Build an RGBA overlay; optionally map near-zero values to white / near-transparent."""
    data = np.asarray(values, dtype=np.float64)
    cmap = plt.get_cmap(cmap_name)
    span = max(float(vmax - vmin), 1e-9)
    normed = np.clip((data - vmin) / span, 0.0, 1.0)
    rgba = cmap(normed)
    if zero_is_white:
        # 0 → white & near-transparent; higher values denser (alpha rises with magnitude).
        alpha = np.clip(normed, 0.0, 1.0) ** 0.85
        rgba[..., 0] = (1.0 - alpha) * 1.0 + alpha * rgba[..., 0]
        rgba[..., 1] = (1.0 - alpha) * 1.0 + alpha * rgba[..., 1]
        rgba[..., 2] = (1.0 - alpha) * 1.0 + alpha * rgba[..., 2]
        rgba[..., 3] = 0.02 + 0.78 * alpha
    else:
        rgba[..., 3] = 0.65
    rgba[~np.isfinite(data)] = (0.0, 0.0, 0.0, 0.0)
    return rgba


def _make_s2_rgb(stack, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    rgb = np.stack(
        [stack.band("red").data, stack.band("green").data, stack.band("blue").data],
        axis=-1,
    ).astype(np.float64)
    out = np.zeros_like(rgb)
    for i in range(3):
        band = rgb[:, :, i]
        lo, hi = np.nanpercentile(band[np.isfinite(band)], [p_lo, p_hi])
        out[:, :, i] = np.clip((band - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return out


def regenerate_cwt_multiscale() -> Path:
    """Sheep Mountain–only CWT board (Oso moved to its own README figure).

    Layout (5 panels):
      Top:    (a) DEM+hillshade | (b) CWT ridges/valleys | (c) mean complexity on hillshade
      Bottom: (d) A–A' profile spanning two columns (width of a+b) | (e) λ_dom on hillshade
    """
    sheep_dem = raster_io.load_geotiff(DATA / "sheep_mtn_dem_3dep.tif")
    sheep_hs = terrain_analysis.compute_hillshade(sheep_dem)
    sheep_cwt = terrain_analysis.compute_multiscale_complexity(
        sheep_dem, scales_m=[30.0, 60.0, 120.0, 220.0]
    )

    fig = plt.figure(figsize=(16.0, 10.2), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.05, hspace=0.08, wspace=0.05)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_prof = fig.add_subplot(gs[1, 0:2])
    ax_lam = fig.add_subplot(gs[1, 2])

    ext_s = [
        sheep_dem.bounds[0] / 1e3,
        sheep_dem.bounds[2] / 1e3,
        sheep_dem.bounds[1] / 1e3,
        sheep_dem.bounds[3] / 1e3,
    ]
    p_start, p_end = (726900.0, 4942800.0), (729700.0, 4944600.0)

    # (a) DEM + hillshade
    ax_a.imshow(sheep_hs.data, cmap="gray", extent=ext_s, origin="upper")
    im0 = ax_a.imshow(sheep_dem.data, cmap="terrain", alpha=0.58, extent=ext_s, origin="upper")
    ax_a.plot(
        [p_start[0] / 1e3, p_end[0] / 1e3],
        [p_start[1] / 1e3, p_end[1] / 1e3],
        "r--",
        lw=2.0,
        label="Swath A–A'",
    )
    ax_a.set_title("(a) Sheep Mountain DEM + hillshade", fontweight="bold", pad=8)
    ax_a.set_xlabel("UTM Easting (km, Zone 12N)")
    ax_a.set_ylabel("UTM Northing (km)")
    ax_a.legend(loc="lower right", fontsize=8)
    fig.colorbar(im0, ax=ax_a, fraction=0.046, pad=0.04, label="Elevation (m)")

    # (b) CWT ridges / valleys (kept as a map panel — not turned into the overlay slot)
    cwt_30 = sheep_cwt["cwt_stack"].band("cwt_30m").data
    vmax_w = float(np.nanpercentile(np.abs(cwt_30), 98))
    im1 = ax_b.imshow(cwt_30, cmap="RdBu_r", vmin=-vmax_w, vmax=vmax_w, extent=ext_s, origin="upper")
    ax_b.set_title("(b) CWT response W (s = 30 m): ridges + / valleys −", fontweight="bold", pad=8)
    ax_b.set_xlabel("UTM Easting (km, Zone 12N)")
    ax_b.set_ylabel("UTM Northing (km)")
    fig.colorbar(im1, ax=ax_b, fraction=0.046, pad=0.04, label="Wavelet Amplitude (m)")

    # (c) Mean complexity as hillshade overlay; 0 → white / near-transparent
    comp = np.asarray(sheep_cwt["mean_complexity"].data, dtype=np.float64)
    vmax_c = float(np.nanpercentile(comp, 98))
    ax_c.imshow(sheep_hs.data, cmap="gray", extent=ext_s, origin="upper")
    rgba_c = _rgba_overlay_from_values(comp, "magma", 0.0, vmax_c, zero_is_white=True)
    ax_c.imshow(rgba_c, extent=ext_s, origin="upper")
    sm_c = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0.0, vmax_c))
    sm_c.set_array([])
    fig.colorbar(sm_c, ax=ax_c, fraction=0.046, pad=0.04, label="Mean |W(s)| / s")
    ax_c.set_title("(c) Mean complexity on hillshade (s = 30–220 m)", fontweight="bold", pad=8)
    ax_c.set_xlabel("UTM Easting (km, Zone 12N)")
    ax_c.set_ylabel("UTM Northing (km)")

    # (d) Swath A–A' profile — two columns wide (a+b), same row height as λ_dom map
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
    ax_prof.plot(d_m, mprof["layers"]["Elevation"]["mean"], "k-", lw=2.0, label="Elevation (m)")
    ax_prof.fill_between(
        d_m,
        mprof["layers"]["Elevation"]["min"],
        mprof["layers"]["Elevation"]["max"],
        color="gray",
        alpha=0.22,
    )
    ax_prof.set_xlabel("Distance Along Profile A–A' (m)")
    ax_prof.set_ylabel("Elevation (m NAVD88)", color="k")
    ax_prof2 = ax_prof.twinx()
    ax_prof2.plot(
        d_m, mprof["layers"]["Energy_30m"]["mean"], color="#d95f02", lw=1.6, label="CWT Energy (s=30 m)"
    )
    ax_prof2.plot(
        d_m,
        mprof["layers"]["Energy_120m"]["mean"],
        color="#7570b3",
        lw=1.8,
        ls="--",
        label="CWT Energy (s=120 m)",
    )
    ax_prof2.set_ylabel("Wavelet Energy |W(s)|/s", color="#7570b3")
    ax_prof.set_title("(d) Swath A–A': elevation vs. CWT energy", fontweight="bold", pad=8)
    ax_prof.grid(True, ls=":", alpha=0.45)
    lines1, labels1 = ax_prof.get_legend_handles_labels()
    lines2, labels2 = ax_prof2.get_legend_handles_labels()
    ax_prof2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7.5)

    # (e) Dominant wavelength on DEM hillshade (old d → old f slot as transparent overlay)
    lam = np.asarray(sheep_cwt["dominant_wavelength_m"].data, dtype=np.float64)
    vmin_l = float(np.nanpercentile(lam, 2))
    vmax_l = float(np.nanpercentile(lam, 98))
    ax_lam.imshow(sheep_hs.data, cmap="gray", extent=ext_s, origin="upper")
    rgba_l = _rgba_overlay_from_values(lam, "viridis", vmin_l, vmax_l, zero_is_white=False)
    ax_lam.imshow(rgba_l, extent=ext_s, origin="upper")
    sm_l = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin_l, vmax_l))
    sm_l.set_array([])
    fig.colorbar(sm_l, ax=ax_lam, fraction=0.046, pad=0.04, label="Wavelength λ (m)")
    ax_lam.set_title("(e) Dominant wavelength λ_dom on hillshade", fontweight="bold", pad=8)
    ax_lam.set_xlabel("UTM Easting (km, Zone 12N)")
    ax_lam.set_ylabel("UTM Northing (km)")

    for a in (ax_a, ax_b, ax_c, ax_lam):
        plotting.use_plain_axes(a)

    fig.suptitle(
        "pyGeoMapper — Rust 2D Mexican-hat CWT and multiscale topographic complexity (Sheep Mountain)",
        fontsize=12,
        fontweight="bold",
        y=1.01,
    )
    out = IMG / "cwt_multiscale.png"
    plotting.save_figure(fig, out, max_width_px=1400, palette=True)
    return out


def regenerate_oso_landslide() -> Path:
    """Dedicated Oso README figure: pre/post DEM+imagery drapes and DoD.

    InSAR around the March 2014 failure is not in the package (Sentinel-1 launched after;
    available ALOS/UAVSAR pairs are pre-creep or 2020–21, not co-event with these DEMs),
    so this ships panels (a)–(c) only.
    """
    from pygeomapper import surface_change

    dem_pre = raster_io.load_geotiff(DATA / "oso_dem_pre_cop30.tif")
    dem_post = raster_io.load_geotiff(DATA / "oso_dem_post_3dep.tif")
    s2_post = raster_io.load_multiband_geotiff(
        DATA / "oso_s2_20170826.tif",
        band_names=["blue", "green", "red", "nir", "swir1", "swir2"],
    )
    naip_path = DATA / "oso_naip_20130910.tif"
    if naip_path.is_file():
        naip = raster_io.load_multiband_geotiff(
            naip_path, band_names=["red", "green", "blue"]
        )
        pre_rgb = np.stack(
            [naip.band("red").data, naip.band("green").data, naip.band("blue").data],
            axis=-1,
        ).astype(np.float64)
        if np.nanmax(pre_rgb) > 1.5:
            pre_rgb = pre_rgb / 255.0
        pre_rgb = np.clip(pre_rgb, 0.0, 1.0)
        pre_photo_note = "USDA NAIP 2013-09-10"
    else:
        # Fallback only if NAIP tile missing: do not invent imagery.
        pre_rgb = None
        pre_photo_note = "hillshade only (no pre-2014 optical tile)"

    hs_pre = terrain_analysis.compute_hillshade(dem_pre)
    hs_post = terrain_analysis.compute_hillshade(dem_post)
    raw_dod = surface_change.compute_dod(dem_pre, dem_post)
    stable = np.abs(raw_dod["raw_dod"].data) < 4.0
    dod_res = surface_change.compute_dod(dem_pre, dem_post, detrend_stable_mask=stable)
    dod = dod_res["dod"]
    vol = surface_change.compute_volume_budget(dod, lod_threshold=2.0)

    ext = [
        dem_post.bounds[0] / 1e3,
        dem_post.bounds[2] / 1e3,
        dem_post.bounds[1] / 1e3,
        dem_post.bounds[3] / 1e3,
    ]
    post_rgb = _make_s2_rgb(s2_post)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), layout="constrained")
    fig.suptitle(
        "2014 Oso landslide (WA) — pre/post DEM drapes and DEM of Difference",
        fontsize=12.5,
        fontweight="bold",
    )

    axes[0].imshow(hs_pre.data, cmap="gray", extent=ext, origin="upper")
    if pre_rgb is not None:
        axes[0].imshow(pre_rgb, alpha=0.40, extent=ext, origin="upper")
    axes[0].set_title(
        f"(a) Pre-slide DEM (COP30 2011–13)\n+ {pre_photo_note} (α=0.4)",
        fontweight="bold",
        fontsize=10,
    )

    axes[1].imshow(hs_post.data, cmap="gray", extent=ext, origin="upper")
    axes[1].imshow(post_rgb, alpha=0.40, extent=ext, origin="upper")
    axes[1].set_title(
        "(b) Post-slide DEM (3DEP LiDAR)\n+ Sentinel-2 2017-08-26 RGB (α=0.4)",
        fontweight="bold",
        fontsize=10,
    )

    axes[2].imshow(hs_post.data, cmap="gray", extent=ext, origin="upper")
    dod_masked = np.ma.masked_inside(dod.data, -2.0, 2.0)
    im = axes[2].imshow(
        dod_masked, cmap=plotting.DOD_CMAP, vmin=-45.0, vmax=45.0, alpha=0.88, extent=ext, origin="upper"
    )
    axes[2].set_title(
        f"(c) DoD (red = erosion, blue = deposit)\n"
        f"eroded {vol['eroded_volume_m3']/1e6:.2f} Mm³ | deposited {vol['deposited_volume_m3']/1e6:.2f} Mm³",
        fontweight="bold",
        fontsize=10,
    )
    cb = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cb.set_label(plotting.DOD_LABEL)

    for ax in axes:
        ax.set_xlabel("UTM Easting (km, Zone 10N)")
        ax.set_ylabel("UTM Northing (km)")
        plotting.use_plain_axes(ax)

    fig.text(
        0.5,
        -0.02,
        "No co-event InSAR shipped: Sentinel-1 launched after 22 Mar 2014; package has no ALOS/UAVSAR "
        "pair spanning this pre/post DEM pair. Panels (a)–(c) only.",
        ha="center",
        fontsize=8,
        style="italic",
    )

    out = IMG / "oso_landslide_dod.png"
    plotting.save_figure(fig, out, max_width_px=1600, palette=True)
    # Supersede the old Oso complexity-only board.
    old = IMG / "oso_cwt_change.png"
    if old.exists():
        old.unlink()
    return out


# A–A' (SW–NE across the anticline) and B–B' (~60° different, nearly N–S across the fold).
SHEEP_AA = ((726900.0, 4942800.0), (729700.0, 4944600.0))
SHEEP_BB = ((728390.0, 4941805.0), (728250.0, 4944795.0))
SHEEP_SECTIONS = (
    (SHEEP_AA[0], SHEEP_AA[1], "A", "A'"),
    (SHEEP_BB[0], SHEEP_BB[1], "B", "B'"),
)
# Shared 3D camera: ~30° clockwise from a SE face-on view so more of the S–N block side is visible.
SHEEP_3D_AZIMUTH_DEG = -88.0
SHEEP_3D_ELEVATION_DEG = 26.0


def _sheep_fault(dem):
    start_xy, end_xy = SHEEP_AA
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
    _, _, fault_3d = _sheep_fault(dem)
    print("cross-section fault published:", fault_3d["published"])
    print("cross-section fault inferred:", fault_3d["inferred"])

    # Stack the same two faulted sections used in the 3D README figure (A–A' above, B–B' below).
    fig, axes = plt.subplots(2, 1, figsize=(13, 9.4), layout="constrained")
    for ax, (p0, p1, lab0, lab1) in zip(axes, SHEEP_SECTIONS):
        fault_2d = cross_section.section_fault_params(fault_3d, p0, p1)
        print(f"section {lab0}–{lab1} params:", fault_2d)
        sec = cross_section.build_geological_cross_section(
            dem, geol, start_xy=p0, end_xy=p1, attitudes=attitudes, depth_m=360.0, faults=[fault_2d]
        )
        cross_section.plot_geological_cross_section(
            sec,
            unit_metadata=unit_metadata,
            vertical_exaggeration=1.6,
            ax=ax,
            dip_tick_color="#1f6feb",
            section_labels=(lab0, lab1),
            title=(
                f"Sheep Mountain Anticline {lab0}–{lab1} with published SW-dipping ~50° basement thrust "
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

    # Opaque solid block (A–A' + B–B' marked) | matching faulted fences; shared camera.
    fig, _ = viewer_3d.plot_3d_block_and_section(
        block_f,
        start_xy,
        end_xy,
        sections=SHEEP_SECTIONS,
        vertical_exaggeration=1.7,
        azimuth_deg=SHEEP_3D_AZIMUTH_DEG,
        elevation_deg=SHEEP_3D_ELEVATION_DEG,
        n_samples=160,
        show_fault_plane=True,
        top_stride=1,
        wall_stride=1,
        titles=(
            "(a) 3D geologic block with A–A′ & B–B′ section locations",
            f"(b) Matching 3D sections A–A′ & B–B′ + SW-dipping ~{fault_spec['dip_deg']:.0f}° basement thrust "
            f"(throw +{fault_spec['throw_m']:.0f} m inferred)",
        ),
    )
    out1 = IMG / "3d_block_diagram.png"
    plotting.save_figure(fig, out1, max_width_px=1350, palette=False)
    plt.close(fig)

    fence_panels = viewer_3d.slice_3d_fence_diagram(
        block_f, transects=[(s[0], s[1]) for s in SHEEP_SECTIONS], n_samples=160
    )
    for fence, sec in zip(fence_panels, SHEEP_SECTIONS):
        fence["name"] = f"Fence {sec[2]}-{sec[3]}"
    fig = plt.figure(figsize=(16, 7.2), layout="constrained")
    ax1 = fig.add_subplot(121, projection="3d")
    viewer_3d.plot_3d_cutaway_block(
        block_f,
        vertical_exaggeration=1.7,
        azimuth_deg=SHEEP_3D_AZIMUTH_DEG,
        elevation_deg=SHEEP_3D_ELEVATION_DEG,
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
        azimuth_deg=SHEEP_3D_AZIMUTH_DEG,
        elevation_deg=SHEEP_3D_ELEVATION_DEG,
        show_fault_plane=True,
        show_wireframe_surface=True,
        show_legend=False,
        ax=ax2,
        title="(b) 3D A–A′ & B–B′ fences + reconstructed fault plane",
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

    fig, axes = plt.subplots(1, 4, figsize=(10.8, 12.2), sharey=True)
    fig.subplots_adjust(left=0.055, right=0.995, top=0.945, bottom=0.10, wspace=0.07)
    fig.suptitle(
        "Case 1A — 2025 Mw 7.7 Myanmar (Sagaing): REAL Sentinel-1 (COMET LiCSAR, us7000pn9s)",
        fontsize=10,
        fontweight="bold",
        color="#0f172a",
        y=0.98,
    )
    panels = [
        (u_north_real, bwr_sar, -5, 5, "N-S displacement (meters)", "a  N–S displacement"),
        (azi_106D, bwr_sar, -5, 5, "Azi offset (meters)", "b  Azimuth offset (106D)"),
        (rng_106D, bwr_sar, -1, 1, "Range offset (meters)", "c  Range offset (106D)"),
        (phase_cm, sar_rainbow, 0, CM_PER_CYCLE, "LOS phase (range increase, cm)", "d  Wrapped phase (106D)"),
    ]
    panel_mappables = []
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
        ax.set_xticks([96.0])
        ax.set_xticklabels(["96°E"])
        ax.set_yticks(np.arange(19, 24))
        if ax is not axes[0]:
            ax.tick_params(labelleft=False)
        else:
            ax.set_yticklabels([f"{v}°N" for v in range(19, 24)])
        panel_mappables.append((im, ax, clabel))
    fig.canvas.draw()
    for im, ax, clabel in panel_mappables:
        plotting.matching_width_colorbar(
            fig, im, ax, orientation="horizontal", thickness=0.013, pad=0.026,
            label=clabel, label_fontsize=8.0, use_position=True,
        )
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
        0.012,
        "a: N–S + vectors (11 km) · b–c: 106D offsets · d: 106D ifg 2025-03-24→04-05 · α=0.60 GLO-30\n"
        "Contains modified Copernicus Sentinel-1 data (2025) processed by COMET LiCSAR (JASMIN). "
        "Copernicus DEM GLO-30. Blank = no coverage / masked by LiCSAR.",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#334155",
    )
    out = IMG / "case1a_myanmar_earthquake.jpg"
    plotting.save_figure(fig, out, max_width_px=1500, quality=75)
    return out


def main() -> None:
    t0 = time.perf_counter()
    outs = []
    print("=== CWT Sheep Mountain board ===")
    outs.append(regenerate_cwt_multiscale())
    print("=== Oso landslide DoD board ===")
    outs.append(regenerate_oso_landslide())
    print("=== cross-section ===")
    outs.append(regenerate_cross_section())
    print("=== 3D A–A′ / B–B′ ===")
    outs.extend(regenerate_3d_block())
    print("=== Case 1A LiCSAR 4-panel ===")
    outs.append(regenerate_case1a())
    for p in outs:
        print(f"saved {p} ({p.stat().st_size / 1e3:.0f} KB)")
        _copy_artifact(p)
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
