# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
2D hillshade draping, 2.5D/3D perspective DEM surface rendering, Rust-accelerated 3D geological
block-diagram & cutaway/fence mesh generation, `forge3d` integration, and interactive local
WebGL / HTML5 3D block-diagram viewer.

Provides hillshade-draped RGB compositing (`drape_overlay_on_hillshade`) and 3D
perspective DEM draping (`plot_3d_terrain_drape`), plus:
- `calibrate_3d_structural_horizon`: Calibrates the 2D structural fold elevation surface
  `h_struct(y, x)` from a DEM, surface geological map (or auto-mapped units), 3-point contact
  attitudes, and optional 3D fault parameters.
- `build_3d_block_diagram`: Evaluates the 3D subsurface voxel volume `(rows, cols, nz)` and
  watertight 3D triangle meshes (top surface, 4 vertical cross-section side walls, base plate,
  optional cutaway internal walls/floor, and 3D fault plane) in parallel Rust
  (`_core.build_3d_block_volume_rs`, `_core.build_block_diagram_meshes_rs`).
- `slice_3d_fence_diagram`: Slices intersecting vertical 3D fence cross-section ribbons across
  the 3D geological block volume in Rust (`_core.slice_3d_block_fence_rs`).
- `export_3d_block_mesh`: Exports 3D block or cutaway meshes to binary glTF 2.0 (`.glb`),
  Stanford `.ply`, or Wavefront `.obj` via Rust (`_core.export_block_glb_rs`, etc.).
- `render_with_forge3d`: High-resolution terrain/block drape adapter integrating `forge3d`
  (https://pypi.org/project/forge3d/) when available, with automatic fallback.
- `plot_3d_block_diagram`, `plot_3d_cutaway_block`, `plot_3d_fence_diagram`,
  `plot_3d_block_and_section`: Publication-ready 3D block / cutaway / fence renderers, plus a
  README-style layout (opaque volumetric block with A–A′ / B–B′ marked | distinct faulted
  3D section panels, shared camera). `include_fault` / `show_fault_plane` toggle loading and
  drawing of a plane from `reconstruct_planar_fault`.
- `export_interactive_block_html` & `launch_block_viewer`: Self-contained, 100% offline
  interactive 3D HTML5/WebGL canvas block-diagram viewer for Jupyter notebooks and browsers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.ndimage import gaussian_filter

from . import _core, cross_section, terrain_analysis
from .raster_io import GeoGrid

try:
    import forge3d  # noqa: F401

    HAS_FORGE3D = True
except ImportError:  # pragma: no cover
    HAS_FORGE3D = False


DEFAULT_UNIT_PALETTE_HEX: tuple[str, ...] = (
    "#6baed6",  # 0: Mm  - Madison Limestone
    "#9ecae1",  # 1: IPta - Tensleep & Amsden
    "#e6550d",  # 2: TRPc - Chugwater & Goose Egg
    "#74c476",  # 3: Jsg  - Sundance & Gypsum Spring
    "#fd8d3c",  # 4: KJcm - Cloverly & Morrison
    "#969696",  # 5: Kmt  - Mowry & Thermopolis
    "#ffffb3",  # 6: Qa   - Quaternary Alluvium
)


def _hex_list_to_rgba_u8(hex_colors: Sequence[str], alpha: int = 255) -> np.ndarray:
    """Convert a sequence of hex strings (`#rrggbb`) to an `(N, 4)` `uint8` RGBA array."""
    rows = []
    for h in hex_colors:
        rgb = mcolors.to_rgb(h)
        rows.append(
            [
                int(round(rgb[0] * 255.0)),
                int(round(rgb[1] * 255.0)),
                int(round(rgb[2] * 255.0)),
                int(alpha),
            ]
        )
    return np.ascontiguousarray(rows, dtype=np.uint8)


def _parse_unit_catalog(
    units_metadata: Sequence[dict[str, Any]] | dict[str, Any] | None,
    surface_units: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Normalize unit metadata into a sorted list of unit dicts and an `(N_units, 4)` `uint8` RGBA palette."""
    if isinstance(units_metadata, dict) and "units" in units_metadata:
        raw_list = list(units_metadata["units"])
    elif isinstance(units_metadata, Sequence):
        raw_list = list(units_metadata)
    else:
        max_u = int(np.nanmax(surface_units)) if np.any(np.isfinite(surface_units)) else 6
        raw_list = []
        for uid in range(max(max_u + 1, 1)):
            col = DEFAULT_UNIT_PALETTE_HEX[uid % len(DEFAULT_UNIT_PALETTE_HEX)]
            raw_list.append(
                {
                    "id": uid,
                    "code": f"U{uid}",
                    "name": f"Unit {uid}",
                    "color": col,
                }
            )

    normalized_list = []
    for d in raw_list:
        d_copy = dict(d)
        uid = int(d_copy.get("id", d_copy.get("unit_id", 0)))
        d_copy["id"] = uid
        d_copy["unit_id"] = uid
        normalized_list.append(d_copy)
    normalized_list.sort(key=lambda d: int(d["id"]))
    max_id = max(int(d["id"]) for d in normalized_list) if normalized_list else 0
    hex_pal = [DEFAULT_UNIT_PALETTE_HEX[i % len(DEFAULT_UNIT_PALETTE_HEX)] for i in range(max_id + 1)]
    for d in normalized_list:
        uid = int(d["id"])
        if "color" in d:
            hex_pal[uid] = str(d["color"])
    return normalized_list, _hex_list_to_rgba_u8(hex_pal)


def drape_overlay_on_hillshade(
    hillshade: GeoGrid | np.ndarray,
    overlay: GeoGrid | np.ndarray,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    alpha: float = 0.65,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Blend a continuous or categorical `overlay` layer over a shaded-relief `hillshade`
    into an `(rows, cols, 3)` float RGB array in `[0, 1]`.
    """
    hs = hillshade.data if isinstance(hillshade, GeoGrid) else np.asarray(hillshade, dtype=np.float64)
    ov = overlay.data if isinstance(overlay, GeoGrid) else np.asarray(overlay, dtype=np.float64)

    hs_clean = np.clip(np.nan_to_num(hs, nan=0.5), 0.0, 1.0)
    base_rgb = np.stack([hs_clean, hs_clean, hs_clean], axis=-1)

    valid = np.isfinite(ov)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    if not np.any(valid):
        return base_rgb

    lo = float(np.nanpercentile(ov[valid], 2.0)) if vmin is None else float(vmin)
    hi = float(np.nanpercentile(ov[valid], 98.0)) if vmax is None else float(vmax)
    if abs(hi - lo) < 1e-12:
        hi = lo + 1.0

    norm = mcolors.Normalize(vmin=lo, vmax=hi, clip=True)
    colormap = plt.get_cmap(cmap)
    ov_rgba = colormap(norm(np.nan_to_num(ov, nan=lo)))
    ov_rgb = ov_rgba[:, :, :3]

    # Luminance-preserving hillshade modulation + alpha blend
    shaded_color = np.clip(ov_rgb * (0.45 + 0.75 * hs_clean[:, :, np.newaxis]), 0.0, 1.0)
    blended = (1.0 - alpha) * base_rgb + alpha * shaded_color
    out = np.where(valid[:, :, np.newaxis], blended, base_rgb)
    return np.clip(out, 0.0, 1.0)


def drape_categorical_units_on_hillshade(
    hillshade: GeoGrid | np.ndarray,
    unit_labels: GeoGrid | np.ndarray,
    unit_palette_rgba: np.ndarray,
    alpha: float = 0.88,
    draw_contacts: bool = True,
) -> np.ndarray:
    """
    Drape integer geological formation IDs over a multi-azimuth hillshade and return an
    `(rows, cols, 4)` `uint8` RGBA image suitable for 3D mesh vertex coloring and WebGL textures.

    High `alpha` keeps stratigraphic unit fills opaque and colorful on solid 3D blocks
    (hillshade only gently modulates the unit colors).
    """
    hs = hillshade.data if isinstance(hillshade, GeoGrid) else np.asarray(hillshade, dtype=np.float64)
    lbl = unit_labels.data if isinstance(unit_labels, GeoGrid) else np.asarray(unit_labels)
    lbl_f = np.asarray(lbl, dtype=np.float64)
    lbl_i32 = np.where(np.isfinite(lbl_f) & (lbl_f >= 0), np.round(lbl_f).astype(np.int32), -1)

    hs_clean = np.clip(np.nan_to_num(hs, nan=0.6), 0.0, 1.0)
    rows, cols = lbl_i32.shape
    # Seed with unit colors (not grey hillshade) so every labeled cell stays painted.
    out_rgb = np.full((rows, cols, 3), 0.85, dtype=np.float64)

    n_pal = len(unit_palette_rgba)
    pal_f = unit_palette_rgba[:, :3].astype(np.float64) / 255.0
    a = float(np.clip(alpha, 0.0, 1.0))
    for uid in range(n_pal):
        m = lbl_i32 == uid
        if np.any(m):
            c_rgb = pal_f[uid]
            # Gentle hillshade modulation on top of solid unit color (keeps faces readable).
            shaded = np.clip(c_rgb[np.newaxis, :] * (0.55 + 0.55 * hs_clean[m, np.newaxis]), 0.0, 1.0)
            solid = np.broadcast_to(c_rgb[np.newaxis, :], shaded.shape)
            lit = 0.70 * solid + 0.30 * shaded
            out_rgb[m] = (1.0 - a) * out_rgb[m] + a * lit

    if draw_contacts:
        contacts, _ = _core.extract_unit_contacts_rs(np.ascontiguousarray(lbl_i32, dtype=np.int32))
        c_mask = contacts > 0
        out_rgb[c_mask] = out_rgb[c_mask] * 0.40

    rgba_u8 = np.empty((rows, cols, 4), dtype=np.uint8)
    rgba_u8[:, :, :3] = np.clip(np.round(out_rgb * 255.0), 0, 255).astype(np.uint8)
    rgba_u8[:, :, 3] = 255
    return np.ascontiguousarray(rgba_u8, dtype=np.uint8)


def plot_3d_terrain_drape(
    dem: GeoGrid,
    overlay: GeoGrid | None = None,
    cmap: str = "terrain",
    vmin: float | None = None,
    vmax: float | None = None,
    vertical_exaggeration: float = 1.5,
    azimuth_deg: float = -55.0,
    elevation_deg: float = 32.0,
    stride: int = 2,
    ax: Any | None = None,
    title: str | None = None,
    cbar_label: str | None = None,
) -> tuple[Any, Any]:
    """
    Render a 3D perspective surface mesh of `dem` draped with `overlay` (or elevation + hillshade).
    """
    hs = terrain_analysis.compute_hillshade(dem, multi_azimuth=True)
    target = overlay if overlay is not None else dem
    rgb = drape_overlay_on_hillshade(hs, target, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.72)

    xs = dem.x_coords()
    ys = dem.y_coords()
    xx, yy = np.meshgrid(xs - xs.min(), ys - ys.min())
    zz = np.nan_to_num(dem.data, nan=float(np.nanmean(dem.data)))

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    s = max(1, int(stride))
    ax.plot_surface(
        xx[::s, ::s],
        yy[::s, ::s],
        zz[::s, ::s],
        facecolors=rgb[::s, ::s],
        rstride=1,
        cstride=1,
        antialiased=True,
        shade=False,
    )

    x_range = float(xx.max() - xx.min())
    y_range = float(yy.max() - yy.min())
    z_range = float(zz.max() - zz.min()) * float(vertical_exaggeration)
    max_xy = max(x_range, y_range, 1.0)
    ax.set_box_aspect((x_range / max_xy, y_range / max_xy, max(z_range / max_xy, 0.22)))
    ax.view_init(elev=float(elevation_deg), azim=float(azimuth_deg))
    ax.set_xlabel("Relative Easting (m)")
    ax.set_ylabel("Relative Northing (m)")
    ax.set_zlabel(f"Elevation ({dem.units})")
    ax.zaxis.set_major_locator(MaxNLocator(nbins=4))
    if title:
        ax.set_title(title, fontweight="bold", pad=10)

    if cbar_label is not None:
        vals = target.data[np.isfinite(target.data)]
        lo = float(np.percentile(vals, 2.0)) if vmin is None else float(vmin)
        hi = float(np.percentile(vals, 98.0)) if vmax is None else float(vmax)
        sm = cm.ScalarMappable(norm=mcolors.Normalize(vmin=lo, vmax=hi), cmap=plt.get_cmap(cmap))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, shrink=0.58, pad=0.16, label=cbar_label)

    return fig, ax


def render_with_forge3d(
    dem: GeoGrid,
    overlay_rgb: np.ndarray | None = None,
    vertical_exaggeration: float = 1.6,
    azimuth_deg: float = -52.0,
    elevation_deg: float = 30.0,
    width: int = 1024,
    height: int = 768,
) -> dict[str, Any]:
    """
    Render a 3D terrain drape using `forge3d` (https://pypi.org/project/forge3d/) when installed,
    or automatically fall back to the built-in Rust hillshade + high-resolution 3D surface renderer.

    Returns a dictionary containing:
    - `backend`: `'forge3d'` or `'pygeomapper_rust_fallback'`
    - `has_forge3d`: `bool`
    - `draped_rgb`: `(rows, cols, 3)` float RGB array in `[0, 1]`
    - `hillshade`: `GeoGrid` multi-azimuth shaded relief
    """
    hs = terrain_analysis.compute_hillshade(dem, multi_azimuth=True)
    if overlay_rgb is None:
        draped_rgb = drape_overlay_on_hillshade(hs, dem, cmap="terrain", alpha=0.72)
    else:
        arr = np.asarray(overlay_rgb)
        if arr.dtype == np.uint8:
            draped_rgb = arr[:, :, :3].astype(np.float64) / 255.0
        else:
            draped_rgb = np.clip(arr[:, :, :3], 0.0, 1.0)

    backend = "pygeomapper_rust_fallback"
    forge_result = None
    if HAS_FORGE3D:  # pragma: no cover
        try:
            if hasattr(forge3d, "render_terrain"):
                forge_result = forge3d.render_terrain(
                    np.ascontiguousarray(dem.data, dtype=np.float32),
                    texture=np.ascontiguousarray((draped_rgb * 255).astype(np.uint8)),
                    z_scale=float(vertical_exaggeration),
                    azimuth=float(azimuth_deg),
                    elevation=float(elevation_deg),
                    width=int(width),
                    height=int(height),
                )
            backend = "forge3d"
        except Exception:
            backend = "pygeomapper_rust_fallback"

    return {
        "backend": backend,
        "has_forge3d": HAS_FORGE3D,
        "draped_rgb": draped_rgb,
        "hillshade": hs,
        "forge_output": forge_result,
    }


def calibrate_3d_structural_horizon(
    dem: GeoGrid,
    geol_labels: GeoGrid,
    bedrock_unit_ids: Sequence[int] | None = None,
    unit_thickness_m: float | Sequence[float] = 105.0,
    surficial_unit_id: int = 6,
    smoothing_sigma_px: float = 5.5,
    fault_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Calibrate a continuous 2D structural fold elevation surface `h_struct(y, x)` and
    stratigraphic boundary thresholds `unit_zeta_thresholds` from surface geological units,
    topography, and optional 3D fault throw.

    In the stratigraphic potential-field formulation:
        `zeta(y, x, z) = z - h_struct(y, x) - fault_offset(y, x, z)`
    where higher `zeta` corresponds to stratigraphically younger formations.
    """
    u_arr = np.where(
        np.isfinite(geol_labels.data) & (geol_labels.data >= 0),
        np.round(geol_labels.data).astype(np.int32),
        -1,
    )
    if bedrock_unit_ids is None:
        unique_u = sorted(int(u) for u in np.unique(u_arr) if u >= 0 and u != surficial_unit_id)
        if not unique_u:
            unique_u = [0]
        bedrock_ids = np.array(unique_u, dtype=np.int32)
    else:
        bedrock_ids = np.array(bedrock_unit_ids, dtype=np.int32)

    n_bed = len(bedrock_ids)
    if isinstance(unit_thickness_m, (int, float)):
        thick = np.full(n_bed, float(unit_thickness_m), dtype=np.float64)
    else:
        thick = np.asarray(unit_thickness_m, dtype=np.float64)
        if len(thick) != n_bed:
            thick = np.resize(thick, n_bed)

    # Cumulative stratigraphic center height `zeta_center` for each bedrock unit (0 = oldest core)
    edges = np.concatenate([[0.0], np.cumsum(thick)])
    zeta_centers = 0.5 * (edges[:-1] + edges[1:])
    zeta_centers -= zeta_centers[0]
    thresholds = 0.5 * (zeta_centers[:-1] + zeta_centers[1:]) if n_bed > 1 else np.array([], dtype=np.float64)

    # Evaluate surface fault offset if a 3D fault is present
    xs = dem.x_coords()
    ys = dem.y_coords()
    xx, yy = np.meshgrid(xs, ys)
    surf_flt_off = np.zeros_like(dem.data, dtype=np.float64)
    fault_tuple: tuple[float, float, float, float, float, float, float] | None = None

    if fault_spec is not None:
        fx0 = float(fault_spec.get("x0", np.percentile(xs, 35.0)))
        fy0 = float(fault_spec.get("y0", np.percentile(ys, 35.0)))
        fz0 = float(fault_spec.get("z0", np.nanmedian(dem.data)))
        fstrike = float(fault_spec.get("strike_deg", 140.0))
        fdip = float(fault_spec.get("dip_deg", 50.0))
        fthrow = float(fault_spec.get("throw_m", 150.0))
        flistric = float(fault_spec.get("listric_curvature", 0.0))
        fault_tuple = (fx0, fy0, fz0, fstrike, fdip, fthrow, flistric)

        dip_dir_rad = np.deg2rad((fstrike + 90.0) % 360.0)
        d_dip = (xx - fx0) * np.sin(dip_dir_rad) + (yy - fy0) * np.cos(dip_dir_rad)
        z_flt = (
            fz0
            - d_dip * np.tan(np.deg2rad(np.clip(abs(fdip), 5.0, 88.0)))
            + flistric * d_dip * np.abs(d_dip)
        )
        surf_flt_off = np.where(dem.data >= z_flt, fthrow, 0.0)

    raw_h = np.zeros_like(dem.data, dtype=np.float64)
    weights = np.zeros_like(dem.data, dtype=np.float64)
    for idx_b, uid in enumerate(bedrock_ids):
        m = (u_arr == uid) & np.isfinite(dem.data)
        if np.any(m):
            raw_h[m] = dem.data[m] - surf_flt_off[m] - zeta_centers[idx_b]
            weights[m] = 1.0

    if not np.any(weights > 0):
        h_struct = np.full_like(dem.data, float(np.nanmean(dem.data)), dtype=np.float64)
    else:
        sigma = max(float(smoothing_sigma_px), 1.0)
        h_num = gaussian_filter(raw_h * weights, sigma=sigma, mode="nearest")
        h_den = np.maximum(gaussian_filter(weights, sigma=sigma, mode="nearest"), 1e-6)
        h_struct = h_num / h_den

    return {
        "structural_horizon": np.ascontiguousarray(h_struct, dtype=np.float64),
        "unit_zeta_centers": np.ascontiguousarray(zeta_centers, dtype=np.float64),
        "unit_zeta_thresholds": np.ascontiguousarray(thresholds, dtype=np.float64),
        "bedrock_unit_ids": np.ascontiguousarray(bedrock_ids, dtype=np.int32),
        "fault_params_tuple": fault_tuple,
    }


def build_3d_block_diagram(
    dem: GeoGrid,
    geol_labels: GeoGrid,
    units_metadata: Sequence[dict[str, Any]] | dict[str, Any] | None = None,
    attitudes: Sequence[dict[str, Any]] | None = None,
    z_bottom: float | None = None,
    depth_m: float = 385.0,
    nz: int = 64,
    unit_thickness_m: float | Sequence[float] = 105.0,
    surficial_unit_id: int = 6,
    surficial_thickness_m: float = 24.0,
    near_surface_lock_m: float = 18.0,
    smoothing_sigma_px: float = 5.5,
    fault_spec: dict[str, Any] | None = None,
    include_fault: bool = True,
    cutaway_fraction: tuple[float, float, float] | None = None,
    surface_drape_rgba: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Build a complete 3D shallow geological block model and watertight 3D triangle meshes in Rust.

    Parameters
    ----------
    dem : GeoGrid
        Digital elevation model in projected metric coordinates (e.g., UTM meters).
    geol_labels : GeoGrid
        Surface geological unit raster (`0..K-1`, ordered oldest to youngest bedrock + optional surficial unit).
    units_metadata : sequence of dict or dict, optional
        Formation metadata (`id`, `code`, `name`, `color`).
    attitudes : sequence of dict, optional
        3-point bed attitudes from `cross_section.estimate_contact_attitudes`. Computed automatically if `None`.
    z_bottom : float, optional
        Base elevation of the 3D block diagram (m). Defaults to `min(dem) - depth_m` (or `750.0 m` for Sheep Mountain).
    depth_m : float, default 385.0
        Depth below minimum DEM elevation when `z_bottom` is not explicitly provided.
    nz : int, default 64
        Number of vertical voxel layers between `z_bottom` and `z_top`.
    fault_spec : dict, optional
        3D planar or listric fault from `cross_section.reconstruct_planar_fault` (preferred) or an equivalent
        dict: `{'x0', 'y0', 'z0', 'strike_deg', 'dip_deg', 'throw_m', 'name'}`.
    include_fault : bool, default True
        When False, the block is built without a fault even if `fault_spec` is provided (API toggle;
        README figures ship the faulted reconstruction).
    cutaway_fraction : tuple of (y_frac, x_frac, z_frac), optional
        When provided (e.g., `(0.45, 0.48, 0.35)`), also generates a cutaway 3D mesh (`cutaway_mesh`)
        with the SW/front sub-block excavated down to `z_frac` so internal folded beds and fault offsets are exposed.
    """
    active_fault = fault_spec if (include_fault and fault_spec is not None) else None
    u_arr = np.where(
        np.isfinite(geol_labels.data) & (geol_labels.data >= 0),
        np.round(geol_labels.data).astype(np.int32),
        -1,
    )
    u_arr = np.ascontiguousarray(u_arr, dtype=np.int32)
    dem_f64 = np.ascontiguousarray(dem.data, dtype=np.float64)

    unit_list, unit_palette_rgba = _parse_unit_catalog(units_metadata, u_arr)
    hs = terrain_analysis.compute_hillshade(dem, multi_azimuth=True)

    if surface_drape_rgba is None:
        drape_rgba = drape_categorical_units_on_hillshade(
            hs, u_arr, unit_palette_rgba, alpha=0.78, draw_contacts=True
        )
    else:
        drape_rgba = np.ascontiguousarray(surface_drape_rgba, dtype=np.uint8)

    if attitudes is None:
        attitudes = cross_section.estimate_contact_attitudes(geol_labels, dem, window_m=320.0)

    cal = calibrate_3d_structural_horizon(
        dem=dem,
        geol_labels=geol_labels,
        unit_thickness_m=unit_thickness_m,
        surficial_unit_id=surficial_unit_id,
        smoothing_sigma_px=smoothing_sigma_px,
        fault_spec=active_fault,
    )

    z_min_dem = float(np.nanmin(dem_f64))
    z_max_dem = float(np.nanmax(dem_f64))
    z_bot = float(z_bottom) if z_bottom is not None else round(z_min_dem - float(depth_m), 1)
    z_top = float(z_max_dem + 12.0)

    vol = _core.build_3d_block_volume_rs(
        dem_f64,
        u_arr,
        cal["structural_horizon"],
        cal["unit_zeta_thresholds"],
        cal["bedrock_unit_ids"],
        float(dem.x0),
        float(dem.y0),
        float(dem.dx),
        float(dem.dy),
        z_bot,
        z_top,
        int(nz),
        int(surficial_unit_id),
        float(surficial_thickness_m),
        float(near_surface_lock_m),
        cal["fault_params_tuple"],
    )

    full_mesh = _core.build_block_diagram_meshes_rs(
        dem_f64,
        u_arr,
        drape_rgba,
        vol["voxel_units"],
        vol["z_levels"],
        unit_palette_rgba,
        vol["fault_surface_z"],
        float(dem.x0),
        float(dem.y0),
        float(dem.dx),
        float(dem.dy),
        None,
    )

    # Also build a cutaway mesh (defaulting to SW quadrant `(r: 52%..100%, c: 0..48%, iz_floor: 35%)` if requested)
    if cutaway_fraction is None:
        cut_fy, cut_fx, cut_fz = (0.48, 0.48, 0.36)
    else:
        cut_fy, cut_fx, cut_fz = (
            float(cutaway_fraction[0]),
            float(cutaway_fraction[1]),
            float(cutaway_fraction[2]),
        )
    r_cut0 = int(round(dem.rows * (1.0 - np.clip(cut_fy, 0.1, 0.9))))
    r_cut1 = dem.rows
    c_cut0 = 0
    c_cut1 = int(round(dem.cols * np.clip(cut_fx, 0.1, 0.9)))
    iz_cut = int(round(len(vol["z_levels"]) * np.clip(cut_fz, 0.05, 0.85)))
    cutaway_bounds = (r_cut0, r_cut1, c_cut0, c_cut1, iz_cut)

    cutaway_mesh = _core.build_block_diagram_meshes_rs(
        dem_f64,
        u_arr,
        drape_rgba,
        vol["voxel_units"],
        vol["z_levels"],
        unit_palette_rgba,
        vol["fault_surface_z"],
        float(dem.x0),
        float(dem.y0),
        float(dem.dx),
        float(dem.dy),
        cutaway_bounds,
    )

    # Compute 3D voxel volume proportions per geological formation
    vox = vol["voxel_units"]
    cell_vol_m3 = abs(float(dem.dx) * float(dem.dy)) * float(
        (z_top - z_bot) / max(len(vol["z_levels"]) - 1, 1)
    )
    subsurface_mask = vox >= 0
    total_subsurface_voxels = int(np.sum(subsurface_mask))
    unit_volume_stats: list[dict[str, Any]] = []
    for u_info in unit_list:
        uid = int(u_info["id"])
        cnt = int(np.sum(vox == uid))
        frac = cnt / max(total_subsurface_voxels, 1)
        unit_volume_stats.append(
            {
                "id": uid,
                "code": u_info.get("code", f"U{uid}"),
                "name": u_info.get("name", f"Unit {uid}"),
                "color": u_info.get("color", "#888888"),
                "voxel_count": cnt,
                "volume_fraction": float(frac),
                "volume_km3": float(cnt * cell_vol_m3 / 1e9),
            }
        )

    return {
        "dem": dem,
        "geol_labels": geol_labels,
        "hillshade": hs,
        "surface_units": u_arr,
        "surface_drape_rgba": drape_rgba,
        "units_metadata": unit_list,
        "unit_palette_rgba": unit_palette_rgba,
        "attitudes": list(attitudes),
        "structural_horizon": cal["structural_horizon"],
        "unit_zeta_thresholds": cal["unit_zeta_thresholds"],
        "bedrock_unit_ids": cal["bedrock_unit_ids"],
        "fault_spec": active_fault,
        "fault_params_tuple": cal["fault_params_tuple"],
        "z_bottom": z_bot,
        "z_top": z_top,
        "z_levels": vol["z_levels"],
        "voxel_units": vol["voxel_units"],
        "zeta_volume": vol["zeta_volume"],
        "fault_offset_volume": vol["fault_offset_volume"],
        "fault_surface_z": vol["fault_surface_z"],
        "horizon_elevations": vol.get("horizon_elevations"),
        "mesh": dict(full_mesh),
        "cutaway_bounds": cutaway_bounds,
        "cutaway_mesh": dict(cutaway_mesh),
        "unit_volume_stats": unit_volume_stats,
    }


def slice_3d_fence_diagram(
    block_model: dict[str, Any],
    transects: Sequence[tuple[tuple[float, float], tuple[float, float]]] | None = None,
    n_samples: int = 160,
) -> list[dict[str, Any]]:
    """
    Slice one or more intersecting vertical 3D fence-diagram cross-section ribbons across
    `block_model` in Rust (`_core.slice_3d_block_fence_rs`).

    If `transects` is `None`, generates 3 intersecting structural + strike fence panels across
    the block domain (SW-NE structural dip panel, WNW-ESE fold-axis strike panel, and S-N panel).
    """
    dem: GeoGrid = block_model["dem"]
    xs = dem.x_coords()
    ys = dem.y_coords()
    x_min, x_max = (float(xs.min()), float(xs.max()))
    y_min, y_max = (float(ys.min()), float(ys.max()))
    dx_span = x_max - x_min
    dy_span = y_max - y_min

    if transects is None:
        transects = [
            # 1. SW -> NE structural dip fence across anticline & forelimb fault
            ((x_min + 0.08 * dx_span, y_min + 0.10 * dy_span), (x_min + 0.92 * dx_span, y_min + 0.90 * dy_span)),
            # 2. NW -> SE strike fence along the doubly plunging hinge & Bighorn canyon
            ((x_min + 0.12 * dx_span, y_min + 0.88 * dy_span), (x_min + 0.88 * dx_span, y_min + 0.12 * dy_span)),
            # 3. Central West -> East fence
            ((x_min + 0.06 * dx_span, y_min + 0.50 * dy_span), (x_min + 0.94 * dx_span, y_min + 0.50 * dy_span)),
        ]

    dem_f64 = np.ascontiguousarray(dem.data, dtype=np.float64)
    vox = np.ascontiguousarray(block_model["voxel_units"], dtype=np.int32)
    z_levels = np.ascontiguousarray(block_model["z_levels"], dtype=np.float64)
    pal = np.ascontiguousarray(block_model["unit_palette_rgba"], dtype=np.uint8)

    panels: list[dict[str, Any]] = []
    for idx, (pt_a, pt_b) in enumerate(transects):
        xa, ya = (float(pt_a[0]), float(pt_a[1]))
        xb, yb = (float(pt_b[0]), float(pt_b[1]))
        res = _core.slice_3d_block_fence_rs(
            dem_f64,
            vox,
            z_levels,
            pal,
            float(dem.x0),
            float(dem.y0),
            float(dem.dx),
            float(dem.dy),
            xa,
            ya,
            xb,
            yb,
            int(n_samples),
        )
        d = dict(res)
        d["mesh"] = dict(d["mesh"])
        d["panel_id"] = idx
        d["name"] = f"Fence {chr(65 + idx)}-{chr(65 + idx)}'"
        d["start_xy"] = (xa, ya)
        d["end_xy"] = (xb, yb)
        panels.append(d)
    return panels


def export_3d_block_mesh(
    mesh_or_block: dict[str, Any],
    output_path: str | Path,
    use_cutaway: bool = False,
    center_coordinates: bool = True,
    vertical_exaggeration: float = 1.0,
) -> dict[str, Any]:
    """
    Export a 3D geological block or cutaway triangle mesh to binary glTF 2.0 (`.glb`),
    ASCII Stanford `.ply`, or Wavefront `.obj` directly via Rust.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if "vertices" in mesh_or_block:
        mesh = mesh_or_block
    else:
        mesh = mesh_or_block["cutaway_mesh"] if use_cutaway else mesh_or_block["mesh"]

    verts = np.array(mesh["vertices"], dtype=np.float64, copy=True)
    norms = np.ascontiguousarray(mesh["normals"], dtype=np.float64)
    tris = np.ascontiguousarray(mesh["triangles"], dtype=np.uint32)
    cols = np.ascontiguousarray(mesh["vertex_colors_rgba"], dtype=np.uint8)

    if center_coordinates and len(verts) > 0:
        verts[:, 0] -= float(np.mean(verts[:, 0]))
        verts[:, 1] -= float(np.mean(verts[:, 1]))
        verts[:, 2] = (verts[:, 2] - float(np.min(verts[:, 2]))) * float(vertical_exaggeration)
    elif vertical_exaggeration != 1.0 and len(verts) > 0:
        z0 = float(np.min(verts[:, 2]))
        verts[:, 2] = z0 + (verts[:, 2] - z0) * float(vertical_exaggeration)

    verts = np.ascontiguousarray(verts, dtype=np.float64)
    ext = path.suffix.lower()
    if ext == ".glb":
        written = _core.export_block_glb_rs(str(path), verts, norms, tris, cols)
    elif ext == ".ply":
        written = _core.export_block_ply_rs(str(path), verts, norms, tris, cols)
    elif ext == ".obj":
        written = _core.export_block_obj_rs(str(path), verts, norms, tris, cols)
    else:
        raise ValueError(f"Unsupported 3D mesh extension '{ext}'. Expected .glb, .ply, or .obj.")

    return {
        "path": str(path),
        "format": ext.lstrip("."),
        "n_vertices": int(len(verts)),
        "n_triangles": int(len(tris)),
        "file_size_bytes": int(path.stat().st_size),
        "written_metric": int(written),
    }


def _render_mesh_poly3d(
    ax: Any,
    mesh: dict[str, Any],
    x_origin: float,
    y_origin: float,
    include_face_groups: Sequence[int] | None = None,
    face_stride: int = 1,
    alpha: float = 1.0,
    edge_alpha: float = 0.0,
    force_rgba: tuple[float, float, float, float] | None = None,
    edge_rgba: tuple[float, float, float, float] | None = None,
    brighten: float = 0.0,
) -> None:
    """Render a subset of triangles from a Rust `Mesh3DData` dictionary onto a Matplotlib 3D Axes."""
    verts = np.asarray(mesh["vertices"], dtype=np.float64)
    tris = np.asarray(mesh["triangles"], dtype=np.uint32)
    cols_u8 = np.asarray(mesh["vertex_colors_rgba"], dtype=np.uint8)
    fg = np.asarray(mesh["face_groups"], dtype=np.uint8)

    if len(tris) == 0:
        return

    tri_fg = fg[tris[:, 0]]
    if include_face_groups is not None:
        mask = np.isin(tri_fg, np.asarray(include_face_groups, dtype=np.uint8))
        tris = tris[mask]
    if len(tris) == 0:
        return

    step = max(1, int(face_stride))
    tris = tris[::step]

    v_rel = verts.copy()
    v_rel[:, 0] -= x_origin
    v_rel[:, 1] -= y_origin

    polys = v_rel[tris]
    if force_rgba is not None:
        tri_rgba = np.tile(np.asarray(force_rgba, dtype=np.float64), (len(tris), 1))
        tri_rgba[:, 3] = np.clip(tri_rgba[:, 3] * float(alpha), 0.02, 1.0)
    else:
        # Per-triangle color = mean of its 3 vertex RGBA colors
        c0 = cols_u8[tris[:, 0]].astype(np.float64)
        c1 = cols_u8[tris[:, 1]].astype(np.float64)
        c2 = cols_u8[tris[:, 2]].astype(np.float64)
        tri_rgba = (c0 + c1 + c2) / (3.0 * 255.0)
        tri_rgba[:, 3] = np.clip(tri_rgba[:, 3] * float(alpha), 0.05, 1.0)
    if brighten > 0.0:
        # Lift wall/section colors so near faces do not read as near-black in MPL 3D.
        b = float(np.clip(brighten, 0.0, 0.45))
        tri_rgba[:, :3] = np.clip(tri_rgba[:, :3] * (1.0 + 0.35 * b) + b, 0.0, 1.0)

    if edge_rgba is not None:
        edgecolors = edge_rgba
        lw = 0.35
    elif edge_alpha > 0:
        edgecolors = (0.15, 0.15, 0.15, float(edge_alpha))
        lw = 0.15
    else:
        edgecolors = "none"
        lw = 0.0
    coll = Poly3DCollection(
        polys,
        facecolors=tri_rgba,
        edgecolors=edgecolors,
        linewidths=lw,
        antialiased=False,
        shade=False,
    )
    # Average z-sort is more stable than "min" for dense block walls (avoids blank gray faces).
    try:
        coll.set_zsort("average")
    except Exception:
        pass
    ax.add_collection3d(coll)


def _plot_3d_attitude_discs(
    ax: Any,
    attitudes: Sequence[dict[str, Any]],
    x_origin: float,
    y_origin: float,
    max_symbols: int = 18,
    stem_len_m: float = 75.0,
) -> None:
    """Plot 3D strike/dip orientation pins and dip-direction vectors on top of the 3D block surface."""
    if not attitudes:
        return
    step = max(1, len(attitudes) // max_symbols)
    sub = list(attitudes)[::step][:max_symbols]
    xs = [float(a["centroid_x"]) - x_origin for a in sub]
    ys = [float(a["centroid_y"]) - y_origin for a in sub]
    zs = [float(a["centroid_z"]) + 14.0 for a in sub]
    ax.scatter(
        xs,
        ys,
        zs,
        c="#111111",
        s=22,
        marker="o",
        depthshade=False,
        zorder=20,
    )
    for a, x, y, z in zip(sub, xs, ys, zs):
        dd_rad = np.deg2rad(float(a["dip_dir_deg"]))
        dip_rad = np.deg2rad(float(a["dip_deg"]))
        dx_v = stem_len_m * np.sin(dd_rad) * np.cos(dip_rad)
        dy_v = stem_len_m * np.cos(dd_rad) * np.cos(dip_rad)
        dz_v = -stem_len_m * np.sin(dip_rad) * 0.5
        ax.plot(
            [x, x + dx_v],
            [y, y + dy_v],
            [z, z + dz_v],
            color="#111111",
            linewidth=1.2,
            zorder=20,
        )


def _plot_3d_surface_contacts(
    ax: Any,
    block_model: dict[str, Any],
    x_origin: float,
    y_origin: float,
    *,
    color: str = "#111111",
    linewidth: float = 1.2,
) -> None:
    """Draw non-fault unit contacts on the top surface as solid black lines (no tick marks)."""
    import contourpy

    dem: GeoGrid = block_model["dem"]
    geol = block_model.get("geol_labels")
    if geol is None:
        return
    labels = np.asarray(geol.data if hasattr(geol, "data") else geol, dtype=np.float64)
    lbl = np.where(np.isfinite(labels) & (labels >= 0), np.round(labels), np.nan)
    if not np.isfinite(lbl).any():
        return
    xs = dem.x_coords()
    ys = dem.y_coords()
    z_dem = np.asarray(dem.data, dtype=np.float64)
    order = np.argsort(ys)
    ys_inc = ys[order]
    lbl_inc = lbl[order]
    z_inc = z_dem[order]
    # Contour half-integer boundaries between integer unit IDs → solid contact polylines.
    levels = np.arange(np.nanmin(lbl_inc) + 0.5, np.nanmax(lbl_inc), 1.0)
    if levels.size == 0:
        return
    gen = contourpy.contour_generator(x=xs, y=ys_inc, z=np.nan_to_num(lbl_inc, nan=-1.0))
    for lev in levels:
        for line in gen.lines(float(lev)):
            if len(line) < 2:
                continue
            ci = np.interp(line[:, 0], xs, np.arange(len(xs)))
            ri = np.interp(line[:, 1], ys_inc, np.arange(len(ys_inc)))
            z_line = z_inc[np.clip(np.round(ri).astype(int), 0, len(ys_inc) - 1),
                           np.clip(np.round(ci).astype(int), 0, len(xs) - 1)] + 5.0
            ax.plot(
                line[:, 0] - x_origin,
                line[:, 1] - y_origin,
                z_line,
                color=color,
                linewidth=linewidth,
                solid_capstyle="round",
                zorder=16,
            )


def _fault_dip_tick_dirs(fault_spec: dict[str, Any]) -> tuple[float, float, float]:
    """Unit dip-direction vector (east, north, down-positive-z uses negative vertical component)."""
    dip_dir = float(fault_spec.get("dip_dir_deg", (float(fault_spec["strike_deg"]) + 90.0) % 360.0))
    dip = float(fault_spec.get("dip_deg", 50.0))
    rad_az = np.deg2rad(dip_dir)
    rad_dip = np.deg2rad(dip)
    dx = float(np.sin(rad_az) * np.cos(rad_dip))
    dy = float(np.cos(rad_az) * np.cos(rad_dip))
    dz = float(-np.sin(rad_dip))
    return dx, dy, dz


def _plot_3d_fault_traces(
    ax: Any,
    block_model: dict[str, Any],
    x_origin: float,
    y_origin: float,
    azimuth_deg: float,
    color: str = "#d62728",
    linewidth: float = 2.6,
    show_dip_ticks: bool = True,
) -> None:
    """
    Draw where the fault plane cuts the block: red trace on the top surface and facing side walls,
    with short ticks indicating fault-plane dip direction. The plane itself is inside the opaque
    block, so it is not filled on the solid-block view.
    """
    import contourpy

    dem: GeoGrid = block_model["dem"]
    z_flt = np.asarray(block_model.get("fault_surface_z"), dtype=np.float64)
    fault_spec = block_model.get("fault_spec")
    if fault_spec is None or z_flt.shape != dem.shape or not np.isfinite(z_flt).any():
        return
    xs, ys = dem.x_coords(), dem.y_coords()
    z_dem = np.asarray(dem.data, dtype=np.float64)
    z_bot = float(block_model["z_bottom"])
    order = np.argsort(ys)
    ys_inc, zd, zf = ys[order], z_dem[order], z_flt[order]
    tick_dx, tick_dy, tick_dz = _fault_dip_tick_dirs(fault_spec)
    tick_len = 120.0

    def _add_dip_ticks(x_line: np.ndarray, y_line: np.ndarray, z_line: np.ndarray) -> None:
        if not show_dip_ticks or len(x_line) < 2:
            return
        idx = np.linspace(0, len(x_line) - 1, num=min(11, max(len(x_line) // 8, 4)), dtype=int)
        for i in idx:
            ax.plot(
                [x_line[i], x_line[i] + tick_dx * tick_len],
                [y_line[i], y_line[i] + tick_dy * tick_len],
                [z_line[i], z_line[i] + tick_dz * tick_len * 0.85],
                color=color,
                linewidth=1.8,
                zorder=19,
            )

    # Top-surface trace: zero contour of (z_dem - z_fault), lifted slightly so it is not hidden by the drape
    gen = contourpy.contour_generator(x=xs, y=ys_inc, z=np.nan_to_num(zd - zf, nan=0.0))
    for line in gen.lines(0.0):
        if len(line) < 2:
            continue
        ci = np.interp(line[:, 0], xs, np.arange(len(xs)))
        ri = np.interp(line[:, 1], ys_inc, np.arange(len(ys_inc)))
        z_line = zd[np.round(ri).astype(int), np.round(ci).astype(int)] + 6.0
        xl = line[:, 0] - x_origin
        yl = line[:, 1] - y_origin
        ax.plot(xl, yl, z_line, color=color, linewidth=linewidth, zorder=18)
        _add_dip_ticks(xl, yl, z_line)

    # Side-wall traces on the walls facing the camera (outward normal toward the eye)
    az = np.deg2rad(float(azimuth_deg))
    eye = (np.cos(az), np.sin(az))
    walls = [
        ((-1.0, 0.0), np.full_like(ys_inc, xs[0]), ys_inc, zd[:, 0], zf[:, 0]),
        ((1.0, 0.0), np.full_like(ys_inc, xs[-1]), ys_inc, zd[:, -1], zf[:, -1]),
        ((0.0, -1.0), xs, np.full_like(xs, ys_inc[0]), zd[0, :], zf[0, :]),
        ((0.0, 1.0), xs, np.full_like(xs, ys_inc[-1]), zd[-1, :], zf[-1, :]),
    ]
    for normal, wx, wy, top, z_wall in walls:
        if normal[0] * eye[0] + normal[1] * eye[1] <= 0.0:
            continue
        z_line = np.where((z_wall >= z_bot) & (z_wall <= top), z_wall, np.nan)
        xl = wx - x_origin
        yl = wy - y_origin
        ax.plot(xl, yl, z_line, color=color, linewidth=linewidth, zorder=18)
        finite = np.isfinite(z_line)
        if np.count_nonzero(finite) >= 2:
            _add_dip_ticks(xl[finite], yl[finite], z_line[finite])


def _add_unit_legend(ax: Any, units_metadata: Sequence[dict[str, Any]], include_fault: bool = True) -> None:
    handles = [
        Patch(
            facecolor=u.get("color", "#888888"),
            edgecolor="#222222",
            linewidth=0.6,
            label=f"{u.get('code', u['id'])}: {u.get('name', '')}",
        )
        for u in units_metadata
    ]
    if include_fault:
        handles.append(
            Patch(
                facecolor="#d62728",
                edgecolor="#222222",
                linewidth=0.6,
                alpha=0.28,
                label="Thrust / Reverse Fault Plane",
            )
        )
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        fontsize=7.5,
        framealpha=0.92,
        title="Stratigraphic Units & Structure",
        title_fontsize=8.0,
    )


def _sample_dem_polyline(
    dem: GeoGrid,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    n: int = 80,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinear DEM samples along a map polyline (x, y, z)."""
    xa, ya = (float(start_xy[0]), float(start_xy[1]))
    xb, yb = (float(end_xy[0]), float(end_xy[1]))
    xs = np.linspace(xa, xb, int(n))
    ys = np.linspace(ya, yb, int(n))
    cols = (xs - dem.x0) / dem.dx - 0.5
    rows = (ys - dem.y0) / dem.dy - 0.5
    c0 = np.clip(np.floor(cols).astype(int), 0, dem.cols - 2)
    r0 = np.clip(np.floor(rows).astype(int), 0, dem.rows - 2)
    tc = np.clip(cols - c0, 0.0, 1.0)
    tr = np.clip(rows - r0, 0.0, 1.0)
    z00 = dem.data[r0, c0]
    z01 = dem.data[r0, c0 + 1]
    z10 = dem.data[r0 + 1, c0]
    z11 = dem.data[r0 + 1, c0 + 1]
    zs = (1 - tr) * ((1 - tc) * z00 + tc * z01) + tr * ((1 - tc) * z10 + tc * z11)
    zs = np.where(np.isfinite(zs), zs, float(np.nanmedian(dem.data)))
    return xs, ys, zs


def mark_section_on_block(
    ax: Any,
    block_model: dict[str, Any],
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    label_a: str = "A",
    label_ap: str = "A'",
    color: str = "#111111",
    linewidth: float = 2.8,
    n_samples: int = 100,
) -> None:
    """Draw the A–A' surface trace on a solid 3D block (map polyline lifted to the DEM)."""
    dem: GeoGrid = block_model["dem"]
    xs = dem.x_coords()
    ys = dem.y_coords()
    x_min, y_min = (float(xs.min()), float(ys.min()))
    px, py, pz = _sample_dem_polyline(dem, start_xy, end_xy, n=n_samples)
    ax.plot(
        px - x_min,
        py - y_min,
        pz + 10.0,
        color=color,
        linewidth=linewidth,
        zorder=25,
        solid_capstyle="round",
    )
    ax.text(
        float(px[0] - x_min),
        float(py[0] - y_min),
        float(pz[0]) + 35.0,
        label_a,
        fontsize=11,
        fontweight="bold",
        color=color,
        zorder=26,
    )
    ax.text(
        float(px[-1] - x_min),
        float(py[-1] - y_min),
        float(pz[-1]) + 35.0,
        label_ap,
        fontsize=11,
        fontweight="bold",
        color=color,
        zorder=26,
    )


def plot_3d_block_diagram(
    block_model: dict[str, Any],
    surface_drape_rgba: np.ndarray | None = None,
    vertical_exaggeration: float = 1.65,
    azimuth_deg: float = -58.0,
    elevation_deg: float = 28.0,
    top_stride: int = 2,
    wall_stride: int = 1,
    show_attitudes: bool = True,
    show_fault_plane: bool = False,
    show_surface_contacts: bool = True,
    show_legend: bool = True,
    ax: Any | None = None,
    title: str | None = "3D Geological Block Diagram (Surface & Projected Subsurface)",
    section_xy: (
        tuple[tuple[float, float], tuple[float, float]]
        | Sequence[tuple[tuple[float, float], tuple[float, float], str, str]]
        | None
    ) = None,
) -> tuple[Any, Any]:
    """
    Render a publication-quality 3D perspective geological block diagram showing:
    - Top topographic surface draped with geological units (or custom RGB/RGBA drape) — opaque
    - Four vertical cross-section side walls (South, North, West, East) down to `z_bottom`
    - Base plate and 3-point strike/dip orientation pins.
    - Non-fault unit contacts as solid black surface lines (no ticks).
    - `show_fault_plane=True`: red fault trace with dip-direction ticks on top/side walls
      (the plane itself is inside the opaque block); the legend lists the fault only then.
    - `section_xy`: one `(start, end)` pair or a sequence of `(start, end, label0, label1)`
      surface traces marked on the solid block top (e.g. A–A' and B–B').
    """
    dem: GeoGrid = block_model["dem"]
    xs = dem.x_coords()
    ys = dem.y_coords()
    x_min, x_max = (float(xs.min()), float(xs.max()))
    y_min, y_max = (float(ys.min()), float(ys.max()))
    z_bot = float(block_model["z_bottom"])
    z_top = float(np.nanmax(dem.data))

    if ax is None:
        fig = plt.figure(figsize=(10, 7.5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    xx, yy = np.meshgrid(xs - x_min, ys - y_min)
    zz = np.nan_to_num(dem.data, nan=float(np.nanmean(dem.data)))
    if surface_drape_rgba is not None:
        drape_u8 = np.ascontiguousarray(surface_drape_rgba, dtype=np.uint8)
    else:
        drape_u8 = np.ascontiguousarray(block_model["surface_drape_rgba"], dtype=np.uint8)
    if drape_u8.shape[2] == 3:
        drape_u8 = np.dstack([drape_u8, np.full(dem.shape, 255, dtype=np.uint8)])
    # Stride>=2 keeps matplotlib's 3D painter algorithm stable (stride=1 → blank gray faces).
    stride = max(2, int(top_stride))
    facecolors = drape_u8[::stride, ::stride, :3].astype(np.float64) / 255.0
    # plot_surface wants (M-1, N-1, 4) facecolors for an (M, N) grid
    fc = np.concatenate(
        [facecolors[:-1, :-1, :], np.ones((*facecolors[:-1, :-1].shape[:2], 1), dtype=np.float64)],
        axis=-1,
    )

    # Base plate first, then opaque unit-colored top, then walls last so near-side beds
    # are not swallowed by axis panes / painter-order artifacts.
    _render_mesh_poly3d(
        ax,
        block_model["mesh"],
        x_origin=x_min,
        y_origin=y_min,
        include_face_groups=[5],
        face_stride=1,
        alpha=1.0,
        edge_alpha=0.0,
        brighten=0.05,
    )
    ax.plot_surface(
        xx[::stride, ::stride],
        yy[::stride, ::stride],
        zz[::stride, ::stride],
        facecolors=fc,
        shade=False,
        linewidth=0,
        antialiased=False,
        rstride=1,
        cstride=1,
    )
    _render_mesh_poly3d(
        ax,
        block_model["mesh"],
        x_origin=x_min,
        y_origin=y_min,
        include_face_groups=[1, 2, 3, 4],
        face_stride=max(2, int(wall_stride)),
        alpha=1.0,
        edge_alpha=0.0,
        brighten=0.16,
    )

    # Draw crisp black bounding rim around top topography and base corners
    ax.plot(xx[-1, :], yy[-1, :], zz[-1, :], color="#1a1a1a", linewidth=1.1, zorder=15)
    ax.plot(xx[0, :], yy[0, :], zz[0, :], color="#1a1a1a", linewidth=0.9, zorder=15)
    ax.plot(xx[:, 0], yy[:, 0], zz[:, 0], color="#1a1a1a", linewidth=1.1, zorder=15)
    ax.plot(xx[:, -1], yy[:, -1], zz[:, -1], color="#1a1a1a", linewidth=0.9, zorder=15)

    if show_surface_contacts:
        _plot_3d_surface_contacts(ax, block_model, x_origin=x_min, y_origin=y_min)

    if show_fault_plane:
        _plot_3d_fault_traces(ax, block_model, x_origin=x_min, y_origin=y_min, azimuth_deg=azimuth_deg)

    if show_attitudes and block_model.get("attitudes"):
        _plot_3d_attitude_discs(ax, block_model["attitudes"], x_origin=x_min, y_origin=y_min)

    if section_xy is not None:
        # Accept a single (start, end) or a list of (start, end, lab0, lab1)
        if (
            isinstance(section_xy, (list, tuple))
            and len(section_xy) >= 2
            and isinstance(section_xy[0], (list, tuple))
            and len(section_xy[0]) >= 2
            and isinstance(section_xy[0][0], (int, float))
        ):
            # Single transect (start_xy, end_xy)
            mark_section_on_block(ax, block_model, section_xy[0], section_xy[1])  # type: ignore[arg-type]
        else:
            for item in section_xy:  # type: ignore[union-attr]
                s0, s1 = item[0], item[1]
                lab0 = item[2] if len(item) > 2 else "A"
                lab1 = item[3] if len(item) > 3 else "A'"
                mark_section_on_block(
                    ax, block_model, s0, s1, label_a=str(lab0), label_ap=str(lab1)
                )

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = (z_top - z_bot) * float(vertical_exaggeration)
    max_xy = max(x_range, y_range, 1.0)
    ax.set_xlim(0.0, x_range)
    ax.set_ylim(0.0, y_range)
    ax.set_zlim(z_bot, z_top + 25.0)
    ax.set_box_aspect((x_range / max_xy, y_range / max_xy, max(z_range / max_xy, 0.28)))
    ax.view_init(elev=float(elevation_deg), azim=float(azimuth_deg))
    # Hide gray axis panes that otherwise read as "empty" uncolored block faces.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((0.75, 0.75, 0.75, 0.35))
        except Exception:
            pass
    ax.set_xlabel("Relative Easting (m)", labelpad=6)
    ax.set_ylabel("Relative Northing (m)", labelpad=6)
    ax.set_zlabel("Elevation (m NAVD88)", labelpad=6)
    if title:
        ax.set_title(title, fontweight="bold", pad=10)

    if show_legend:
        _add_unit_legend(
            ax,
            block_model["units_metadata"],
            include_fault=show_fault_plane and block_model.get("fault_spec") is not None,
        )
    return fig, ax


def plot_3d_cutaway_block(
    block_model: dict[str, Any],
    vertical_exaggeration: float = 1.65,
    azimuth_deg: float = -54.0,
    elevation_deg: float = 30.0,
    face_stride: int = 1,
    show_fault_plane: bool = True,
    show_legend: bool = False,
    ax: Any | None = None,
    title: str | None = "3D Cutaway Block Diagram (Internal Fold & Thrust Architecture)",
) -> tuple[Any, Any]:
    """
    Render the excavated 3D cutaway geological block diagram (`block_model['cutaway_mesh']`),
    revealing the internal folded stratigraphy, horizontal bench slice, and 3D fault plane.
    """
    dem: GeoGrid = block_model["dem"]
    xs = dem.x_coords()
    ys = dem.y_coords()
    x_min, x_max = (float(xs.min()), float(xs.max()))
    y_min, y_max = (float(ys.min()), float(ys.max()))
    z_bot = float(block_model["z_bottom"])
    z_top = float(np.nanmax(dem.data))

    if ax is None:
        fig = plt.figure(figsize=(9.5, 7.2))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    c_mesh = block_model["cutaway_mesh"]
    # Render base, outer walls, and internal cutaway walls/floor at full resolution
    _render_mesh_poly3d(
        ax,
        c_mesh,
        x_origin=x_min,
        y_origin=y_min,
        include_face_groups=[1, 2, 3, 4, 5, 6],
        face_stride=face_stride,
        alpha=1.0,
        edge_alpha=0.04,
    )
    # Render remaining top surface (`face_group=0`)
    _render_mesh_poly3d(
        ax,
        c_mesh,
        x_origin=x_min,
        y_origin=y_min,
        include_face_groups=[0],
        face_stride=max(2, face_stride * 2),
        alpha=1.0,
        edge_alpha=0.0,
    )
    if show_fault_plane:
        _render_mesh_poly3d(
            ax,
            c_mesh,
            x_origin=x_min,
            y_origin=y_min,
            include_face_groups=[7],
            face_stride=3,
            alpha=0.68,
            edge_alpha=0.22,
        )

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = (z_top - z_bot) * float(vertical_exaggeration)
    max_xy = max(x_range, y_range, 1.0)
    ax.set_xlim(0.0, x_range)
    ax.set_ylim(0.0, y_range)
    ax.set_zlim(z_bot, z_top + 25.0)
    ax.set_box_aspect((x_range / max_xy, y_range / max_xy, max(z_range / max_xy, 0.28)))
    ax.view_init(elev=float(elevation_deg), azim=float(azimuth_deg))
    ax.set_xlabel("Relative Easting (m)", labelpad=6)
    ax.set_ylabel("Relative Northing (m)", labelpad=6)
    ax.set_zlabel("Elevation (m NAVD88)", labelpad=6)
    if title:
        ax.set_title(title, fontweight="bold", pad=10)
    if show_legend:
        _add_unit_legend(
            ax,
            block_model["units_metadata"],
            include_fault=block_model.get("fault_spec") is not None,
        )
    return fig, ax


def plot_3d_fence_diagram(
    block_model: dict[str, Any],
    fence_panels: Sequence[dict[str, Any]] | None = None,
    vertical_exaggeration: float = 1.65,
    azimuth_deg: float = -52.0,
    elevation_deg: float = 28.0,
    show_fault_plane: bool = True,
    show_wireframe_surface: bool = True,
    show_legend: bool = False,
    ax: Any | None = None,
    title: str | None = "3D Intersecting Structural Fence Diagram",
    section_alpha: float = 1.0,
    fault_alpha: float = 0.14,
    fault_face_stride: int = 4,
) -> tuple[Any, Any]:
    """
    Render vertical 3D cross-section fence ribbons (`slice_3d_fence_diagram`) inside the
    3D block bounding frame, with optional translucent 3D fault surface and topographic
    wireframe. Prefer one transect (e.g. A–A') when comparing with a 2D section; pass
    `fence_panels` explicitly to avoid the default three-panel layout.

    Section faces default to fully opaque (`section_alpha=1`); the fault plane is a light,
    highly transparent red mesh (`fault_alpha≈0.14`) so stratigraphy stays readable.
    """
    if fence_panels is None:
        fence_panels = slice_3d_fence_diagram(block_model)

    dem: GeoGrid = block_model["dem"]
    xs = dem.x_coords()
    ys = dem.y_coords()
    x_min, x_max = (float(xs.min()), float(xs.max()))
    y_min, y_max = (float(ys.min()), float(ys.max()))
    z_bot = float(block_model["z_bottom"])
    z_top = float(np.nanmax(dem.data))

    if ax is None:
        fig = plt.figure(figsize=(9.5, 7.2))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Base plate + faint topo wireframe FIRST so opaque section faces can cover them
    # (Matplotlib 3D has no real depth buffer; later artists otherwise "ghost" through).
    _render_mesh_poly3d(
        ax,
        block_model["mesh"],
        x_origin=x_min,
        y_origin=y_min,
        include_face_groups=[5],
        face_stride=1,
        alpha=0.22,
        edge_alpha=0.12,
    )
    if show_wireframe_surface:
        xx, yy = np.meshgrid(xs - x_min, ys - y_min)
        zz = np.nan_to_num(dem.data, nan=float(np.nanmean(dem.data)))
        ax.plot_wireframe(
            xx,
            yy,
            zz,
            rstride=20,
            cstride=20,
            color="#777777",
            linewidth=0.18,
            alpha=0.10,
        )

    # Opaque vertical fence ribbon(s) next — solid stratigraphic colors.
    for panel in fence_panels:
        _render_mesh_poly3d(
            ax,
            panel["mesh"],
            x_origin=x_min,
            y_origin=y_min,
            include_face_groups=None,
            face_stride=1,
            alpha=float(np.clip(section_alpha, 0.05, 1.0)),
            edge_alpha=0.08,
            brighten=0.12,
        )
        # Top topographic trace of the fence panel
        px = np.asarray(panel["x"]) - x_min
        py = np.asarray(panel["y"]) - y_min
        pz = np.asarray(panel["z_topo_m"])
        ax.plot(px, py, pz + 3.0, color="#111111", linewidth=1.8, zorder=18)
        name = str(panel.get("name", "Fence A-A'"))
        # Prefer "Fence A-A'" → labels A / A'
        parts = name.replace("Fence ", "").split("-")
        lab0 = parts[0] if parts else "A"
        lab1 = parts[-1] if len(parts) > 1 else "A'"
        ax.text(
            float(px[0]),
            float(py[0]),
            float(pz[0]) + 28.0,
            lab0,
            fontsize=9.5,
            fontweight="bold",
            color="#111111",
        )
        ax.text(
            float(px[-1]),
            float(py[-1]),
            float(pz[-1]) + 28.0,
            lab1,
            fontsize=9.5,
            fontweight="bold",
            color="#111111",
        )

    # Light transparent red fault last (like the "good" figure): sections stay readable through it.
    if show_fault_plane:
        fa = float(np.clip(fault_alpha, 0.02, 1.0))
        _render_mesh_poly3d(
            ax,
            block_model["mesh"],
            x_origin=x_min,
            y_origin=y_min,
            include_face_groups=[7],
            face_stride=max(1, int(fault_face_stride)),
            alpha=1.0,
            edge_alpha=0.0,
            force_rgba=(0.839, 0.153, 0.157, fa),
            edge_rgba=(0.70, 0.05, 0.05, min(0.22, fa + 0.06)),
        )

    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = (z_top - z_bot) * float(vertical_exaggeration)
    max_xy = max(x_range, y_range, 1.0)
    ax.set_xlim(0.0, x_range)
    ax.set_ylim(0.0, y_range)
    ax.set_zlim(z_bot, z_top + 25.0)
    ax.set_box_aspect((x_range / max_xy, y_range / max_xy, max(z_range / max_xy, 0.28)))
    ax.view_init(elev=float(elevation_deg), azim=float(azimuth_deg))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
            axis.pane.set_edgecolor((0.75, 0.75, 0.75, 0.35))
        except Exception:
            pass
    ax.set_xlabel("Relative Easting (m)", labelpad=6)
    ax.set_ylabel("Relative Northing (m)", labelpad=6)
    ax.set_zlabel("Elevation (m NAVD88)", labelpad=6)
    if title:
        ax.set_title(title, fontweight="bold", pad=10)
    if show_legend:
        _add_unit_legend(
            ax,
            block_model["units_metadata"],
            include_fault=show_fault_plane and block_model.get("fault_spec") is not None,
        )
    return fig, ax


def plot_3d_block_and_section(
    block_model: dict[str, Any],
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    sections: Sequence[tuple[tuple[float, float], tuple[float, float], str, str]] | None = None,
    vertical_exaggeration: float = 1.65,
    azimuth_deg: float = -78.0,
    elevation_deg: float = 28.0,
    block_azimuth_deg: float | None = None,
    block_elevation_deg: float | None = None,
    section_azimuth_deg: float | None = None,
    section_elevation_deg: float | None = None,
    n_samples: int = 160,
    show_fault_plane: bool = True,
    top_stride: int = 2,
    wall_stride: int = 2,
    titles: Sequence[str] | None = None,
    section_alpha: float = 1.0,
    fault_alpha: float = 0.14,
) -> tuple[Any, tuple[Any, Any]]:
    """
    README-style two-panel figure: opaque solid 3D geologic block | dual-section 3D view.

    Left (a): opaque volumetric block with section-position traces marked (default A–A', or
    pass `sections=` for A–A' + B–B'). Non-fault contacts are solid black; the fault is a red
    surface trace with dip ticks (plane itself is inside the opaque block).

    Right (b): **one** 3D axes containing all requested section fences together (both A–A′ and
    B–B′ when provided), with opaque stratigraphic colors and a light, highly transparent red
    fault plane — not separate stacked subplots.

    Both panels share one camera (`azimuth_deg` / `elevation_deg`). Default ≈ original SE
    face-on view (−58°) rotated 20° clockwise (−78°) so more of the S–N side is visible.
    """
    if sections is None:
        sections = [(start_xy, end_xy, "A", "A'")]
    sections = list(sections)

    # Shared camera (legacy kwargs still accepted but default to the shared view).
    az = float(azimuth_deg if block_azimuth_deg is None else block_azimuth_deg)
    el = float(elevation_deg if block_elevation_deg is None else block_elevation_deg)
    if section_azimuth_deg is not None:
        az = float(section_azimuth_deg)
    if section_elevation_deg is not None:
        el = float(section_elevation_deg)

    fences = slice_3d_fence_diagram(
        block_model,
        transects=[(s[0], s[1]) for s in sections],
        n_samples=n_samples,
    )
    for fence, sec in zip(fences, sections):
        fence["name"] = f"Fence {sec[2]}-{sec[3]}"

    labels = " & ".join(f"{s[2]}–{s[3]}" for s in sections)
    if titles is None:
        titles = (
            f"(a) 3D geologic block with {labels} section locations",
            f"(b) Matching 3D sections {labels} (published ~50° SW thrust; tip/throw inferred)",
        )
    titles = list(titles)

    # Exactly two panels: left solid block | right one box with all section fences.
    fig = plt.figure(figsize=(14.8, 6.8), layout="constrained")
    ax0 = fig.add_subplot(121, projection="3d")
    ax1 = fig.add_subplot(122, projection="3d")

    plot_3d_block_diagram(
        block_model,
        vertical_exaggeration=vertical_exaggeration,
        azimuth_deg=az,
        elevation_deg=el,
        top_stride=max(2, int(top_stride)),
        wall_stride=max(2, int(wall_stride)),
        show_fault_plane=show_fault_plane,
        show_surface_contacts=True,
        show_attitudes=False,
        show_legend=False,
        ax=ax0,
        title=titles[0],
        section_xy=list(sections),
    )
    plot_3d_fence_diagram(
        block_model,
        fence_panels=fences,
        vertical_exaggeration=vertical_exaggeration,
        azimuth_deg=az,
        elevation_deg=el,
        show_fault_plane=show_fault_plane,
        # No topo wireframe: with two opaque fences it only ghosts through MPL faces.
        show_wireframe_surface=False,
        show_legend=True,
        ax=ax1,
        title=titles[1] if len(titles) > 1 else f"(b) Matching 3D sections {labels}",
        section_alpha=section_alpha,
        fault_alpha=fault_alpha,
        fault_face_stride=5,
    )
    return fig, (ax0, ax1)


# Backward-compatible alias (older notebooks/tests). Prefer `plot_3d_block_and_section`.
def plot_3d_section_comparison(
    block_unfaulted: dict[str, Any],
    block_faulted: dict[str, Any],
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    vertical_exaggeration: float = 1.65,
    azimuth_deg: float | None = None,
    elevation_deg: float = 28.0,
    n_samples: int = 160,
    show_fault_plane: bool = True,
    titles: tuple[str, str] | None = None,
) -> tuple[Any, tuple[Any, Any]]:
    """Deprecated alias: opaque faulted block + matching faulted section fence(s)."""
    kwargs: dict[str, Any] = dict(
        vertical_exaggeration=vertical_exaggeration,
        elevation_deg=elevation_deg,
        n_samples=n_samples,
        show_fault_plane=show_fault_plane,
        titles=titles,
    )
    if azimuth_deg is not None:
        kwargs["azimuth_deg"] = azimuth_deg
    return plot_3d_block_and_section(block_faulted, start_xy, end_xy, **kwargs)


def export_interactive_block_html(
    block_model: dict[str, Any],
    output_html: str | Path,
    extra_surface_drapes: dict[str, np.ndarray] | None = None,
    grid_stride: int = 2,
    title: str = "pyGeoMapper — Interactive 3D Geological Block & Subsurface Viewer",
) -> dict[str, Any]:
    """
    Generate a 100% self-contained, offline interactive 3D HTML5/WebGL-style block-diagram viewer
    that opens directly in any local browser or Jupyter notebook without external CDN dependencies.

    Interactive controls include:
    - Mouse drag to orbit (azimuth / elevation), Shift+drag or right-drag to pan, scroll wheel to zoom
    - View mode selector: Full 3D Block, Cutaway Quadrant (SW Excavated), or Intersecting Fence Mode
    - Surface Drape selector: e.g. USGS Reference Geology, Auto-Mapped Units, Sentinel-2 False-Color,
      or 2D CWT Multi-Scale Complexity
    - Vertical Exaggeration slider (`1.0x` to `3.5x`)
    - Cutaway X / Y / Z excavation sliders to slice into the 3D fold & thrust architecture in real time
    - Toggle switches for 3D Thrust/Reverse Fault Plane, 3-Point Strike/Dip Discs, and individual
      stratigraphic formations (`Mm` through `Qa`).
    """
    out_path = Path(output_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    dem: GeoGrid = block_model["dem"]
    s = max(1, int(grid_stride))
    dem_sub = np.nan_to_num(dem.data[::s, ::s], nan=float(np.nanmean(dem.data)))
    rows_s, cols_s = dem_sub.shape

    xs = dem.x_coords()[::s]
    ys = dem.y_coords()[::s]
    x_center = float(0.5 * (xs.min() + xs.max()))
    y_center = float(0.5 * (ys.min() + ys.max()))
    xs_rel = np.round(xs - x_center, 1).tolist()
    ys_rel = np.round(ys - y_center, 1).tolist()

    # Subsample 3D voxel units (rows_s, cols_s, nz_s)
    vox_full = np.asarray(block_model["voxel_units"], dtype=np.int32)
    z_levels_full = np.asarray(block_model["z_levels"], dtype=np.float64)
    z_step = max(1, len(z_levels_full) // 24)
    vox_sub = vox_full[::s, ::s, ::z_step]
    z_levels_sub = np.round(z_levels_full[::z_step], 1).tolist()

    # Fault surface subsampled
    flt_sub = np.asarray(block_model["fault_surface_z"], dtype=np.float64)[::s, ::s]
    flt_list = np.where(np.isfinite(flt_sub), np.round(flt_sub, 1), None).tolist()

    # Prepare surface drapes as compact hex color grids or RGB arrays (rows_s, cols_s, 3)
    drapes_dict: dict[str, list[Any]] = {}
    base_drape = np.asarray(block_model["surface_drape_rgba"], dtype=np.uint8)[::s, ::s, :3]
    drapes_dict["Geological Map (Primary)"] = base_drape.tolist()

    if extra_surface_drapes:
        for d_name, d_arr in extra_surface_drapes.items():
            arr = np.asarray(d_arr)
            if arr.dtype != np.uint8:
                arr = np.clip(np.round(arr * 255.0), 0, 255).astype(np.uint8)
            drapes_dict[str(d_name)] = arr[::s, ::s, :3].tolist()

    # Attitude symbols
    att_list = []
    for a in block_model.get("attitudes", [])[:28]:
        att_list.append(
            {
                "x": round(float(a["centroid_x"]) - x_center, 1),
                "y": round(float(a["centroid_y"]) - y_center, 1),
                "z": round(float(a["centroid_z"]), 1),
                "strike": round(float(a["strike_deg"]), 1),
                "dip": round(float(a["dip_deg"]), 1),
                "dip_dir": round(float(a["dip_dir_deg"]), 1),
            }
        )

    payload = {
        "title": title,
        "rows": int(rows_s),
        "cols": int(cols_s),
        "nz": int(len(z_levels_sub)),
        "xs": xs_rel,
        "ys": ys_rel,
        "z_levels": z_levels_sub,
        "z_bottom": round(float(block_model["z_bottom"]), 1),
        "z_top": round(float(np.max(dem_sub)), 1),
        "dem": np.round(dem_sub, 1).tolist(),
        "surface_units": np.asarray(block_model["surface_units"], dtype=np.int32)[::s, ::s].tolist(),
        "voxel_units": vox_sub.tolist(),
        "fault_z": flt_list,
        "units": block_model["units_metadata"],
        "unit_stats": block_model.get("unit_volume_stats", []),
        "drapes": drapes_dict,
        "attitudes": att_list,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>{title}</title>
<style>
  :root {{
    --bg: #0f172a;
    --panel: #1e293b;
    --border: #334155;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --accent: #38bdf8;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    height: 100vh;
    overflow: hidden;
  }}
  #sidebar {{
    width: 340px;
    min-width: 320px;
    background: var(--panel);
    border-right: 1px solid var(--border);
    padding: 16px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }}
  h1 {{ font-size: 15px; font-weight: 700; color: var(--accent); line-height: 1.3; }}
  .subtitle {{ font-size: 11.5px; color: var(--muted); }}
  .card {{
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 11px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .card-title {{
    font-size: 11.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--muted);
  }}
  label {{
    font-size: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
  }}
  select, input[type="range"] {{
    width: 100%;
    accent-color: var(--accent);
  }}
  select {{
    background: #0f172a;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 6px 8px;
    font-size: 12px;
  }}
  .unit-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
    padding: 3px 0;
  }}
  .swatch {{
    width: 13px;
    height: 13px;
    border-radius: 3px;
    display: inline-block;
    margin-right: 7px;
    border: 1px solid rgba(255,255,255,0.35);
    vertical-align: middle;
  }}
  .btn-row {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  button {{
    background: #0f172a;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 5px 9px;
    font-size: 11.5px;
    cursor: pointer;
  }}
  button:hover {{ border-color: var(--accent); color: var(--accent); }}
  #viewport {{
    flex: 1;
    position: relative;
    display: flex;
    flex-direction: column;
  }}
  canvas {{
    width: 100%;
    height: 100%;
    display: block;
    cursor: grab;
  }}
  canvas:active {{ cursor: grabbing; }}
  #hud {{
    position: absolute;
    top: 12px;
    right: 14px;
    background: rgba(15, 23, 42, 0.84);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 9px 12px;
    font-size: 11.5px;
    color: var(--muted);
    pointer-events: none;
    line-height: 1.45;
  }}
</style>
</head>
<body>
<div id="sidebar">
  <div>
    <h1>{title}</h1>
    <div class="subtitle">Rust 3D Stratigraphic &amp; Faulted Block Engine (Offline WebGL/Canvas)</div>
  </div>

  <div class="card">
    <div class="card-title">Display Mode &amp; Surface Drape</div>
    <label>3D View Mode</label>
    <select id="modeSelect">
      <option value="block">Full 3D Block Diagram</option>
      <option value="cutaway" selected>3D Cutaway Block (SW Excavated)</option>
      <option value="fence">3D Intersecting Fence Slices</option>
    </select>
    <label>Topographic Surface Drape</label>
    <select id="drapeSelect"></select>
  </div>

  <div class="card">
    <div class="card-title">Camera &amp; Vertical Scale</div>
    <label>Vertical Exaggeration <span id="veVal">1.8x</span></label>
    <input type="range" id="veSlider" min="1.0" max="3.5" step="0.1" value="1.8" />
    <div class="btn-row">
      <button id="btnSW">SW Oblique</button>
      <button id="btnNW">NW Canyon</button>
      <button id="btnSE">SE Forelimb</button>
      <button id="btnTop">Map View</button>
    </div>
  </div>

  <div class="card">
    <div class="card-title">3D Cutaway &amp; Structural Overlays</div>
    <label>Cutaway X Extent <span id="cutXVal">48%</span></label>
    <input type="range" id="cutXSlider" min="0" max="85" step="2" value="48" />
    <label>Cutaway Y Extent <span id="cutYVal">48%</span></label>
    <input type="range" id="cutYSlider" min="0" max="85" step="2" value="48" />
    <label>Cutaway Bench Elevation <span id="cutZVal">1025 m</span></label>
    <input type="range" id="cutZSlider" min="0" max="85" step="5" value="35" />
    <label><input type="checkbox" id="chkFault" checked /> Show 3D Thrust Fault Surface</label>
    <label><input type="checkbox" id="chkAttitudes" checked /> Show 3-Point Strike/Dip Pins</label>
  </div>

  <div class="card">
    <div class="card-title">Stratigraphic Formations</div>
    <div id="unitList"></div>
  </div>
</div>

<div id="viewport">
  <canvas id="glCanvas"></canvas>
  <div id="hud">
    <div><strong style="color:#f1f5f9;">Orbit:</strong> Left-drag &nbsp;|&nbsp; <strong style="color:#f1f5f9;">Pan:</strong> Shift+drag &nbsp;|&nbsp; <strong style="color:#f1f5f9;">Zoom:</strong> Scroll</div>
    <div id="hudStats"></div>
  </div>
</div>

<script>
const DATA = {payload_json};

const canvas = document.getElementById("glCanvas");
const ctx = canvas.getContext("2d");

let state = {{
  azimuth: -54.0,
  elevation: 28.0,
  zoom: 1.0,
  panX: 0.0,
  panY: 0.0,
  ve: 1.8,
  mode: "cutaway",
  drapeName: Object.keys(DATA.drapes)[0],
  cutX: 0.48,
  cutY: 0.48,
  cutZIdx: Math.floor(DATA.nz * 0.35),
  showFault: true,
  showAttitudes: true,
  visibleUnits: {{}}
}};

const unitColorsRgb = {{}};
DATA.units.forEach(u => {{
  state.visibleUnits[u.id] = true;
  const hex = u.color.replace("#", "");
  unitColorsRgb[u.id] = [
    parseInt(hex.substring(0,2), 16),
    parseInt(hex.substring(2,4), 16),
    parseInt(hex.substring(4,6), 16)
  ];
}});

const drapeSelect = document.getElementById("drapeSelect");
Object.keys(DATA.drapes).forEach(k => {{
  const opt = document.createElement("option");
  opt.value = k;
  opt.textContent = k;
  drapeSelect.appendChild(opt);
}});

const unitListEl = document.getElementById("unitList");
DATA.units.forEach(u => {{
  const row = document.createElement("div");
  row.className = "unit-row";
  const stat = DATA.unit_stats.find(s => s.id === u.id);
  const pct = stat ? (stat.volume_fraction * 100).toFixed(1) + "%" : "";
  row.innerHTML = `<label style="justify-content:flex-start;cursor:pointer;">
    <input type="checkbox" checked data-uid="${{u.id}}" />
    <span class="swatch" style="background:${{u.color}}"></span>
    <span><strong>${{u.code}}</strong> — ${{u.name.split("(")[0]}}</span>
  </label><span style="color:#94a3b8;font-size:11px;">${{pct}}</span>`;
  row.querySelector("input").addEventListener("change", (e) => {{
    state.visibleUnits[u.id] = e.target.checked;
    requestRender();
  }});
  unitListEl.appendChild(row);
}});

function resizeCanvas() {{
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * window.devicePixelRatio;
  canvas.height = rect.height * window.devicePixelRatio;
  requestRender();
}}
window.addEventListener("resize", resizeCanvas);

let isDragging = false, lastX = 0, lastY = 0, isPan = false;
canvas.addEventListener("mousedown", e => {{
  isDragging = true;
  isPan = e.shiftKey || e.button === 2;
  lastX = e.clientX;
  lastY = e.clientY;
}});
window.addEventListener("mouseup", () => {{ isDragging = false; }});
window.addEventListener("mousemove", e => {{
  if (!isDragging) return;
  const dx = e.clientX - lastX;
  const dy = e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;
  if (isPan) {{
    state.panX += dx;
    state.panY += dy;
  }} else {{
    state.azimuth = (state.azimuth - dx * 0.45) % 360;
    state.elevation = Math.max(5, Math.min(85, state.elevation + dy * 0.35));
  }}
  requestRender();
}});
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("wheel", e => {{
  e.preventDefault();
  state.zoom = Math.max(0.45, Math.min(2.8, state.zoom * (e.deltaY < 0 ? 1.08 : 0.92)));
  requestRender();
}}, {{ passive: false }});

document.getElementById("modeSelect").addEventListener("change", e => {{ state.mode = e.target.value; requestRender(); }});
drapeSelect.addEventListener("change", e => {{ state.drapeName = e.target.value; requestRender(); }});
document.getElementById("veSlider").addEventListener("input", e => {{
  state.ve = parseFloat(e.target.value);
  document.getElementById("veVal").textContent = state.ve.toFixed(1) + "x";
  requestRender();
}});
document.getElementById("cutXSlider").addEventListener("input", e => {{
  state.cutX = parseFloat(e.target.value) / 100.0;
  document.getElementById("cutXVal").textContent = e.target.value + "%";
  requestRender();
}});
document.getElementById("cutYSlider").addEventListener("input", e => {{
  state.cutY = parseFloat(e.target.value) / 100.0;
  document.getElementById("cutYVal").textContent = e.target.value + "%";
  requestRender();
}});
document.getElementById("cutZSlider").addEventListener("input", e => {{
  const frac = parseFloat(e.target.value) / 100.0;
  state.cutZIdx = Math.min(DATA.nz - 2, Math.floor(frac * DATA.nz));
  document.getElementById("cutZVal").textContent = Math.round(DATA.z_levels[state.cutZIdx]) + " m";
  requestRender();
}});
document.getElementById("chkFault").addEventListener("change", e => {{ state.showFault = e.target.checked; requestRender(); }});
document.getElementById("chkAttitudes").addEventListener("change", e => {{ state.showAttitudes = e.target.checked; requestRender(); }});

document.getElementById("btnSW").addEventListener("click", () => {{ state.azimuth = -54; state.elevation = 28; requestRender(); }});
document.getElementById("btnNW").addEventListener("click", () => {{ state.azimuth = -135; state.elevation = 30; requestRender(); }});
document.getElementById("btnSE").addEventListener("click", () => {{ state.azimuth = 38; state.elevation = 26; requestRender(); }});
document.getElementById("btnTop").addEventListener("click", () => {{ state.azimuth = -90; state.elevation = 82; requestRender(); }});

function projectPoint(x, y, z, cx, cy, scale, cosA, sinA, cosE, sinE, zMid) {{
  const zr = (z - zMid) * state.ve;
  const xr = x * cosA - y * sinA;
  const yr = x * sinA + y * cosA;
  const sx = cx + (xr * scale) + state.panX;
  const sy = cy - ((yr * sinE + zr * cosE) * scale) + state.panY;
  const depth = yr * cosE - zr * sinE;
  return [sx, sy, depth];
}}

function rgbStr(rgb, shade) {{
  const r = Math.min(255, Math.max(0, Math.round(rgb[0] * shade)));
  const g = Math.min(255, Math.max(0, Math.round(rgb[1] * shade)));
  const b = Math.min(255, Math.max(0, Math.round(rgb[2] * shade)));
  return `rgb(${{r}},${{g}},${{b}})`;
}}

let renderPending = false;
function requestRender() {{
  if (!renderPending) {{
    renderPending = true;
    requestAnimationFrame(renderScene);
  }}
}}

function renderScene() {{
  renderPending = false;
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const R = DATA.rows, C = DATA.cols, NZ = DATA.nz;
  const spanXY = Math.max(
    Math.abs(DATA.xs[C - 1] - DATA.xs[0]),
    Math.abs(DATA.ys[R - 1] - DATA.ys[0]),
    1.0
  );
  const scale = (Math.min(W, H) * 0.56 / spanXY) * state.zoom;
  const cx = W * 0.50, cy = H * 0.54;
  const zMid = 0.5 * (DATA.z_bottom + DATA.z_top);

  const azRad = (state.azimuth + 90.0) * Math.PI / 180.0;
  const elRad = state.elevation * Math.PI / 180.0;
  const cosA = Math.cos(azRad), sinA = Math.sin(azRad);
  const cosE = Math.cos(elRad), sinE = Math.sin(elRad);

  const cutCMax = state.mode === "cutaway" ? Math.floor(C * state.cutX) : 0;
  const cutRMin = state.mode === "cutaway" ? Math.floor(R * (1.0 - state.cutY)) : R;
  const zCut = DATA.z_levels[state.cutZIdx];

  function inCutaway(r, c) {{
    return state.mode === "cutaway" && r >= cutRMin && c < cutCMax;
  }}

  const quads = [];
  const drape = DATA.drapes[state.drapeName] || DATA.drapes[Object.keys(DATA.drapes)[0]];

  // 1. Top surface or cutaway horizontal bench
  if (state.mode !== "fence") {{
    for (let r = 0; r < R - 1; r++) {{
      for (let c = 0; c < C - 1; c++) {{
        const isCut = inCutaway(r, c);
        const uid = isCut
          ? DATA.voxel_units[r][c][state.cutZIdx]
          : DATA.surface_units[r][c];
        if (uid >= 0 && !state.visibleUnits[uid]) continue;

        const z00 = isCut ? Math.min(zCut, DATA.dem[r][c]) : DATA.dem[r][c];
        const z10 = isCut ? Math.min(zCut, DATA.dem[r+1][c]) : DATA.dem[r+1][c];
        const z11 = isCut ? Math.min(zCut, DATA.dem[r+1][c+1]) : DATA.dem[r+1][c+1];
        const z01 = isCut ? Math.min(zCut, DATA.dem[r][c+1]) : DATA.dem[r][c+1];

        const p0 = projectPoint(DATA.xs[c], DATA.ys[r], z00, cx, cy, scale, cosA, sinA, cosE, sinE, zMid);
        const p1 = projectPoint(DATA.xs[c+1], DATA.ys[r], z01, cx, cy, scale, cosA, sinA, cosE, sinE, zMid);
        const p2 = projectPoint(DATA.xs[c+1], DATA.ys[r+1], z11, cx, cy, scale, cosA, sinA, cosE, sinE, zMid);
        const p3 = projectPoint(DATA.xs[c], DATA.ys[r+1], z10, cx, cy, scale, cosA, sinA, cosE, sinE, zMid);

        const col = isCut ? (unitColorsRgb[uid] || [180,180,180]) : drape[r][c];
        const shade = isCut ? 0.90 : 1.0;
        quads.push({{
          pts: [p0, p1, p2, p3],
          depth: 0.25 * (p0[2] + p1[2] + p2[2] + p3[2]),
          fill: rgbStr(col, shade),
          stroke: isCut ? "rgba(15,23,42,0.18)" : null
        }});
      }}
    }}
  }}

  // Helper to emit vertical wall column strips
  function addWallStrip(r0, c0, r1, c1, izStart, zCapOverride, shade) {{
    const x0 = DATA.xs[c0], y0 = DATA.ys[r0];
    const x1 = DATA.xs[c1], y1 = DATA.ys[r1];
    const cap0 = zCapOverride !== null ? Math.min(zCapOverride, DATA.dem[r0][c0]) : DATA.dem[r0][c0];
    const cap1 = zCapOverride !== null ? Math.min(zCapOverride, DATA.dem[r1][c1]) : DATA.dem[r1][c1];
    for (let iz = izStart; iz < NZ - 1; iz++) {{
      const zlo = DATA.z_levels[iz], zhi = DATA.z_levels[iz + 1];
      if (zlo >= cap0 && zlo >= cap1) break;
      let uid = DATA.voxel_units[r0][c0][iz];
      if (uid < 0 && iz > 0) uid = DATA.voxel_units[r0][c0][iz - 1];
      if (uid < 0 || !state.visibleUnits[uid]) continue;
      const za0 = Math.min(zlo, cap0), zb0 = Math.min(zlo, cap1);
      const za1 = Math.min(zhi, cap0), zb1 = Math.min(zhi, cap1);
      const p0 = projectPoint(x0, y0, za0, cx, cy, scale, cosA, sinA, cosE, sinE, zMid);
      const p1 = projectPoint(x1, y1, zb0, cx, cy, scale, cosA, sinA, cosE, sinE, zMid);
      const p2 = projectPoint(x1, y1, zb1, cx, cy, scale, cosA, sinA, cosE, sinE, zMid);
      const p3 = projectPoint(x0, y0, za1, cx, cy, scale, cosA, sinA, cosE, sinE, zMid);
      const prevUid = iz > 0 ? DATA.voxel_units[r0][c0][iz - 1] : uid;
      const localShade = (prevUid >= 0 && prevUid !== uid) ? shade * 0.76 : shade;
      quads.push({{
        pts: [p0, p1, p2, p3],
        depth: 0.25 * (p0[2] + p1[2] + p2[2] + p3[2]),
        fill: rgbStr(unitColorsRgb[uid] || [160,160,160], localShade),
        stroke: null
      }});
    }}
  }}

  if (state.mode !== "fence") {{
    // Outer 4 walls: South (r=R-1), North (r=0), West (c=0), East (c=C-1)
    for (let c = 0; c < C - 1; c++) {{
      addWallStrip(R - 1, c, R - 1, c + 1, 0, inCutaway(R - 1, c) ? zCut : null, 0.90);
      addWallStrip(0, c, 0, c + 1, 0, inCutaway(0, c) ? zCut : null, 0.78);
    }}
    for (let r = 0; r < R - 1; r++) {{
      addWallStrip(r, 0, r + 1, 0, 0, inCutaway(r, 0) ? zCut : null, 0.85);
      addWallStrip(r, C - 1, r + 1, C - 1, 0, inCutaway(r, C - 1) ? zCut : null, 0.80);
    }}
    // Cutaway internal vertical walls
    if (state.mode === "cutaway" && cutCMax > 0 && cutRMin < R) {{
      for (let c = 0; c < Math.min(cutCMax, C - 1); c++) {{
        addWallStrip(cutRMin, c, cutRMin, c + 1, state.cutZIdx, null, 0.94);
      }}
      for (let r = cutRMin; r < R - 1; r++) {{
        addWallStrip(r, cutCMax, r + 1, cutCMax, state.cutZIdx, null, 0.88);
      }}
    }}
  }} else {{
    // Intersecting 3D Fence Mode: 3 structural & strike fence ribbons
    const midR = Math.floor(R * 0.5), midC = Math.floor(C * 0.5);
    for (let c = 0; c < C - 1; c++) addWallStrip(midR, c, midR, c + 1, 0, null, 0.92);
    for (let r = 0; r < R - 1; r++) addWallStrip(r, midC, r + 1, midC, 0, null, 0.88);
    const nDiag = Math.min(R, C) - 1;
    for (let i = 0; i < nDiag; i++) {{
      addWallStrip(R - 1 - i, i, R - 2 - i, i + 1, 0, null, 0.95);
    }}
  }}

  // 3. Optional 3D Thrust Fault Plane
  if (state.showFault) {{
    for (let r = 0; r < R - 1; r += 2) {{
      for (let c = 0; c < C - 1; c += 2) {{
        const zf = DATA.fault_z[r][c];
        if (zf === null || zf < DATA.z_bottom || zf > DATA.dem[r][c]) continue;
        const r1 = Math.min(R - 1, r + 2), c1 = Math.min(C - 1, c + 2);
        const zf10 = DATA.fault_z[r1][c], zf11 = DATA.fault_z[r1][c1], zf01 = DATA.fault_z[r][c1];
        if (zf10 === null || zf11 === null || zf01 === null) continue;
        const p0 = projectPoint(DATA.xs[c], DATA.ys[r], Math.min(zf, DATA.dem[r][c]), cx, cy, scale, cosA, sinA, cosE, sinE, zMid);
        const p1 = projectPoint(DATA.xs[c1], DATA.ys[r], Math.min(zf01, DATA.dem[r][c1]), cx, cy, scale, cosA, sinA, cosE, sinE, zMid);
        const p2 = projectPoint(DATA.xs[c1], DATA.ys[r1], Math.min(zf11, DATA.dem[r1][c1]), cx, cy, scale, cosA, sinA, cosE, sinE, zMid);
        const p3 = projectPoint(DATA.xs[c], DATA.ys[r1], Math.min(zf10, DATA.dem[r1][c]), cx, cy, scale, cosA, sinA, cosE, sinE, zMid);
        quads.push({{
          pts: [p0, p1, p2, p3],
          depth: 0.25 * (p0[2] + p1[2] + p2[2] + p3[2]),
          fill: "rgba(220, 38, 38, 0.62)",
          stroke: "rgba(254, 202, 202, 0.35)"
        }});
      }}
    }}
  }}

  // Painter's algorithm sort (back to front)
  quads.sort((a, b) => b.depth - a.depth);

  for (let i = 0; i < quads.length; i++) {{
    const q = quads[i];
    ctx.beginPath();
    ctx.moveTo(q.pts[0][0], q.pts[0][1]);
    ctx.lineTo(q.pts[1][0], q.pts[1][1]);
    ctx.lineTo(q.pts[2][0], q.pts[2][1]);
    ctx.lineTo(q.pts[3][0], q.pts[3][1]);
    ctx.closePath();
    ctx.fillStyle = q.fill;
    ctx.fill();
    if (q.stroke) {{
      ctx.strokeStyle = q.stroke;
      ctx.lineWidth = 0.6;
      ctx.stroke();
    }}
  }}

  // 4. 3-Point Strike/Dip Pins
  if (state.showAttitudes) {{
    DATA.attitudes.forEach(a => {{
      const p = projectPoint(a.x, a.y, a.z + 12.0, cx, cy, scale, cosA, sinA, cosE, sinE, zMid);
      const ddRad = a.dip_dir * Math.PI / 180.0;
      const pTip = projectPoint(
        a.x + 85.0 * Math.sin(ddRad),
        a.y + 85.0 * Math.cos(ddRad),
        a.z - 20.0,
        cx, cy, scale, cosA, sinA, cosE, sinE, zMid
      );
      ctx.beginPath();
      ctx.arc(p[0], p[1], 3.2 * window.devicePixelRatio, 0, Math.PI * 2);
      ctx.fillStyle = "#f8fafc";
      ctx.fill();
      ctx.strokeStyle = "#0f172a";
      ctx.lineWidth = 1.4;
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(p[0], p[1]);
      ctx.lineTo(pTip[0], pTip[1]);
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 1.8;
      ctx.stroke();
    }});
  }}

  document.getElementById("hudStats").textContent =
    `Azimuth: ${{Math.round(state.azimuth)}}° | Elev: ${{Math.round(state.elevation)}}° | V.E.: ${{state.ve.toFixed(1)}}x | Polygons: ${{quads.length.toLocaleString()}}`;
}}

resizeCanvas();
</script>
</body>
</html>
"""
    out_path.write_text(html_content, encoding="utf-8")
    return {
        "html_path": str(out_path),
        "file_size_bytes": int(out_path.stat().st_size),
        "grid_shape": (rows_s, cols_s, len(z_levels_sub)),
        "surface_drapes": list(drapes_dict.keys()),
    }


def launch_block_viewer(
    block_model: dict[str, Any],
    output_html: str | Path = "outputs/interactive_3d_block_viewer.html",
    extra_surface_drapes: dict[str, np.ndarray] | None = None,
    grid_stride: int = 2,
    width: str = "100%",
    height: int = 680,
) -> Any:
    """
    Build the self-contained interactive 3D HTML5/WebGL geological block viewer (`export_interactive_block_html`)
    and return an inline `IPython.display.IFrame` (or HTML metadata dict when IPython is not active)
    for immediate interactive inspection inside a Jupyter Notebook.
    """
    info = export_interactive_block_html(
        block_model=block_model,
        output_html=output_html,
        extra_surface_drapes=extra_surface_drapes,
        grid_stride=grid_stride,
    )
    try:
        from IPython.display import IFrame

        return IFrame(src=str(info["html_path"]), width=width, height=height)
    except ImportError:  # pragma: no cover
        return info
