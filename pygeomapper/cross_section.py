# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Topographic swath profiling, 3-point/least-squares bed attitude estimation, and shallow
geological cross-section construction backed by Rust.

Includes:
- Bilinear centerline and perpendicular swath corridor sampling (`extract_swath_profile`,
  `extract_multilayer_profile`)
- 3-point / least-squares / robust-trimmed planar attitude solver (`estimate_bed_attitude`)
  and automated contact attitude scanner (`estimate_contact_attitudes`) in Rust
- Apparent dip projection (`compute_apparent_dip`)
- Potential-field shallow geological cross-section builder (`build_geological_cross_section`)
  and publication-ready 2D cross-section renderer (`plot_geological_cross_section`).
"""

from __future__ import annotations

from itertools import product
from typing import Any, Sequence

import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from . import _core
from .raster_io import GeoGrid


def extract_swath_profile(
    grid: GeoGrid,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    n_samples: int = 200,
    swath_width_m: float = 150.0,
    n_cross: int = 11,
) -> dict[str, np.ndarray]:
    """
    Sample a centerline profile and perpendicular swath corridor (`mean`, `min`, `max`, `std`)
    across `grid` from `start_xy = (xa, ya)` to `end_xy = (xb, yb)` using bilinear interpolation in Rust.
    """
    xa, ya = (float(start_xy[0]), float(start_xy[1]))
    xb, yb = (float(end_xy[0]), float(end_xy[1]))
    arr = np.ascontiguousarray(grid.data, dtype=np.float64)
    res = _core.sample_swath_profile_rs(
        arr,
        float(grid.x0),
        float(grid.y0),
        float(grid.dx),
        float(grid.dy),
        xa,
        ya,
        xb,
        yb,
        int(n_samples),
        float(swath_width_m),
        int(n_cross),
        float(grid.nodata),
    )
    return dict(res)


def extract_multilayer_profile(
    grids: dict[str, GeoGrid],
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    n_samples: int = 200,
    swath_width_m: float = 150.0,
    n_cross: int = 11,
) -> dict[str, Any]:
    """
    Sample multiple co-registered `GeoGrid` layers along the exact same transect `(start_xy -> end_xy)`.
    """
    if not grids:
        raise ValueError("grids dictionary must not be empty")
    out: dict[str, Any] = {"layers": {}}
    for i, (name, g) in enumerate(grids.items()):
        prof = extract_swath_profile(
            g,
            start_xy=start_xy,
            end_xy=end_xy,
            n_samples=n_samples,
            swath_width_m=swath_width_m,
            n_cross=n_cross,
        )
        if i == 0:
            out["distance_m"] = prof["distance_m"]
            out["x"] = prof["x"]
            out["y"] = prof["y"]
            out["start_xy"] = start_xy
            out["end_xy"] = end_xy
            out["swath_width_m"] = swath_width_m
        out["layers"][name] = {
            "center": prof["center"],
            "mean": prof["mean"],
            "min": prof["min"],
            "max": prof["max"],
            "std": prof["std"],
            "units": g.units,
        }
    return out


def estimate_bed_attitude(
    xs: Sequence[float] | np.ndarray,
    ys: Sequence[float] | np.ndarray,
    zs: Sequence[float] | np.ndarray,
    robust_trim: bool = True,
) -> dict[str, Any] | None:
    """
    Fit a geological plane `z(x, y) = dz_dx * x + dz_dy * y + intercept_c` to 3 or more
    3D contact points `(xs, ys, zs)` in Rust (`_core.estimate_bed_attitude_rs`).

    Returns Right-Hand-Rule `strike_deg` `[0, 360)`, `dip_deg` `[0, 90]`, `dip_dir_deg` `[0, 360)`,
    `r_squared`, `rmse_m`, and `collinearity_ratio`.
    """
    xa = np.ascontiguousarray(np.asarray(xs, dtype=np.float64))
    ya = np.ascontiguousarray(np.asarray(ys, dtype=np.float64))
    za = np.ascontiguousarray(np.asarray(zs, dtype=np.float64))
    res = _core.estimate_bed_attitude_rs(xa, ya, za, bool(robust_trim))
    return dict(res) if res is not None else None


def estimate_contact_attitudes(
    labels: GeoGrid | np.ndarray,
    dem: GeoGrid | np.ndarray,
    window_m: float = 350.0,
    min_points: int = 6,
    min_relief_m: float = 8.0,
    min_r2: float = 0.50,
    min_collinearity: float = 0.05,
) -> list[dict[str, Any]]:
    """
    Automatically extract lithologic contact traces between adjacent formations in `labels`
    and fit local 3D planes using `dem` across sliding windows of size `window_m` in Rust.
    """
    ref = labels if isinstance(labels, GeoGrid) else (dem if isinstance(dem, GeoGrid) else None)
    l_arr = labels.data if isinstance(labels, GeoGrid) else np.asarray(labels)
    d_arr = dem.data if isinstance(dem, GeoGrid) else np.asarray(dem, dtype=np.float64)

    lbl_i32 = np.where(np.isfinite(l_arr) & (l_arr >= 0), np.round(l_arr).astype(np.int32), -1)
    lbl_i32 = np.ascontiguousarray(lbl_i32, dtype=np.int32)
    dem_f64 = np.ascontiguousarray(d_arr, dtype=np.float64)

    if ref is not None:
        x0, y0, dx, dy, nodata = (ref.x0, ref.y0, ref.dx, ref.dy, ref.nodata)
    else:
        x0, y0, dx, dy, nodata = (0.0, 0.0, 10.0, -10.0, np.nan)

    recs = _core.estimate_contact_attitudes_grid_rs(
        lbl_i32,
        dem_f64,
        float(x0),
        float(y0),
        float(dx),
        float(dy),
        float(window_m),
        int(min_points),
        float(min_relief_m),
        float(min_r2),
        float(min_collinearity),
        float(nodata),
    )
    return [dict(r) for r in recs]


def compute_apparent_dip(
    dip_deg: float,
    dip_dir_deg: float,
    section_azimuth_deg: float,
) -> float:
    """
    Compute apparent dip angle (degrees, signed: `> 0` dipping forward in the direction of
    `section_azimuth_deg`, `< 0` dipping backward) in Rust.
    """
    return float(
        _core.compute_apparent_dip_deg_rs(
            float(dip_deg), float(dip_dir_deg), float(section_azimuth_deg)
        )
    )


def _in_hanging_wall(fault: dict[str, float], s: np.ndarray, z: np.ndarray) -> np.ndarray:
    """True where section point `(s, z)` lies on the hanging-wall side of a planar section fault."""
    s_fault = fault["s0"] + (fault["z0"] - z) * fault["cot_dip"]
    return s >= s_fault if fault["dip_deg"] >= 0 else s <= s_fault


def build_geological_cross_section(
    dem: GeoGrid,
    geol_labels: GeoGrid,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    attitudes: Sequence[dict[str, Any]] | None = None,
    depth_m: float = 350.0,
    n_samples: int = 240,
    n_depth: int = 160,
    swath_width_m: float = 120.0,
    enforce_superposition: bool = True,
    faults: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Construct a 2D shallow geological cross-section along `start_xy -> end_xy` by combining:
    1. Bilinear topographic swath profile (`center`, `min`, `max`) from `dem`
    2. Surface geological unit sampling and contact intersection detection from `geol_labels`
    3. 3-point / least-squares bed attitudes projected into apparent dip along the section azimuth
    4. Continuous structural horizon integration (`zeta(s, z) = z - h_struct(s) - fault_offset(s, z)`)
       projecting stratigraphic units and bedding planes to `depth_m` below the surface.

    Parameters
    ----------
    enforce_superposition : bool, default True
        When True and unit IDs are ordered oldest (`0`) to youngest (`K-1`), ensures the local
        apparent dip sign at each contact crossing dips toward the younger stratigraphic unit, and
        keeps the projected horizons in stratigraphic order (each unit at least 5 m thick).
    faults : sequence of dict, optional
        Each fault dict may specify `{'distance_m': float, 'dip_deg': float, 'throw_m': float, 'name': str}`,
        where `dip_deg` is apparent dip in section (>0 dipping right, <0 dipping left) and `throw_m`
        is vertical structural offset of the hanging wall relative to the footwall. Unit colors and
        projected contacts are both offset along the dipping fault plane (not a vertical step).
    """
    topo_prof = extract_swath_profile(
        dem,
        start_xy=start_xy,
        end_xy=end_xy,
        n_samples=n_samples,
        swath_width_m=swath_width_m,
    )
    dist_m = topo_prof["distance_m"]
    xs = topo_prof["x"]
    ys = topo_prof["y"]
    z_topo = topo_prof["center"]

    xa, ya = (float(start_xy[0]), float(start_xy[1]))
    xb, yb = (float(end_xy[0]), float(end_xy[1]))
    section_az_deg = float(np.rad2deg(np.arctan2(xb - xa, yb - ya)) % 360.0)

    # Nearest-pixel sampling of integer geological unit labels along the profile
    cols_idx = np.clip(
        np.round((xs - geol_labels.x0) / geol_labels.dx).astype(int), 0, geol_labels.cols - 1
    )
    rows_idx = np.clip(
        np.round((ys - geol_labels.y0) / geol_labels.dy).astype(int), 0, geol_labels.rows - 1
    )
    raw_u = geol_labels.data[rows_idx, cols_idx]
    surf_units = np.where(np.isfinite(raw_u) & (raw_u >= 0), np.round(raw_u).astype(int), -1)

    if attitudes is None:
        attitudes = estimate_contact_attitudes(geol_labels, dem)

    # Detect surface contact crossings along the profile
    contact_crossings: list[dict[str, Any]] = []
    search_radius_m = max(abs(dem.dx) * 45.0, 500.0)

    for i in range(len(dist_m) - 1):
        u_left = int(surf_units[i])
        u_right = int(surf_units[i + 1])
        if u_left >= 0 and u_right >= 0 and u_left != u_right:
            s_c = 0.5 * (dist_m[i] + dist_m[i + 1])
            x_c = 0.5 * (xs[i] + xs[i + 1])
            y_c = 0.5 * (ys[i] + ys[i + 1])
            z_c = 0.5 * (z_topo[i] + z_topo[i + 1])
            ua, ub = (min(u_left, u_right), max(u_left, u_right))

            # Match nearby 3-point attitude measurements
            best_app_dip: float | None = None
            best_true_dip: float = 20.0
            best_strike: float = (section_az_deg - 90.0) % 360.0
            best_dip_dir: float = section_az_deg
            best_r2: float = 0.0

            if attitudes:
                weighted_tan = 0.0
                weight_sum = 0.0
                for att in attitudes:
                    d_xy = float(
                        np.hypot(att["centroid_x"] - x_c, att["centroid_y"] - y_c)
                    )
                    pair_match = (att.get("unit_a") == ua and att.get("unit_b") == ub)
                    eff_dist = d_xy * (0.45 if pair_match else 1.0)
                    if eff_dist <= search_radius_m:
                        w = float(att.get("r_squared", 0.7)) / max(eff_dist, 25.0)
                        app_d = compute_apparent_dip(
                            float(att["dip_deg"]),
                            float(att["dip_dir_deg"]),
                            section_az_deg,
                        )
                        weighted_tan += w * np.tan(np.deg2rad(app_d))
                        weight_sum += w
                        if w > best_r2:
                            best_r2 = w
                            best_true_dip = float(att["dip_deg"])
                            best_strike = float(att["strike_deg"])
                            best_dip_dir = float(att["dip_dir_deg"])
                if weight_sum > 0.0:
                    best_app_dip = float(np.rad2deg(np.arctan(weighted_tan / weight_sum)))

            if best_app_dip is None:
                best_app_dip = 18.0 if u_right > u_left else -18.0

            if enforce_superposition:
                mag = max(abs(best_app_dip), 8.0)
                # If u_right > u_left (younger to the right), beds dip to the right (positive apparent dip)
                best_app_dip = mag if u_right > u_left else -mag

            contact_crossings.append(
                {
                    "distance_m": float(s_c),
                    "x": float(x_c),
                    "y": float(y_c),
                    "z": float(z_c),
                    "unit_left": u_left,
                    "unit_right": u_right,
                    "unit_a": ua,
                    "unit_b": ub,
                    "apparent_dip_deg": float(best_app_dip),
                    "true_dip_deg": float(best_true_dip),
                    "strike_deg": float(best_strike),
                    "dip_dir_deg": float(best_dip_dir),
                }
            )

    # Interpolate continuous structural slope dz_bed/ds = -tan(apparent_dip) along profile
    if contact_crossings:
        c_dists = np.array([c["distance_m"] for c in contact_crossings], dtype=np.float64)
        c_slopes = np.array(
            [-np.tan(np.deg2rad(c["apparent_dip_deg"])) for c in contact_crossings],
            dtype=np.float64,
        )
        raw_slope = np.interp(dist_m, c_dists, c_slopes)
        bed_slope = gaussian_filter1d(raw_slope, sigma=max(2.0, n_samples / 30.0))
    else:
        bed_slope = np.zeros_like(dist_m)

    # Structural elevation horizon h_struct(s) = integral_0^s bed_slope(s') ds'
    ds = np.gradient(dist_m)
    h_struct = np.cumsum(bed_slope * ds)
    h_struct -= h_struct[0]

    # Build 2D elevation grid (n_depth, n_samples)
    z_min = float(np.nanmin(z_topo) - abs(depth_m))
    z_max = float(np.nanmax(z_topo) + 25.0)
    z_levels = np.linspace(z_min, z_max, int(n_depth))
    ss_2d, zz_2d = np.meshgrid(dist_m, z_levels)

    # Optional fault offset field delta_h_fault(z, s)
    fault_offset_2d = np.zeros_like(ss_2d)
    fault_offset_surf = np.zeros_like(dist_m)
    fault_traces: list[dict[str, Any]] = []
    fault_geoms: list[dict[str, float]] = []
    if faults:
        for f in faults:
            f_s0 = float(f["distance_m"])
            f_dip = float(f.get("dip_deg", 65.0))
            f_throw = float(f.get("throw_m", 40.0))
            f_name = str(f.get("name", "Fault"))
            f_z0 = float(np.interp(f_s0, dist_m, z_topo))
            cot_dip = 1.0 / np.tan(np.deg2rad(f_dip if abs(f_dip) > 5.0 else 65.0))
            # Fault trace position s_fault(z) = f_s0 + (f_z0 - z) * cot_dip
            geom = {"s0": f_s0, "z0": f_z0, "cot_dip": cot_dip, "dip_deg": f_dip, "throw_m": f_throw}
            fault_geoms.append(geom)
            fault_offset_2d[_in_hanging_wall(geom, ss_2d, zz_2d)] += f_throw
            fault_offset_surf[_in_hanging_wall(geom, dist_m, z_topo)] += f_throw

            z_trace = np.linspace(z_min, f_z0, 60)
            s_trace = f_s0 + (f_z0 - z_trace) * cot_dip
            fault_traces.append(
                {
                    "name": f_name,
                    "s_trace_m": s_trace,
                    "z_trace_m": z_trace,
                    "surface_distance_m": f_s0,
                    "surface_z_m": f_z0,
                    "dip_deg": f_dip,
                    "throw_m": f_throw,
                }
            )

    # Stratigraphic elevation coordinate zeta(z, s) = z - h_struct(s) - fault_offset(z, s)
    # Note: higher zeta corresponds to stratigraphically younger (higher) units!
    zeta_surf = z_topo - h_struct - fault_offset_surf
    zeta_2d = zz_2d - h_struct[np.newaxis, :] - fault_offset_2d

    # Calibrate stratigraphic unit thresholds along zeta from contact crossings (or unit medians)
    unique_units = sorted(int(u) for u in np.unique(surf_units) if u >= 0)
    unit_thresholds: dict[tuple[int, int], list[float]] = {}
    for c in contact_crossings:
        ua, ub = (c["unit_a"], c["unit_b"])
        idx_c = int(np.argmin(np.abs(dist_m - c["distance_m"])))
        z_val = float(zeta_surf[idx_c])
        unit_thresholds.setdefault((ua, ub), []).append(z_val)

    # Assign subsurface unit at each (iz, is) by comparing zeta_2d(iz, is) to the surface unit's
    # local contact boundaries, propagating beds smoothly beneath the surface:
    subsurface_units = np.full(ss_2d.shape, np.nan, dtype=np.float64)

    # Compute sorted boundary levels between consecutive stratigraphic units
    boundary_levels: list[tuple[float, int, int]] = []
    for (ua, ub), z_list in sorted(unit_thresholds.items()):
        boundary_levels.append((float(np.mean(z_list)), ua, ub))
    if enforce_superposition and boundary_levels:
        # Older-below-younger: order horizons by the unit they cap and keep each unit >= 5 m thick
        boundary_levels.sort(key=lambda item: (max(item[1], item[2]), min(item[1], item[2])))
        ordered: list[tuple[float, int, int]] = []
        for b_zeta, ua, ub in boundary_levels:
            if ordered:
                b_zeta = max(b_zeta, ordered[-1][0] + 5.0)
            ordered.append((b_zeta, ua, ub))
        boundary_levels = ordered
    boundary_levels.sort(key=lambda item: item[0])

    for j in range(len(dist_m)):
        dz_col = zz_2d[:, j] - z_topo[j]
        below = dz_col <= 0.0
        if not np.any(below):
            continue
        zeta_col = zeta_2d[below, j]
        if boundary_levels:
            # Classify each depth pixel by which stratigraphic horizon interval zeta_col falls into
            col_units = np.full(zeta_col.shape, boundary_levels[0][1], dtype=np.float64)
            for b_zeta, ua, ub in boundary_levels:
                higher_unit = ub if ub > ua else ua
                col_units = np.where(zeta_col >= b_zeta, float(higher_unit), col_units)
            # Force shallowest 15 m to match exact mapped surface unit at column j
            near_surf = (z_topo[j] - zz_2d[below, j]) <= 15.0
            if surf_units[j] >= 0:
                col_units = np.where(near_surf, float(surf_units[j]), col_units)
            subsurface_units[below, j] = col_units
        else:
            subsurface_units[below, j] = float(surf_units[j])

    # Project bedding contact curves z_contact(s) = zeta_c + h_struct(s) + fault_offset(s, z_contact).
    # Each fault-block combination gives one branch, kept only where (s, z) really lies in that block, so
    # contacts are cut along the dipping fault plane (and repeated beneath a reverse fault) like the unit colors.
    projected_contacts: list[dict[str, Any]] = []
    for (b_zeta, ua, ub) in boundary_levels:
        branches: list[np.ndarray] = []
        for in_hw in product((False, True), repeat=len(fault_geoms)):
            z_b = b_zeta + h_struct + sum(g["throw_m"] for g, on in zip(fault_geoms, in_hw) if on)
            valid = np.ones(dist_m.shape, dtype=bool)
            for g, on in zip(fault_geoms, in_hw):
                valid &= _in_hanging_wall(g, dist_m, z_b) == on
            branches.append(np.where(valid, z_b, np.nan))
        z_curve = branches[0].copy()
        for z_b in branches[1:]:
            z_curve = np.where(np.isfinite(z_curve), z_curve, z_b)
        # Mask curve where it rises above the topographic surface (eroded in air)
        z_curve_sub = np.where(z_curve <= z_topo + 5.0, z_curve, np.nan)
        z_curve_air = np.where(z_curve > z_topo, z_curve, np.nan)
        projected_contacts.append(
            {
                "unit_a": ua,
                "unit_b": ub,
                "zeta_m": b_zeta,
                "z_subsurface_m": z_curve_sub,
                "z_projected_air_m": z_curve_air,
                "z_subsurface_branches_m": [np.where(z_b <= z_topo + 5.0, z_b, np.nan) for z_b in branches],
                "z_projected_air_branches_m": [np.where(z_b > z_topo, z_b, np.nan) for z_b in branches],
            }
        )

    return {
        "distance_m": dist_m,
        "x": xs,
        "y": ys,
        "z_topo_m": z_topo,
        "z_swath_min_m": topo_prof["min"],
        "z_swath_max_m": topo_prof["max"],
        "surface_units": surf_units,
        "unique_units": unique_units,
        "section_azimuth_deg": section_az_deg,
        "start_xy": start_xy,
        "end_xy": end_xy,
        "depth_m": depth_m,
        "z_levels_m": z_levels,
        "subsurface_units": subsurface_units,
        "h_struct_m": h_struct,
        "apparent_bed_slope": bed_slope,
        "contact_crossings": contact_crossings,
        "projected_contacts": projected_contacts,
        "fault_traces": fault_traces,
    }


def plot_geological_cross_section(
    section: dict[str, Any],
    unit_metadata: dict[int, dict[str, Any]] | None = None,
    vertical_exaggeration: float = 1.5,
    ax: Any | None = None,
    title: str | None = None,
    show_dip_ticks: bool = True,
    show_swath_envelope: bool = True,
    show_eroded_projection: bool = True,
) -> tuple[Any, Any]:
    """
    Render a 2D geological cross-section showing the topographic profile, colored subsurface
    stratigraphic units, projected bedding horizons, eroded air-line projections, fault traces,
    and surface contact apparent dip ticks.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.8))
    else:
        fig = ax.figure

    dist_m = section["distance_m"]
    z_topo = section["z_topo_m"]
    z_levels = section["z_levels_m"]
    sub_units = section["subsurface_units"]
    unique_units = section["unique_units"]

    default_palette = [
        "#8c6bb1",
        "#6baed6",
        "#fd8d3c",
        "#e6550d",
        "#31a354",
        "#74c476",
        "#fdd0a2",
        "#969696",
    ]
    unit_colors: dict[int, str] = {}
    unit_labels: dict[int, str] = {}
    for idx, uid in enumerate(unique_units):
        if unit_metadata and uid in unit_metadata:
            info = unit_metadata[uid]
            unit_colors[uid] = info.get("color", default_palette[idx % len(default_palette)])
            code = info.get("code", f"U{uid}")
            name = info.get("name", f"Unit {uid}")
            unit_labels[uid] = f"{code}: {name}"
        else:
            unit_colors[uid] = default_palette[idx % len(default_palette)]
            unit_labels[uid] = f"Unit {uid}"

    # Map subsurface_units into discrete indices 0..K-1 for pcolormesh / contourf
    if unique_units:
        uid_to_idx = {uid: i for i, uid in enumerate(unique_units)}
        idx_grid = np.full(sub_units.shape, np.nan, dtype=np.float64)
        for uid, i_val in uid_to_idx.items():
            idx_grid[sub_units == uid] = float(i_val)
        cmap = mcolors.ListedColormap([unit_colors[uid] for uid in unique_units])
        bounds = np.arange(-0.5, len(unique_units) + 0.5, 1.0)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        ax.pcolormesh(
            dist_m,
            z_levels,
            idx_grid,
            cmap=cmap,
            norm=norm,
            shading="nearest",
            alpha=0.88,
        )

    # Plot projected subsurface bedding contacts and dashed above-surface fold projections
    for pc in section.get("projected_contacts", []):
        for z_sub in pc.get("z_subsurface_branches_m", [pc["z_subsurface_m"]]):
            ax.plot(
                dist_m,
                z_sub,
                color="#1f1f1f",
                linewidth=1.25,
                linestyle="-",
                alpha=0.85,
            )
        if show_eroded_projection:
            z_cap = np.nanmax(z_topo) + 140.0
            for z_air in pc.get("z_projected_air_branches_m", [pc["z_projected_air_m"]]):
                z_air_clipped = np.where(z_air <= z_cap, z_air, np.nan)
                ax.plot(
                    dist_m,
                    z_air_clipped,
                    color="#555555",
                    linewidth=0.95,
                    linestyle="--",
                    alpha=0.60,
                )

    # Plot fault traces if present
    for ft in section.get("fault_traces", []):
        ax.plot(
            ft["s_trace_m"],
            ft["z_trace_m"],
            color="#b30000",
            linewidth=2.4,
            linestyle="-",
            label=ft["name"],
        )

    # Plot swath min-max topographic corridor and centerline topographic profile
    if show_swath_envelope:
        ax.fill_between(
            dist_m,
            section["z_swath_min_m"],
            section["z_swath_max_m"],
            color="#4a4a4a",
            alpha=0.18,
            label="Swath Min–Max Envelope",
        )
    ax.plot(dist_m, z_topo, color="black", linewidth=2.2, label="Topographic Profile")

    # Plot surface contact pins and apparent dip ticks
    if show_dip_ticks:
        tick_len_s = max(float(dist_m[-1] - dist_m[0]) * 0.032, 45.0)
        for c in section.get("contact_crossings", []):
            s0 = c["distance_m"]
            z0 = c["z"]
            app_dip = c["apparent_dip_deg"]
            ds_tick = np.sign(app_dip) * tick_len_s if abs(app_dip) > 1e-3 else tick_len_s
            dz_tick = -abs(np.tan(np.deg2rad(app_dip))) * tick_len_s
            ax.plot(
                [s0, s0 + ds_tick],
                [z0, z0 + dz_tick],
                color="#d90000",
                linewidth=2.0,
                zorder=6,
            )
            ax.scatter([s0], [z0], color="#d90000", s=26, zorder=7)
            ax.annotate(
                f"{abs(app_dip):.0f}°",
                xy=(s0, z0),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#990000",
            )

    z_min_plot = float(z_levels.min())
    z_max_plot = float(np.nanmax(z_topo) + (155.0 if show_eroded_projection else 45.0))
    ax.set_xlim(float(dist_m[0]), float(dist_m[-1]))
    ax.set_ylim(z_min_plot, z_max_plot)
    if vertical_exaggeration > 0:
        ax.set_aspect(float(vertical_exaggeration))

    ax.set_xlabel("Distance Along Cross-Section Profile (m)")
    ax.set_ylabel("Elevation (m NAVD88)")
    if title:
        ax.set_title(title, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.45)

    handles = [Patch(facecolor=unit_colors[uid], edgecolor="k", label=unit_labels[uid]) for uid in unique_units]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.92,
        title="Stratigraphic Units",
        title_fontsize=8.5,
    )
    return fig, ax
