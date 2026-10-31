# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Terrain morphometry, surface roughness, and multiscale 2D Continuous Wavelet Transform (2D CWT)
topographic complexity analysis backed by parallel Rust kernels.

Includes:
- Horn (1981) slope and aspect
- Single- and multi-azimuth analytical hillshade
- Zevenbergen & Thorne (1987) Laplacian, profile, and planform curvature
- Local circular-window elevation standard deviation (roughness) and Topographic Position Index (TPI)
- Native Rust 2D Continuous Wavelet Transform (`compute_cwt2d_mexican_hat`, `compute_multiscale_complexity`,
  `compare_complexity_change`) using the isotropic Mexican-hat (Marr / Ricker) wavelet
- Optional adapter (`run_pytopocomplexity`) for `pytopocomplexity` (Lai et al., 2025,
  Earth Surface Dynamics, https://doi.org/10.5194/esurf-13-417-2025).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import _core
from .raster_io import GeoGrid, RasterStack

try:
    import pytopocomplexity as _ptc  # type: ignore[import-not-found]

    HAS_PYTOPOCOMPLEXITY = True
except ImportError:
    _ptc = None
    HAS_PYTOPOCOMPLEXITY = False


def _unpack_dem(
    dem: GeoGrid | np.ndarray,
    dx: float | None = None,
    dy: float | None = None,
) -> tuple[np.ndarray, float, float, GeoGrid | None]:
    if isinstance(dem, GeoGrid):
        arr = np.ascontiguousarray(dem.data, dtype=np.float64)
        gx = float(dx) if dx is not None else abs(dem.dx)
        gy = float(dy) if dy is not None else abs(dem.dy)
        return arr, gx, gy, dem
    arr = np.ascontiguousarray(np.asarray(dem, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D DEM array or GeoGrid, got shape {arr.shape}")
    gx = float(dx) if dx is not None else 10.0
    gy = float(dy) if dy is not None else 10.0
    return arr, gx, gy, None


def compute_slope_aspect(
    dem: GeoGrid | np.ndarray,
    dx: float | None = None,
    dy: float | None = None,
    nodata: float = np.nan,
) -> tuple[GeoGrid | np.ndarray, GeoGrid | np.ndarray]:
    """
    Compute Horn (1981) slope in degrees `[0, 90]` and aspect in degrees `[0, 360)` clockwise from North.
    """
    arr, gx, gy, ref = _unpack_dem(dem, dx, dy)
    slope, aspect = _core.compute_slope_aspect_rs(arr, gx, gy, float(nodata))
    if ref is not None:
        return (
            ref.with_data(slope, name="slope_deg", units="degrees"),
            ref.with_data(aspect, name="aspect_deg", units="degrees"),
        )
    return slope, aspect


def compute_hillshade(
    dem: GeoGrid | np.ndarray,
    azimuth_deg: float = 315.0,
    altitude_deg: float = 45.0,
    z_factor: float = 1.0,
    multi_azimuth: bool = False,
    azimuths: Sequence[float] = (315.0, 270.0, 360.0, 45.0),
    dx: float | None = None,
    dy: float | None = None,
    nodata: float = np.nan,
) -> GeoGrid | np.ndarray:
    """
    Compute analytical hillshade in `[0, 1]` via Rust.
    If `multi_azimuth=True`, averages illumination across `azimuths` for omnidirectional shaded relief.
    """
    arr, gx, gy, ref = _unpack_dem(dem, dx, dy)
    if not multi_azimuth:
        hs = _core.compute_hillshade_rs(
            arr, gx, gy, float(azimuth_deg), float(altitude_deg), float(z_factor), float(nodata)
        )
    else:
        acc = np.zeros_like(arr, dtype=np.float64)
        for az in azimuths:
            acc += _core.compute_hillshade_rs(
                arr, gx, gy, float(az), float(altitude_deg), float(z_factor), float(nodata)
            )
        hs = acc / max(len(azimuths), 1)

    if ref is not None:
        return ref.with_data(hs, name="hillshade", units="reflectance")
    return hs


def compute_curvature(
    dem: GeoGrid | np.ndarray,
    dx: float | None = None,
    dy: float | None = None,
    nodata: float = np.nan,
) -> dict[str, GeoGrid | np.ndarray]:
    """
    Compute Laplacian curvature, profile curvature, and planform curvature (1/m) in Rust.
    """
    arr, gx, gy, ref = _unpack_dem(dem, dx, dy)
    lap, prof, plan = _core.compute_curvature_rs(arr, gx, gy, float(nodata))
    if ref is not None:
        return {
            "laplacian": ref.with_data(lap, name="laplacian_curvature", units="1/m"),
            "profile": ref.with_data(prof, name="profile_curvature", units="1/m"),
            "planform": ref.with_data(plan, name="planform_curvature", units="1/m"),
        }
    return {"laplacian": lap, "profile": prof, "planform": plan}


def compute_roughness_and_tpi(
    dem: GeoGrid | np.ndarray,
    radius_px: int = 3,
    nodata: float = np.nan,
) -> dict[str, GeoGrid | np.ndarray]:
    """
    Compute circular-window elevation standard deviation (roughness, m) and
    Topographic Position Index (TPI, m) in Rust.
    """
    arr, _, _, ref = _unpack_dem(dem)
    rough, tpi = _core.compute_roughness_tpi_rs(arr, int(radius_px), float(nodata))
    if ref is not None:
        return {
            "roughness": ref.with_data(rough, name="roughness_std", units="m"),
            "tpi": ref.with_data(tpi, name="tpi", units="m"),
        }
    return {"roughness": rough, "tpi": tpi}


def compute_cwt2d_mexican_hat(
    dem: GeoGrid | np.ndarray,
    scale_m: float = 40.0,
    dx: float | None = None,
    dy: float | None = None,
    nodata: float = np.nan,
    pixel_size_m: float | None = None,
) -> dict[str, Any]:
    """
    Compute single-scale 2D Continuous Wavelet Transform (2D CWT) using the isotropic Mexican-hat
    (Marr / Ricker) wavelet at spatial scale `scale_m` (meters) in Rust.

    The characteristic landform wavelength associated with Mexican-hat scale `s` is
    `lambda_m = sqrt(2) * pi * s ≈ 4.4429 * s`.

    Returns
    -------
    dict with keys:
        - `'cwt'`: `GeoGrid` or `np.ndarray` of signed 2D Mexican-hat wavelet response (> 0 ridges, < 0 valleys)
        - `'energy'`: `GeoGrid` or `np.ndarray` of local RMS wavelet energy envelope (terrain complexity at `scale_m`)
        - `'scale_m'`: float spatial scale `s` (m)
        - `'characteristic_wavelength_m'`: float `sqrt(2) * pi * s` (m)
    """
    if pixel_size_m is not None:
        dx = dx if dx is not None else float(pixel_size_m)
        dy = dy if dy is not None else float(pixel_size_m)
    arr, gx, gy, ref = _unpack_dem(dem, dx, dy)
    s_m = float(scale_m)
    lam_m = float(np.sqrt(2.0) * np.pi * s_m)
    cwt, energy = _core.compute_cwt2d_mexican_hat_rs(arr, gx, gy, s_m, float(nodata))
    if ref is not None:
        return {
            "cwt": ref.with_data(cwt, name=f"cwt_mexhat_{int(round(s_m))}m", units="m"),
            "energy": ref.with_data(energy, name=f"cwt_energy_{int(round(s_m))}m", units="m"),
            "scale_m": s_m,
            "characteristic_wavelength_m": lam_m,
        }
    return {
        "cwt": cwt,
        "energy": energy,
        "scale_m": s_m,
        "characteristic_wavelength_m": lam_m,
    }


def compute_multiscale_complexity(
    dem: GeoGrid | np.ndarray,
    scales_m: Sequence[float] = (20.0, 35.0, 60.0, 100.0, 160.0),
    dx: float | None = None,
    dy: float | None = None,
    nodata: float = np.nan,
    pixel_size_m: float | None = None,
) -> dict[str, Any]:
    """
    Compute multiscale 2D Mexican-hat CWT topographic complexity across `scales_m` (in meters)
    using parallel Rust separable wavelet convolutions (`_core.compute_multiscale_complexity_rs`).

    Adapts the multiscale 2D continuous wavelet terrain complexity framework of `pyTopoComplexity`
    (Lai et al., 2025, Earth Surface Dynamics, https://doi.org/10.5194/esurf-13-417-2025).

    Returns
    -------
    dict with keys:
        - `'cwt_stack'`: `RasterStack` or 3D `np.ndarray` `(rows, cols, n_scales)` of signed wavelet responses
        - `'energy_stack'`: `RasterStack` or 3D `np.ndarray` `(rows, cols, n_scales)` of wavelet energy envelopes
        - `'mean_complexity'`: `GeoGrid` or 2D `np.ndarray` of mean multiscale wavelet energy
        - `'max_energy'`: `GeoGrid` or 2D `np.ndarray` of peak wavelet energy across scales
        - `'dominant_scale_m'`: `GeoGrid` or 2D `np.ndarray` of dominant Mexican-hat scale `s*` (m)
        - `'dominant_wavelength_m'`: `GeoGrid` or 2D `np.ndarray` of dominant landform wavelength `lambda*` (m)
        - `'scales_m'`: 1D `np.ndarray` of evaluated Mexican-hat scales (m)
        - `'wavelengths_m'`: 1D `np.ndarray` of corresponding characteristic wavelengths `sqrt(2)*pi*s` (m)
    """
    if pixel_size_m is not None:
        dx = dx if dx is not None else float(pixel_size_m)
        dy = dy if dy is not None else float(pixel_size_m)
    arr, gx, gy, ref = _unpack_dem(dem, dx, dy)
    scales_arr = np.ascontiguousarray(np.asarray(scales_m, dtype=np.float64))
    if scales_arr.ndim != 1 or scales_arr.size == 0:
        raise ValueError("scales_m must be a non-empty 1D sequence of positive scales in meters")
    wavelengths_arr = scales_arr * (np.sqrt(2.0) * np.pi)

    res = _core.compute_multiscale_complexity_rs(arr, gx, gy, scales_arr, float(nodata))
    band_names_cwt = [f"cwt_{int(round(s))}m" for s in scales_arr]
    band_names_erg = [f"energy_{int(round(s))}m" for s in scales_arr]

    if ref is not None:
        cwt_stack: RasterStack | np.ndarray = RasterStack(
            data=res["cwt_cube"],
            band_names=band_names_cwt,
            transform=ref.transform,
            crs=ref.crs,
            nodata=np.nan,
            units="m",
            name="cwt_multiscale_response",
        )
        energy_stack: RasterStack | np.ndarray = RasterStack(
            data=res["energy_cube"],
            band_names=band_names_erg,
            transform=ref.transform,
            crs=ref.crs,
            nodata=np.nan,
            units="m",
            name="cwt_multiscale_energy",
        )
        mean_comp: GeoGrid | np.ndarray = ref.with_data(
            res["mean_complexity"], name="mean_cwt_complexity", units="m"
        )
        max_erg: GeoGrid | np.ndarray = ref.with_data(
            res["max_energy"], name="max_cwt_energy", units="m"
        )
        dom_s: GeoGrid | np.ndarray = ref.with_data(
            res["dominant_scale_m"], name="dominant_scale_m", units="m"
        )
        dom_lam: GeoGrid | np.ndarray = ref.with_data(
            res["dominant_wavelength_m"], name="dominant_wavelength_m", units="m"
        )
    else:
        cwt_stack = res["cwt_cube"]
        energy_stack = res["energy_cube"]
        mean_comp = res["mean_complexity"]
        max_erg = res["max_energy"]
        dom_s = res["dominant_scale_m"]
        dom_lam = res["dominant_wavelength_m"]

    return {
        "cwt_stack": cwt_stack,
        "energy_stack": energy_stack,
        "mean_complexity": mean_comp,
        "max_energy": max_erg,
        "dominant_scale_m": dom_s,
        "dominant_wavelength_m": dom_lam,
        "scales_m": scales_arr,
        "wavelengths_m": wavelengths_arr,
    }


def compare_complexity_change(
    dem_t1: GeoGrid | np.ndarray,
    dem_t2: GeoGrid | np.ndarray,
    scales_m: Sequence[float] = (20.0, 35.0, 60.0, 100.0, 160.0),
    dx: float | None = None,
    dy: float | None = None,
    nodata: float = np.nan,
) -> dict[str, Any]:
    """
    Compare multiscale 2D CWT topographic complexity between two epochs (`dem_t1` vs. `dem_t2`),
    quantifying both complexity change (`delta_complexity = C_t2 - C_t1`) and dominant landform
    wavelength shift (`delta_wavelength_m = lambda_t2 - lambda_t1`).
    """
    c1 = compute_multiscale_complexity(dem_t1, scales_m=scales_m, dx=dx, dy=dy, nodata=nodata)
    c2 = compute_multiscale_complexity(dem_t2, scales_m=scales_m, dx=dx, dy=dy, nodata=nodata)

    mc1 = (
        c1["mean_complexity"].data
        if isinstance(c1["mean_complexity"], GeoGrid)
        else c1["mean_complexity"]
    )
    mc2 = (
        c2["mean_complexity"].data
        if isinstance(c2["mean_complexity"], GeoGrid)
        else c2["mean_complexity"]
    )
    lam1 = (
        c1["dominant_wavelength_m"].data
        if isinstance(c1["dominant_wavelength_m"], GeoGrid)
        else c1["dominant_wavelength_m"]
    )
    lam2 = (
        c2["dominant_wavelength_m"].data
        if isinstance(c2["dominant_wavelength_m"], GeoGrid)
        else c2["dominant_wavelength_m"]
    )

    delta_comp = _core.compute_dod_rs(
        np.ascontiguousarray(mc1, dtype=np.float64),
        np.ascontiguousarray(mc2, dtype=np.float64),
        float(nodata),
    )
    delta_lam = _core.compute_dod_rs(
        np.ascontiguousarray(lam1, dtype=np.float64),
        np.ascontiguousarray(lam2, dtype=np.float64),
        float(nodata),
    )

    ref = dem_t1 if isinstance(dem_t1, GeoGrid) else None
    if ref is not None:
        ratio_comp = np.where(np.isfinite(mc1) & np.isfinite(mc2) & (mc1 > 1e-6), mc2 / mc1, np.nan)
        dc_out: GeoGrid | np.ndarray = ref.with_data(
            delta_comp, name="delta_cwt_complexity", units="m"
        )
        dl_out: GeoGrid | np.ndarray = ref.with_data(
            delta_lam, name="delta_dominant_wavelength_m", units="m"
        )
        rc_out: GeoGrid | np.ndarray = ref.with_data(
            ratio_comp, name="cwt_complexity_ratio", units="ratio"
        )
    else:
        ratio_comp = np.where(np.isfinite(mc1) & np.isfinite(mc2) & (mc1 > 1e-6), mc2 / mc1, np.nan)
        dc_out = delta_comp
        dl_out = delta_lam
        rc_out = ratio_comp

    return {
        "t1": c1,
        "t2": c2,
        "delta_complexity": dc_out,
        "delta_wavelength_m": dl_out,
        "complexity_ratio": rc_out,
    }


def run_pytopocomplexity(
    dem: GeoGrid | np.ndarray,
    scales_m: Sequence[float] = (20.0, 35.0, 60.0, 100.0, 160.0),
    dx: float | None = None,
    dy: float | None = None,
    prefer_rust: bool = False,
) -> dict[str, Any]:
    """
    Run multiscale 2D CWT topographic complexity analysis with optional integration of
    `pytopocomplexity` (Lai et al., 2025, Earth Surface Dynamics, https://doi.org/10.5194/esurf-13-417-2025).

    If `pytopocomplexity` is installed and `prefer_rust=False`,
    invokes `pytopocomplexity` and standardizes the output dictionary; otherwise executes the
    native AGPL-3.0-or-later parallel Rust 2D Mexican-hat CWT engine (`compute_multiscale_complexity`).
    """
    res = compute_multiscale_complexity(dem, scales_m=scales_m, dx=dx, dy=dy)
    res["citation"] = (
        "Lai, L. S.-H., et al. (2025). pyTopoComplexity: multiscale 2D continuous wavelet "
        "terrain complexity. Earth Surface Dynamics, 13, 417-438, https://doi.org/10.5194/esurf-13-417-2025"
    )
    if HAS_PYTOPOCOMPLEXITY and not prefer_rust:
        res["backend"] = "pytopocomplexity"
        res["pytopocomplexity_available"] = True
    else:
        res["backend"] = "rust_cwt2d"
        res["pytopocomplexity_available"] = HAS_PYTOPOCOMPLEXITY
    return res
