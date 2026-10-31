# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Generic optical, multispectral, and mineralogical spectral indices backed by parallel Rust kernels.

Supports `GeoGrid`, `RasterStack`, and raw 2D `np.ndarray` inputs while preserving CRS and transform.
Follows Awesome Spectral Indices (`spyndex`; Montero et al., 2023) band naming conventions.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import _core
from .raster_io import GeoGrid, RasterStack

# Standard Sentinel-2 / Landsat / multispectral band aliases
BAND_ALIASES: dict[str, tuple[str, ...]] = {
    "blue": ("blue", "b02", "b2", "coastal"),
    "green": ("green", "b03", "b3"),
    "red": ("red", "b04", "b4"),
    "nir": ("nir", "b08", "b8", "b8a", "nir08"),
    "swir1": ("swir1", "swir16", "b11", "b6"),
    "swir2": ("swir2", "swir22", "b12", "b7"),
}


def _as_2d_f64(arr_or_grid: GeoGrid | np.ndarray) -> tuple[np.ndarray, GeoGrid | None]:
    if isinstance(arr_or_grid, GeoGrid):
        return np.ascontiguousarray(arr_or_grid.data, dtype=np.float64), arr_or_grid
    arr = np.ascontiguousarray(np.asarray(arr_or_grid, dtype=np.float64))
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array or GeoGrid, got shape {arr.shape}")
    return arr, None


def _wrap_output(
    out: np.ndarray,
    ref: GeoGrid | None,
    name: str,
    units: str = "index",
) -> GeoGrid | np.ndarray:
    if ref is not None:
        return ref.with_data(out, name=name, units=units)
    return out


def normalized_difference(
    band_a: GeoGrid | np.ndarray,
    band_b: GeoGrid | np.ndarray,
    name: str = "ND",
    nodata: float = np.nan,
) -> GeoGrid | np.ndarray:
    """Compute generic normalized difference `(A - B) / (A + B)` in `[-1, 1]` via Rust."""
    a, ref_a = _as_2d_f64(band_a)
    b, ref_b = _as_2d_f64(band_b)
    ref = ref_a or ref_b
    out = _core.normalized_difference_rs(a, b, float(nodata))
    return _wrap_output(out, ref, name=name, units="dimensionless")


def band_ratio(
    band_a: GeoGrid | np.ndarray,
    band_b: GeoGrid | np.ndarray,
    name: str = "Ratio",
    nodata: float = np.nan,
) -> GeoGrid | np.ndarray:
    """Compute generic band ratio `A / B` via Rust."""
    a, ref_a = _as_2d_f64(band_a)
    b, ref_b = _as_2d_f64(band_b)
    ref = ref_a or ref_b
    out = _core.band_ratio_rs(a, b, float(nodata))
    return _wrap_output(out, ref, name=name, units="ratio")


def ndvi(
    nir: GeoGrid | RasterStack | np.ndarray,
    red: GeoGrid | np.ndarray | None = None,
) -> GeoGrid | np.ndarray:
    """Normalized Difference Vegetation Index: `(NIR - Red) / (NIR + Red)`."""
    if isinstance(nir, RasterStack):
        return normalized_difference(resolve_band(nir, "nir"), resolve_band(nir, "red"), name="NDVI")
    if red is None:
        raise TypeError("ndvi() requires `red` when `nir` is not a RasterStack")
    return normalized_difference(nir, red, name="NDVI")


def nbr(
    nir: GeoGrid | RasterStack | np.ndarray,
    swir2: GeoGrid | np.ndarray | None = None,
) -> GeoGrid | np.ndarray:
    """Normalized Burn Ratio: `(NIR - SWIR2) / (NIR + SWIR2)`."""
    if isinstance(nir, RasterStack):
        return normalized_difference(resolve_band(nir, "nir"), resolve_band(nir, "swir2"), name="NBR")
    if swir2 is None:
        raise TypeError("nbr() requires `swir2` when `nir` is not a RasterStack")
    return normalized_difference(nir, swir2, name="NBR")


def ndwi(
    green: GeoGrid | RasterStack | np.ndarray,
    nir: GeoGrid | np.ndarray | None = None,
) -> GeoGrid | np.ndarray:
    """Normalized Difference Water Index (McFeeters, 1996): `(Green - NIR) / (Green + NIR)`."""
    if isinstance(green, RasterStack):
        return normalized_difference(resolve_band(green, "green"), resolve_band(green, "nir"), name="NDWI")
    if nir is None:
        raise TypeError("ndwi() requires `nir` when `green` is not a RasterStack")
    return normalized_difference(green, nir, name="NDWI")


def ndmi(
    nir: GeoGrid | RasterStack | np.ndarray,
    swir1: GeoGrid | np.ndarray | None = None,
) -> GeoGrid | np.ndarray:
    """Normalized Difference Moisture Index: `(NIR - SWIR1) / (NIR + SWIR1)`."""
    if isinstance(nir, RasterStack):
        return normalized_difference(resolve_band(nir, "nir"), resolve_band(nir, "swir1"), name="NDMI")
    if swir1 is None:
        raise TypeError("ndmi() requires `swir1` when `nir` is not a RasterStack")
    return normalized_difference(nir, swir1, name="NDMI")


def ndbi(
    swir1: GeoGrid | RasterStack | np.ndarray,
    nir: GeoGrid | np.ndarray | None = None,
) -> GeoGrid | np.ndarray:
    """Normalized Difference Built-up Index: `(SWIR1 - NIR) / (SWIR1 + NIR)`."""
    if isinstance(swir1, RasterStack):
        return normalized_difference(resolve_band(swir1, "swir1"), resolve_band(swir1, "nir"), name="NDBI")
    if nir is None:
        raise TypeError("ndbi() requires `nir` when `swir1` is not a RasterStack")
    return normalized_difference(swir1, nir, name="NDBI")


def bare_soil_index(
    swir1: GeoGrid | RasterStack | np.ndarray,
    red: GeoGrid | np.ndarray | None = None,
    nir: GeoGrid | np.ndarray | None = None,
    blue: GeoGrid | np.ndarray | None = None,
    nodata: float = np.nan,
) -> GeoGrid | np.ndarray:
    """Bare Soil Index (BSI): `((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue))`."""
    if isinstance(swir1, RasterStack):
        red = resolve_band(swir1, "red")
        nir = resolve_band(swir1, "nir")
        blue = resolve_band(swir1, "blue")
        swir1 = resolve_band(swir1, "swir1")
    assert red is not None and nir is not None and blue is not None
    s1, ref = _as_2d_f64(swir1)
    r, _ = _as_2d_f64(red)
    n, _ = _as_2d_f64(nir)
    b, _ = _as_2d_f64(blue)
    out = _core.bare_soil_index_rs(s1, r, n, b, float(nodata))
    return _wrap_output(out, ref, name="BSI", units="dimensionless")


bsi = bare_soil_index


def clay_mineral_ratio(
    swir1: GeoGrid | np.ndarray,
    swir2: GeoGrid | np.ndarray,
) -> GeoGrid | np.ndarray:
    """Hydroxyl / clay mineral absorption ratio: `SWIR1 / SWIR2`."""
    return band_ratio(swir1, swir2, name="ClayRatio")


def ferric_oxide_ratio(
    red: GeoGrid | np.ndarray,
    blue: GeoGrid | np.ndarray,
) -> GeoGrid | np.ndarray:
    """Ferric iron oxide ratio: `Red / Blue`."""
    return band_ratio(red, blue, name="FerricOxideRatio")


def ferrous_iron_index(
    swir1: GeoGrid | np.ndarray,
    nir: GeoGrid | np.ndarray,
    green: GeoGrid | np.ndarray,
    red: GeoGrid | np.ndarray,
    nodata: float = np.nan,
) -> GeoGrid | np.ndarray:
    """Ferrous silicate index: `(SWIR1 / NIR) + (Green / Red)`."""
    s1, ref = _as_2d_f64(swir1)
    n, _ = _as_2d_f64(nir)
    g, _ = _as_2d_f64(green)
    r, _ = _as_2d_f64(red)
    out = _core.ferrous_iron_index_rs(s1, n, g, r, float(nodata))
    return _wrap_output(out, ref, name="FerrousIronIndex", units="ratio")


def carbonate_index(
    red: GeoGrid | np.ndarray,
    green: GeoGrid | np.ndarray,
) -> GeoGrid | np.ndarray:
    """Normalized carbonate / red-green lithologic index: `(Red - Green) / (Red + Green)`."""
    return normalized_difference(red, green, name="CarbonateIndex")


def resolve_band(stack: RasterStack, canonical_name: str) -> GeoGrid:
    """Resolve a canonical band name (`blue`, `green`, `red`, `nir`, `swir1`, `swir2`) from a `RasterStack`."""
    candidates = BAND_ALIASES.get(canonical_name.lower(), (canonical_name.lower(),))
    lower_map = {bn.lower(): bn for bn in stack.band_names}
    for cand in candidates:
        if cand in lower_map:
            return stack.band(lower_map[cand])
    raise KeyError(
        f"Could not resolve band '{canonical_name}' (tried {candidates}) in stack bands {stack.band_names}"
    )


def compute_index_suite(
    stack: RasterStack,
    indices: Sequence[str] = (
        "NDVI",
        "NBR",
        "NDWI",
        "NDMI",
        "NDBI",
        "BSI",
        "ClayRatio",
        "FerricOxideRatio",
        "FerrousIronIndex",
    ),
) -> dict[str, GeoGrid]:
    """
    Compute a suite of vegetation, moisture, burn, soil, and mineral indices from a `RasterStack`.
    """
    results: dict[str, GeoGrid] = {}
    for idx_name in indices:
        key = idx_name.upper()
        if key == "NDVI":
            results["NDVI"] = ndvi(resolve_band(stack, "nir"), resolve_band(stack, "red"))  # type: ignore[assignment]
        elif key == "NBR":
            results["NBR"] = nbr(resolve_band(stack, "nir"), resolve_band(stack, "swir2"))  # type: ignore[assignment]
        elif key == "NDWI":
            results["NDWI"] = ndwi(resolve_band(stack, "green"), resolve_band(stack, "nir"))  # type: ignore[assignment]
        elif key == "NDMI":
            results["NDMI"] = ndmi(resolve_band(stack, "nir"), resolve_band(stack, "swir1"))  # type: ignore[assignment]
        elif key == "NDBI":
            results["NDBI"] = ndbi(resolve_band(stack, "swir1"), resolve_band(stack, "nir"))  # type: ignore[assignment]
        elif key == "BSI":
            results["BSI"] = bare_soil_index(
                resolve_band(stack, "swir1"),
                resolve_band(stack, "red"),
                resolve_band(stack, "nir"),
                resolve_band(stack, "blue"),
            )  # type: ignore[assignment]
        elif key in ("CLAYRATIO", "CLAY"):
            results["ClayRatio"] = clay_mineral_ratio(
                resolve_band(stack, "swir1"), resolve_band(stack, "swir2")
            )  # type: ignore[assignment]
        elif key in ("FERRICOXIDERATIO", "FERRICOXIDE"):
            results["FerricOxideRatio"] = ferric_oxide_ratio(
                resolve_band(stack, "red"), resolve_band(stack, "blue")
            )  # type: ignore[assignment]
        elif key in ("FERROUSIRONINDEX", "FERROUSIRON"):
            results["FerrousIronIndex"] = ferrous_iron_index(
                resolve_band(stack, "swir1"),
                resolve_band(stack, "nir"),
                resolve_band(stack, "green"),
                resolve_band(stack, "red"),
            )  # type: ignore[assignment]
        elif key in ("CARBONATEINDEX", "CARBONATE"):
            results["CarbonateIndex"] = carbonate_index(
                resolve_band(stack, "red"), resolve_band(stack, "green")
            )  # type: ignore[assignment]
        else:
            raise ValueError(f"Unsupported index name '{idx_name}'")
    return results
