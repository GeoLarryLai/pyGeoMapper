# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Shrink the full-resolution 2025 Myanmar example rasters into the lean copies shipped in example_data/hazards.

Inputs (full resolution, rebuilt by scripts/fetch_myanmar_licsar.py and raster_io.fetch_copernicus_dem):
  myanmar_licsar_106D_ifg.tif, myanmar_licsar_offsets.tif : 0.001 deg grid (1000 x 5300 px)
  myanmar_eq_dem_north.tif, myanmar_eq_dem_south.tif    : Copernicus GLO-30, 1 arc-sec (3600 x 9540 px each)
Outputs (same names, same bands / scale_factor tags, overwritten in place):
  LiCSAR products -> 0.002 deg (2 x 2 blocks): wrapped phase = circular mean angle(sum exp(i*phi));
                     every other band = NaN-aware block mean.
  DEM halves      -> 3 arc-sec (~90 m, 3 x 3 block mean).
Files already at the lean resolution are left untouched, so the script is safe to re-run.
Notebook 5 (Case 1A) and the tests read the pixel size from each file, so either resolution works.

Run: python scripts/fetch_myanmar_licsar.py && python scripts/make_myanmar_demo.py
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine

ROOT = Path(__file__).resolve().parents[1]
HAZ = ROOT / "example_data" / "hazards"
PRODUCT_RES, PRODUCT_FACTOR = 0.002, 2
DEM_RES, DEM_FACTOR = 3.0 / 3600.0, 3
NODATA = -32768


def block_mean(a, k):
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:h, :w].reshape(h // k, k, w // k, k)
    with np.errstate(invalid="ignore"):
        return np.nanmean(b, axis=(1, 3))


def block_circular_mean(phase, k):
    z = np.exp(1j * phase)
    z[~np.isfinite(phase)] = np.nan
    h, w = (z.shape[0] // k) * k, (z.shape[1] // k) * k
    b = z[:h, :w].reshape(h // k, k, w // k, k)
    with np.errstate(invalid="ignore"):
        return np.angle(np.nanmean(b, axis=(1, 3)))


def shrink(path, target_res, k, circular_bands=()):
    with rasterio.open(path) as src:
        if abs(src.res[0] - target_res) < 1e-9:
            print(path.name, "already lean:", src.width, "x", src.height)
            return
        profile = src.profile
        tags = src.tags()
        band_tags = [src.tags(b) for b in range(1, src.count + 1)]
        names = src.descriptions
        bands = []
        for b in range(1, src.count + 1):
            raw = src.read(b).astype(np.float64)
            raw[raw == NODATA] = np.nan
            sc = float(band_tags[b - 1].get("scale_factor", 1.0))
            reducer = block_circular_mean if names[b - 1] in circular_bands else block_mean
            bands.append(reducer(raw * sc, k) / sc)
        transform = src.transform * Affine.scale(k)
    h, w = bands[0].shape
    tags["demo_downsample"] = f"{k}x{k} block mean of the full-resolution file (scripts/make_myanmar_demo.py)"
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", driver="GTiff", height=h, width=w, count=len(bands), dtype="int16",
                       crs=profile["crs"], transform=transform, nodata=NODATA, compress="DEFLATE", predictor=2,
                       zlevel=9, tiled=True, blockxsize=512, blockysize=512) as dst:
        for i, (a, bt, name) in enumerate(zip(bands, band_tags, names), 1):
            q = np.where(np.isfinite(a), np.clip(np.round(a), -32767, 32767), NODATA).astype(np.int16)
            dst.write(q, i)
            if name:
                dst.set_band_description(i, name)
            if bt:
                dst.update_tags(i, **bt)
        dst.update_tags(**tags)
    tmp.replace(path)
    print(path.name, "->", w, "x", h, f"{path.stat().st_size / 1e6:.1f} MB")


def main():
    shrink(HAZ / "myanmar_licsar_106D_ifg.tif", PRODUCT_RES, PRODUCT_FACTOR, circular_bands=("wrapped_phase_rad",))
    shrink(HAZ / "myanmar_licsar_offsets.tif", PRODUCT_RES, PRODUCT_FACTOR)
    for half in ("north", "south"):
        shrink(HAZ / f"myanmar_eq_dem_{half}.tif", DEM_RES, DEM_FACTOR)


if __name__ == "__main__":
    main()
