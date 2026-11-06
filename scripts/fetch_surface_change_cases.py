# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Fetch the REAL Sentinel-2 + DEM inputs for notebook 5 Cases 3-5 (flood, wildfire, agri-urban).

For each case: two Sentinel-2 L2A scenes (pre / post) from AWS Element84 Earth Search (public, no login),
cropped to a 5 km x 5 km UTM window at 20 m, six bands in the order blue, green, red, nir, swir1, swir2
(Earth Search assets blue/green/red/nir = 10 m averaged to 20 m, swir16/swir22 = native 20 m), written as
float32 surface reflectance (DN * 1e-4, minus the 0.1 BOA offset only when Earth Search has not already
applied it). The DEM is Copernicus GLO-30 (one vintage), bilinear-resampled to the same 20 m grid and written
twice (`*_dem_pre.tif`, `*_dem_post.tif`), so the DEM of Difference is zero by construction: these cases
show spectral change (dNDWI, dNBR, dNDVI/dNDBI), not measured elevation change.

Output (example_data/hazards): {flood,wildfire,agri_urban}_{s2_pre,s2_post,dem_pre,dem_post}.tif
"Contains modified Copernicus Sentinel data (2017-2024)". Copernicus DEM GLO-30: (c) DLR e.V. 2010-2014 and
(c) Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the EU and ESA.
"""
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject

from pygeomapper import raster_io

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "example_data" / "hazards"
STAC = "https://earth-search.aws.element84.com/v1"
RES = 20.0
SIZE = 5000.0
BANDS = [("blue", "blue"), ("green", "green"), ("red", "red"), ("nir", "nir"), ("swir1", "swir16"), ("swir2", "swir22")]

# name: (EPSG, window centre easting/northing, pre item, post item)
CASES = {
    # February 2019 Yolo Bypass flood (Fremont Weir spill after the mid-February atmospheric river),
    # beside West Sacramento, California. The 2019 Missouri River flood water (Percival, IA) was too
    # sediment-laden for the green-vs-NIR NDWI, so this clearer-water flood is used instead.
    "flood": (32610, (622012.0, 4266661.0), "S2B_10SFH_20190125_0_L2A", "S2A_10SFH_20190219_0_L2A"),
    # 2020 Holiday Farm Fire, McKenzie River canyon, Lane County, Oregon (ignited 2020-09-07)
    "wildfire": (32610, (547120.0, 4889120.0), "S2A_10TEP_20200726_1_L2A", "S2B_10TEP_20200929_1_L2A"),
    # Center-pivot fields and suburban growth south of Kuna / Nampa, Boise Valley, Idaho
    "agri_urban": (32611, (543120.0, 4819120.0), "S2B_11TNJ_20170731_0_L2A", "S2A_11TNJ_20240709_0_L2A"),
}


def window(cx, cy):
    h = SIZE / 2.0
    return cx - h, cy - h, cx + h, cy + h


def read_s2(client, item_id, epsg, bounds):
    item = next(client.search(ids=[item_id], collections=["sentinel-2-l2a"]).items())
    baseline = float(item.properties.get("s2:processing_baseline", "0") or 0)
    offset = -0.1 if baseline >= 4.0 and not item.properties.get("earthsearch:boa_offset_applied", False) else 0.0
    w = int(round((bounds[2] - bounds[0]) / RES))
    h = int(round((bounds[3] - bounds[1]) / RES))
    out = []
    for _, key in BANDS:
        with rasterio.open(item.assets[key].href) as src, WarpedVRT(
            src, crs=f"EPSG:{epsg}", transform=from_origin(bounds[0], bounds[3], RES, RES),
            width=w, height=h, resampling=Resampling.average,
        ) as vrt:
            dn = vrt.read(1).astype(np.float32)
        refl = np.where(dn > 0, dn * 1e-4 + offset, np.nan)
        out.append(np.round(refl, 4).astype(np.float32))
    return np.stack(out), item


def read_dem(epsg, bounds):
    t = Transformer.from_crs(epsg, 4326, always_xy=True)
    xs, ys = zip(*[t.transform(x, y) for x in bounds[0::2] for y in bounds[1::2]])
    pad = 0.01
    g = raster_io.fetch_copernicus_dem((min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad), resolution_m=30)
    w = int(round((bounds[2] - bounds[0]) / RES))
    h = int(round((bounds[3] - bounds[1]) / RES))
    dst = np.full((h, w), np.nan, dtype=np.float32)
    x0, dx, rx, y0, ry, dy = g.transform
    reproject(g.data.astype(np.float32), dst, src_transform=rasterio.Affine(dx, rx, x0, ry, dy, y0), src_crs=g.crs,
              dst_transform=from_origin(bounds[0], bounds[3], RES, RES), dst_crs=f"EPSG:{epsg}",
              resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    return np.round(dst, 2)


def write(path, arr, epsg, bounds, names, tags):
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0], dtype="float32",
        crs=f"EPSG:{epsg}", transform=from_origin(bounds[0], bounds[3], RES, RES),
        compress="DEFLATE", predictor=3, tiled=True, blockxsize=256, blockysize=256,
    ) as ds:
        ds.write(arr)
        ds.descriptions = tuple(names)
        ds.update_tags(**tags)


def main(names=None):
    client = Client.open(STAC)
    for name, (epsg, (cx, cy), pre_id, post_id) in CASES.items():
        if names and name not in names:
            continue
        b = window(cx, cy)
        for tag, iid in (("pre", pre_id), ("post", post_id)):
            arr, item = read_s2(client, iid, epsg, b)
            write(OUT / f"{name}_s2_{tag}.tif", arr, epsg, b, [n for n, _ in BANDS],
                  {"source": f"Sentinel-2 L2A {iid} (Earth Search)", "datetime": str(item.datetime),
                   "units": "surface reflectance"})
            print(name, tag, iid, np.nanmean(arr, axis=(1, 2)).round(3))
        dem = read_dem(epsg, b)
        for tag in ("pre", "post"):
            write(OUT / f"{name}_dem_{tag}.tif", dem, epsg, b, [f"elevation_{tag}_m"],
                  {"source": "Copernicus DEM GLO-30 (same DEM for pre and post)", "units": "m"})
        print(name, "dem", float(np.nanmin(dem)), float(np.nanmax(dem)), "bounds", b)


if __name__ == "__main__":
    import sys

    main(sys.argv[1:] or None)  # e.g. `python scripts/fetch_surface_change_cases.py flood`
