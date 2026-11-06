# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Fetch a coarse pre-event Sentinel-2 true-color mosaic under the 2025-03-28 Mw 7.7 Myanmar rupture swath
(95.6-96.6 E, 17.8-23.1 N) on the same 0.002 deg grid as the lean LiCSAR products, as the imagery backdrop for
the fringes-on-imagery figure in notebook 5 (`image/myanmar_insar_3d.png`).

Source (public, no login): ESA Copernicus Sentinel-2 L2A via AWS Element84 Earth Search STAC
  https://earth-search.aws.element84.com/v1 (collection `sentinel-2-l2a`, asset `visual` = 10 m TCI COG).
  For each MGRS tile the 4 best scenes (< 40% cloud, ranked by cloud % + empty-pixel %) of 2025-01-01..2025-03-27
  are used; each COG is read at its 1/16 overview (~160 m) and averaged onto the 0.002 deg grid; better scenes
  are placed first and later ones only fill empty pixels.
Output (example_data/hazards):
  myanmar_s2_rgb_pre.tif : uint8 RGB (0 = no data), EPSG:4326, 0.002 deg, 500 x 2650 px.
"Contains modified Copernicus Sentinel data (2025)".
"""
import json
from pathlib import Path
import urllib.request

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "example_data" / "hazards" / "myanmar_s2_rgb_pre.tif"
STAC = "https://earth-search.aws.element84.com/v1/search"
LON0, LON1, LAT0, LAT1, RES = 95.6, 96.6, 17.8, 23.1, 0.002
W, H = int(round((LON1 - LON0) / RES)), int(round((LAT1 - LAT0) / RES))
DST_T = from_origin(LON0, LAT1, RES, RES)
DATES = "2025-01-01T00:00:00Z/2025-03-27T23:59:59Z"
OVERVIEW_LEVEL = 3  # 0 -> 1/2, 1 -> 1/4, 2 -> 1/8, 3 -> 1/16 of the 10 m grid


def search_items():
    body = {"collections": ["sentinel-2-l2a"], "bbox": [LON0, LAT0, LON1, LAT1], "datetime": DATES,
            "query": {"eo:cloud_cover": {"lt": 40}}, "limit": 100}
    items, url, payload = [], STAC, json.dumps(body).encode()
    while url:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.load(r)
        items += page["features"]
        nxt = [lk for lk in page.get("links", []) if lk.get("rel") == "next"]
        url, payload = (nxt[0]["href"], json.dumps(nxt[0].get("body", body)).encode()) if nxt else (None, None)
    return items


def score(it):
    """Lower is better: cloudy % plus empty (outside-swath) % of the tile."""
    p = it["properties"]
    return p["eo:cloud_cover"] + p.get("s2:nodata_pixel_percentage", 0.0)


def ranked_per_tile(items, n=4):
    """Up to `n` best scenes per MGRS tile (edge-of-swath tiles need a second pass to fill)."""
    by_tile = {}
    for it in items:
        p = it["properties"]
        tile = p.get("grid:code") or p.get("s2:mgrs_tile") or it["id"].split("_")[1]
        by_tile.setdefault(tile, []).append(it)
    picks = [it for its in by_tile.values() for it in sorted(its, key=score)[:n]]
    return sorted(picks, key=score), len(by_tile)


def main():
    chosen, n_tiles = ranked_per_tile(search_items())
    print(f"{n_tiles} MGRS tiles, {len(chosen)} scenes")
    mosaic = np.zeros((3, H, W), np.uint8)
    used = []
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    with rasterio.Env(**env):
        for it in chosen:
            href = it["assets"]["visual"]["href"]
            with rasterio.open(href, OVERVIEW_LEVEL=OVERVIEW_LEVEL) as src:
                tile = np.zeros((3, H, W), np.float32)
                reproject(src.read().astype(np.float32), tile, src_transform=src.transform, src_crs=src.crs,
                          dst_transform=DST_T, dst_crs="EPSG:4326", src_nodata=0, dst_nodata=0,
                          resampling=Resampling.average)
            fill = (mosaic.max(0) == 0) & (tile.max(0) > 0)
            if fill.any():
                mosaic[:, fill] = np.clip(np.round(tile[:, fill]), 1, 255).astype(np.uint8)
                used.append(f"{it['id']} ({it['properties']['eo:cloud_cover']:.1f}% cloud)")
                print("  +", used[-1], f"{fill.mean():.1%} of grid")
    with rasterio.open(OUT, "w", driver="GTiff", height=H, width=W, count=3, dtype="uint8", crs="EPSG:4326",
                       transform=DST_T, nodata=0, compress="DEFLATE", predictor=2, zlevel=9, tiled=True,
                       blockxsize=512, blockysize=512, photometric="RGB") as d:
        d.write(mosaic)
        for i, n in enumerate(("red", "green", "blue"), 1):
            d.set_band_description(i, n)
        d.update_tags(lon_min=LON0, lon_max=LON1, lat_min=LAT0, lat_max=LAT1, dates=DATES,
                      source="ESA Copernicus Sentinel-2 L2A TCI via AWS Earth Search (pre-event, REAL DATA)",
                      scenes="; ".join(used))
    print(OUT, f"{OUT.stat().st_size / 1e6:.1f} MB", f"coverage {(mosaic.max(0) > 0).mean():.1%}")


if __name__ == "__main__":
    main()
