# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Slumgullion Earthflow (Hinsdale County, CO): REAL surface velocity from optical feature tracking
(COSI-Corr-style normalized cross-correlation) of public USDA NAIP orthoimagery (2017, 2019, 2021, 2023),
plus a USGS 3DEP 1/3 arc-second (~10 m) DEM. No login is needed:
  * NAIP COGs come from the Microsoft Planetary Computer (anonymous SAS token from
    https://planetarycomputer.microsoft.com/api/sas/v1/token/naipeuwest/naip).
  * 3DEP comes from https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage

Method:
  1. Mosaic each year's RGB onto a common 1 m EPSG:26913 grid as a gray (mean RGB) image, then high-pass it
     (minus a 10 m Gaussian blur) to suppress illumination differences.
  2. For each 48 m template on a 16 m step, compute the normalized cross-correlation (TM_CCOEFF_NORMED)
     against a +/-24 m search window in the later image, with parabolic sub-pixel peak refinement.
  3. Keep matches with peak NCC > 0.5. Remove the residual orthorectification misregistration as a robust
     affine (bias + tilt) fit over the whole scene (the landslide is <2% of the area).
  4. velocity = displacement / dt for each consecutive pair (17-19, 19-21, 21-23). The final value is the
     per-pixel median of the pairs where at least 2 pairs are valid and their spread is < 1.5 m/yr.
Output: example_data/hazards/slumgullion_naip_velocity.tif (16 m) and slumgullion_dem_3dep10m.tif.
"""
import json, sys, tempfile, urllib.request
from datetime import date
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "example_data" / "hazards"
WORK = Path(tempfile.gettempdir()) / "slum"
BOUNDS = (298000, 4204000, 306000, 4210500)  # EPSG:26913 (NAD83 / UTM 13N)
DATES = {"2017": date(2017, 10, 15), "2019": date(2019, 9, 12), "2021": date(2021, 8, 22), "2023": date(2023, 9, 8)}
PAIRS = [("2017", "2019"), ("2019", "2021"), ("2021", "2023")]
T, R, S = 48, 24, 16


def fetch_naip(yr):
    tok = json.load(urllib.request.urlopen("https://planetarycomputer.microsoft.com/api/sas/v1/token/naipeuwest/naip"))["token"]
    q = {"collections": ["naip"], "bbox": [-107.30, 37.96, -107.20, 38.025], "datetime": f"{yr}-01-01/{yr}-12-31", "limit": 20}
    req = urllib.request.Request("https://planetarycomputer.microsoft.com/api/stac/v1/search", data=json.dumps(q).encode(),
                                 headers={"Content-Type": "application/json"})
    feats = json.load(urllib.request.urlopen(req))["features"]
    srcs = [rasterio.open(f["assets"]["image"]["href"] + "?" + tok) for f in feats]
    arr, tr = merge(srcs, bounds=BOUNDS, res=1.0, indexes=[1, 2, 3], resampling=Resampling.average)
    g = np.clip(arr.astype(np.float32).mean(0), 0, 255).astype("uint8")
    prof = dict(driver="GTiff", height=g.shape[0], width=g.shape[1], count=1, dtype="uint8", crs=srcs[0].crs, transform=tr, compress="deflate")
    with rasterio.open(WORK / f"naip_{yr}.tif", "w", **prof) as o:
        o.write(g, 1)


def _load(y):
    import cv2
    a = rasterio.open(WORK / f"naip_{y}.tif").read(1).astype(np.float32)
    return a - cv2.GaussianBlur(a, (0, 0), 10)


def ncc_pair(pair):
    import cv2
    A, B = _load(pair[0]), _load(pair[1])
    H, W = A.shape
    h = T // 2
    rows = np.arange(R + h, H - R - h, S)
    cols = np.arange(R + h, W - R - h, S)
    out = np.full((len(rows), len(cols), 3), np.nan, np.float32)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            t = A[r - h:r + h, c - h:c + h]
            if t.std() < 3:
                continue
            m = cv2.matchTemplate(B[r - h - R:r + h + R, c - h - R:c + h + R], t, cv2.TM_CCOEFF_NORMED)
            _, pk, _, (x, y) = cv2.minMaxLoc(m)
            if 0 < x < m.shape[1] - 1 and 0 < y < m.shape[0] - 1:
                dx = (m[y, x - 1] - m[y, x + 1]) / (2 * (m[y, x - 1] - 2 * m[y, x] + m[y, x + 1]))
                dy = (m[y - 1, x] - m[y + 1, x]) / (2 * (m[y - 1, x] - 2 * m[y, x] + m[y + 1, x]))
                out[i, j] = (x + dx - R, -(y + dy - R), pk)  # east, north displacement (m), peak NCC
    return out


def deramp(comp, good):
    rr, cc = np.mgrid[:comp.shape[0], :comp.shape[1]]
    msk = good & np.isfinite(comp)
    res = comp
    for _ in range(3):
        G = np.c_[np.ones(msk.sum()), rr[msk], cc[msk]]
        c, *_ = np.linalg.lstsq(G, comp[msk], rcond=None)
        res = comp - (c[0] + c[1] * rr + c[2] * cc)
        msk = good & np.isfinite(res) & (np.abs(res) < 3 * np.nanstd(res[msk]))
    return res


def main():
    WORK.mkdir(exist_ok=True)
    for y in DATES:
        if not (WORK / f"naip_{y}.tif").exists():
            fetch_naip(y)
    with Pool(3) as P:
        offs = P.map(ncc_pair, PAIRS)
    V, bands, names = [], [], []
    for (y1, y2), o in zip(PAIRS, offs):
        dt = (DATES[y2] - DATES[y1]).days / 365.25
        good = o[..., 2] > 0.5
        e, n = deramp(o[..., 0], good), deramp(o[..., 1], good)
        e[~good] = np.nan
        n[~good] = np.nan
        V.append(np.stack([e / dt, n / dt]))
        bands += [e / dt, n / dt]
        names += [f"ve_{y1}_{y2}_m_per_yr", f"vn_{y1}_{y2}_m_per_yr"]
    V = np.stack(V)
    with np.errstate(all="ignore"):
        ve, vn = np.nanmedian(V[:, 0], 0), np.nanmedian(V[:, 1], 0)
        spread = np.nanmax(np.hypot(V[:, 0] - ve, V[:, 1] - vn), 0)
    cnt = np.isfinite(V[:, 0]).sum(0)
    ok = (cnt >= 2) & (spread < 1.5)
    ve[~ok] = np.nan
    vn[~ok] = np.nan
    bands = [ve, vn, spread] + bands
    names = ["ve_m_per_yr", "vn_m_per_yr", "pair_spread_m_per_yr"] + names
    with rasterio.open(WORK / "naip_2019.tif") as s:
        crs = s.crs
    tr = from_origin(BOUNDS[0] + R + T // 2 - S / 2, BOUNDS[3] - (R + T // 2) + S / 2, S, S)
    prof = dict(driver="GTiff", height=ve.shape[0], width=ve.shape[1], count=len(bands), dtype="int16", nodata=-32768,
                crs=crs, transform=tr, compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(OUT / "slumgullion_naip_velocity.tif", "w", **prof) as d:
        for i, (b, nm) in enumerate(zip(bands, names), 1):
            d.write(np.where(np.isfinite(b), np.round(b / 0.001), -32768).astype("int16"), i)
            d.set_band_description(i, nm)
            d.update_tags(i, scale_factor="0.001", name=nm)
        d.update_tags(source="USDA NAIP via Microsoft Planetary Computer; NCC feature tracking (pyGeoMapper)",
                      pairs="2017-10-15/2019-09-12, 2019-09-12/2021-08-22, 2021-08-22/2023-09-08", units="m/yr")
    # 3DEP 10 m DEM
    w, h = (BOUNDS[2] - BOUNDS[0]) // 10, (BOUNDS[3] - BOUNDS[1]) // 10
    url = ("https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage?"
           f"bbox={BOUNDS[0]},{BOUNDS[1]},{BOUNDS[2]},{BOUNDS[3]}&bboxSR=26913&imageSR=26913&size={w},{h}"
           "&format=tiff&pixelType=F32&interpolation=RSP_BilinearInterpolation&f=image")
    raw = WORK / "dem_raw.tif"
    raw.write_bytes(urllib.request.urlopen(url, timeout=300).read())
    with rasterio.open(raw) as s:
        z = s.read(1).astype(np.float32)
    prof = dict(driver="GTiff", height=h, width=w, count=1, dtype="float32", crs=crs,
                transform=from_origin(BOUNDS[0], BOUNDS[3], 10, 10), compress="deflate", predictor=3, nodata=-9999)
    with rasterio.open(OUT / "slumgullion_dem_3dep10m.tif", "w", **prof) as d:
        d.write(z, 1)
    print("done", ve.shape, z.shape, np.nanmin(z), np.nanmax(z))


if __name__ == "__main__":
    sys.exit(main())
