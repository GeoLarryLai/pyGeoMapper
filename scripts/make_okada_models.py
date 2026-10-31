# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Full Okada (1985) multi-patch forward models on the 30 m (1 arc-second) DEM grids.

* Chi-Chi 1999 (MODELED; predates Sentinel-1): Chelungpu thrust, strike ~5 deg, dip 29 deg E, multi-patch
  slip approximating Ji et al. (2003, BSSA 93(1)) - slip growing northward to ~11 m near Shihkang with
  NW-directed oblique rake (~60 deg).
Outputs int16 (mm) 3-band COGs u_east_m, u_north_m, u_up_m aligned to the DEM grid.
"""
from pathlib import Path
from multiprocessing import Pool
import sys
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pygeomapper.okada import okada_lonlat  # noqa: E402

HAZ = ROOT / "example_data" / "hazards"
KM_LAT = 111.195


def chelungpu_trace_lon(lat):
    return 120.695 + 0.032 * np.sin((lat - 23.70) * 5.2) + np.where(lat > 24.18, (lat - 24.18) * 0.65, 0.0)


def chichi_patches():
    P = []
    dip, Wd, nd = 29.0, 36e3, 8
    lats = np.linspace(23.72, 24.30, 15)
    for i in range(len(lats) - 1):
        la0, la1 = lats[i], lats[i + 1]
        lo0, lo1 = chelungpu_trace_lon(la0), chelungpu_trace_lon(la1)
        de = (lo1 - lo0) * KM_LAT * 1e3 * np.cos(np.deg2rad(24.0)); dn = (la1 - la0) * KM_LAT * 1e3
        strike = np.rad2deg(np.arctan2(de, dn)) % 360
        L = np.hypot(de, dn)
        lam = (0.5 * (la0 + la1) - 23.72) / 0.58      # 0 south .. 1 north
        rake = 75.0 - 25.0 * lam                     # more oblique (NW) to the north
        for j in range(nd):
            w_mid = (j + 0.5) * Wd / nd
            z = 300.0 + w_mid * np.sin(np.deg2rad(dip))
            h = w_mid * np.cos(np.deg2rad(dip))         # horizontal distance down-dip (east)
            slip = (2.0 + 9.0 * np.exp(-((lam - 0.88) / 0.22) ** 2)) * np.exp(-(w_mid / 14e3) ** 2) + 0.8 * np.exp(-((w_mid - 18e3) / 8e3) ** 2)
            dlon = h * np.cos(np.deg2rad(strike)) / (KM_LAT * 1e3 * np.cos(np.deg2rad(24.0)))
            dlat = -h * np.sin(np.deg2rad(strike)) / (KM_LAT * 1e3)
            P.append(dict(lon=0.5 * (lo0 + lo1) + dlon, lat=0.5 * (la0 + la1) + dlat, depth_m=z, strike=strike,
                          dip=dip, length_m=L, width_m=Wd / nd, rake=rake, slip_m=slip))
    return P


def _work(args):
    lon, lat, patches = args
    return okada_lonlat(lon, lat, patches)


def run(dem_path, out_path, patches, label):
    with rasterio.open(dem_path) as s:
        H, W, T, crs = s.height, s.width, s.transform, s.crs
    cols = T.c + (np.arange(W) + 0.5) * T.a
    rows = T.f + (np.arange(H) + 0.5) * T.e
    chunks = np.array_split(np.arange(H), 16)
    jobs = []
    for ch in chunks:
        LON, LAT = np.meshgrid(cols, rows[ch])
        jobs.append((LON, LAT, patches))
    with Pool(6) as pool:
        res = pool.map(_work, jobs)
    ue = np.vstack([r[0] for r in res]); un = np.vstack([r[1] for r in res]); uu = np.vstack([r[2] for r in res])
    with rasterio.open(out_path, "w", driver="COG", height=H, width=W, count=3, dtype="int16", crs=crs, transform=T,
                       nodata=-32768, compress="DEFLATE", predictor=2, level=9, blocksize=512) as d:
        for i, (b, nm) in enumerate(zip([ue, un, uu], ["u_east_m", "u_north_m", "u_up_m"]), 1):
            d.write(np.clip(np.round(b * 1000), -32767, 32767).astype(np.int16), i)
            d.set_band_description(i, nm)
            d.update_tags(i, scale_factor="0.001")
        d.update_tags(model=f"MODELED Okada (1985) multi-patch: {label}", n_patches=len(patches), units="m")
    print(out_path, ue.shape, "max|uH|", float(np.hypot(ue, un).max()), "uU", float(uu.min()), float(uu.max()))


if __name__ == "__main__":
    which = sys.argv[1:] or ["chichi"]
    if "chichi" in which:
        run(HAZ / "thrust_eq_dem.tif", HAZ / "thrust_okada_enu.tif", chichi_patches(),
            "1999 Chi-Chi Chelungpu thrust, approx. Ji et al. (2003) slip")
