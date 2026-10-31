# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Fetch REAL COMET LiCSAR Sentinel-1 products for the 2025-03-28 Mw 7.7 Myanmar earthquake (USGS us7000pn9s)
and clip them to the Sagaing rupture swath 95.6-96.6 E, 17.8-23.1 N on a common 0.001 deg grid.

Sources (public, no login; contains modified Copernicus Sentinel-1 data processed by COMET/LiCSAR, JASMIN):
  * Interferogram, descending track 106D, frames 106D_06445/06644/06844/07044_131313, pair 20250324_20250405:
    https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products.public/106/<frame>/interferograms/20250324_20250405/
    -> geo.diff_pha.tif (wrapped, rad, GACOS-free, Goldstein-filtered), geo.cc.tif (coherence 0-255), geo.unw.tif (SNAPHU, rad)
  * Event offset-tracking mosaics + LOS unit vectors:
    https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/EQ/us7000pn9s/
    -> 106D/070A.geo.azi.masked.tif, .geo.rng.masked.tif (m), .geo.E/N/U.tif (LOS unit vector, ground->satellite)
Outputs (example_data/hazards):
  myanmar_licsar_106D_ifg.tif     : int16 bands wrapped_phase (1e-4 rad), unwrapped_los_m (0.1 mm), coherence (0-1 *1e4)
  myanmar_licsar_offsets.tif      : int16 (cm) azi_106D, rng_106D, azi_070A, rng_070A, u_north_from_azi
"""
from pathlib import Path
import subprocess
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path.home() / ".cache" / "pygeomapper" / "licsar"
OUT = ROOT / "example_data" / "hazards"
LON0, LON1, LAT0, LAT1, RES = 95.6, 96.6, 17.8, 23.1, 0.001
W, H = int(round((LON1 - LON0) / RES)), int(round((LAT1 - LAT0) / RES))
DST_T = from_origin(LON0, LAT1, RES, RES)
PAIR = "20250324_20250405"
FRAMES = ["106D_06445_131313", "106D_06644_131313", "106D_06844_131313", "106D_07044_131313"]
IFG = "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products.public/106/{f}/interferograms/{p}/{p}.{e}"
EQ = "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products/EQ/us7000pn9s/{n}"
WAVELENGTH = 0.0554658


def get(url, name):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists() or p.stat().st_size < 10000:
        subprocess.run(["curl", "-sSL", "-o", str(p), url], check=True)
    return p


def to_grid(path, resampling=Resampling.nearest):
    dst = np.full((H, W), np.nan, np.float32)
    with rasterio.open(path) as s:
        src = s.read(1).astype(np.float32)
        if s.nodata is not None and np.isfinite(s.nodata):
            src[src == s.nodata] = np.nan
        reproject(src, dst, src_transform=s.transform, src_crs=s.crs, dst_transform=DST_T, dst_crs="EPSG:4326",
                  resampling=resampling, src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def mosaic(ext, align=False):
    out = np.full((H, W), np.nan, np.float32)
    for f in FRAMES:
        a = to_grid(get(IFG.format(f=f, p=PAIR, e=ext), f"{f}_{PAIR}.{ext}"))
        a[a == 0] = np.nan  # LiCSAR uses 0 as masked
        both = np.isfinite(out) & np.isfinite(a)
        if align and both.sum() > 1000:
            a = a - np.nanmedian(a[both] - out[both])  # tie each frame's arbitrary unwrapping reference
        fill = ~np.isfinite(out) & np.isfinite(a)
        out[fill] = a[fill]
    return out


def write(path, bands, names, scale, tags):
    with rasterio.open(path, "w", driver="COG", height=H, width=W, count=len(bands), dtype="int16",
                       crs="EPSG:4326", transform=DST_T, nodata=-32768, compress="DEFLATE", predictor=2,
                       level=9, blocksize=512) as d:
        for i, (b, n, sc) in enumerate(zip(bands, names, scale), 1):
            q = np.where(np.isfinite(b), np.clip(np.round(b / sc), -32767, 32767), -32768).astype(np.int16)
            d.write(q, i)
            d.set_band_description(i, n)
            d.update_tags(i, scale_factor=str(sc), name=n)
        d.update_tags(lon_min=LON0, lon_max=LON1, lat_min=LAT0, lat_max=LAT1, **tags)
    print(path, path.stat().st_size / 1e6, "MB")


def main():
    wrapped = mosaic("geo.diff_pha.tif")
    cc = mosaic("geo.cc.tif") / 255.0
    unw = mosaic("geo.unw.tif", align=True)
    unw_los = -unw * WAVELENGTH / (4 * np.pi)  # LiCSAR: positive phase = range increase -> LOS toward-satellite (m)
    write(OUT / "myanmar_licsar_106D_ifg.tif", [wrapped, unw_los, cc],
          ["wrapped_phase_rad", "unwrapped_los_m", "coherence"], [1e-4, 1e-4, 1e-4],
          dict(source="COMET LiCSAR Sentinel-1 106D " + PAIR + " (REAL DATA)", wavelength_m=WAVELENGTH))

    g = {k: to_grid(get(EQ.format(n=k), "EQ_" + k)) for k in
         ["106D.geo.azi.masked.tif", "106D.geo.rng.masked.tif", "070A.geo.azi.masked.tif",
          "070A.geo.rng.masked.tif", "106D.geo.E.tif", "106D.geo.N.tif", "106D.geo.U.tif",
          "070A.geo.E.tif", "070A.geo.N.tif", "070A.geo.U.tif"]}

    def flight(e, n, ascending):
        # azimuth (along-track) unit vector = horizontal LOS rotated 90 deg; desc flies S, asc flies N
        v = np.stack([-n, e]) if not ascending else np.stack([n, -e])
        v = v / np.hypot(*v)
        return v if (v[1] < 0).mean() == (not ascending) or True else -v

    fd = flight(np.nanmedian(g["106D.geo.E.tif"]), np.nanmedian(g["106D.geo.N.tif"]), False)
    fa = flight(np.nanmedian(g["070A.geo.E.tif"]), np.nanmedian(g["070A.geo.N.tif"]), True)
    if fd[1] > 0: fd = -fd
    if fa[1] < 0: fa = -fa
    A = np.array([fd, fa])                               # rows: [sin h, cos h]
    Ainv = np.linalg.inv(A)
    azd, aza = g["106D.geo.azi.masked.tif"], g["070A.geo.azi.masked.tif"]
    u_n = Ainv[1, 0] * azd + Ainv[1, 1] * aza            # 2x2 solve of two along-track offsets for (E, N); keep N
    # Full 3D: least squares over all four offsets (2 azimuth + 2 range) for (u_E, u_N, u_U).
    # Range-offset sign: GAMMA/LiCSAR offsets are (secondary - primary) pixel shifts, so +range = range increase
    # = motion AWAY from the satellite -> p = -e_LOS. (Flipping the sign of both range rows only flips (u_E, u_U),
    # so the LSQ residual cannot decide it; it is fixed from the convention, consistent with the Kilauea check
    # where the same convention gives the known seaward (east) motion of the south flank.)
    from pygeomapper.insar3d import lsq_enu
    rgd, rga = g["106D.geo.rng.masked.tif"], g["070A.geo.rng.masked.tif"]
    e = {t: [g[f"{t}.geo.{c}.tif"] for c in "ENU"] for t in ("106D", "070A")}
    best = None
    for sgn in (-1.0,):
        obs = [(azd, fd[0], fd[1], 0.0), (aza, fa[0], fa[1], 0.0),
               (rgd, *[sgn * v for v in e["106D"]]), (rga, *[sgn * v for v in e["070A"]])]
        r = lsq_enu(obs, sigmas=[0.3, 0.3, 0.15, 0.15], min_obs=4)
        res = sum(np.nansum((o[0] - (o[1] * r["E"] + o[2] * r["N"] + o[3] * r["U"])) ** 2) for o in obs)
        print("range sign", sgn, "residual", res)
        if best is None or res < best[0]:
            best = (res, sgn, r)
    _, sgn, r3 = best
    write(OUT / "myanmar_licsar_offsets.tif",
          [azd, rgd, aza, rga, u_n, r3["E"], r3["U"]],
          ["azi_106D_m", "rng_106D_m", "azi_070A_m", "rng_070A_m", "u_north_m", "u_east_m", "u_up_m"], [1e-2] * 7,
          dict(source="COMET LiCSAR event us7000pn9s offset tracking (REAL DATA); u_north from 106D+070A azimuth "
                      "offsets; u_east/u_up from 4-offset weighted least squares",
               range_sign=str(sgn),
               heading_106D_vec=str(fd.round(3).tolist()), heading_070A_vec=str(fa.round(3).tolist())))

if __name__ == "__main__":
    main()
