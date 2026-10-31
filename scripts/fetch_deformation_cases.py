# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Fetch REAL public deformation products for the ground-deformation cases (no login needed) and clip them.

  * Kilauea 2018 (LiCSAR Sentinel-1, CEDA/JASMIN, CC-BY 4.0):
      087D_07004_060904 20180423_20180505 and 124A_06996_091406 20180420_20180508
      (both span the 2018-05-04 Mw 6.9 south-flank earthquake and the LERZ dike intrusion)
  * Jakobshavn Isbrae / Sermeq Kujalleq (NASA ITS_LIVE v2 annual 120 m mosaics 2016, 2018, EPSG:3413)
  * Copernicus GLO-30 DEMs for both.
Products: https://dap.ceda.ac.uk/neodc/comet/data/licsar_products/<track>/<frame>/<pair>/<pair>.geo.{diff_pha,unw,cc}.tif
          https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products.public/<track>/<frame>/metadata/<frame>.geo.{E,N,U}.tif
          https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/annual/ITS_LIVE_velocity_120m_RGI05A_<yr>_v02.nc
"""
from pathlib import Path
import subprocess, sys
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pygeomapper import raster_io                         # noqa: E402
from pygeomapper.insar3d import lsq_enu                   # noqa: E402

CACHE = Path.home() / ".cache" / "pygeomapper" / "licsar"
OUT = ROOT / "example_data" / "hazards"
CEDA = "https://dap.ceda.ac.uk/neodc/comet/data/licsar_products/{t}/{f}/{p}/{p}.geo.{e}.tif"
META = "https://gws-access.jasmin.ac.uk/public/nceo_geohazards/LiCSAR_products.public/{t}/{f}/metadata/{f}.geo.{e}.tif"
WAVELENGTH = 0.0554658
ITS = 'NETCDF:"/vsicurl/https://its-live-data.s3.amazonaws.com/velocity_mosaic/v2/annual/ITS_LIVE_velocity_120m_RGI05A_{y}_v02.nc":{v}'


def get(url, name):
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists() or p.stat().st_size < 10000:
        subprocess.run(["curl", "-sSfL", "-o", str(p), url], check=True)
    return p


def to_grid(path, T, H, W, crs="EPSG:4326", zero_nan=False):
    dst = np.full((H, W), np.nan, np.float32)
    with rasterio.open(path) as s:
        a = s.read(1).astype(np.float32)
        if s.nodata is not None and np.isfinite(s.nodata):
            a[a == s.nodata] = np.nan
        if zero_nan:
            a[a == 0] = np.nan
        reproject(a, dst, src_transform=s.transform, src_crs=s.crs, dst_transform=T, dst_crs=crs,
                  resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    return dst


def write(path, bands, names, scale, T, crs="EPSG:4326", tags=None):
    H, W = bands[0].shape
    with rasterio.open(path, "w", driver="COG", height=H, width=W, count=len(bands), dtype="int16", crs=crs,
                       transform=T, nodata=-32768, compress="DEFLATE", predictor=2, level=9, blocksize=512) as d:
        for i, (b, n, sc) in enumerate(zip(bands, names, scale), 1):
            q = np.where(np.isfinite(b), np.clip(np.round(b / sc), -32767, 32767), -32768).astype(np.int16)
            d.write(q, i)
            d.set_band_description(i, n)
            d.update_tags(i, scale_factor=str(sc), name=n)
        d.update_tags(**(tags or {}))
    print(path.name, round(path.stat().st_size / 1e6, 2), "MB")


def licsar_pair(track, frame, pair, T, H, W):
    g = {e: to_grid(get(CEDA.format(t=track, f=frame, p=pair, e=e), f"{frame}_{pair}.geo.{e}.tif"), T, H, W,
                    zero_nan=True) for e in ["diff_pha", "unw", "cc"]}
    los = -g["unw"] * WAVELENGTH / (4 * np.pi)            # +ve = toward satellite (m), same convention as Myanmar
    los -= np.nanmedian(los[g["cc"] > 100]) if (g["cc"] > 100).sum() > 100 else 0  # no absolute reference
    return g["diff_pha"], los, g["cc"] / 255.0


def los_vec(track, frame, T, H, W):
    return [to_grid(get(META.format(t=track, f=frame, e=e), f"{frame}.geo.{e}.tif"), T, H, W) for e in "ENU"]


def asc_desc_EU(los_a, va, los_d, vd, cc_a, cc_d, cmin=0.25):
    """Two-LOS E/U inversion (N dropped: LOS has ~0.1 N sensitivity). d = e_LOS . u."""
    la = np.where(cc_a >= cmin, los_a, np.nan)
    ld = np.where(cc_d >= cmin, los_d, np.nan)
    r = lsq_enu([(la, va[0], va[1], va[2]), (ld, vd[0], vd[1], vd[2])], components=("E", "U"))
    return r["E"], r["U"]


def grid(lon0, lat0, lon1, lat1, res=0.001):
    W, H = int(round((lon1 - lon0) / res)), int(round((lat1 - lat0) / res))
    return from_origin(lon0, lat1, res, res), H, W


def kilauea():
    b = (-155.45, 19.1, -154.75, 19.6)
    T, H, W = grid(*b)
    wd, ld, cd = licsar_pair(87, "087D_07004_060904", "20180423_20180505", T, H, W)
    wa, la, ca = licsar_pair(124, "124A_06996_091406", "20180420_20180508", T, H, W)
    vd, va = los_vec(87, "087D_07004_060904", T, H, W), los_vec(124, "124A_06996_091406", T, H, W)
    uE, uU = asc_desc_EU(la, va, ld, vd, ca, cd)
    write(OUT / "kilauea_licsar_ifg.tif", [wd, ld, cd, wa, la, ca, uE, uU],
          ["wrapped_087D_rad", "los_087D_m", "coh_087D", "wrapped_124A_rad", "los_124A_m", "coh_124A",
           "u_east_m", "u_up_m"], [1e-4, 1e-3, 1e-4, 1e-4, 1e-3, 1e-4, 1e-3, 1e-3], T,
          tags=dict(source="COMET LiCSAR Sentinel-1 087D 20180423_20180505 + 124A 20180420_20180508 (REAL DATA)",
                    lon_min=b[0], lat_min=b[1], lon_max=b[2], lat_max=b[3],
                    los_vec_087D=str([round(float(np.nanmedian(v)), 3) for v in vd]),
                    los_vec_124A=str([round(float(np.nanmedian(v)), 3) for v in va])))
    raster_io.fetch_copernicus_dem(bounds=b, resolution_m=30, out_path=OUT / "kilauea_dem.tif")


def jakobshavn():
    b = (-260e3, -2340e3, -120e3, -2220e3)          # EPSG:3413 (m)
    bands, names = [], []
    for y in (2016, 2018):
        for v in ("vx", "vy", "v_error"):
            with rasterio.open(ITS.format(y=y, v=v)) as s:
                w = from_bounds(*b, s.transform)
                a = s.read(1, window=w).astype(np.float32)
                if s.nodata is not None:
                    a[a == s.nodata] = np.nan
                T = s.window_transform(w)
            bands.append(a)
            names.append(f"{v}_{y}_m_per_yr")
    write(OUT / "jakobshavn_itslive_velocity.tif", bands, names, [1.0] * 6, T, crs="EPSG:3413",
          tags=dict(source="NASA MEaSUREs ITS_LIVE v2 annual 120 m velocity mosaics, RGI05A, 2016 and 2018 (REAL DATA)"))
    # Copernicus GLO-30 reprojected onto an exact 30 m EPSG:3413 grid of the same window
    ll = transform_bounds("EPSG:3413", "EPSG:4326", *b, densify_pts=50)
    tmp = CACHE / "jakobshavn_glo30_ll.tif"
    raster_io.fetch_copernicus_dem(bounds=ll, resolution_m=30, out_path=tmp)
    T30 = from_origin(b[0], b[3], 30, 30)
    H30, W30 = int((b[3] - b[1]) / 30), int((b[2] - b[0]) / 30)
    dem = np.full((H30, W30), np.nan, np.float32)
    with rasterio.open(tmp) as s:
        a = s.read(1).astype(np.float32)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        reproject(a, dem, src_transform=s.transform, src_crs=s.crs, dst_transform=T30, dst_crs="EPSG:3413",
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    write(OUT / "jakobshavn_dem.tif", [dem], ["elevation_m"], [1.0], T30, crs="EPSG:3413",
          tags=dict(source="Copernicus DEM GLO-30, reprojected to EPSG:3413 30 m"))


if __name__ == "__main__":
    for name in (sys.argv[1:] or ["kilauea", "jakobshavn"]):
        globals()[name]()
