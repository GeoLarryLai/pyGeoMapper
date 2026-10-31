# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Build the MODELED (synthetic, not real SAR) 2025 Mw 7.7 Myanmar / Sagaing Fault displacement stack
`example_data/hazards/myanmar_eq_insar_gnss.tif` on a 0.004 deg (~440 m) grid covering the
95.6-96.6 E, 17.8-23.1 N swath of the Copernicus GLO-90 DEM.

Physics: finite-length vertical right-lateral strike-slip fault (~450 km, 18.6-22.65 N) in an elastic
half-space, using the Okada-like 2-D screw-dislocation profile u_par = -(s/pi) * atan(x / D)
(Savage & Burford 1973) with locking depth D, an along-strike slip taper s(lat) peaking at ~5 m,
and a finite-length decay so displacement fades beyond the rupture tips.
Bands are noise-free; decorrelation, atmosphere and offset-tracking noise are added in Notebook 5
via `simulate_sar_observables()`.
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

OUT = Path(__file__).resolve().parents[1] / "example_data/hazards/myanmar_eq_insar_gnss.tif"
LON0, LON1, LAT0, LAT1, RES = 95.6, 96.6, 17.8, 23.1, 0.004
S_ASC = np.array([-0.608, -0.112, 0.786])
S_DESC = np.array([0.618, -0.114, 0.778])
LAT_N, LAT_S, D_KM = 22.65, 18.60, 12.0


def fault_lon(lat):
    """Sagaing Fault surface trace (approx.): ~96.03 E at Mandalay, ~96.18 E near Naypyitaw."""
    return 96.035 + 0.065 * (22.0 - lat)


def slip_m(lat):
    """Along-strike right-lateral slip (m): ~3 m north, ~5 m peak 20-21 N, tapering at both tips."""
    t = np.clip((lat - LAT_S) / (LAT_N - LAT_S), 0, 1)
    taper = np.sin(np.pi * t) ** 0.35
    return (3.2 + 1.8 * np.exp(-((lat - 20.6) / 0.9) ** 2)) * taper


def model_fields(lon, lat):
    km_lon = 111.32 * np.cos(np.deg2rad(lat))
    x_km = (lon - fault_lon(lat)) * km_lon            # + east of fault
    s = slip_m(np.clip(lat, LAT_S, LAT_N))
    # finite-length decay beyond the tips (Okada-like far-field fall-off)
    beyond = np.maximum(lat - LAT_N, 0) + np.maximum(LAT_S - lat, 0)
    s = s * np.exp(-(beyond * 111.0 / 25.0) ** 2) + 0 * lat
    s = np.where(beyond > 0, slip_m(np.clip(lat, LAT_S + 0.05, LAT_N - 0.05)) * np.exp(-(beyond * 111 / 25) ** 2), s)
    u_par = -(s / np.pi) * np.arctan(x_km / D_KM)    # west block -> north, east block -> south (dextral)
    dlon = fault_lon(lat + 0.01) - fault_lon(lat - 0.01)
    strike = np.arctan2(dlon * km_lon, 0.02 * 111.32)  # azimuth of strike from north (rad), pointing north
    u_e = u_par * np.sin(strike)
    u_n = u_par * np.cos(strike)
    u_u = 0.25 * np.exp(-(x_km / 6.0) ** 2) * np.sin((lat - 18.0) * 3.3) * np.sign(x_km + 1e-9) * (s / 5.0)
    return u_e, u_n, u_u


def main():
    lons = np.arange(LON0 + RES / 2, LON1, RES)
    lats = np.arange(LAT1 - RES / 2, LAT0, -RES)
    LON, LAT = np.meshgrid(lons, lats)
    u_e, u_n, u_u = model_fields(LON, LAT)
    los_a = S_ASC[0] * u_e + S_ASC[1] * u_n + S_ASC[2] * u_u
    los_d = S_DESC[0] * u_e + S_DESC[1] * u_n + S_DESC[2] * u_u
    wrapped = np.angle(np.exp(1j * 4 * np.pi / 0.05546 * los_d))   # Sentinel-1 C-band, 2.77 cm / cycle
    bands = [los_a, los_d, u_e, u_n, u_u, wrapped]
    names = ["los_asc_m", "los_desc_m", "gnss_east_m", "gnss_north_m", "gnss_up_m", "wrapped_phase_rad"]
    with rasterio.open(OUT, "w", driver="GTiff", height=LAT.shape[0], width=LAT.shape[1], count=6,
                       dtype="float32", crs="EPSG:4326", transform=from_origin(LON0, LAT1, RES, RES),
                       compress="DEFLATE", predictor=3, zlevel=9, tiled=True) as dst:
        for i, (b, n) in enumerate(zip(bands, names), 1):
            q = 0.01 if n.endswith('_rad') else 0.001  # quantize to 1 mm / 0.01 rad for compression
            dst.write((np.round(b / q) * q).astype(np.float32), i)
            dst.set_band_description(i, n)
        dst.update_tags(units="m", model="MODELED Okada-like screw dislocation; NOT real SAR",
                        lon_min=LON0, lon_max=LON1, lat_min=LAT0, lat_max=LAT1, locking_depth_km=D_KM)
    print(OUT, LAT.shape, float(np.hypot(u_e, u_n).max()), float(np.abs(u_u).max()))


if __name__ == "__main__":
    main()
