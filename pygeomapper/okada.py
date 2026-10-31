# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
Okada (1985) surface displacements due to a finite rectangular dislocation in an elastic half-space.

Vectorised NumPy port of the closed-form expressions (Okada, Y., 1985, BSSA 75(4), 1135-1154),
following the widely used formulation of F. Beauducel's `okada85.m` (centroid-referenced geometry).

Conventions
-----------
* `e, n` : observation east / north coordinates (m) relative to the fault centroid's surface projection.
* `depth`: centroid depth (m, positive down). `strike`, `dip`, `rake` in degrees (Aki & Richards).
  rake = 0 left-lateral, 180 right-lateral, 90 thrust, -90 normal.
* `L`, `W`: fault length along strike and width down dip (m). `slip`, `opening` in m.
* Returns (u_e, u_n, u_u) in m.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-14


def _I5(xi, eta, q, dip, nu, R, db):
    X = np.sqrt(xi**2 + q**2)
    cd, sd = np.cos(dip), np.sin(dip)
    if cd > _EPS:
        with np.errstate(divide="ignore", invalid="ignore"):
            I = (1 - 2 * nu) * 2 / cd * np.arctan((eta * (X + q * cd) + X * (R + X) * sd) / (xi * (R + X) * cd))
        I = np.where(xi == 0, 0.0, I)
    else:
        I = -(1 - 2 * nu) * xi * sd / (R + db)
    return I


def _I4(db, eta, q, dip, nu, R):
    cd, sd = np.cos(dip), np.sin(dip)
    if cd > _EPS:
        return (1 - 2 * nu) / cd * (np.log(R + db) - sd * np.log(R + eta))
    return -(1 - 2 * nu) * q / (R + db)


def _I3(eta, q, dip, nu, R):
    cd, sd = np.cos(dip), np.sin(dip)
    yb = eta * cd + q * sd
    db = eta * sd - q * cd
    if cd > _EPS:
        return (1 - 2 * nu) * (yb / (cd * (R + db)) - np.log(R + eta)) + sd / cd * _I4(db, eta, q, dip, nu, R)
    return (1 - 2 * nu) / 2 * (eta / (R + db) + yb * q / (R + db) ** 2 - np.log(R + eta))


def _I2(eta, q, dip, nu, R):
    return (1 - 2 * nu) * (-np.log(R + eta)) - _I3(eta, q, dip, nu, R)


def _I1(xi, eta, q, dip, nu, R):
    cd, sd = np.cos(dip), np.sin(dip)
    db = eta * sd - q * cd
    if cd > _EPS:
        return (1 - 2 * nu) * (-xi / (cd * (R + db))) - sd / cd * _I5(xi, eta, q, dip, nu, R, db)
    return -(1 - 2 * nu) / 2 * xi * q / (R + db) ** 2


def _atan_term(xi, eta, q, R):
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.arctan(xi * eta / (q * R))
    return np.where(q != 0, t, 0.0)


def _ss(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    cd, sd = np.cos(dip), np.sin(dip)
    db = eta * sd - q * cd
    ux = xi * q / (R * (R + eta)) + _atan_term(xi, eta, q, R) + _I1(xi, eta, q, dip, nu, R) * sd
    uy = (eta * cd + q * sd) * q / (R * (R + eta)) + q * cd / (R + eta) + _I2(eta, q, dip, nu, R) * sd
    uz = db * q / (R * (R + eta)) + q * sd / (R + eta) + _I4(db, eta, q, dip, nu, R) * sd
    return ux, uy, uz


def _ds(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    cd, sd = np.cos(dip), np.sin(dip)
    db = eta * sd - q * cd
    at = _atan_term(xi, eta, q, R)
    ux = q / R - _I3(eta, q, dip, nu, R) * sd * cd
    uy = (eta * cd + q * sd) * q / (R * (R + xi)) + cd * at - _I1(xi, eta, q, dip, nu, R) * sd * cd
    uz = db * q / (R * (R + xi)) + sd * at - _I5(xi, eta, q, dip, nu, R, db) * sd * cd
    return ux, uy, uz


def _tf(xi, eta, q, dip, nu):
    R = np.sqrt(xi**2 + eta**2 + q**2)
    cd, sd = np.cos(dip), np.sin(dip)
    db = eta * sd - q * cd
    at = _atan_term(xi, eta, q, R)
    ux = q**2 / (R * (R + eta)) - _I3(eta, q, dip, nu, R) * sd**2
    uy = -(eta * sd - q * cd) * q / (R * (R + xi)) - sd * (xi * q / (R * (R + eta)) - at) - _I1(xi, eta, q, dip, nu, R) * sd**2
    uz = (eta * cd + q * sd) * q / (R * (R + xi)) + cd * (xi * q / (R * (R + eta)) - at) - _I5(xi, eta, q, dip, nu, R, db) * sd**2
    return ux, uy, uz


def _chinnery(f, x, p, L, W, q, dip, nu):
    a = f(x, p, q, dip, nu)
    b = f(x, p - W, q, dip, nu)
    c = f(x - L, p, q, dip, nu)
    d = f(x - L, p - W, q, dip, nu)
    return [ai - bi - ci + di for ai, bi, ci, di in zip(a, b, c, d)]


def okada85(e, n, depth, strike, dip, L, W, rake, slip, opening=0.0, nu=0.25):
    """Surface displacement (u_e, u_n, u_u) of one rectangular dislocation (Okada 1985)."""
    e = np.asarray(e, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    phi = np.deg2rad(strike)
    dp = np.deg2rad(dip)
    rk = np.deg2rad(rake)
    sp, cp = np.sin(phi), np.cos(phi)
    cd, sd = np.cos(dp), np.sin(dp)
    U1, U2, U3 = np.cos(rk) * slip, np.sin(rk) * slip, opening
    d = depth + sd * W / 2.0                       # depth of the lower edge
    # Okada frame: x along strike, y = left of strike (fault dips toward -y), origin above lower-edge start
    x = e * sp + n * cp + L / 2.0
    y = -e * cp + n * sp + cd * W / 2.0
    p = y * cd + d * sd
    q = y * sd - d * cd
    ux = np.zeros_like(e); uy = np.zeros_like(e); uz = np.zeros_like(e)
    with np.errstate(divide="ignore", invalid="ignore"):
        if U1 != 0:
            a = _chinnery(_ss, x, p, L, W, q, dp, nu)
            ux += -U1 / (2 * np.pi) * a[0]; uy += -U1 / (2 * np.pi) * a[1]; uz += -U1 / (2 * np.pi) * a[2]
        if U2 != 0:
            a = _chinnery(_ds, x, p, L, W, q, dp, nu)
            ux += -U2 / (2 * np.pi) * a[0]; uy += -U2 / (2 * np.pi) * a[1]; uz += -U2 / (2 * np.pi) * a[2]
        if U3 != 0:
            a = _chinnery(_tf, x, p, L, W, q, dp, nu)
            ux += U3 / (2 * np.pi) * a[0]; uy += U3 / (2 * np.pi) * a[1]; uz += U3 / (2 * np.pi) * a[2]
    ue = sp * ux - cp * uy
    un = cp * ux + sp * uy
    return np.nan_to_num(ue), np.nan_to_num(un), np.nan_to_num(uz)


def okada_lonlat(lon, lat, patches, nu=0.25):
    """
    Sum Okada (1985) displacements of many rectangular patches on a lon/lat grid (local tangent-plane approx).

    `patches`: iterable of dicts with keys lon, lat (centroid surface projection), depth_m, strike, dip,
    length_m, width_m, rake, slip_m (optional opening_m).
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    ue = np.zeros_like(lon); un = np.zeros_like(lon); uu = np.zeros_like(lon)
    R_E = 6371008.8
    for pt in patches:
        e = np.deg2rad(lon - pt["lon"]) * R_E * np.cos(np.deg2rad(pt["lat"]))
        n = np.deg2rad(lat - pt["lat"]) * R_E
        a, b, c = okada85(e, n, pt["depth_m"], pt["strike"], pt["dip"], pt["length_m"], pt["width_m"],
                          pt["rake"], pt["slip_m"], pt.get("opening_m", 0.0), nu)
        ue += a; un += b; uu += c
    return ue, un, uu
