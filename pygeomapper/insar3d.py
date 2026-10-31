# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Per-pixel least-squares inversion of SAR observables (LOS phase, range offsets, azimuth offsets)
for 3D ground displacement (u_E, u_N, u_U).

Each observation k is a projection d_k = p_k . u + noise, where p_k is a unit vector:
  * LOS / range-offset:  p = -e_LOS  (e_LOS = ground->satellite unit vector; range increase is positive), or +e_LOS
                         for "toward satellite" positive LOS displacement.
  * azimuth offset:      p = flight (along-track) horizontal unit vector.
Solve min ||W^(1/2)(G u - d)||^2 pixel-wise; unknown components can be dropped (e.g. N for two LOS tracks).
"""
import numpy as np


def flight_vector(e_los, n_los, ascending):
    """Horizontal along-track unit vector (E, N) from the horizontal part of the LOS vector."""
    v = np.array([-n_los, e_los], float)
    v /= np.hypot(*v)
    if ascending and v[1] < 0 or (not ascending) and v[1] > 0:
        v = -v
    return v


def lsq_enu(observations, components=("E", "N", "U"), sigmas=None, min_obs=None):
    """observations: list of (d, pE, pN, pU) with d an array and p* scalars or arrays of the same shape.
    Returns dict component -> array (NaN where fewer than len(components) valid observations)."""
    idx = {"E": 1, "N": 2, "U": 3}
    shape = np.shape(observations[0][0])
    m, k = len(observations), len(components)
    sig = np.ones(m) if sigmas is None else np.asarray(sigmas, float)
    G = np.zeros(shape + (m, k))
    d = np.zeros(shape + (m,))
    valid = np.zeros(shape + (m,), bool)
    for i, ob in enumerate(observations):
        di = np.asarray(ob[0], float)
        valid[..., i] = np.isfinite(di)
        d[..., i] = np.where(valid[..., i], di, 0) / sig[i]
        for j, c in enumerate(components):
            G[..., i, j] = np.broadcast_to(ob[idx[c]], shape) / sig[i] * valid[..., i]
    GtG = np.einsum("...ij,...ik->...jk", G, G)
    Gtd = np.einsum("...ij,...i->...j", G, d)
    ok = valid.sum(-1) >= (min_obs or k)
    det = np.linalg.det(GtG)
    ok &= np.abs(det) > 1e-6
    GtG[~ok] = np.eye(k)
    u = np.linalg.solve(GtG, Gtd[..., None])[..., 0]
    u[~ok] = np.nan
    return {c: u[..., j] for j, c in enumerate(components)}
