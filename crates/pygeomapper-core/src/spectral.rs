// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
use rayon::prelude::*;

#[inline]
fn is_valid(v: f64, nodata: f64) -> bool {
    v.is_finite() && (nodata.is_nan() || (v - nodata).abs() > 1e-9)
}

/// Generic normalized difference: (a - b) / (a + b), clamped to [-1.0, 1.0].
pub fn normalized_difference(band_a: &[f64], band_b: &[f64], nodata: f64) -> Vec<f64> {
    let n = band_a.len().min(band_b.len());
    let mut out = vec![f64::NAN; n];
    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        let a = band_a[i];
        let b = band_b[i];
        if is_valid(a, nodata) && is_valid(b, nodata) {
            let denom = a + b;
            if denom.abs() > 1e-12 {
                *val = ((a - b) / denom).clamp(-1.0, 1.0);
            }
        }
    });
    out
}

/// Generic band ratio: a / b.
pub fn band_ratio(band_a: &[f64], band_b: &[f64], nodata: f64) -> Vec<f64> {
    let n = band_a.len().min(band_b.len());
    let mut out = vec![f64::NAN; n];
    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        let a = band_a[i];
        let b = band_b[i];
        if is_valid(a, nodata) && is_valid(b, nodata) && b.abs() > 1e-12 {
            *val = a / b;
        }
    });
    out
}

/// Bare Soil Index (BSI): ((SWIR1 + Red) - (NIR + Blue)) / ((SWIR1 + Red) + (NIR + Blue)).
pub fn bare_soil_index(
    swir1: &[f64],
    red: &[f64],
    nir: &[f64],
    blue: &[f64],
    nodata: f64,
) -> Vec<f64> {
    let n = swir1.len().min(red.len()).min(nir.len()).min(blue.len());
    let mut out = vec![f64::NAN; n];
    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        let s1 = swir1[i];
        let r = red[i];
        let ni = nir[i];
        let bl = blue[i];
        if is_valid(s1, nodata)
            && is_valid(r, nodata)
            && is_valid(ni, nodata)
            && is_valid(bl, nodata)
        {
            let num = (s1 + r) - (ni + bl);
            let den = (s1 + r) + (ni + bl);
            if den.abs() > 1e-12 {
                *val = (num / den).clamp(-1.0, 1.0);
            }
        }
    });
    out
}

/// Ferrous iron / silicate index: (SWIR1 / NIR) + (Green / Red).
pub fn ferrous_iron_index(
    swir1: &[f64],
    nir: &[f64],
    green: &[f64],
    red: &[f64],
    nodata: f64,
) -> Vec<f64> {
    let n = swir1.len().min(nir.len()).min(green.len()).min(red.len());
    let mut out = vec![f64::NAN; n];
    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        let s1 = swir1[i];
        let ni = nir[i];
        let g = green[i];
        let r = red[i];
        if is_valid(s1, nodata)
            && is_valid(ni, nodata)
            && is_valid(g, nodata)
            && is_valid(r, nodata)
            && ni.abs() > 1e-12
            && r.abs() > 1e-12
        {
            *val = (s1 / ni) + (g / r);
        }
    });
    out
}

/// Linear combination / affine spectral index: (c1*b1 + c2*b2 + c0) / (d1*b1 + d2*b2 + d0).
pub fn rational_two_band_index(
    band1: &[f64],
    band2: &[f64],
    c1: f64,
    c2: f64,
    c0: f64,
    d1: f64,
    d2: f64,
    d0: f64,
    nodata: f64,
) -> Vec<f64> {
    let n = band1.len().min(band2.len());
    let mut out = vec![f64::NAN; n];
    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        let b1 = band1[i];
        let b2 = band2[i];
        if is_valid(b1, nodata) && is_valid(b2, nodata) {
            let num = c1 * b1 + c2 * b2 + c0;
            let den = d1 * b1 + d2 * b2 + d0;
            if den.abs() > 1e-12 {
                *val = num / den;
            }
        }
    });
    out
}
