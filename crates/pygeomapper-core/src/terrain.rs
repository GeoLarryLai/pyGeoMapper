// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
use rayon::prelude::*;
use std::f64::consts::PI;

#[inline]
fn is_valid(v: f64, nodata: f64) -> bool {
    v.is_finite() && (nodata.is_nan() || (v - nodata).abs() > 1e-9)
}

#[inline]
fn get_clamped(dem: &[f64], rows: usize, cols: usize, r: isize, c: isize) -> f64 {
    let rr = r.clamp(0, (rows as isize) - 1) as usize;
    let cc = c.clamp(0, (cols as isize) - 1) as usize;
    dem[rr * cols + cc]
}

/// Compute Horn (1981) slope (degrees, [0, 90]) and aspect (degrees clockwise from North, [0, 360), -1 for flat).
pub fn compute_slope_aspect(
    dem: &[f64],
    rows: usize,
    cols: usize,
    dx: f64,
    dy: f64,
    nodata: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rows * cols;
    let mut slope = vec![f64::NAN; n];
    let mut aspect = vec![f64::NAN; n];
    if rows == 0 || cols == 0 || dx <= 0.0 || dy <= 0.0 {
        return (slope, aspect);
    }

    slope
        .par_chunks_mut(cols)
        .zip(aspect.par_chunks_mut(cols))
        .enumerate()
        .for_each(|(r, (s_row, a_row))| {
            let ri = r as isize;
            for c in 0..cols {
                let ci = c as isize;
                let z0 = dem[r * cols + c];
                if !is_valid(z0, nodata) {
                    continue;
                }
                let mut win = [0.0f64; 9];
                let mut valid = true;
                for dr in -1..=1 {
                    for dc in -1..=1 {
                        let v = get_clamped(dem, rows, cols, ri + dr, ci + dc);
                        let zv = if is_valid(v, nodata) { v } else { z0 };
                        if !zv.is_finite() {
                            valid = false;
                        }
                        win[((dr + 1) * 3 + (dc + 1)) as usize] = zv;
                    }
                }
                if !valid {
                    continue;
                }
                // Horn (1981) 3x3 finite differences:
                // Row index increases southward (top-down raster), col index increases eastward:
                let dz_dx = ((win[2] + 2.0 * win[5] + win[8]) - (win[0] + 2.0 * win[3] + win[6]))
                    / (8.0 * dx);
                // Northward gradient dz_dy (row - 1 is North, row + 1 is South):
                let dz_dy = ((win[0] + 2.0 * win[1] + win[2]) - (win[6] + 2.0 * win[7] + win[8]))
                    / (8.0 * dy);

                let grad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt();
                s_row[c] = grad.atan() * (180.0 / PI);

                if grad < 1e-12 {
                    a_row[c] = -1.0;
                } else {
                    // Downslope direction clockwise from North:
                    // Downslope vector is (-dz_dx, -dz_dy) where +x is East, +y is North
                    let mut deg = (-dz_dx).atan2(-dz_dy) * (180.0 / PI);
                    if deg < 0.0 {
                        deg += 360.0;
                    }
                    a_row[c] = deg;
                }
            }
        });

    (slope, aspect)
}

/// Compute multi-azimuth or single-azimuth analytical hillshade in [0, 1].
pub fn compute_hillshade(
    dem: &[f64],
    rows: usize,
    cols: usize,
    dx: f64,
    dy: f64,
    azimuth_deg: f64,
    altitude_deg: f64,
    z_factor: f64,
    nodata: f64,
) -> Vec<f64> {
    let n = rows * cols;
    let mut hs = vec![f64::NAN; n];
    if rows == 0 || cols == 0 || dx <= 0.0 || dy <= 0.0 {
        return hs;
    }

    let zenith_rad = (90.0 - altitude_deg.clamp(0.0, 90.0)) * (PI / 180.0);
    let az_rad = azimuth_deg * (PI / 180.0);
    let cos_z = zenith_rad.cos();
    let sin_z = zenith_rad.sin();

    hs.par_chunks_mut(cols)
        .enumerate()
        .for_each(|(r, hs_row)| {
            let ri = r as isize;
            for c in 0..cols {
                let ci = c as isize;
                let z0 = dem[r * cols + c];
                if !is_valid(z0, nodata) {
                    continue;
                }
                let mut win = [0.0f64; 9];
                for dr in -1..=1 {
                    for dc in -1..=1 {
                        let v = get_clamped(dem, rows, cols, ri + dr, ci + dc);
                        win[((dr + 1) * 3 + (dc + 1)) as usize] =
                            if is_valid(v, nodata) { v } else { z0 };
                    }
                }
                let dz_dx = z_factor
                    * ((win[2] + 2.0 * win[5] + win[8]) - (win[0] + 2.0 * win[3] + win[6]))
                    / (8.0 * dx);
                let dz_dy = z_factor
                    * ((win[0] + 2.0 * win[1] + win[2]) - (win[6] + 2.0 * win[7] + win[8]))
                    / (8.0 * dy);

                let slope_rad = (dz_dx * dz_dx + dz_dy * dz_dy).sqrt().atan();
                let aspect_rad = if dz_dx.abs() < 1e-12 && dz_dy.abs() < 1e-12 {
                    0.0
                } else {
                    (-dz_dx).atan2(-dz_dy)
                };
                let illum = cos_z * slope_rad.cos()
                    + sin_z * slope_rad.sin() * (az_rad - aspect_rad).cos();
                hs_row[c] = illum.clamp(0.0, 1.0);
            }
        });

    hs
}

/// Compute Laplacian curvature, profile curvature, and planform curvature.
pub fn compute_curvature(
    dem: &[f64],
    rows: usize,
    cols: usize,
    dx: f64,
    dy: f64,
    nodata: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = rows * cols;
    let mut laplacian = vec![f64::NAN; n];
    let mut profile = vec![f64::NAN; n];
    let mut planform = vec![f64::NAN; n];
    if rows == 0 || cols == 0 || dx <= 0.0 || dy <= 0.0 {
        return (laplacian, profile, planform);
    }

    laplacian
        .par_chunks_mut(cols)
        .zip(profile.par_chunks_mut(cols).zip(planform.par_chunks_mut(cols)))
        .enumerate()
        .for_each(|(r, (lap_row, (prof_row, plan_row)))| {
            let ri = r as isize;
            for c in 0..cols {
                let ci = c as isize;
                let z0 = dem[r * cols + c];
                if !is_valid(z0, nodata) {
                    continue;
                }
                let mut w = [0.0f64; 9];
                for dr in -1..=1 {
                    for dc in -1..=1 {
                        let v = get_clamped(dem, rows, cols, ri + dr, ci + dc);
                        w[((dr + 1) * 3 + (dc + 1)) as usize] =
                            if is_valid(v, nodata) { v } else { z0 };
                    }
                }
                // Second derivatives (Zevenbergen & Thorne, 1987)
                let d2z_dx2 = (w[3] - 2.0 * w[4] + w[5]) / (dx * dx);
                let d2z_dy2 = (w[1] - 2.0 * w[4] + w[7]) / (dy * dy);
                let d2z_dxdy = ((w[2] - w[0]) - (w[8] - w[6])) / (4.0 * dx * dy);
                let dz_dx = (w[5] - w[3]) / (2.0 * dx);
                let dz_dy = (w[1] - w[7]) / (2.0 * dy);

                lap_row[c] = d2z_dx2 + d2z_dy2;

                let p = dz_dx * dz_dx + dz_dy * dz_dy;
                if p < 1e-12 {
                    prof_row[c] = 0.0;
                    plan_row[c] = 0.0;
                } else {
                    let q = p + 1.0;
                    prof_row[c] = -(d2z_dx2 * dz_dx * dz_dx
                        + 2.0 * d2z_dxdy * dz_dx * dz_dy
                        + d2z_dy2 * dz_dy * dz_dy)
                        / (p * q.powf(1.5));
                    plan_row[c] = -(d2z_dx2 * dz_dy * dz_dy
                        - 2.0 * d2z_dxdy * dz_dx * dz_dy
                        + d2z_dy2 * dz_dx * dz_dx)
                        / p.powf(1.5);
                }
            }
        });

    (laplacian, profile, planform)
}

/// Compute local elevation standard deviation (roughness) and Topographic Position Index (TPI)
/// within a circular moving window of radius `radius_px`.
pub fn compute_roughness_tpi(
    dem: &[f64],
    rows: usize,
    cols: usize,
    radius_px: usize,
    nodata: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rows * cols;
    let mut roughness = vec![f64::NAN; n];
    let mut tpi = vec![f64::NAN; n];
    if rows == 0 || cols == 0 {
        return (roughness, tpi);
    }
    let rad = radius_px.max(1) as isize;
    let r2 = rad * rad;

    roughness
        .par_chunks_mut(cols)
        .zip(tpi.par_chunks_mut(cols))
        .enumerate()
        .for_each(|(r, (rough_row, tpi_row))| {
            let ri = r as isize;
            for c in 0..cols {
                let ci = c as isize;
                let z0 = dem[r * cols + c];
                if !is_valid(z0, nodata) {
                    continue;
                }
                let mut sum = 0.0;
                let mut sum_sq = 0.0;
                let mut count = 0usize;
                for dr in -rad..=rad {
                    for dc in -rad..=rad {
                        if dr * dr + dc * dc > r2 {
                            continue;
                        }
                        let rr = ri + dr;
                        let cc = ci + dc;
                        if rr >= 0 && rr < (rows as isize) && cc >= 0 && cc < (cols as isize) {
                            let v = dem[(rr as usize) * cols + (cc as usize)];
                            if is_valid(v, nodata) {
                                sum += v;
                                sum_sq += v * v;
                                count += 1;
                            }
                        }
                    }
                }
                if count > 1 {
                    let mean = sum / (count as f64);
                    let var = (sum_sq / (count as f64) - mean * mean).max(0.0);
                    rough_row[c] = var.sqrt();
                    tpi_row[c] = z0 - mean;
                } else {
                    rough_row[c] = 0.0;
                    tpi_row[c] = 0.0;
                }
            }
        });

    (roughness, tpi)
}

/// Fill any non-finite / nodata pixels with the mean of finite pixels so separable 1D convolutions
/// do not propagate NaNs across the grid; original invalid pixels are restored to NaN afterward.
fn prepare_clean_dem(dem: &[f64], nodata: f64) -> (Vec<f64>, Vec<bool>) {
    let n = dem.len();
    let mut valid = vec![false; n];
    let mut sum = 0.0f64;
    let mut cnt = 0usize;
    for i in 0..n {
        if is_valid(dem[i], nodata) {
            valid[i] = true;
            sum += dem[i];
            cnt += 1;
        }
    }
    let fill = if cnt > 0 { sum / (cnt as f64) } else { 0.0 };
    let clean: Vec<f64> = (0..n)
        .map(|i| if valid[i] { dem[i] } else { fill })
        .collect();
    (clean, valid)
}

/// Build 1D Gaussian kernel `g` (sum = 1) and zero-mean 1D second-derivative Gaussian kernel `h2` (sum = 0)
/// for spatial scale `scale_m` and grid spacing `step_m`.
fn build_1d_cwt_kernels(scale_m: f64, step_m: f64, cutoff_sigma: f64) -> (Vec<f64>, Vec<f64>, isize) {
    let s = scale_m.max(step_m * 0.5);
    let sigma_px = s / step_m.max(1e-9);
    let radius = ((cutoff_sigma.max(3.0) * sigma_px).ceil() as isize).clamp(2, 128);
    let len = (2 * radius + 1) as usize;
    let mut g = vec![0.0f64; len];
    let mut h2 = vec![0.0f64; len];

    let mut sum_g = 0.0f64;
    let mut sum_pos = 0.0f64;
    let mut sum_neg = 0.0f64;

    for idx in 0..len {
        let d = (idx as isize - radius) as f64 * step_m;
        let u2 = (d * d) / (s * s);
        let exp_term = (-0.5 * u2).exp();
        g[idx] = exp_term;
        sum_g += exp_term;

        let val = (1.0 - u2) * exp_term;
        h2[idx] = val;
        if val > 0.0 {
            sum_pos += val;
        } else if val < 0.0 {
            sum_neg -= val; // positive magnitude of negative lobe
        }
    }

    if sum_g > 1e-15 {
        for v in g.iter_mut() {
            *v /= sum_g;
        }
    }

    // Balance positive and negative lobes of h2 so sum(h2) == 0 exactly and sum_{h2 > 0} h2 == 1.0:
    // Because h2[-d] == h2[+d], any constant or linear ramp a*x + b has exact zero response,
    // and a sinusoid cos(k*x) produces scale-invariant dimensionless response (k*s)^2 * exp(-(k*s)^2 / 2).
    if sum_pos > 1e-15 && sum_neg > 1e-15 {
        let norm_factor = step_m / ((2.0 * PI).sqrt() * s);
        let neg_scale = sum_pos / sum_neg;
        for v in h2.iter_mut() {
            if *v < 0.0 {
                *v *= neg_scale * norm_factor;
            } else {
                *v *= norm_factor;
            }
        }
    }

    (g, h2, radius)
}

fn convolve_1d_horiz(input: &[f64], rows: usize, cols: usize, kernel: &[f64], radius: isize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * cols];
    out.par_chunks_mut(cols)
        .enumerate()
        .for_each(|(r, out_row)| {
            let row_slice = &input[r * cols..(r + 1) * cols];
            for c in 0..cols {
                let ci = c as isize;
                let mut acc = 0.0f64;
                for k_idx in 0..kernel.len() {
                    let dc = (k_idx as isize) - radius;
                    // Reflect at boundary to minimize edge gradient artifacts
                    let mut cc = ci + dc;
                    if cc < 0 {
                        cc = -cc;
                    } else if cc >= cols as isize {
                        cc = 2 * (cols as isize - 1) - cc;
                    }
                    let cc_clamped = cc.clamp(0, cols as isize - 1) as usize;
                    acc += row_slice[cc_clamped] * kernel[k_idx];
                }
                out_row[c] = acc;
            }
        });
    out
}

fn convolve_1d_vert(input: &[f64], rows: usize, cols: usize, kernel: &[f64], radius: isize) -> Vec<f64> {
    let mut out = vec![0.0f64; rows * cols];
    out.par_chunks_mut(cols)
        .enumerate()
        .for_each(|(r, out_row)| {
            let ri = r as isize;
            for c in 0..cols {
                let mut acc = 0.0f64;
                for k_idx in 0..kernel.len() {
                    let dr = (k_idx as isize) - radius;
                    let mut rr = ri + dr;
                    if rr < 0 {
                        rr = -rr;
                    } else if rr >= rows as isize {
                        rr = 2 * (rows as isize - 1) - rr;
                    }
                    let rr_clamped = rr.clamp(0, rows as isize - 1) as usize;
                    acc += input[rr_clamped * cols + c] * kernel[k_idx];
                }
                out_row[c] = acc;
            }
        });
    out
}

/// Compute single-scale 2D Continuous Wavelet Transform (2D CWT) using the isotropic Mexican-hat
/// (Marr / Ricker) wavelet at spatial scale `scale_m` (in meters; characteristic wavelength `lambda = sqrt(2)*pi*scale_m`).
///
/// Returns `(cwt_response, wavelet_energy_envelope)`:
/// - `cwt_response`: signed 2D Mexican-hat coefficient `W_s(x, y)` (> 0 on convex ridges, < 0 in concave valleys).
/// - `wavelet_energy_envelope`: local RMS wavelet energy `E_s(x, y) = sqrt(2 * (G_s * W_s^2))` measuring multiscale terrain complexity.
pub fn compute_cwt2d_mexican_hat(
    dem: &[f64],
    rows: usize,
    cols: usize,
    dx: f64,
    dy: f64,
    scale_m: f64,
    nodata: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = rows * cols;
    if rows == 0 || cols == 0 || dx <= 0.0 || dy <= 0.0 || scale_m <= 0.0 {
        return (vec![f64::NAN; n], vec![f64::NAN; n]);
    }
    let (clean, valid) = prepare_clean_dem(dem, nodata);
    let (gx, h2x, rx) = build_1d_cwt_kernels(scale_m, dx, 3.5);
    let (gy, h2y, ry) = build_1d_cwt_kernels(scale_m, dy, 3.5);

    // Separable 2D Mexican-hat decomposition:
    // W_s = (z *_x h2x) *_y gy + (z *_x gx) *_y h2y
    let zx_h2 = convolve_1d_horiz(&clean, rows, cols, &h2x, rx);
    let term_x = convolve_1d_vert(&zx_h2, rows, cols, &gy, ry);

    let zx_g = convolve_1d_horiz(&clean, rows, cols, &gx, rx);
    let term_y = convolve_1d_vert(&zx_g, rows, cols, &h2y, ry);

    let mut cwt = vec![0.0f64; n];
    let mut cwt_sq = vec![0.0f64; n];
    for i in 0..n {
        let w = term_x[i] + term_y[i];
        cwt[i] = w;
        cwt_sq[i] = w * w;
    }

    // Smooth W_s^2 with the Gaussian envelope G_s so periodic ridge-valley trains have uniform energy across zero-crossings
    let env_x = convolve_1d_horiz(&cwt_sq, rows, cols, &gx, rx);
    let env_xy = convolve_1d_vert(&env_x, rows, cols, &gy, ry);

    let mut energy = vec![f64::NAN; n];
    for i in 0..n {
        if valid[i] {
            energy[i] = (2.0 * env_xy[i].max(0.0)).sqrt();
        } else {
            cwt[i] = f64::NAN;
        }
    }

    (cwt, energy)
}

#[derive(Debug, Clone)]
pub struct MultiscaleComplexityResult {
    pub cwt_cube: Vec<f64>,
    pub energy_cube: Vec<f64>,
    pub mean_complexity: Vec<f64>,
    pub max_energy: Vec<f64>,
    pub dominant_scale_m: Vec<f64>,
    pub dominant_wavelength_m: Vec<f64>,
}

/// Compute multiscale 2D Mexican-hat CWT complexity across `scales_m` (in meters).
///
/// Characteristic landform wavelength corresponding to Mexican-hat scale `s` is `lambda = sqrt(2) * pi * s`.
pub fn compute_multiscale_complexity(
    dem: &[f64],
    rows: usize,
    cols: usize,
    dx: f64,
    dy: f64,
    scales_m: &[f64],
    nodata: f64,
) -> MultiscaleComplexityResult {
    let n = rows * cols;
    let k = scales_m.len();
    if rows == 0 || cols == 0 || k == 0 {
        return MultiscaleComplexityResult {
            cwt_cube: Vec::new(),
            energy_cube: Vec::new(),
            mean_complexity: vec![f64::NAN; n],
            max_energy: vec![f64::NAN; n],
            dominant_scale_m: vec![f64::NAN; n],
            dominant_wavelength_m: vec![f64::NAN; n],
        };
    }

    // Evaluate all scales in parallel
    let per_scale: Vec<(Vec<f64>, Vec<f64>)> = scales_m
        .par_iter()
        .map(|&s| compute_cwt2d_mexican_hat(dem, rows, cols, dx, dy, s, nodata))
        .collect();

    let mut cwt_cube = vec![f64::NAN; n * k];
    let mut energy_cube = vec![f64::NAN; n * k];
    let mut mean_complexity = vec![f64::NAN; n];
    let mut max_energy = vec![f64::NAN; n];
    let mut dominant_scale_m = vec![f64::NAN; n];
    let mut dominant_wavelength_m = vec![f64::NAN; n];

    let lambda_factor = 2.0f64.sqrt() * PI;

    for i in 0..n {
        if !per_scale[0].1[i].is_finite() {
            continue;
        }
        let mut sum_e = 0.0f64;
        let mut best_e = -1.0f64;
        let mut best_idx = 0usize;
        for s_idx in 0..k {
            let w = per_scale[s_idx].0[i];
            let e = per_scale[s_idx].1[i];
            cwt_cube[i * k + s_idx] = w;
            energy_cube[i * k + s_idx] = e;
            sum_e += e;
            if e > best_e {
                best_e = e;
                best_idx = s_idx;
            }
        }
        mean_complexity[i] = sum_e / (k as f64);
        max_energy[i] = best_e;

        // Parabolic 3-point sub-grid refinement in log-scale space when peak is an interior scale bin
        let mut dom_s = scales_m[best_idx];
        if k >= 3 && best_idx > 0 && best_idx + 1 < k {
            let e_prev = per_scale[best_idx - 1].1[i].max(1e-15).ln();
            let e_curr = per_scale[best_idx].1[i].max(1e-15).ln();
            let e_next = per_scale[best_idx + 1].1[i].max(1e-15).ln();
            let denom = e_prev - 2.0 * e_curr + e_next;
            if denom < -1e-12 {
                let delta = 0.5 * (e_prev - e_next) / denom;
                let clamped_delta = delta.clamp(-0.5, 0.5);
                let ls_prev = scales_m[best_idx - 1].ln();
                let ls_curr = scales_m[best_idx].ln();
                let ls_next = scales_m[best_idx + 1].ln();
                let step_ls = if clamped_delta >= 0.0 {
                    ls_next - ls_curr
                } else {
                    ls_curr - ls_prev
                };
                dom_s = (ls_curr + clamped_delta * step_ls).exp();
            }
        }
        dominant_scale_m[i] = dom_s;
        dominant_wavelength_m[i] = dom_s * lambda_factor;
    }

    MultiscaleComplexityResult {
        cwt_cube,
        energy_cube,
        mean_complexity,
        max_energy,
        dominant_scale_m,
        dominant_wavelength_m,
    }
}
