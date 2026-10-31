// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
#[inline]
fn is_valid(v: f64, nodata: f64) -> bool {
    v.is_finite() && (nodata.is_nan() || (v - nodata).abs() > 1e-9)
}

/// Bilinear interpolation at continuous map coordinates `(x, y)` where `(x0, y0)` is center of pixel `(0, 0)`
/// and `dx`, `dy` are signed pixel spacings (e.g. `dx > 0`, `dy < 0` for north-up rasters).
pub fn sample_bilinear_at(
    grid: &[f64],
    rows: usize,
    cols: usize,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    x: f64,
    y: f64,
    nodata: f64,
) -> f64 {
    if rows == 0 || cols == 0 || dx.abs() < 1e-12 || dy.abs() < 1e-12 {
        return f64::NAN;
    }
    let col_f = (x - x0) / dx;
    let row_f = (y - y0) / dy;
    if col_f < 0.0
        || row_f < 0.0
        || col_f > ((cols - 1) as f64)
        || row_f > ((rows - 1) as f64)
    {
        return f64::NAN;
    }
    let c0 = col_f.floor() as usize;
    let r0 = row_f.floor() as usize;
    let c1 = (c0 + 1).min(cols - 1);
    let r1 = (r0 + 1).min(rows - 1);
    let fc = col_f - (c0 as f64);
    let fr = row_f - (r0 as f64);

    let v00 = grid[r0 * cols + c0];
    let v01 = grid[r0 * cols + c1];
    let v10 = grid[r1 * cols + c0];
    let v11 = grid[r1 * cols + c1];

    if !is_valid(v00, nodata)
        || !is_valid(v01, nodata)
        || !is_valid(v10, nodata)
        || !is_valid(v11, nodata)
    {
        return f64::NAN;
    }

    let top = v00 * (1.0 - fc) + v01 * fc;
    let bot = v10 * (1.0 - fc) + v11 * fc;
    top * (1.0 - fr) + bot * fr
}

#[derive(Debug, Clone)]
pub struct SwathProfileResult {
    pub distances_m: Vec<f64>,
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub center_values: Vec<f64>,
    pub swath_mean: Vec<f64>,
    pub swath_min: Vec<f64>,
    pub swath_max: Vec<f64>,
    pub swath_std: Vec<f64>,
}

/// Sample a centerline and perpendicular swath corridor of width `swath_width_m` along `(xa, ya) -> (xb, yb)`.
#[allow(clippy::too_many_arguments)]
pub fn sample_swath_profile(
    grid: &[f64],
    rows: usize,
    cols: usize,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    xa: f64,
    ya: f64,
    xb: f64,
    yb: f64,
    n_samples: usize,
    swath_width_m: f64,
    n_cross: usize,
    nodata: f64,
) -> SwathProfileResult {
    let n = n_samples.max(2);
    let total_len = ((xb - xa).powi(2) + (yb - ya).powi(2)).sqrt();
    let ux = if total_len > 1e-12 {
        (xb - xa) / total_len
    } else {
        1.0
    };
    let uy = if total_len > 1e-12 {
        (yb - ya) / total_len
    } else {
        0.0
    };
    // Unit normal perpendicular to transect
    let nx = -uy;
    let ny = ux;

    let cross_samples = if swath_width_m <= 0.0 {
        1
    } else {
        n_cross.max(3)
    };
    let half_w = 0.5 * swath_width_m.max(0.0);

    let mut distances_m = Vec::with_capacity(n);
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    let mut center_values = Vec::with_capacity(n);
    let mut swath_mean = Vec::with_capacity(n);
    let mut swath_min = Vec::with_capacity(n);
    let mut swath_max = Vec::with_capacity(n);
    let mut swath_std = Vec::with_capacity(n);

    for i in 0..n {
        let t = (i as f64) / ((n - 1) as f64);
        let dist = t * total_len;
        let cx = xa + t * (xb - xa);
        let cy = ya + t * (yb - ya);
        let cval = sample_bilinear_at(grid, rows, cols, x0, y0, dx, dy, cx, cy, nodata);

        distances_m.push(dist);
        xs.push(cx);
        ys.push(cy);
        center_values.push(cval);

        if cross_samples <= 1 {
            swath_mean.push(cval);
            swath_min.push(cval);
            swath_max.push(cval);
            swath_std.push(0.0);
        } else {
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            let mut vmin = f64::INFINITY;
            let mut vmax = f64::NEG_INFINITY;
            let mut cnt = 0usize;
            for j in 0..cross_samples {
                let s = -half_w + (2.0 * half_w) * (j as f64) / ((cross_samples - 1) as f64);
                let sx = cx + s * nx;
                let sy = cy + s * ny;
                let val = sample_bilinear_at(grid, rows, cols, x0, y0, dx, dy, sx, sy, nodata);
                if val.is_finite() {
                    sum += val;
                    sum_sq += val * val;
                    if val < vmin {
                        vmin = val;
                    }
                    if val > vmax {
                        vmax = val;
                    }
                    cnt += 1;
                }
            }
            if cnt > 0 {
                let mean = sum / (cnt as f64);
                let var = (sum_sq / (cnt as f64) - mean * mean).max(0.0);
                swath_mean.push(mean);
                swath_min.push(vmin);
                swath_max.push(vmax);
                swath_std.push(var.sqrt());
            } else {
                swath_mean.push(f64::NAN);
                swath_min.push(f64::NAN);
                swath_max.push(f64::NAN);
                swath_std.push(f64::NAN);
            }
        }
    }

    SwathProfileResult {
        distances_m,
        xs,
        ys,
        center_values,
        swath_mean,
        swath_min,
        swath_max,
        swath_std,
    }
}

use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct PlaneAttitudeResult {
    pub strike_deg: f64,
    pub dip_deg: f64,
    pub dip_dir_deg: f64,
    pub dz_dx: f64,
    pub dz_dy: f64,
    pub intercept_c: f64,
    pub r_squared: f64,
    pub rmse_m: f64,
    pub collinearity_ratio: f64,
    pub centroid_x: f64,
    pub centroid_y: f64,
    pub centroid_z: f64,
    pub relief_m: f64,
    pub n_points: usize,
}

fn solve_ols_plane(pts: &[(f64, f64, f64)]) -> Option<(f64, f64, f64, f64, f64, f64, f64)> {
    let n = pts.len();
    if n < 3 {
        return None;
    }
    let inv_n = 1.0 / (n as f64);
    let mut mx = 0.0f64;
    let mut my = 0.0f64;
    let mut mz = 0.0f64;
    for &(x, y, z) in pts {
        mx += x;
        my += y;
        mz += z;
    }
    mx *= inv_n;
    my *= inv_n;
    mz *= inv_n;

    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    let mut sxy = 0.0f64;
    let mut sxz = 0.0f64;
    let mut syz = 0.0f64;
    for &(x, y, z) in pts {
        let dx = x - mx;
        let dy = y - my;
        let dz = z - mz;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
        sxz += dx * dz;
        syz += dy * dz;
    }

    let det = sxx * syy - sxy * sxy;
    if det.abs() < 1e-12 {
        return None;
    }
    let a = (syy * sxz - sxy * syz) / det;
    let b = (sxx * syz - sxy * sxz) / det;
    let c = mz - a * mx - b * my;

    // Eigenvalues of 2x2 horizontal covariance [[sxx, sxy], [sxy, syy]] for collinearity check:
    let trace = sxx + syy;
    let disc = ((sxx - syy).powi(2) + 4.0 * sxy * sxy).sqrt();
    let l_max = 0.5 * (trace + disc);
    let l_min = (0.5 * (trace - disc)).max(0.0);
    let col_ratio = if l_max > 1e-12 {
        (l_min / l_max).sqrt()
    } else {
        0.0
    };

    Some((a, b, c, mx, my, mz, col_ratio))
}

/// Fit a geological plane `z(x, y) = a*x + b*y + c` to 3D contact coordinates `(xs, ys, zs)`
/// and return Right-Hand-Rule `strike_deg` (`[0, 360)`), `dip_deg` (`[0, 90]`), and `dip_dir_deg` (`[0, 360)`).
pub fn estimate_bed_attitude(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    robust_trim: bool,
) -> Option<PlaneAttitudeResult> {
    let m = xs.len().min(ys.len()).min(zs.len());
    let mut valid_pts: Vec<(f64, f64, f64)> = Vec::with_capacity(m);
    let mut z_min = f64::INFINITY;
    let mut z_max = f64::NEG_INFINITY;
    for i in 0..m {
        if xs[i].is_finite() && ys[i].is_finite() && zs[i].is_finite() {
            valid_pts.push((xs[i], ys[i], zs[i]));
            if zs[i] < z_min {
                z_min = zs[i];
            }
            if zs[i] > z_max {
                z_max = zs[i];
            }
        }
    }
    if valid_pts.len() < 3 {
        return None;
    }

    let (mut a, mut b, mut c, mut mx, mut my, mut mz, mut col_ratio) =
        solve_ols_plane(&valid_pts)?;

    // Optional iterative outlier trimming (keeps >= 75% of points with smallest vertical residuals)
    let mut active_pts = valid_pts;
    if robust_trim && active_pts.len() >= 8 {
        for _ in 0..2 {
            let mut residuals: Vec<(f64, (f64, f64, f64))> = active_pts
                .iter()
                .map(|&(x, y, z)| {
                    let r = (z - (a * x + b * y + c)).abs();
                    (r, (x, y, z))
                })
                .collect();
            residuals.sort_by(|u, v| u.0.partial_cmp(&v.0).unwrap_or(std::cmp::Ordering::Equal));
            // Only trim if residual spread is non-trivial
            if residuals.last().map(|x| x.0).unwrap_or(0.0) > 0.25 {
                let keep_n = ((active_pts.len() * 8) / 10).max(4);
                let trimmed: Vec<(f64, f64, f64)> =
                    residuals.into_iter().take(keep_n).map(|x| x.1).collect();
                if let Some(sol) = solve_ols_plane(&trimmed) {
                    a = sol.0;
                    b = sol.1;
                    c = sol.2;
                    mx = sol.3;
                    my = sol.4;
                    mz = sol.5;
                    col_ratio = sol.6;
                    active_pts = trimmed;
                }
            }
        }
    }

    let mut ss_res = 0.0f64;
    let mut ss_tot = 0.0f64;
    for &(x, y, z) in &active_pts {
        let z_hat = a * x + b * y + c;
        let err = z - z_hat;
        let dev = z - mz;
        ss_res += err * err;
        ss_tot += dev * dev;
    }
    let rmse_m = (ss_res / (active_pts.len() as f64)).sqrt();
    let r_squared = if ss_tot > 1e-12 {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    } else if ss_res < 1e-10 {
        1.0
    } else {
        0.0
    };

    let grad = (a * a + b * b).sqrt();
    let dip_deg = grad.atan() * (180.0 / PI);
    let (dip_dir_deg, strike_deg) = if grad < 1e-12 {
        (0.0, 0.0)
    } else {
        // Steepest descent vector in horizontal (East, North) is (-a, -b)
        let mut dd = (-a).atan2(-b) * (180.0 / PI);
        if dd < 0.0 {
            dd += 360.0;
        }
        // Right-Hand Rule: strike is 90 degrees counter-clockwise from dip direction
        let mut st = dd - 90.0;
        if st < 0.0 {
            st += 360.0;
        }
        (dd, st)
    };

    Some(PlaneAttitudeResult {
        strike_deg,
        dip_deg,
        dip_dir_deg,
        dz_dx: a,
        dz_dy: b,
        intercept_c: c,
        r_squared,
        rmse_m,
        collinearity_ratio: col_ratio,
        centroid_x: mx,
        centroid_y: my,
        centroid_z: mz,
        relief_m: (z_max - z_min).max(0.0),
        n_points: active_pts.len(),
    })
}

/// Compute apparent dip (degrees, signed: > 0 dipping forward along section azimuth, < 0 dipping backward)
/// given true `dip_deg`, `dip_dir_deg` (clockwise from North), and `section_azimuth_deg` (clockwise from North).
pub fn compute_apparent_dip_deg(dip_deg: f64, dip_dir_deg: f64, section_azimuth_deg: f64) -> f64 {
    let dip_rad = dip_deg * (PI / 180.0);
    let delta_az = (dip_dir_deg - section_azimuth_deg) * (PI / 180.0);
    (dip_rad.tan() * delta_az.cos()).atan() * (180.0 / PI)
}

#[derive(Debug, Clone)]
pub struct ContactAttitudeRecord {
    pub unit_a: i32,
    pub unit_b: i32,
    pub attitude: PlaneAttitudeResult,
}

/// Slide spatial windows across mapped unit contacts in `labels` + `dem` and estimate local bed attitudes
/// wherever topographic relief and horizontal spread (V-rule) are sufficient.
#[allow(clippy::too_many_arguments)]
pub fn estimate_contact_attitudes_grid(
    labels: &[i32],
    dem: &[f64],
    rows: usize,
    cols: usize,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    window_m: f64,
    min_points: usize,
    min_relief_m: f64,
    min_r2: f64,
    min_collinearity: f64,
    nodata: f64,
) -> Vec<ContactAttitudeRecord> {
    let (_, contacts) = crate::mapping::extract_unit_contacts(labels, rows, cols, x0, y0, dx, dy);
    if contacts.is_empty() {
        return Vec::new();
    }

    // Group contact pixels by (unit_a, unit_b) and spatial window cell (wx, wy)
    let win = window_m.max(dx.abs() * 4.0);
    let xmin = x0.min(x0 + ((cols - 1) as f64) * dx);
    let ymin = y0.min(y0 + ((rows - 1) as f64) * dy);

    let mut groups: std::collections::BTreeMap<(i32, i32, isize, isize), Vec<(f64, f64, f64)>> =
        std::collections::BTreeMap::new();

    for cp in contacts {
        let z = dem[cp.row * cols + cp.col];
        if !is_valid(z, nodata) {
            continue;
        }
        let wx = ((cp.x - xmin) / win).floor() as isize;
        let wy = ((cp.y - ymin) / win).floor() as isize;
        groups
            .entry((cp.unit_a, cp.unit_b, wx, wy))
            .or_default()
            .push((cp.x, cp.y, z));
    }

    let mut records = Vec::new();
    let req_pts = min_points.max(4);
    for ((ua, ub, _, _), pts) in groups {
        if pts.len() < req_pts {
            continue;
        }
        let xs: Vec<f64> = pts.iter().map(|p| p.0).collect();
        let ys: Vec<f64> = pts.iter().map(|p| p.1).collect();
        let zs: Vec<f64> = pts.iter().map(|p| p.2).collect();
        if let Some(att) = estimate_bed_attitude(&xs, &ys, &zs, true) {
            if att.relief_m >= min_relief_m
                && att.r_squared >= min_r2
                && att.collinearity_ratio >= min_collinearity
                && att.dip_deg <= 88.0
            {
                records.push(ContactAttitudeRecord {
                    unit_a: ua,
                    unit_b: ub,
                    attitude: att,
                });
            }
        }
    }

    records
}
