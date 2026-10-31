// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
use rayon::prelude::*;
use std::collections::VecDeque;
use std::f64::consts::PI;

#[inline]
fn is_valid(v: f64, nodata: f64) -> bool {
    v.is_finite() && (nodata.is_nan() || (v - nodata).abs() > 1e-9)
}

/// Compute DEM of Difference (DoD) or raster difference: t2 - t1.
pub fn compute_dod(t1: &[f64], t2: &[f64], nodata: f64) -> Vec<f64> {
    let n = t1.len().min(t2.len());
    let mut out = vec![f64::NAN; n];
    out.par_iter_mut().enumerate().for_each(|(i, val)| {
        let z1 = t1[i];
        let z2 = t2[i];
        if is_valid(z1, nodata) && is_valid(z2, nodata) {
            *val = z2 - z1;
        }
    });
    out
}

/// Solve a 3x3 linear system A * x = b using Gaussian elimination with partial pivoting.
fn solve_3x3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for i in 0..3 {
        let mut max_row = i;
        let mut max_val = a[i][i].abs();
        for r in (i + 1)..3 {
            if a[r][i].abs() > max_val {
                max_val = a[r][i].abs();
                max_row = r;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }
        let piv = a[i][i];
        for r in (i + 1)..3 {
            let factor = a[r][i] / piv;
            for c in i..3 {
                a[r][c] -= factor * a[i][c];
            }
            b[r] -= factor * b[i];
        }
    }
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        let mut s = b[i];
        for c in (i + 1)..3 {
            s -= a[i][c] * x[c];
        }
        x[i] = s / a[i][i];
    }
    Some(x)
}

/// Fit and remove a planar tilt / vertical bias z_bias(x, y) = c0 + cx*x + cy*y
/// estimated over stable pixels (`stable_mask[i] != 0`).
/// Coordinates x = c * dx, y = r * dy relative to top-left corner.
pub fn detrend_dod_planar(
    dod: &[f64],
    stable_mask: &[u8],
    rows: usize,
    cols: usize,
    dx: f64,
    dy: f64,
) -> (Vec<f64>, f64, f64, f64) {
    let n = rows * cols;
    if dod.len() < n || stable_mask.len() < n {
        return (dod.to_vec(), 0.0, 0.0, 0.0);
    }

    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    let mut count = 0usize;

    for r in 0..rows {
        let y = (r as f64) * dy;
        for c in 0..cols {
            let idx = r * cols + c;
            let z = dod[idx];
            if stable_mask[idx] != 0 && z.is_finite() {
                let x = (c as f64) * dx;
                let p = [1.0, x, y];
                for i in 0..3 {
                    atb[i] += p[i] * z;
                    for j in 0..3 {
                        ata[i][j] += p[i] * p[j];
                    }
                }
                count += 1;
            }
        }
    }

    let coeffs = if count >= 3 {
        solve_3x3(ata, atb).unwrap_or([0.0, 0.0, 0.0])
    } else {
        [0.0, 0.0, 0.0]
    };
    let (c0, cx, cy) = (coeffs[0], coeffs[1], coeffs[2]);

    let mut out = vec![f64::NAN; n];
    out.par_chunks_mut(cols)
        .enumerate()
        .for_each(|(r, out_row)| {
            let y = (r as f64) * dy;
            for c in 0..cols {
                let z = dod[r * cols + c];
                if z.is_finite() {
                    let x = (c as f64) * dx;
                    out_row[c] = z - (c0 + cx * x + cy * y);
                }
            }
        });

    (out, c0, cx, cy)
}

/// Estimate horizontal & vertical co-registration shift (dx_shift_m, dy_shift_m, dz_shift_m)
/// using Nuth & Kääb (2011): dz = dx_shift * tan(slope) * sin(aspect) + dy_shift * tan(slope) * cos(aspect) + dz_shift.
pub fn estimate_nuth_kaab_shift(
    dod: &[f64],
    slope_deg: &[f64],
    aspect_deg: &[f64],
    stable_mask: &[u8],
) -> (f64, f64, f64) {
    let n = dod
        .len()
        .min(slope_deg.len())
        .min(aspect_deg.len())
        .min(stable_mask.len());
    let mut ata = [[0.0f64; 3]; 3];
    let mut atb = [0.0f64; 3];
    let mut count = 0usize;

    for i in 0..n {
        if stable_mask[i] == 0 {
            continue;
        }
        let dz = dod[i];
        let s = slope_deg[i];
        let a = aspect_deg[i];
        if !dz.is_finite() || !s.is_finite() || !a.is_finite() || s < 2.0 || a < 0.0 {
            continue;
        }
        let tan_s = (s * PI / 180.0).tan();
        let a_rad = a * PI / 180.0;
        let p = [tan_s * a_rad.sin(), tan_s * a_rad.cos(), 1.0];
        for r in 0..3 {
            atb[r] += p[r] * dz;
            for c in 0..3 {
                ata[r][c] += p[r] * p[c];
            }
        }
        count += 1;
    }

    if count < 10 {
        return (0.0, 0.0, 0.0);
    }
    if let Some(sol) = solve_3x3(ata, atb) {
        (sol[0], sol[1], sol[2])
    } else {
        (0.0, 0.0, 0.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VolumeBudget {
    pub eroded_pixels: usize,
    pub deposited_pixels: usize,
    pub stable_pixels: usize,
    pub eroded_area_m2: f64,
    pub deposited_area_m2: f64,
    pub eroded_volume_m3: f64,
    pub deposited_volume_m3: f64,
    pub net_volume_m3: f64,
    pub volume_uncertainty_m3: f64,
    pub mean_dz_m: f64,
    pub max_erosion_m: f64,
    pub max_deposition_m: f64,
}

/// Integrate eroded and deposited area and volume above level of detection `lod_threshold` (m),
/// and propagate uncorrelated vertical uncertainty `sigma_dz` (m).
pub fn compute_volume_budget(
    dod: &[f64],
    cell_area_m2: f64,
    lod_threshold: f64,
    sigma_dz: f64,
) -> VolumeBudget {
    let lod = lod_threshold.abs();
    let mut eroded_pixels = 0usize;
    let mut deposited_pixels = 0usize;
    let mut stable_pixels = 0usize;
    let mut eroded_sum = 0.0f64;
    let mut deposited_sum = 0.0f64;
    let mut valid_sum = 0.0f64;
    let mut valid_count = 0usize;
    let mut max_erosion = 0.0f64;
    let mut max_deposition = 0.0f64;

    for &dz in dod {
        if !dz.is_finite() {
            continue;
        }
        valid_sum += dz;
        valid_count += 1;
        if dz < -lod {
            eroded_pixels += 1;
            let depth = -dz;
            eroded_sum += depth;
            if depth > max_erosion {
                max_erosion = depth;
            }
        } else if dz > lod {
            deposited_pixels += 1;
            deposited_sum += dz;
            if dz > max_deposition {
                max_deposition = dz;
            }
        } else {
            stable_pixels += 1;
        }
    }

    let eroded_area_m2 = (eroded_pixels as f64) * cell_area_m2;
    let deposited_area_m2 = (deposited_pixels as f64) * cell_area_m2;
    let eroded_volume_m3 = eroded_sum * cell_area_m2;
    let deposited_volume_m3 = deposited_sum * cell_area_m2;
    let net_volume_m3 = deposited_volume_m3 - eroded_volume_m3;
    let active_pixels = eroded_pixels + deposited_pixels;
    let volume_uncertainty_m3 =
        (active_pixels as f64).sqrt() * sigma_dz.abs() * cell_area_m2;
    let mean_dz_m = if valid_count > 0 {
        valid_sum / (valid_count as f64)
    } else {
        0.0
    };

    VolumeBudget {
        eroded_pixels,
        deposited_pixels,
        stable_pixels,
        eroded_area_m2,
        deposited_area_m2,
        eroded_volume_m3,
        deposited_volume_m3,
        net_volume_m3,
        volume_uncertainty_m3,
        mean_dz_m,
        max_erosion_m: max_erosion,
        max_deposition_m: max_deposition,
    }
}

/// Compute Otsu threshold maximizing inter-class variance over finite pixels.
pub fn otsu_threshold(values: &[f64], n_bins: usize) -> f64 {
    let bins = n_bins.max(16);
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    let mut count = 0usize;
    for &v in values {
        if v.is_finite() {
            if v < vmin {
                vmin = v;
            }
            if v > vmax {
                vmax = v;
            }
            count += 1;
        }
    }
    if count == 0 || (vmax - vmin).abs() < 1e-12 {
        return if vmin.is_finite() { vmin } else { 0.0 };
    }

    let mut hist = vec![0usize; bins];
    let scale = (bins as f64) / (vmax - vmin);
    for &v in values {
        if v.is_finite() {
            let mut b = ((v - vmin) * scale).floor() as isize;
            if b < 0 {
                b = 0;
            }
            if b >= bins as isize {
                b = (bins as isize) - 1;
            }
            hist[b as usize] += 1;
        }
    }

    let total = count as f64;
    let mut sum_total = 0.0f64;
    for (i, &h) in hist.iter().enumerate() {
        let bin_center = vmin + ((i as f64) + 0.5) / scale;
        sum_total += (h as f64) * bin_center;
    }

    let mut weight_b = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut max_var = -1.0f64;
    let mut best_sum_thresh = 0.0f64;
    let mut best_count = 0usize;

    for (i, &h) in hist.iter().enumerate() {
        weight_b += h as f64;
        if weight_b == 0.0 {
            continue;
        }
        let weight_f = total - weight_b;
        if weight_f <= 0.0 {
            break;
        }
        let bin_center = vmin + ((i as f64) + 0.5) / scale;
        sum_b += (h as f64) * bin_center;
        let mean_b = sum_b / weight_b;
        let mean_f = (sum_total - sum_b) / weight_f;
        let diff = mean_b - mean_f;
        let between_var = weight_b * weight_f * diff * diff;
        let thr_i = vmin + ((i as f64) + 1.0) / scale;
        if between_var > max_var + 1e-12 {
            max_var = between_var;
            best_sum_thresh = thr_i;
            best_count = 1;
        } else if (between_var - max_var).abs() <= 1e-12 {
            best_sum_thresh += thr_i;
            best_count += 1;
        }
    }

    if best_count > 0 {
        best_sum_thresh / (best_count as f64)
    } else {
        0.5 * (vmin + vmax)
    }
}

/// Classify continuous change values into discrete severity classes 0..=thresholds.len() (255 for NaN).
pub fn classify_change_severity(values: &[f64], thresholds: &[f64]) -> Vec<u8> {
    let mut out = vec![255u8; values.len()];
    out.par_iter_mut().enumerate().for_each(|(i, cls)| {
        let v = values[i];
        if v.is_finite() {
            let mut c = 0u8;
            for &t in thresholds {
                if v >= t {
                    c = c.saturating_add(1);
                } else {
                    break;
                }
            }
            *cls = c;
        }
    });
    out
}

#[derive(Debug, Clone)]
pub struct PatchStat {
    pub patch_id: i32,
    pub pixel_count: usize,
    pub area_m2: f64,
    pub mean_magnitude: f64,
    pub max_magnitude: f64,
    pub integrated_volume_or_sum: f64,
    pub centroid_x: f64,
    pub centroid_y: f64,
    pub bbox_r0: usize,
    pub bbox_c0: usize,
    pub bbox_r1: usize,
    pub bbox_c1: usize,
}

/// 8-connected component labeling and per-patch statistics for binary change mask.
/// `x0, y0` are the center coordinates of pixel (0, 0); `dx` (>0 East) and `dy` (<0 South or >0) are pixel spacings.
pub fn connected_change_patches(
    mask: &[u8],
    magnitude: &[f64],
    rows: usize,
    cols: usize,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    min_pixels: usize,
) -> (Vec<i32>, Vec<PatchStat>) {
    let n = rows * cols;
    let mut labels = vec![0i32; n];
    if rows == 0 || cols == 0 || mask.len() < n || magnitude.len() < n {
        return (labels, Vec::new());
    }

    let cell_area = (dx * dy).abs();
    let mut visited = vec![false; n];
    let mut raw_patches: Vec<(Vec<usize>, PatchStat)> = Vec::new();
    let mut queue = VecDeque::new();
    let min_px = min_pixels.max(1);

    for r in 0..rows {
        for c in 0..cols {
            let start = r * cols + c;
            if mask[start] == 0 || visited[start] {
                continue;
            }
            visited[start] = true;
            queue.push_back((r, c));

            let mut pixels = Vec::new();
            let mut sum_mag = 0.0f64;
            let mut max_mag = f64::NEG_INFINITY;
            let mut valid_mag_count = 0usize;
            let mut sum_r = 0.0f64;
            let mut sum_c = 0.0f64;
            let mut r0 = r;
            let mut r1 = r;
            let mut c0 = c;
            let mut c1 = c;

            while let Some((cr, cc)) = queue.pop_front() {
                let idx = cr * cols + cc;
                pixels.push(idx);
                sum_r += cr as f64;
                sum_c += cc as f64;
                if cr < r0 {
                    r0 = cr;
                }
                if cr > r1 {
                    r1 = cr;
                }
                if cc < c0 {
                    c0 = cc;
                }
                if cc > c1 {
                    c1 = cc;
                }
                let m = magnitude[idx];
                if m.is_finite() {
                    sum_mag += m;
                    if m.abs() > max_mag {
                        max_mag = m.abs();
                    }
                    valid_mag_count += 1;
                }

                let ri = cr as isize;
                let ci = cc as isize;
                for dr in -1..=1 {
                    for dc in -1..=1 {
                        if dr == 0 && dc == 0 {
                            continue;
                        }
                        let nr = ri + dr;
                        let nc = ci + dc;
                        if nr >= 0 && nr < (rows as isize) && nc >= 0 && nc < (cols as isize) {
                            let nidx = (nr as usize) * cols + (nc as usize);
                            if !visited[nidx] && mask[nidx] != 0 {
                                visited[nidx] = true;
                                queue.push_back((nr as usize, nc as usize));
                            }
                        }
                    }
                }
            }

            if pixels.len() >= min_px {
                let px_count = pixels.len();
                let mean_r = sum_r / (px_count as f64);
                let mean_c = sum_c / (px_count as f64);
                let cx = x0 + mean_c * dx;
                let cy = y0 + mean_r * dy;
                let mean_mag = if valid_mag_count > 0 {
                    sum_mag / (valid_mag_count as f64)
                } else {
                    0.0
                };
                let max_m = if max_mag.is_finite() { max_mag } else { 0.0 };
                let stat = PatchStat {
                    patch_id: 0,
                    pixel_count: px_count,
                    area_m2: (px_count as f64) * cell_area,
                    mean_magnitude: mean_mag,
                    max_magnitude: max_m,
                    integrated_volume_or_sum: sum_mag * cell_area,
                    centroid_x: cx,
                    centroid_y: cy,
                    bbox_r0: r0,
                    bbox_c0: c0,
                    bbox_r1: r1,
                    bbox_c1: c1,
                };
                raw_patches.push((pixels, stat));
            }
        }
    }

    // Sort patches by pixel_count descending so patch 1 is the largest change feature
    raw_patches.sort_by(|a, b| b.1.pixel_count.cmp(&a.1.pixel_count));

    let mut stats = Vec::with_capacity(raw_patches.len());
    for (i, (pixels, mut stat)) in raw_patches.into_iter().enumerate() {
        let pid = (i + 1) as i32;
        stat.patch_id = pid;
        for idx in pixels {
            labels[idx] = pid;
        }
        stats.push(stat);
    }

    (labels, stats)
}

/// Output of `compute_displacement_strain`.
#[derive(Debug, Clone)]
pub struct DisplacementStrainResult {
    pub du_dx: Vec<f64>,
    pub du_dy: Vec<f64>,
    pub gradient_mag: Vec<f64>,
    pub microstrain: Vec<f64>,
    pub scarp_curvature: Vec<f64>,
    pub gradient_azimuth_deg: Vec<f64>,
}

/// Compute 2D spatial displacement gradients (`du/dx`, `du/dy`), strain magnitude `|grad u|` (m/m),
/// microstrain (`1e6 * |grad u|`), Laplacian fault-scarp curvature (`d2u/dx2 + d2u/dy2`), and
/// gradient azimuth (`[0, 360)` degrees clockwise from North) in parallel across rows.
pub fn compute_displacement_strain(
    disp: &[f64],
    rows: usize,
    cols: usize,
    dx: f64,
    dy: f64,
    nodata: f64,
) -> DisplacementStrainResult {
    let n = rows * cols;
    let mut du_dx = vec![f64::NAN; n];
    let mut du_dy = vec![f64::NAN; n];
    let mut gradient_mag = vec![f64::NAN; n];
    let mut microstrain = vec![f64::NAN; n];
    let mut scarp_curvature = vec![f64::NAN; n];
    let mut gradient_azimuth_deg = vec![f64::NAN; n];

    if rows < 2 || cols < 2 || disp.len() < n {
        return DisplacementStrainResult {
            du_dx,
            du_dy,
            gradient_mag,
            microstrain,
            scarp_curvature,
            gradient_azimuth_deg,
        };
    }

    let abs_dx = dx.abs().max(1e-9);
    let abs_dy = dy.abs().max(1e-9);
    let dy_sign = if dy < 0.0 { -1.0 } else { 1.0 };

    du_dx
        .par_chunks_mut(cols)
        .zip(du_dy.par_chunks_mut(cols))
        .zip(gradient_mag.par_chunks_mut(cols))
        .zip(microstrain.par_chunks_mut(cols))
        .zip(scarp_curvature.par_chunks_mut(cols))
        .zip(gradient_azimuth_deg.par_chunks_mut(cols))
        .enumerate()
        .for_each(|(r, (((((gx_row, gy_row), gm_row), ms_row), cur_row), az_row))| {
            let r_prev = r.saturating_sub(1);
            let r_next = (r + 1).min(rows - 1);
            let span_y = ((r_next - r_prev).max(1) as f64) * abs_dy;

            for c in 0..cols {
                let u_c = disp[r * cols + c];
                if !is_valid(u_c, nodata) {
                    continue;
                }
                let c_prev = c.saturating_sub(1);
                let c_next = (c + 1).min(cols - 1);
                let span_x = ((c_next - c_prev).max(1) as f64) * abs_dx;

                let u_w = disp[r * cols + c_prev];
                let u_e = disp[r * cols + c_next];
                let u_rprev = disp[r_prev * cols + c];
                let u_rnext = disp[r_next * cols + c];

                if !is_valid(u_w, nodata)
                    || !is_valid(u_e, nodata)
                    || !is_valid(u_rprev, nodata)
                    || !is_valid(u_rnext, nodata)
                {
                    continue;
                }

                // Easting derivative du/dx and Northing derivative du/dy
                let dudx = (u_e - u_w) / span_x;
                let dudy = ((u_rnext - u_rprev) / span_y) * dy_sign;
                let gmag = dudx.hypot(dudy);
                let d2x = (u_e - 2.0 * u_c + u_w) / (abs_dx * abs_dx);
                let d2y = (u_rnext - 2.0 * u_c + u_rprev) / (abs_dy * abs_dy);
                let az = (dudx.atan2(dudy) * 180.0 / PI).rem_euclid(360.0);

                gx_row[c] = dudx;
                gy_row[c] = dudy;
                gm_row[c] = gmag;
                ms_row[c] = gmag * 1.0e6;
                cur_row[c] = d2x + d2y;
                az_row[c] = az;
            }
        });

    DisplacementStrainResult {
        du_dx,
        du_dy,
        gradient_mag,
        microstrain,
        scarp_curvature,
        gradient_azimuth_deg,
    }
}

fn finite_percentile(vals: &[f64], q: f64) -> f64 {
    let mut clean: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
    if clean.is_empty() {
        return 0.0;
    }
    clean.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = (q.clamp(0.0, 100.0) / 100.0) * ((clean.len() - 1) as f64);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - (lo as f64);
    clean[lo] * (1.0 - frac) + clean[hi] * frac
}

fn finite_mean(vals: &[f64]) -> f64 {
    let mut s = 0.0f64;
    let mut cnt = 0usize;
    for &v in vals {
        if v.is_finite() {
            s += v;
            cnt += 1;
        }
    }
    if cnt > 0 { s / (cnt as f64) } else { 0.0 }
}

/// Standardized physical & dimensionless summary signature for a surface-change regime.
#[derive(Debug, Clone)]
pub struct MultiHazardSignature {
    pub total_area_km2: f64,
    pub max_abs_dz_or_disp_m: f64,
    pub p95_abs_dz_or_disp_m: f64,
    pub eroded_volume_m3: f64,
    pub deposited_volume_m3: f64,
    pub active_elevation_area_km2: f64,
    pub max_strain_m_per_m: f64,
    pub p99_strain_m_per_m: f64,
    pub high_strain_area_km2: f64,
    pub mean_d_ndvi: f64,
    pub p05_d_ndvi: f64,
    pub p95_d_ndvi: f64,
    pub veg_loss_area_km2: f64,
    pub veg_gain_area_km2: f64,
    pub mean_d_nbr: f64,
    pub p95_d_nbr: f64,
    pub burned_area_km2: f64,
    pub mean_d_ndwi: f64,
    pub p95_d_ndwi: f64,
    pub inundated_area_km2: f64,
    pub mean_d_ndmi: f64,
    pub p95_d_ndmi: f64,
    pub wetted_area_km2: f64,
    pub mean_d_ndbi: f64,
    pub p95_d_ndbi: f64,
    pub built_expansion_area_km2: f64,
    pub mean_d_bsi: f64,
    pub p95_d_bsi: f64,
    pub bare_soil_expansion_area_km2: f64,
    pub mean_d_cwt: f64,
    pub p95_abs_d_cwt: f64,
    pub mean_emb_cosine_dist: f64,
    pub p95_emb_cosine_dist: f64,
    /// 6-axis dimensionless process fingerprint in `[0, 1]`:
    /// `[tectonic_strain, mass_wasting, flood_inundation, wildfire_burn, agri_urban, cwt_roughness]`
    pub fingerprint_scores: [f64; 6],
    pub dominant_regime: String,
}

/// Evaluate physical area/volume change budgets and a standardized 6-axis dimensionless
/// process fingerprint in `[0, 1]` across multi-modal change rasters in Rust.
#[allow(clippy::too_many_arguments)]
pub fn compute_multi_hazard_signature(
    dz_or_disp: &[f64],
    strain_mag: &[f64],
    d_ndvi: &[f64],
    d_nbr: &[f64],
    d_ndwi: &[f64],
    d_ndmi: &[f64],
    d_ndbi: &[f64],
    d_bsi: &[f64],
    d_cwt: &[f64],
    emb_cos_dist: &[f64],
    cell_area_m2: f64,
    lod_m: f64,
    strain_threshold: f64,
) -> MultiHazardSignature {
    let n = dz_or_disp.len();
    let total_area_km2 = (n as f64) * cell_area_m2 * 1.0e-6;
    let area_scale_km2 = cell_area_m2 * 1.0e-6;

    let abs_dz: Vec<f64> = dz_or_disp
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .collect();
    let max_abs_dz_or_disp_m = abs_dz.iter().copied().fold(0.0f64, f64::max);
    let p95_abs_dz_or_disp_m = finite_percentile(&abs_dz, 95.0);

    let vbud = compute_volume_budget(dz_or_disp, cell_area_m2, lod_m, 0.15);
    let active_elevation_area_km2 = (vbud.eroded_area_m2 + vbud.deposited_area_m2) * 1.0e-6;

    let max_strain_m_per_m = strain_mag
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(0.0f64, f64::max);
    let p99_strain_m_per_m = finite_percentile(strain_mag, 99.0);
    let high_strain_px = strain_mag
        .iter()
        .filter(|&&v| v.is_finite() && v >= strain_threshold)
        .count();
    let high_strain_area_km2 = (high_strain_px as f64) * area_scale_km2;

    let mean_d_ndvi = finite_mean(d_ndvi);
    let p05_d_ndvi = finite_percentile(d_ndvi, 5.0);
    let p95_d_ndvi = finite_percentile(d_ndvi, 95.0);
    let veg_loss_px = d_ndvi
        .iter()
        .filter(|&&v| v.is_finite() && v <= -0.15)
        .count();
    let veg_gain_px = d_ndvi
        .iter()
        .filter(|&&v| v.is_finite() && v >= 0.15)
        .count();
    let veg_loss_area_km2 = (veg_loss_px as f64) * area_scale_km2;
    let veg_gain_area_km2 = (veg_gain_px as f64) * area_scale_km2;

    let mean_d_nbr = finite_mean(d_nbr);
    let p95_d_nbr = finite_percentile(d_nbr, 95.0);
    let burned_px = d_nbr
        .iter()
        .filter(|&&v| v.is_finite() && v >= 0.27)
        .count();
    let burned_area_km2 = (burned_px as f64) * area_scale_km2;

    let mean_d_ndwi = finite_mean(d_ndwi);
    let p95_d_ndwi = finite_percentile(d_ndwi, 95.0);
    let inundated_px = d_ndwi
        .iter()
        .filter(|&&v| v.is_finite() && v >= 0.12)
        .count();
    let inundated_area_km2 = (inundated_px as f64) * area_scale_km2;

    let mean_d_ndmi = finite_mean(d_ndmi);
    let p95_d_ndmi = finite_percentile(d_ndmi, 95.0);
    let wetted_px = d_ndmi
        .iter()
        .filter(|&&v| v.is_finite() && v >= 0.10)
        .count();
    let wetted_area_km2 = (wetted_px as f64) * area_scale_km2;

    let mean_d_ndbi = finite_mean(d_ndbi);
    let p95_d_ndbi = finite_percentile(d_ndbi, 95.0);
    let built_px = d_ndbi
        .iter()
        .filter(|&&v| v.is_finite() && v >= 0.08)
        .count();
    let built_expansion_area_km2 = (built_px as f64) * area_scale_km2;

    let mean_d_bsi = finite_mean(d_bsi);
    let p95_d_bsi = finite_percentile(d_bsi, 95.0);
    let bsi_px = d_bsi
        .iter()
        .filter(|&&v| v.is_finite() && v >= 0.08)
        .count();
    let bare_soil_expansion_area_km2 = (bsi_px as f64) * area_scale_km2;

    let mean_d_cwt = finite_mean(d_cwt);
    let abs_cwt: Vec<f64> = d_cwt
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .collect();
    let p95_abs_d_cwt = finite_percentile(&abs_cwt, 95.0);

    let mean_emb_cosine_dist = finite_mean(emb_cos_dist);
    let p95_emb_cosine_dist = finite_percentile(emb_cos_dist, 95.0);

    // Standardized 6-axis dimensionless process scores in [0, 1]
    // 1. Tectonic fault-scarp strain score (high localized |grad u|, minimal areal canopy loss, moderate scarp height)
    let veg_stability = (1.0 - (veg_loss_area_km2 / total_area_km2.max(1e-6)) * 3.0).clamp(0.05, 1.0);
    let scarp_bound = (1.0 - ((p95_abs_dz_or_disp_m - 2.5).max(0.0) / 8.0)).clamp(0.1, 1.0);
    let tectonic_strain = ((p99_strain_m_per_m / 0.018).clamp(0.0, 1.0) * veg_stability * scarp_bound).clamp(0.0, 1.0);

    // 2. Mass-wasting / landslide volumetric + stripping score (large DoD (>3m) + bare earth stripping + CWT roughness)
    let dz_heavy = ((p95_abs_dz_or_disp_m - 2.2).max(0.0) / 12.0).clamp(0.0, 1.0);
    let strip_factor = ((-p05_d_ndvi) / 0.35).clamp(0.0, 1.0);
    let mass_wasting = (0.60 * dz_heavy + 0.40 * (dz_heavy * strip_factor).sqrt()).clamp(0.0, 1.0);

    // 3. Hydrologic flood inundation score (strong dNDWI water emergence + SWIR absorption gate: low dNDBI & low dNBR & low DoD)
    let ndwi_gain = ((p95_d_ndwi - 0.25).max(0.0) / 0.65).clamp(0.0, 1.0);
    let swir_water_gate = (1.0 - (p95_d_ndbi.max(0.0) / 0.14)).clamp(0.0, 1.0);
    let non_burn_water_gate = (1.0 - ((p95_d_nbr - 0.30).max(0.0) / 0.30)).clamp(0.0, 1.0);
    let low_relief_gate = (1.0 - ((p95_abs_dz_or_disp_m - 2.0).max(0.0) / 4.0)).clamp(0.0, 1.0);
    let flood_inundation = (ndwi_gain * swir_water_gate * non_burn_water_gate * low_relief_gate).clamp(0.0, 1.0);

    // 4. Pyrogenic wildfire burn score (high positive dNBR + widespread negative mean dNDVI + low elevation change)
    let topo_stability = (1.0 - (p95_abs_dz_or_disp_m / 3.5)).clamp(0.05, 1.0);
    let canopy_loss_gate = ((-mean_d_ndvi) / 0.18).clamp(0.0, 1.0);
    let wildfire_burn = ((p95_d_nbr / 0.52).clamp(0.0, 1.0) * canopy_loss_gate.sqrt() * topo_stability).clamp(0.0, 1.0);

    // 5. Anthropogenic agricultural + urban/quarry expansion score (positive dNDBI + positive crop dNDVI, gated against wildfire/landslide)
    let built_score = (p95_d_ndbi / 0.24).clamp(0.0, 1.0);
    let crop_greening = (p95_d_ndvi / 0.30).clamp(0.0, 1.0);
    let non_burn_gate = (1.0 - (mean_d_nbr.max(0.0) / 0.22)).clamp(0.1, 1.0);
    let non_slide_gate = (1.0 - ((p95_abs_dz_or_disp_m - 4.5).max(0.0) / 10.0)).clamp(0.1, 1.0);
    let agri_urban = ((0.52 * built_score + 0.48 * crop_greening) * crop_greening.sqrt().max(0.25) * non_burn_gate * non_slide_gate).clamp(0.0, 1.0);

    // 6. Multiscale 2D CWT topographic/texture complexity shift score
    let cwt_roughness = (p95_abs_d_cwt / 1.5).clamp(0.0, 1.0);

    let scores = [
        tectonic_strain,
        mass_wasting,
        flood_inundation,
        wildfire_burn,
        agri_urban,
        cwt_roughness,
    ];

    let regime_names = [
        "earthquake_deformation",
        "landslide_mass_wasting",
        "flood_inundation",
        "wildfire_burn",
        "agricultural_urban_change",
    ];
    let mut best_idx = 0usize;
    let mut best_val = scores[0];
    for i in 1..5 {
        if scores[i] > best_val {
            best_val = scores[i];
            best_idx = i;
        }
    }

    MultiHazardSignature {
        total_area_km2,
        max_abs_dz_or_disp_m,
        p95_abs_dz_or_disp_m,
        eroded_volume_m3: vbud.eroded_volume_m3,
        deposited_volume_m3: vbud.deposited_volume_m3,
        active_elevation_area_km2,
        max_strain_m_per_m,
        p99_strain_m_per_m,
        high_strain_area_km2,
        mean_d_ndvi,
        p05_d_ndvi,
        p95_d_ndvi,
        veg_loss_area_km2,
        veg_gain_area_km2,
        mean_d_nbr,
        p95_d_nbr,
        burned_area_km2,
        mean_d_ndwi,
        p95_d_ndwi,
        inundated_area_km2,
        mean_d_ndmi,
        p95_d_ndmi,
        wetted_area_km2,
        mean_d_ndbi,
        p95_d_ndbi,
        built_expansion_area_km2,
        mean_d_bsi,
        p95_d_bsi,
        bare_soil_expansion_area_km2,
        mean_d_cwt,
        p95_abs_d_cwt,
        mean_emb_cosine_dist,
        p95_emb_cosine_dist,
        fingerprint_scores: scores,
        dominant_regime: regime_names[best_idx].to_string(),
    }
}

/// Compute wrapped InSAR interferometric phase `[-pi, pi)` radians from a Line-of-Sight (LOS)
/// displacement field `los_disp_m` (meters) and radar wavelength `wavelength_m` (meters):
/// `dphi = wrap((4 * pi / wavelength_m) * u_los)`.
pub fn compute_insar_wrapped_phase(
    los_disp_m: &[f64],
    wavelength_m: f64,
    nodata: f64,
) -> Vec<f64> {
    let wl = wavelength_m.abs().max(1e-6);
    let scale = 4.0 * PI / wl;
    let two_pi = 2.0 * PI;
    los_disp_m
        .par_iter()
        .map(|&u| {
            if !is_valid(u, nodata) {
                f64::NAN
            } else {
                let raw = scale * u;
                (raw + PI).rem_euclid(two_pi) - PI
            }
        })
        .collect()
}

/// Output of `decompose_insar_3d`.
#[derive(Debug, Clone)]
pub struct Insar3DDecompositionResult {
    pub u_east: Vec<f64>,
    pub u_north: Vec<f64>,
    pub u_up: Vec<f64>,
    pub horiz_mag: Vec<f64>,
}

#[inline]
fn solve_3x3_sym(a: [[f64; 3]; 3], b: [f64; 3]) -> [f64; 3] {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    if det.abs() < 1e-12 {
        return [f64::NAN, f64::NAN, f64::NAN];
    }
    let inv_det = 1.0 / det;
    let x0 = (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]))
        * inv_det;
    let x1 = (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]))
        * inv_det;
    let x2 = (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        * inv_det;
    [x0, x1, x2]
}

/// Invert Ascending + Descending InSAR Line-of-Sight (LOS) displacement fields together with
/// regularized/interpolated 3-component GNSS priors `(gnss_e, gnss_n, gnss_u)` into a full 3D
/// coseismic surface displacement vector field `(u_east, u_north, u_up)` and horizontal magnitude
/// `horiz_mag = sqrt(u_east^2 + u_north^2)` in parallel across pixels.
#[allow(clippy::too_many_arguments)]
pub fn decompose_insar_3d(
    los_asc: &[f64],
    los_desc: &[f64],
    gnss_e: &[f64],
    gnss_n: &[f64],
    gnss_u: &[f64],
    s_asc: [f64; 3],
    s_desc: [f64; 3],
    gnss_weight_en: f64,
    gnss_weight_u: f64,
    nodata: f64,
) -> Insar3DDecompositionResult {
    let n = los_asc.len();
    let mut u_east = vec![f64::NAN; n];
    let mut u_north = vec![f64::NAN; n];
    let mut u_up = vec![f64::NAN; n];
    let mut horiz_mag = vec![f64::NAN; n];

    let w_en = gnss_weight_en.max(1e-6);
    let w_u = gnss_weight_u.max(1e-6);

    let mut ata = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            ata[i][j] = s_asc[i] * s_asc[j] + s_desc[i] * s_desc[j];
        }
    }
    ata[0][0] += w_en;
    ata[1][1] += w_en;
    ata[2][2] += w_u;

    u_east
        .par_iter_mut()
        .zip(u_north.par_iter_mut())
        .zip(u_up.par_iter_mut())
        .zip(horiz_mag.par_iter_mut())
        .enumerate()
        .for_each(|(idx, (((ue, un), uu), hm))| {
            let da = los_asc[idx];
            let dd = los_desc[idx];
            let ge = gnss_e[idx];
            let gn = gnss_n[idx];
            let gu = gnss_u[idx];
            if !is_valid(da, nodata)
                || !is_valid(dd, nodata)
                || !ge.is_finite()
                || !gn.is_finite()
                || !gu.is_finite()
            {
                return;
            }
            let rhs = [
                s_asc[0] * da + s_desc[0] * dd + w_en * ge,
                s_asc[1] * da + s_desc[1] * dd + w_en * gn,
                s_asc[2] * da + s_desc[2] * dd + w_u * gu,
            ];
            let sol = solve_3x3_sym(ata, rhs);
            *ue = sol[0];
            *un = sol[1];
            *uu = sol[2];
            *hm = sol[0].hypot(sol[1]);
        });

    Insar3DDecompositionResult {
        u_east,
        u_north,
        u_up,
        horiz_mag,
    }
}
