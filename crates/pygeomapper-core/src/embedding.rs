// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
use rayon::prelude::*;

#[inline]
fn is_valid(v: f64, nodata: f64) -> bool {
    v.is_finite() && (nodata.is_nan() || (v - nodata).abs() > 1e-9)
}

/// Compute pixel-wise cosine distance (1 - cos_sim in [0, 2]), cosine similarity ([-1, 1]),
/// and Euclidean L2 distance across D-dimensional embedding vectors (shape: n_pixels * dim).
pub fn embedding_change_metrics(
    emb_t1: &[f64],
    emb_t2: &[f64],
    n_pixels: usize,
    dim: usize,
    nodata: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut cos_dist = vec![f64::NAN; n_pixels];
    let mut cos_sim = vec![f64::NAN; n_pixels];
    let mut l2_dist = vec![f64::NAN; n_pixels];
    if n_pixels == 0 || dim == 0 || emb_t1.len() < n_pixels * dim || emb_t2.len() < n_pixels * dim {
        return (cos_dist, cos_sim, l2_dist);
    }

    cos_dist
        .par_iter_mut()
        .zip(cos_sim.par_iter_mut().zip(l2_dist.par_iter_mut()))
        .enumerate()
        .for_each(|(i, (cd, (cs, l2)))| {
            let base = i * dim;
            let u = &emb_t1[base..base + dim];
            let v = &emb_t2[base..base + dim];

            let mut dot = 0.0f64;
            let mut norm_u2 = 0.0f64;
            let mut norm_v2 = 0.0f64;
            let mut diff2 = 0.0f64;
            let mut ok = true;

            for d in 0..dim {
                let ud = u[d];
                let vd = v[d];
                if !is_valid(ud, nodata) || !is_valid(vd, nodata) {
                    ok = false;
                    break;
                }
                dot += ud * vd;
                norm_u2 += ud * ud;
                norm_v2 += vd * vd;
                let dv = vd - ud;
                diff2 += dv * dv;
            }

            if ok && norm_u2 > 1e-20 && norm_v2 > 1e-20 {
                let denom = (norm_u2 * norm_v2).sqrt();
                let sim = (dot / denom).clamp(-1.0, 1.0);
                *cs = sim;
                *cd = (1.0 - sim).clamp(0.0, 2.0);
                *l2 = diff2.sqrt();
            }
        });

    (cos_dist, cos_sim, l2_dist)
}

/// Compute per-dimension RMS difference, mean signed difference, sorted dimension indices (descending by RMS),
/// and 1st dominant change component projection across D-dimensional embeddings.
pub fn embedding_dimension_importance(
    emb_t1: &[f64],
    emb_t2: &[f64],
    n_pixels: usize,
    dim: usize,
    nodata: f64,
) -> (Vec<usize>, Vec<f64>, Vec<f64>, Vec<f64>) {
    if n_pixels == 0 || dim == 0 || emb_t1.len() < n_pixels * dim || emb_t2.len() < n_pixels * dim {
        return (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    }

    let mut sum_sq = vec![0.0f64; dim];
    let mut sum_signed = vec![0.0f64; dim];
    let mut valid_count = 0usize;

    for i in 0..n_pixels {
        let base = i * dim;
        let u = &emb_t1[base..base + dim];
        let v = &emb_t2[base..base + dim];
        let mut ok = true;
        for d in 0..dim {
            if !is_valid(u[d], nodata) || !is_valid(v[d], nodata) {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        valid_count += 1;
        for d in 0..dim {
            let dv = v[d] - u[d];
            sum_sq[d] += dv * dv;
            sum_signed[d] += dv;
        }
    }

    let denom = (valid_count.max(1)) as f64;
    let rms_per_dim: Vec<f64> = sum_sq.iter().map(|&s| (s / denom).sqrt()).collect();
    let mean_diff_per_dim: Vec<f64> = sum_signed.iter().map(|&s| s / denom).collect();

    let mut ranked_dims: Vec<usize> = (0..dim).collect();
    ranked_dims.sort_by(|&a, &b| {
        rms_per_dim[b]
            .partial_cmp(&rms_per_dim[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Compute unit-norm direction of dominant change via power iteration (8 steps) on centered differences:
    let mut dir = mean_diff_per_dim.clone();
    let mut dir_norm: f64 = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
    if dir_norm < 1e-12 {
        for (d, val) in dir.iter_mut().enumerate() {
            *val = rms_per_dim[d];
        }
        dir_norm = dir.iter().map(|x| x * x).sum::<f64>().sqrt();
    }
    if dir_norm > 1e-12 {
        for x in dir.iter_mut() {
            *x /= dir_norm;
        }
    }

    for _ in 0..6 {
        let mut next_dir = vec![0.0f64; dim];
        for i in 0..n_pixels {
            let base = i * dim;
            let u = &emb_t1[base..base + dim];
            let v = &emb_t2[base..base + dim];
            if !is_valid(u[0], nodata) || !is_valid(v[0], nodata) {
                continue;
            }
            let mut proj = 0.0f64;
            for d in 0..dim {
                let dv = (v[d] - u[d]) - mean_diff_per_dim[d];
                proj += dv * dir[d];
            }
            for d in 0..dim {
                let dv = (v[d] - u[d]) - mean_diff_per_dim[d];
                next_dir[d] += proj * dv;
            }
        }
        let nrm: f64 = next_dir.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nrm > 1e-12 {
            for d in 0..dim {
                dir[d] = next_dir[d] / nrm;
            }
        }
    }
    // Align sign of dominant change vector with mean difference vector
    let sign_dot: f64 = dir
        .iter()
        .zip(mean_diff_per_dim.iter())
        .map(|(a, b)| a * b)
        .sum();
    if sign_dot < 0.0 {
        for x in dir.iter_mut() {
            *x = -*x;
        }
    }

    let mut pc1_proj = vec![f64::NAN; n_pixels];
    pc1_proj
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out_val)| {
            let base = i * dim;
            let u = &emb_t1[base..base + dim];
            let v = &emb_t2[base..base + dim];
            if is_valid(u[0], nodata) && is_valid(v[0], nodata) {
                let mut proj = 0.0f64;
                for d in 0..dim {
                    proj += (v[d] - u[d]) * dir[d];
                }
                *out_val = proj;
            }
        });

    (ranked_dims, rms_per_dim, mean_diff_per_dim, pc1_proj)
}

/// Deterministic k-means++ clustering on multi-band / embedding features (`n_pixels * dim`).
pub fn kmeans_features(
    features: &[f64],
    n_pixels: usize,
    dim: usize,
    n_clusters: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<i32>, Vec<f64>) {
    let mut labels = vec![-1i32; n_pixels];
    let k = n_clusters.max(1);
    if n_pixels == 0 || dim == 0 || features.len() < n_pixels * dim {
        return (labels, vec![0.0; k * dim]);
    }

    // Collect valid pixel indices
    let mut valid_indices = Vec::with_capacity(n_pixels);
    for i in 0..n_pixels {
        let base = i * dim;
        if features[base..base + dim].iter().all(|v| v.is_finite()) {
            valid_indices.push(i);
        }
    }
    if valid_indices.is_empty() {
        return (labels, vec![0.0; k * dim]);
    }

    let mut centroids = vec![0.0f64; k * dim];
    // Deterministic LCG PRNG for k-means++ initialization
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next_u64 = || -> u64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state
    };

    let first_idx = valid_indices[(next_u64() as usize) % valid_indices.len()];
    centroids[0..dim].copy_from_slice(&features[first_idx * dim..(first_idx + 1) * dim]);

    let mut min_dist2 = vec![f64::INFINITY; valid_indices.len()];
    for c in 1..k {
        let prev_c = &centroids[(c - 1) * dim..c * dim];
        let mut sum_d2 = 0.0f64;
        for (vi, &pidx) in valid_indices.iter().enumerate() {
            let f = &features[pidx * dim..(pidx + 1) * dim];
            let mut d2 = 0.0f64;
            for d in 0..dim {
                let diff = f[d] - prev_c[d];
                d2 += diff * diff;
            }
            if d2 < min_dist2[vi] {
                min_dist2[vi] = d2;
            }
            sum_d2 += min_dist2[vi];
        }
        let target = ((next_u64() >> 11) as f64 / ((1u64 << 53) as f64)) * sum_d2;
        let mut accum = 0.0f64;
        let mut chosen_pidx = valid_indices[0];
        for (vi, &pidx) in valid_indices.iter().enumerate() {
            accum += min_dist2[vi];
            if accum >= target {
                chosen_pidx = pidx;
                break;
            }
        }
        centroids[c * dim..(c + 1) * dim]
            .copy_from_slice(&features[chosen_pidx * dim..(chosen_pidx + 1) * dim]);
    }

    let iters = max_iter.max(1);
    for _ in 0..iters {
        // Assign step in parallel
        labels.par_iter_mut().enumerate().for_each(|(i, lbl)| {
            let f = &features[i * dim..(i + 1) * dim];
            if !f.iter().all(|v| v.is_finite()) {
                *lbl = -1;
                return;
            }
            let mut best_c = 0i32;
            let mut best_d2 = f64::INFINITY;
            for c in 0..k {
                let cent = &centroids[c * dim..(c + 1) * dim];
                let mut d2 = 0.0f64;
                for d in 0..dim {
                    let diff = f[d] - cent[d];
                    d2 += diff * diff;
                }
                if d2 < best_d2 {
                    best_d2 = d2;
                    best_c = c as i32;
                }
            }
            *lbl = best_c;
        });

        // Update centroids
        let mut new_centroids = vec![0.0f64; k * dim];
        let mut counts = vec![0usize; k];
        for i in 0..n_pixels {
            let lbl = labels[i];
            if lbl >= 0 && (lbl as usize) < k {
                let c = lbl as usize;
                counts[c] += 1;
                let f = &features[i * dim..(i + 1) * dim];
                let cent = &mut new_centroids[c * dim..(c + 1) * dim];
                for d in 0..dim {
                    cent[d] += f[d];
                }
            }
        }
        let mut max_shift2 = 0.0f64;
        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / (counts[c] as f64);
                let mut shift2 = 0.0f64;
                for d in 0..dim {
                    let updated = new_centroids[c * dim + d] * inv;
                    let diff = updated - centroids[c * dim + d];
                    shift2 += diff * diff;
                    centroids[c * dim + d] = updated;
                }
                if shift2 > max_shift2 {
                    max_shift2 = shift2;
                }
            }
        }
        if max_shift2 < 1e-10 {
            break;
        }
    }

    (labels, centroids)
}
