// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
use rayon::prelude::*;
use std::collections::HashMap;

/// Spatial majority (mode) filter over a circular window of radius `radius_px` for `iterations` passes.
/// Negative labels (`< 0`) are treated as nodata and ignored when voting.
pub fn smooth_labels_majority(
    labels: &[i32],
    rows: usize,
    cols: usize,
    radius_px: usize,
    iterations: usize,
) -> Vec<i32> {
    let n = rows * cols;
    if rows == 0 || cols == 0 || labels.len() < n {
        return labels.to_vec();
    }
    let rad = radius_px.max(1) as isize;
    let r2 = rad * rad;
    let iters = iterations.max(1);

    let mut current = labels[0..n].to_vec();
    let mut next = current.clone();

    for _ in 0..iters {
        next.par_chunks_mut(cols)
            .enumerate()
            .for_each(|(r, out_row)| {
                let ri = r as isize;
                let mut counts: Vec<(i32, usize)> = Vec::with_capacity(16);
                for c in 0..cols {
                    let center_lbl = current[r * cols + c];
                    if center_lbl < 0 {
                        out_row[c] = -1;
                        continue;
                    }
                    counts.clear();
                    let ci = c as isize;
                    for dr in -rad..=rad {
                        for dc in -rad..=rad {
                            if dr * dr + dc * dc > r2 {
                                continue;
                            }
                            let rr = ri + dr;
                            let cc = ci + dc;
                            if rr >= 0
                                && rr < (rows as isize)
                                && cc >= 0
                                && cc < (cols as isize)
                            {
                                let lbl = current[(rr as usize) * cols + (cc as usize)];
                                if lbl >= 0 {
                                    if let Some(entry) =
                                        counts.iter_mut().find(|(k, _)| *k == lbl)
                                    {
                                        entry.1 += 1;
                                    } else {
                                        counts.push((lbl, 1));
                                    }
                                }
                            }
                        }
                    }
                    // Pick majority label; break ties in favor of center_lbl
                    let mut best_lbl = center_lbl;
                    let mut best_cnt = 0usize;
                    for &(lbl, cnt) in &counts {
                        if cnt > best_cnt || (cnt == best_cnt && lbl == center_lbl) {
                            best_cnt = cnt;
                            best_lbl = lbl;
                        }
                    }
                    out_row[c] = best_lbl;
                }
            });
        std::mem::swap(&mut current, &mut next);
    }

    current
}

#[derive(Debug, Clone)]
pub struct ContactPixel {
    pub row: usize,
    pub col: usize,
    pub x: f64,
    pub y: f64,
    pub unit_a: i32,
    pub unit_b: i32,
}

/// Extract lithologic / geomorphic contact boundary pixels where 4-connected adjacent pixels
/// have distinct non-negative unit labels (`unit_a < unit_b`).
pub fn extract_unit_contacts(
    labels: &[i32],
    rows: usize,
    cols: usize,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
) -> (Vec<u8>, Vec<ContactPixel>) {
    let n = rows * cols;
    let mut contact_mask = vec![0u8; n];
    let mut contacts = Vec::new();
    if rows == 0 || cols == 0 || labels.len() < n {
        return (contact_mask, contacts);
    }

    let dirs: [(isize, isize); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
    for r in 0..rows {
        let ri = r as isize;
        for c in 0..cols {
            let u0 = labels[r * cols + c];
            if u0 < 0 {
                continue;
            }
            let ci = c as isize;
            let mut neighbor_unit: Option<i32> = None;
            for &(dr, dc) in &dirs {
                let nr = ri + dr;
                let nc = ci + dc;
                if nr >= 0 && nr < (rows as isize) && nc >= 0 && nc < (cols as isize) {
                    let u1 = labels[(nr as usize) * cols + (nc as usize)];
                    if u1 >= 0 && u1 != u0 {
                        neighbor_unit = Some(match neighbor_unit {
                            Some(prev) => prev.min(u1),
                            None => u1,
                        });
                    }
                }
            }
            if let Some(u1) = neighbor_unit {
                contact_mask[r * cols + c] = 1;
                let unit_a = u0.min(u1);
                let unit_b = u0.max(u1);
                // Record each boundary pixel once on the lower-ID side (or if both neighbors differ)
                if u0 == unit_a {
                    contacts.push(ContactPixel {
                        row: r,
                        col: c,
                        x: x0 + (c as f64) * dx,
                        y: y0 + (r as f64) * dy,
                        unit_a,
                        unit_b,
                    });
                }
            }
        }
    }

    (contact_mask, contacts)
}

#[derive(Debug, Clone)]
pub struct GaussianBayesResult {
    pub pred_labels: Vec<i32>,
    pub probabilities: Vec<f64>,
    pub confidence: Vec<f64>,
    pub entropy: Vec<f64>,
    pub class_ids: Vec<i32>,
}

/// Fast parallel Gaussian Naive Bayes / Diagonal-Mahalanobis probabilistic classifier in Rust.
/// Trains on pixels where `train_mask[i] != 0` and `train_labels[i] >= 0`, and predicts
/// unit labels, posterior probabilities, maximum confidence `[0, 1]`, and normalized Shannon entropy `[0, 1]`
/// across all `n_pixels`.
pub fn fit_predict_gaussian_bayes(
    features: &[f64],
    n_pixels: usize,
    dim: usize,
    train_mask: &[u8],
    train_labels: &[i32],
    var_smoothing: f64,
) -> GaussianBayesResult {
    if n_pixels == 0
        || dim == 0
        || features.len() < n_pixels * dim
        || train_mask.len() < n_pixels
        || train_labels.len() < n_pixels
    {
        return GaussianBayesResult {
            pred_labels: vec![-1; n_pixels],
            probabilities: Vec::new(),
            confidence: vec![f64::NAN; n_pixels],
            entropy: vec![f64::NAN; n_pixels],
            class_ids: Vec::new(),
        };
    }

    // Discover sorted unique class IDs in training set
    let mut class_map: HashMap<i32, usize> = HashMap::new();
    let mut class_ids: Vec<i32> = Vec::new();
    for i in 0..n_pixels {
        if train_mask[i] != 0 && train_labels[i] >= 0 {
            let f = &features[i * dim..(i + 1) * dim];
            if f.iter().all(|v| v.is_finite()) && !class_map.contains_key(&train_labels[i]) {
                class_map.insert(train_labels[i], class_ids.len());
                class_ids.push(train_labels[i]);
            }
        }
    }
    class_ids.sort();
    for (idx, &cid) in class_ids.iter().enumerate() {
        class_map.insert(cid, idx);
    }

    let n_classes = class_ids.len();
    if n_classes == 0 {
        return GaussianBayesResult {
            pred_labels: vec![-1; n_pixels],
            probabilities: Vec::new(),
            confidence: vec![f64::NAN; n_pixels],
            entropy: vec![f64::NAN; n_pixels],
            class_ids,
        };
    }

    let mut means = vec![0.0f64; n_classes * dim];
    let mut sq_sums = vec![0.0f64; n_classes * dim];
    let mut counts = vec![0usize; n_classes];
    let mut total_train = 0usize;

    for i in 0..n_pixels {
        if train_mask[i] != 0 && train_labels[i] >= 0 {
            let f = &features[i * dim..(i + 1) * dim];
            if !f.iter().all(|v| v.is_finite()) {
                continue;
            }
            if let Some(&c_idx) = class_map.get(&train_labels[i]) {
                counts[c_idx] += 1;
                total_train += 1;
                let m_slice = &mut means[c_idx * dim..(c_idx + 1) * dim];
                let s_slice = &mut sq_sums[c_idx * dim..(c_idx + 1) * dim];
                for d in 0..dim {
                    m_slice[d] += f[d];
                    s_slice[d] += f[d] * f[d];
                }
            }
        }
    }

    let mut vars = vec![1.0f64; n_classes * dim];
    let mut log_priors = vec![0.0f64; n_classes];
    let mut max_var = 1e-9f64;

    for c in 0..n_classes {
        let cnt = counts[c].max(1) as f64;
        log_priors[c] = (cnt / (total_train.max(1) as f64)).ln();
        for d in 0..dim {
            let mu = means[c * dim + d] / cnt;
            means[c * dim + d] = mu;
            let v = (sq_sums[c * dim + d] / cnt - mu * mu).max(1e-9);
            vars[c * dim + d] = v;
            if v > max_var {
                max_var = v;
            }
        }
    }
    let eps = var_smoothing.max(1e-9) * max_var;
    for v in vars.iter_mut() {
        *v += eps;
    }

    let mut pred_labels = vec![-1i32; n_pixels];
    let mut probabilities = vec![0.0f64; n_pixels * n_classes];
    let mut confidence = vec![f64::NAN; n_pixels];
    let mut entropy = vec![f64::NAN; n_pixels];
    let inv_log_c = if n_classes > 1 {
        1.0 / (n_classes as f64).ln()
    } else {
        1.0
    };

    pred_labels
        .par_iter_mut()
        .zip(
            probabilities
                .par_chunks_mut(n_classes)
                .zip(confidence.par_iter_mut().zip(entropy.par_iter_mut())),
        )
        .enumerate()
        .for_each(|(i, (lbl_out, (prob_row, (conf_out, ent_out))))| {
            let f = &features[i * dim..(i + 1) * dim];
            if !f.iter().all(|v| v.is_finite()) {
                return;
            }
            let mut max_logp = f64::NEG_INFINITY;
            for c in 0..n_classes {
                let mut logp = log_priors[c];
                let m_slice = &means[c * dim..(c + 1) * dim];
                let v_slice = &vars[c * dim..(c + 1) * dim];
                for d in 0..dim {
                    let diff = f[d] - m_slice[d];
                    let var = v_slice[d];
                    logp -= 0.5 * (var.ln() + (diff * diff) / var);
                }
                prob_row[c] = logp;
                if logp > max_logp {
                    max_logp = logp;
                }
            }
            let mut sum_exp = 0.0f64;
            for c in 0..n_classes {
                let p = (prob_row[c] - max_logp).exp();
                prob_row[c] = p;
                sum_exp += p;
            }
            let mut best_c = 0usize;
            let mut best_p = -1.0f64;
            let mut shannon = 0.0f64;
            for c in 0..n_classes {
                let p = if sum_exp > 0.0 {
                    prob_row[c] / sum_exp
                } else {
                    1.0 / (n_classes as f64)
                };
                prob_row[c] = p;
                if p > best_p {
                    best_p = p;
                    best_c = c;
                }
                if p > 1e-15 {
                    shannon -= p * p.ln();
                }
            }
            *lbl_out = class_ids[best_c];
            *conf_out = best_p.clamp(0.0, 1.0);
            *ent_out = if n_classes > 1 {
                (shannon * inv_log_c).clamp(0.0, 1.0)
            } else {
                0.0
            };
        });

    GaussianBayesResult {
        pred_labels,
        probabilities,
        confidence,
        entropy,
        class_ids,
    }
}
