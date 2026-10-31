// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
pub mod mesh3d;
pub mod change;
pub mod embedding;
pub mod mapping;
pub mod profile;
pub mod spectral;
pub mod terrain;

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_slope_aspect_and_hillshade() {
        let rows = 5;
        let cols = 5;
        let mut dem = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                dem[r * cols + c] = 100.0 - (c as f64);
            }
        }
        let (slope, aspect) = terrain::compute_slope_aspect(&dem, rows, cols, 1.0, 1.0, f64::NAN);
        let center_idx = 2 * cols + 2;
        assert!((slope[center_idx] - 45.0).abs() < 1e-6);
        assert!((aspect[center_idx] - 90.0).abs() < 1e-6);

        let hs = terrain::compute_hillshade(&dem, rows, cols, 1.0, 1.0, 90.0, 45.0, 1.0, f64::NAN);
        assert!((hs[center_idx] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_spectral_indices() {
        let nir = vec![0.5, 0.2, 0.0];
        let red = vec![0.1, 0.2, 0.0];
        let ndvi = spectral::normalized_difference(&nir, &red, f64::NAN);
        assert!((ndvi[0] - (0.4 / 0.6)).abs() < 1e-9);
        assert!(ndvi[1].abs() < 1e-9);
        assert!(ndvi[2].is_nan());

        let ratio = spectral::band_ratio(&nir, &red, f64::NAN);
        assert!((ratio[0] - 5.0).abs() < 1e-9);
        assert!((ratio[1] - 1.0).abs() < 1e-9);
        assert!(ratio[2].is_nan());
    }

    #[test]
    fn test_dod_detrend_and_volume_budget() {
        let rows = 10;
        let cols = 10;
        let dx = 10.0;
        let dy = 10.0;
        let mut dod = vec![0.0; rows * cols];
        let mut stable = vec![1u8; rows * cols];

        for r in 0..rows {
            for c in 0..cols {
                let x = (c as f64) * dx;
                let y = (r as f64) * dy;
                dod[r * cols + c] = 0.5 + 0.01 * x - 0.02 * y;
            }
        }
        dod[3 * cols + 3] -= 4.0;
        stable[3 * cols + 3] = 0;
        dod[6 * cols + 6] += 5.0;
        stable[6 * cols + 6] = 0;

        let (detrended, c0, cx, cy) = change::detrend_dod_planar(&dod, &stable, rows, cols, dx, dy);
        assert!((c0 - 0.5).abs() < 1e-9);
        assert!((cx - 0.01).abs() < 1e-9);
        assert!((cy - (-0.02)).abs() < 1e-9);
        assert!((detrended[3 * cols + 3] - (-4.0)).abs() < 1e-8);
        assert!((detrended[6 * cols + 6] - 5.0).abs() < 1e-8);

        let budget = change::compute_volume_budget(&detrended, dx * dy, 0.2, 0.1);
        assert_eq!(budget.eroded_pixels, 1);
        assert_eq!(budget.deposited_pixels, 1);
        assert!((budget.eroded_volume_m3 - 400.0).abs() < 1e-6);
        assert!((budget.deposited_volume_m3 - 500.0).abs() < 1e-6);
        assert!((budget.net_volume_m3 - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_otsu_and_connected_patches() {
        let mut vals = vec![0.1; 50];
        vals.extend(vec![0.9; 50]);
        let thr = change::otsu_threshold(&vals, 64);
        assert!(thr > 0.15 && thr < 0.85);

        let rows = 6;
        let cols = 6;
        let mut mask = vec![0u8; rows * cols];
        let mut mag = vec![0.0f64; rows * cols];
        for r in 1..=2 {
            for c in 1..=3 {
                mask[r * cols + c] = 1;
                mag[r * cols + c] = 2.5;
            }
        }
        let (labels, stats) =
            change::connected_change_patches(&mask, &mag, rows, cols, 0.0, 0.0, 10.0, -10.0, 2);
        assert_eq!(stats.len(), 1);
        assert_eq!(stats[0].pixel_count, 6);
        assert!((stats[0].area_m2 - 600.0).abs() < 1e-6);
        assert_eq!(labels[1 * cols + 1], 1);
    }

    #[test]
    fn test_embedding_change_metrics() {
        let dim = 64;
        let n_pixels = 3;
        let mut e1 = vec![0.0; n_pixels * dim];
        let mut e2 = vec![0.0; n_pixels * dim];
        e1[0] = 1.0;
        e2[0] = 1.0;
        e1[dim] = 1.0;
        e2[dim + 1] = 1.0;
        e1[2 * dim] = 1.0;
        e2[2 * dim] = -1.0;

        let (cd, cs, l2) = embedding::embedding_change_metrics(&e1, &e2, n_pixels, dim, f64::NAN);
        assert!(cd[0].abs() < 1e-9 && (cs[0] - 1.0).abs() < 1e-9 && l2[0].abs() < 1e-9);
        assert!((cd[1] - 1.0).abs() < 1e-9 && cs[1].abs() < 1e-9 && (l2[1] - 2.0f64.sqrt()).abs() < 1e-9);
        assert!((cd[2] - 2.0).abs() < 1e-9 && (cs[2] + 1.0).abs() < 1e-9 && (l2[2] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_cwt2d_mexican_hat_wavelength_recovery() {
        // Synthetic 2D sinusoid z(x, y) = 25.0 * cos(2*pi*x / lambda_true) with lambda_true = 200 m
        // Analytical peak Mexican-hat scale is s_true = lambda_true / (sqrt(2)*pi) = 45.0158 m
        let rows = 80;
        let cols = 120;
        let dx = 10.0;
        let dy = 10.0;
        let lambda_true = 200.0;
        let mut dem = vec![0.0f64; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let x = (c as f64) * dx;
                dem[r * cols + c] = 25.0 * (2.0 * PI * x / lambda_true).cos();
            }
        }
        let scales = vec![20.0, 30.0, 45.0, 65.0, 95.0];
        let res = terrain::compute_multiscale_complexity(&dem, rows, cols, dx, dy, &scales, f64::NAN);
        let center_idx = (rows / 2) * cols + (cols / 2);
        let recovered_lambda = res.dominant_wavelength_m[center_idx];
        assert!(
            (recovered_lambda - lambda_true).abs() < 12.0,
            "Expected wavelength ~200 m, got {recovered_lambda}"
        );
    }

    #[test]
    fn test_3point_bed_attitude_recovery() {
        // Plane with true dip = 30.0 deg, dip direction = 60.0 deg (ENE), RHR strike = 330.0 deg (NNW)
        let dip_true = 30.0f64;
        let dip_dir_true = 60.0f64;
        let strike_true = 330.0f64;
        let grad = (dip_true * PI / 180.0).tan();
        let a = -grad * (dip_dir_true * PI / 180.0).sin();
        let b = -grad * (dip_dir_true * PI / 180.0).cos();
        let c = 1500.0;

        let xs = vec![0.0, 200.0, 100.0, 350.0, 50.0];
        let ys = vec![0.0, 50.0, 250.0, 180.0, 400.0];
        let zs: Vec<f64> = xs.iter().zip(ys.iter()).map(|(&x, &y)| a * x + b * y + c).collect();

        let att = profile::estimate_bed_attitude(&xs, &ys, &zs, false).expect("Should solve plane");
        assert!((att.dip_deg - dip_true).abs() < 1e-6);
        assert!((att.dip_dir_deg - dip_dir_true).abs() < 1e-6);
        assert!((att.strike_deg - strike_true).abs() < 1e-6);
        assert!((att.r_squared - 1.0).abs() < 1e-9);
        assert!(att.rmse_m < 1e-9);
    }

    #[test]
    fn test_mapping_smoothing_contacts_and_bayes() {
        let rows = 8;
        let cols = 8;
        let mut labels = vec![0i32; rows * cols];
        for r in 0..rows {
            for c in 4..cols {
                labels[r * cols + c] = 1;
            }
        }
        // Inject one noisy pixel inside unit 0
        labels[2 * cols + 1] = 1;
        let smoothed = mapping::smooth_labels_majority(&labels, rows, cols, 1, 1);
        assert_eq!(smoothed[2 * cols + 1], 0);

        let (mask, contacts) =
            mapping::extract_unit_contacts(&smoothed, rows, cols, 0.0, 0.0, 10.0, -10.0);
        assert_eq!(contacts.len(), rows);
        assert_eq!(mask[3 * cols + 3], 1);
        assert_eq!(mask[3 * cols + 4], 1);
        assert_eq!(mask[3 * cols + 1], 0);
    }

    #[test]
    fn test_3d_block_volume_and_meshes() {
        let rows = 12;
        let cols = 14;
        let nz = 10;
        let dx = 20.0;
        let dy = -20.0;
        let mut dem = vec![1200.0; rows * cols];
        let mut surf_u = vec![1i32; rows * cols];
        let mut h_struct = vec![0.0f64; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                dem[r * cols + c] = 1200.0 + (c as f64) * 5.0;
                h_struct[r * cols + c] = -((c as f64) * 8.0);
                surf_u[r * cols + c] = if c < 5 { 0 } else if c < 10 { 1 } else { 2 };
            }
        }
        let thresholds = vec![1020.0, 1100.0];
        let bedrock_ids = vec![0, 1, 2];
        let fault = Some(mesh3d::Fault3DParams {
            x0: 140.0,
            y0: -120.0,
            z0: 1230.0,
            strike_deg: 0.0,
            dip_deg: 60.0,
            throw_m: 45.0,
            listric_curvature: 0.0,
        });

        let vol = mesh3d::build_3d_block_volume(
            &dem,
            &surf_u,
            &h_struct,
            &thresholds,
            &bedrock_ids,
            -1,
            0.0,
            12.0,
            rows,
            cols,
            0.0,
            0.0,
            dx,
            dy,
            900.0,
            1300.0,
            nz,
            fault,
        );
        assert_eq!(vol.voxel_units.len(), rows * cols * nz);
        assert_eq!(vol.horizon_elevations.len(), 2 * rows * cols);
        // Bottom voxel at (0, 0, 0) must be below topography (>= 0)
        assert!(vol.voxel_units[0] >= 0);
        // Top voxel at (0, 0, nz-1) is at z=1300 > dem(0,0)=1200, so must be -1 (air)
        assert_eq!(vol.voxel_units[nz - 1], -1);

        let palette = vec![
            107u8, 174, 214, 255,
            158, 202, 225, 255,
            230, 85, 13, 255,
        ];
        let drape = vec![180u8; rows * cols * 4];
        let mesh = mesh3d::build_block_diagram_meshes(
            &dem,
            &surf_u,
            &drape,
            &vol.voxel_units,
            &vol.z_levels,
            &palette,
            &vol.fault_surface_z,
            rows,
            cols,
            0.0,
            0.0,
            dx,
            dy,
            Some((0, 5, 0, 6, 4)),
        );
        assert!(mesh.n_vertices() > 200);
        assert!(mesh.n_triangles() > 100);
        // Verify all unit normals have unit length
        for i in 0..mesh.n_vertices() {
            let nx = mesh.normals[i * 3];
            let ny = mesh.normals[i * 3 + 1];
            let nz_v = mesh.normals[i * 3 + 2];
            let len = (nx * nx + ny * ny + nz_v * nz_v).sqrt();
            assert!((len - 1.0).abs() < 1e-5);
        }

        // Test fence slice
        let fence = mesh3d::slice_3d_block_fence(
            &dem,
            &vol.voxel_units,
            &vol.z_levels,
            &palette,
            rows,
            cols,
            0.0,
            0.0,
            dx,
            dy,
            10.0,
            -10.0,
            240.0,
            -200.0,
            25,
        );
        assert_eq!(fence.distances_m.len(), 25);
        assert!(fence.mesh.n_triangles() > 20);
    }

    #[test]
    fn test_displacement_strain_and_multi_hazard() {
        let rows = 24;
        let cols = 24;
        let dx = 10.0;
        let dy = -10.0;
        // Synthetic fault scarp step across column 12
        let mut disp = vec![0.0f64; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let x_rel = (c as f64) - 11.5;
                disp[r * cols + c] = 1.2 * (x_rel / 1.5).tanh();
            }
        }
        let st = change::compute_displacement_strain(&disp, rows, cols, dx, dy, f64::NAN);
        assert_eq!(st.gradient_mag.len(), rows * cols);
        // Strain magnitude peaks at fault trace (c = 11 or 12) and is small away from scarp
        let peak_strain = st.gradient_mag[12 * cols + 11];
        let far_strain = st.gradient_mag[12 * cols + 2];
        assert!(peak_strain > 0.04);
        assert!(far_strain < 0.005);

        let zeros = vec![0.0f64; rows * cols];
        let sig = change::compute_multi_hazard_signature(
            &disp,
            &st.gradient_mag,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
            &zeros,
            100.0,
            0.25,
            0.015,
        );
        assert_eq!(sig.dominant_regime, "earthquake_deformation");
        assert!(sig.fingerprint_scores[0] > 0.6);
    }

    #[test]
    fn test_insar_wrapped_phase_and_3d_decomposition() {
        let rows = 8;
        let cols = 8;
        let n = rows * cols;
        // True displacement: u_e = 0.6 m, u_n = -0.8 m, u_u = 0.2 m
        let u_e_true = vec![0.6f64; n];
        let u_n_true = vec![-0.8f64; n];
        let u_u_true = vec![0.2f64; n];

        let s_asc = [-0.608, -0.112, 0.786];
        let s_desc = [0.618, -0.114, 0.778];

        let mut los_asc = vec![0.0f64; n];
        let mut los_desc = vec![0.0f64; n];
        for i in 0..n {
            los_asc[i] = s_asc[0] * u_e_true[i] + s_asc[1] * u_n_true[i] + s_asc[2] * u_u_true[i];
            los_desc[i] = s_desc[0] * u_e_true[i] + s_desc[1] * u_n_true[i] + s_desc[2] * u_u_true[i];
        }

        let phase = change::compute_insar_wrapped_phase(&los_asc, 0.05546576, f64::NAN);
        assert_eq!(phase.len(), n);
        assert!(phase[0] >= -std::f64::consts::PI && phase[0] < std::f64::consts::PI);

        let dec = change::decompose_insar_3d(
            &los_asc,
            &los_desc,
            &u_e_true,
            &u_n_true,
            &u_u_true,
            s_asc,
            s_desc,
            0.25,
            0.10,
            f64::NAN,
        );
        assert!((dec.u_east[0] - 0.6).abs() < 1e-6);
        assert!((dec.u_north[0] + 0.8).abs() < 1e-6);
        assert!((dec.u_up[0] - 0.2).abs() < 1e-6);
        assert!((dec.horiz_mag[0] - 1.0).abs() < 1e-6);
    }
}
