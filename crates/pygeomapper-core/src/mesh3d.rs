// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (C) 2026 Larry Syu-Heng Lai
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};

/// 3D planar or listric fault specification in projected metric coordinates (e.g., UTM meters).
#[derive(Debug, Clone, Copy)]
pub struct Fault3DParams {
    pub x0: f64,
    pub y0: f64,
    pub z0: f64,
    pub strike_deg: f64,
    pub dip_deg: f64,
    pub throw_m: f64,
    pub listric_curvature: f64,
}

impl Fault3DParams {
    /// Evaluate fault surface elevation `z_fault(x, y)` at horizontal position `(x, y)`.
    #[inline]
    pub fn fault_elevation(&self, x: f64, y: f64) -> f64 {
        let dip_dir_rad = ((self.strike_deg + 90.0).rem_euclid(360.0)) * PI / 180.0;
        let sin_dd = dip_dir_rad.sin();
        let cos_dd = dip_dir_rad.cos();
        let d_dip = (x - self.x0) * sin_dd + (y - self.y0) * cos_dd;
        let dip_clamped = self.dip_deg.abs().clamp(5.0, 88.0) * PI / 180.0;
        let tan_dip = dip_clamped.tan();
        self.z0 - d_dip * tan_dip + self.listric_curvature * d_dip * d_dip.abs()
    }

    /// Outward unit normal vector of the fault plane pointing into the hanging wall.
    #[inline]
    pub fn unit_normal(&self) -> [f64; 3] {
        let dip_dir_rad = ((self.strike_deg + 90.0).rem_euclid(360.0)) * PI / 180.0;
        let dip_rad = self.dip_deg.abs().clamp(5.0, 88.0) * PI / 180.0;
        let nx = dip_rad.sin() * dip_dir_rad.sin();
        let ny = dip_rad.sin() * dip_dir_rad.cos();
        let nz = dip_rad.cos();
        [nx, ny, nz]
    }
}

/// Output of `build_3d_block_volume`.
#[derive(Debug, Clone)]
pub struct BlockVolumeResult {
    pub rows: usize,
    pub cols: usize,
    pub nz: usize,
    pub z_levels: Vec<f64>,
    /// Flattened `(rows, cols, nz)`: `-1` above topography, `>= 0` formation ID below topography.
    pub voxel_units: Vec<i32>,
    /// Flattened `(rows, cols, nz)`: stratigraphic potential coordinate `zeta(y, x, z)` (m).
    pub zeta_volume: Vec<f64>,
    /// Flattened `(rows, cols, nz)`: 3D structural fault throw offset (m).
    pub fault_offset_volume: Vec<f64>,
    /// Flattened `(rows, cols)` fault plane elevation `z_fault(y, x)` (m), or `NaN` if no fault.
    pub fault_surface_z: Vec<f64>,
    /// Flattened `(n_horizons, rows, cols)` 3D elevation `z_k(y, x)` of each stratigraphic contact horizon.
    pub horizon_elevations: Vec<f64>,
}

/// Evaluate a 3D shallow geological block volume `(rows, cols, nz)` from `z_bottom` to `z_top`
/// in parallel across rows using Rayon.
#[allow(clippy::too_many_arguments)]
pub fn build_3d_block_volume(
    dem: &[f64],
    surface_units: &[i32],
    structural_horizon: &[f64],
    unit_zeta_thresholds: &[f64],
    bedrock_unit_ids: &[i32],
    surficial_unit_id: i32,
    surficial_thickness_m: f64,
    near_surface_lock_m: f64,
    rows: usize,
    cols: usize,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    z_bottom: f64,
    z_top: f64,
    nz: usize,
    fault: Option<Fault3DParams>,
) -> BlockVolumeResult {
    let nz_clamped = nz.max(4);
    let n_cells = rows * cols;
    let total_voxels = n_cells * nz_clamped;

    let mut z_levels = Vec::with_capacity(nz_clamped);
    let dz = if nz_clamped > 1 {
        (z_top - z_bottom) / ((nz_clamped - 1) as f64)
    } else {
        1.0
    };
    for iz in 0..nz_clamped {
        z_levels.push(z_bottom + (iz as f64) * dz);
    }

    let mut voxel_units = vec![-1i32; total_voxels];
    let mut zeta_volume = vec![f64::NAN; total_voxels];
    let mut fault_offset_volume = vec![0.0f64; total_voxels];
    let mut fault_surface_z = vec![f64::NAN; n_cells];

    let default_oldest = bedrock_unit_ids.first().copied().unwrap_or(0);

    voxel_units
        .par_chunks_mut(cols * nz_clamped)
        .zip(zeta_volume.par_chunks_mut(cols * nz_clamped))
        .zip(fault_offset_volume.par_chunks_mut(cols * nz_clamped))
        .zip(fault_surface_z.par_chunks_mut(cols))
        .enumerate()
        .for_each(|(r, (((u_row, z_row), f_row), fs_row))| {
            let y = y0 + ((r as f64) + 0.5) * dy;
            for c in 0..cols {
                let idx2d = r * cols + c;
                let x = x0 + ((c as f64) + 0.5) * dx;
                let z_dem = dem[idx2d];
                let surf_u = surface_units[idx2d];
                let h_st = structural_horizon[idx2d];

                let z_flt = fault.map(|f| f.fault_elevation(x, y)).unwrap_or(f64::NAN);
                fs_row[c] = z_flt;

                if !z_dem.is_finite() {
                    continue;
                }

                // Evaluate surface fault offset so near-surface stratigraphy aligns with surface contacts
                let surf_fault_off = match fault {
                    Some(f) if z_dem >= z_flt => f.throw_m,
                    _ => 0.0,
                };

                for iz in 0..nz_clamped {
                    let z = z_levels[iz];
                    let v_idx = c * nz_clamped + iz;
                    if z > z_dem {
                        u_row[v_idx] = -1;
                        continue;
                    }

                    let flt_off = match fault {
                        Some(f) if z >= z_flt => f.throw_m,
                        _ => 0.0,
                    };
                    f_row[v_idx] = flt_off;

                    let zeta = z - h_st - flt_off;
                    z_row[v_idx] = zeta;

                    let depth_below_dem = z_dem - z;
                    // Surficial alluvial veneer check
                    if surficial_unit_id >= 0
                        && surf_u == surficial_unit_id
                        && depth_below_dem <= surficial_thickness_m
                    {
                        u_row[v_idx] = surficial_unit_id;
                        continue;
                    }

                    // Near-surface bedrock lock when not crossing the fault plane between z_dem and z
                    if depth_below_dem <= near_surface_lock_m
                        && surf_u >= 0
                        && surf_u != surficial_unit_id
                        && (flt_off - surf_fault_off).abs() < 1e-6
                    {
                        u_row[v_idx] = surf_u;
                        continue;
                    }

                    // Classify bedrock formation from sorted `unit_zeta_thresholds`
                    let mut uid = default_oldest;
                    for (t_idx, &thr) in unit_zeta_thresholds.iter().enumerate() {
                        if zeta >= thr {
                            if let Some(&next_u) = bedrock_unit_ids.get(t_idx + 1) {
                                uid = next_u;
                            }
                        } else {
                            break;
                        }
                    }
                    u_row[v_idx] = uid;
                }
            }
        });

    // Compute 3D horizon elevations z_k(r, c) for each stratigraphic boundary threshold
    let n_horizons = unit_zeta_thresholds.len();
    let mut horizon_elevations = vec![f64::NAN; n_horizons * n_cells];
    for (h_idx, &zeta_k) in unit_zeta_thresholds.iter().enumerate() {
        for r in 0..rows {
            for c in 0..cols {
                let idx2d = r * cols + c;
                let z_dem = dem[idx2d];
                if !z_dem.is_finite() {
                    continue;
                }
                let h_st = structural_horizon[idx2d];
                let z_footwall = zeta_k + h_st;
                let z_flt = fault_surface_z[idx2d];
                let z_hor = if let Some(f) = fault {
                    let z_hanging = zeta_k + h_st + f.throw_m;
                    if z_hanging >= z_flt {
                        z_hanging
                    } else {
                        z_footwall.min(z_flt)
                    }
                } else {
                    z_footwall
                };
                if z_hor >= z_bottom && z_hor <= z_dem + 2.0 {
                    horizon_elevations[h_idx * n_cells + idx2d] = z_hor.min(z_dem);
                }
            }
        }
    }

    BlockVolumeResult {
        rows,
        cols,
        nz: nz_clamped,
        z_levels,
        voxel_units,
        zeta_volume,
        fault_offset_volume,
        fault_surface_z,
        horizon_elevations,
    }
}

/// Complete 3D triangle mesh container returned by `build_block_diagram_meshes` and `slice_3d_block_fence`.
#[derive(Debug, Clone)]
pub struct Mesh3DData {
    /// Flattened `(n_vertices, 3)` `f64` positions `(x, y, z)`.
    pub vertices: Vec<f64>,
    /// Flattened `(n_vertices, 3)` `f64` unit normals `(nx, ny, nz)`.
    pub normals: Vec<f64>,
    /// Flattened `(n_triangles, 3)` `u32` counter-clockwise vertex indices.
    pub triangles: Vec<u32>,
    /// Flattened `(n_vertices, 4)` `u8` RGBA colors.
    pub vertex_colors_rgba: Vec<u8>,
    /// `(n_vertices,)` `i32` geological unit IDs (`-2` base, `-3` fault plane).
    pub unit_ids: Vec<i32>,
    /// `(n_vertices,)` `u8` face group codes:
    /// `0` = top surface, `1` = South wall, `2` = North wall, `3` = West wall, `4` = East wall,
    /// `5` = bottom base, `6` = cutaway internal walls/floor, `7` = fault plane, `8` = horizon surface.
    pub face_groups: Vec<u8>,
}

impl Mesh3DData {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            normals: Vec::new(),
            triangles: Vec::new(),
            vertex_colors_rgba: Vec::new(),
            unit_ids: Vec::new(),
            face_groups: Vec::new(),
        }
    }

    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.unit_ids.len()
    }

    #[inline]
    pub fn n_triangles(&self) -> usize {
        self.triangles.len() / 3
    }

    #[inline]
    pub fn push_vertex(
        &mut self,
        pos: [f64; 3],
        norm: [f64; 3],
        rgba: [u8; 4],
        unit_id: i32,
        face_group: u8,
    ) -> u32 {
        let idx = self.n_vertices() as u32;
        self.vertices.extend_from_slice(&pos);
        self.normals.extend_from_slice(&norm);
        self.vertex_colors_rgba.extend_from_slice(&rgba);
        self.unit_ids.push(unit_id);
        self.face_groups.push(face_group);
        idx
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn push_quad(
        &mut self,
        p0: [f64; 3],
        p1: [f64; 3],
        p2: [f64; 3],
        p3: [f64; 3],
        norm: [f64; 3],
        rgba: [u8; 4],
        unit_id: i32,
        face_group: u8,
    ) {
        let i0 = self.push_vertex(p0, norm, rgba, unit_id, face_group);
        let i1 = self.push_vertex(p1, norm, rgba, unit_id, face_group);
        let i2 = self.push_vertex(p2, norm, rgba, unit_id, face_group);
        let i3 = self.push_vertex(p3, norm, rgba, unit_id, face_group);
        self.triangles.extend_from_slice(&[i0, i1, i2, i0, i2, i3]);
    }
}

impl Default for Mesh3DData {
    fn default() -> Self {
        Self::new()
    }
}

#[inline]
fn shade_rgba(base: [u8; 4], shade_factor: f64) -> [u8; 4] {
    let r = ((base[0] as f64) * shade_factor).round().clamp(0.0, 255.0) as u8;
    let g = ((base[1] as f64) * shade_factor).round().clamp(0.0, 255.0) as u8;
    let b = ((base[2] as f64) * shade_factor).round().clamp(0.0, 255.0) as u8;
    [r, g, b, base[3]]
}

#[inline]
fn lookup_unit_color(unit_palette_rgba: &[u8], unit_id: i32, shade_factor: f64) -> [u8; 4] {
    let n_pal = unit_palette_rgba.len() / 4;
    if unit_id >= 0 && (unit_id as usize) < n_pal {
        let off = (unit_id as usize) * 4;
        let base = [
            unit_palette_rgba[off],
            unit_palette_rgba[off + 1],
            unit_palette_rgba[off + 2],
            unit_palette_rgba[off + 3],
        ];
        shade_rgba(base, shade_factor)
    } else {
        shade_rgba([160, 160, 160, 255], shade_factor)
    }
}

/// Emit a vertical wall strip along a sequence of `(r, c)` grid nodes from `z_start_idx` up to
/// `z_cap(r, c)`, colored by `voxel_units(r, c, iz)`.
#[allow(clippy::too_many_arguments)]
fn emit_vertical_wall_strip(
    mesh: &mut Mesh3DData,
    nodes_rc: &[(usize, usize)],
    dem: &[f64],
    voxel_units: &[i32],
    z_levels: &[f64],
    unit_palette_rgba: &[u8],
    cols: usize,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    iz_start: usize,
    z_top_override: Option<f64>,
    outward_normal: [f64; 3],
    shade_factor: f64,
    face_group: u8,
) {
    let nz = z_levels.len();
    if nodes_rc.len() < 2 || nz < 2 || iz_start >= nz - 1 {
        return;
    }

    for seg in 0..(nodes_rc.len() - 1) {
        let (r0, c0) = nodes_rc[seg];
        let (r1, c1) = nodes_rc[seg + 1];
        let idx0 = r0 * cols + c0;
        let idx1 = r1 * cols + c1;

        let xa = x0 + ((c0 as f64) + 0.5) * dx;
        let ya = y0 + ((r0 as f64) + 0.5) * dy;
        let xb = x0 + ((c1 as f64) + 0.5) * dx;
        let yb = y0 + ((r1 as f64) + 0.5) * dy;

        let z_cap0 = z_top_override.unwrap_or(dem[idx0]).min(dem[idx0]);
        let z_cap1 = z_top_override.unwrap_or(dem[idx1]).min(dem[idx1]);
        if !z_cap0.is_finite() || !z_cap1.is_finite() {
            continue;
        }

        for iz in iz_start..(nz - 1) {
            let z_lo = z_levels[iz];
            let z_hi = z_levels[iz + 1];
            if z_lo >= z_cap0 && z_lo >= z_cap1 {
                break;
            }
            let za0 = z_lo.min(z_cap0);
            let zb0 = z_lo.min(z_cap1);
            let za1 = z_hi.min(z_cap0);
            let zb1 = z_hi.min(z_cap1);
            if (za1 - za0) < 1e-3 && (zb1 - zb0) < 1e-3 {
                continue;
            }

            // Pick the voxel unit at (r0, c0, iz), or fall back downward if near the surface cap
            let mut uid = voxel_units[idx0 * nz + iz];
            if uid < 0 && iz > 0 {
                uid = voxel_units[idx0 * nz + (iz - 1)];
            }
            if uid < 0 {
                uid = voxel_units[idx1 * nz + iz];
            }
            if uid < 0 {
                uid = 0;
            }

            // Modulate shade slightly at stratigraphic contact transitions for visual crispness
            let prev_uid = if iz > 0 {
                voxel_units[idx0 * nz + (iz - 1)]
            } else {
                uid
            };
            let local_shade = if prev_uid >= 0 && prev_uid != uid {
                (shade_factor * 0.78).max(0.25)
            } else {
                shade_factor
            };
            let rgba = lookup_unit_color(unit_palette_rgba, uid, local_shade);

            mesh.push_quad(
                [xa, ya, za0],
                [xb, yb, zb0],
                [xb, yb, zb1],
                [xa, ya, za1],
                outward_normal,
                rgba,
                uid,
                face_group,
            );
        }
    }
}

/// Build complete 3D block-diagram triangle meshes in Rust, including:
/// - `face_group = 0`: Top topographic surface (with optional cutaway window)
/// - `face_group = 1..4`: South, North, West, and East vertical geological cross-section walls
/// - `face_group = 5`: Bottom base plate (`z = z_bottom`)
/// - `face_group = 6`: Internal cutaway vertical walls and horizontal cutaway floor (when `cutaway` is active)
/// - `face_group = 7`: 3D fault plane surface mesh (when `fault_surface_z` has finite pixels)
#[allow(clippy::too_many_arguments)]
pub fn build_block_diagram_meshes(
    dem: &[f64],
    surface_units: &[i32],
    surface_drape_rgba: &[u8],
    voxel_units: &[i32],
    z_levels: &[f64],
    unit_palette_rgba: &[u8],
    fault_surface_z: &[f64],
    rows: usize,
    cols: usize,
    x0: f64,
    y0: f64,
    dx: f64,
    dy: f64,
    cutaway_bounds: Option<(usize, usize, usize, usize, usize)>,
) -> Mesh3DData {
    let mut mesh = Mesh3DData::new();
    let nz = z_levels.len();
    if rows < 2 || cols < 2 || nz < 2 {
        return mesh;
    }

    let z_bottom = z_levels[0];
    let (cut_r0, cut_r1, cut_c0, cut_c1, cut_iz) = cutaway_bounds.unwrap_or((0, 0, 0, 0, nz));
    let has_cutaway = cut_r1 > cut_r0 && cut_c1 > cut_c0 && cut_iz < nz;
    let z_cut = if has_cutaway {
        z_levels[cut_iz]
    } else {
        f64::INFINITY
    };

    let is_in_cutaway = |r: usize, c: usize| -> bool {
        has_cutaway && r >= cut_r0 && r < cut_r1 && c >= cut_c0 && c < cut_c1
    };

    // 1. Top topographic surface (`face_group = 0`) and cutaway horizontal floor (`face_group = 6`)
    let mut top_vert_idx = vec![u32::MAX; rows * cols];
    for r in 0..rows {
        let y = y0 + ((r as f64) + 0.5) * dy;
        for c in 0..cols {
            if is_in_cutaway(r, c) {
                continue;
            }
            let idx2d = r * cols + c;
            let x = x0 + ((c as f64) + 0.5) * dx;
            let z = dem[idx2d];
            if !z.is_finite() {
                continue;
            }
            // Central-difference surface normal
            let c_prev = c.saturating_sub(1);
            let c_next = (c + 1).min(cols - 1);
            let r_prev = r.saturating_sub(1);
            let r_next = (r + 1).min(rows - 1);
            let dz_dx = (dem[r * cols + c_next] - dem[r * cols + c_prev])
                / (((c_next - c_prev).max(1) as f64) * dx);
            let dz_dy = (dem[r_next * cols + c] - dem[r_prev * cols + c])
                / (((r_next - r_prev).max(1) as f64) * dy);
            let n_len = (dz_dx * dz_dx + dz_dy * dz_dy + 1.0).sqrt();
            let norm = [-dz_dx / n_len, -dz_dy / n_len, 1.0 / n_len];

            let off = idx2d * 4;
            let rgba = if off + 3 < surface_drape_rgba.len() {
                [
                    surface_drape_rgba[off],
                    surface_drape_rgba[off + 1],
                    surface_drape_rgba[off + 2],
                    surface_drape_rgba[off + 3],
                ]
            } else {
                lookup_unit_color(unit_palette_rgba, surface_units[idx2d], 1.0)
            };
            top_vert_idx[idx2d] =
                mesh.push_vertex([x, y, z], norm, rgba, surface_units[idx2d], 0);
        }
    }

    for r in 0..(rows - 1) {
        for c in 0..(cols - 1) {
            let i00 = top_vert_idx[r * cols + c];
            let i10 = top_vert_idx[(r + 1) * cols + c];
            let i11 = top_vert_idx[(r + 1) * cols + (c + 1)];
            let i01 = top_vert_idx[r * cols + (c + 1)];
            if i00 != u32::MAX && i10 != u32::MAX && i11 != u32::MAX && i01 != u32::MAX {
                mesh.triangles
                    .extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
            }
        }
    }

    // If cutaway is active, emit the horizontal cutaway floor at `z_cut` (`face_group = 6`)
    if has_cutaway {
        let r_end = cut_r1.min(rows - 1);
        let c_end = cut_c1.min(cols - 1);
        for r in cut_r0..r_end {
            let ya = y0 + ((r as f64) + 0.5) * dy;
            let yb = y0 + (((r + 1) as f64) + 0.5) * dy;
            for c in cut_c0..c_end {
                let xa = x0 + ((c as f64) + 0.5) * dx;
                let xb = x0 + (((c + 1) as f64) + 0.5) * dx;
                let idx2d = r * cols + c;
                let z_fl = z_cut.min(dem[idx2d]);
                let mut uid = voxel_units[idx2d * nz + cut_iz];
                if uid < 0 && cut_iz > 0 {
                    uid = voxel_units[idx2d * nz + (cut_iz - 1)];
                }
                let rgba = lookup_unit_color(unit_palette_rgba, uid, 0.92);
                mesh.push_quad(
                    [xa, ya, z_fl],
                    [xa, yb, z_fl],
                    [xb, yb, z_fl],
                    [xb, ya, z_fl],
                    [0.0, 0.0, 1.0],
                    rgba,
                    uid,
                    6,
                );
            }
        }
    }

    // 2. Four outer vertical cross-section walls (`face_group = 1..4`)
    // Helper closure to emit a boundary wall while clamping top elevation to `z_cut` inside cutaway
    let emit_outer_wall = |mesh: &mut Mesh3DData,
                           nodes: Vec<(usize, usize)>,
                           normal: [f64; 3],
                           shade: f64,
                           fgroup: u8| {
        for w in nodes.windows(2) {
            let (r0, c0) = w[0];
            let (r1, c1) = w[1];
            let in_cut = is_in_cutaway(r0, c0) || is_in_cutaway(r1, c1);
            let z_override = if in_cut { Some(z_cut) } else { None };
            emit_vertical_wall_strip(
                mesh,
                &[(r0, c0), (r1, c1)],
                dem,
                voxel_units,
                z_levels,
                unit_palette_rgba,
                cols,
                x0,
                y0,
                dx,
                dy,
                0,
                z_override,
                normal,
                shade,
                fgroup,
            );
        }
    };

    // South wall (`r = rows - 1`, `c = 0..cols`, outward normal `[0, -1, 0]`)
    let south_nodes: Vec<(usize, usize)> = (0..cols).map(|c| (rows - 1, c)).collect();
    emit_outer_wall(&mut mesh, south_nodes, [0.0, -1.0, 0.0], 0.90, 1);

    // North wall (`r = 0`, `c = (0..cols).rev()`, outward normal `[0, 1, 0]`)
    let north_nodes: Vec<(usize, usize)> = (0..cols).rev().map(|c| (0, c)).collect();
    emit_outer_wall(&mut mesh, north_nodes, [0.0, 1.0, 0.0], 0.78, 2);

    // West wall (`c = 0`, `r = 0..rows`, outward normal `[-1, 0, 0]`)
    let west_nodes: Vec<(usize, usize)> = (0..rows).map(|r| (r, 0)).collect();
    emit_outer_wall(&mut mesh, west_nodes, [-1.0, 0.0, 0.0], 0.86, 3);

    // East wall (`c = cols - 1`, `r = (0..rows).rev()`, outward normal `[1, 0, 0]`)
    let east_nodes: Vec<(usize, usize)> = (0..rows).rev().map(|r| (r, cols - 1)).collect();
    emit_outer_wall(&mut mesh, east_nodes, [1.0, 0.0, 0.0], 0.82, 4);

    // 3. Internal cutaway vertical walls (`face_group = 6`, from `iz = cut_iz` up to `dem`)
    if has_cutaway {
        let r0 = cut_r0.min(rows - 1);
        let r1 = cut_r1.min(rows - 1);
        let c0 = cut_c0.min(cols - 1);
        let c1 = cut_c1.min(cols - 1);

        if r0 > 0 && r0 < rows - 1 {
            let wall_r0: Vec<(usize, usize)> = (c0..=c1).map(|c| (r0, c)).collect();
            emit_vertical_wall_strip(
                &mut mesh,
                &wall_r0,
                dem,
                voxel_units,
                z_levels,
                unit_palette_rgba,
                cols,
                x0,
                y0,
                dx,
                dy,
                cut_iz,
                None,
                [0.0, -1.0, 0.0],
                0.94,
                6,
            );
        }
        if r1 > r0 && r1 < rows - 1 {
            let wall_r1: Vec<(usize, usize)> = (c0..=c1).rev().map(|c| (r1, c)).collect();
            emit_vertical_wall_strip(
                &mut mesh,
                &wall_r1,
                dem,
                voxel_units,
                z_levels,
                unit_palette_rgba,
                cols,
                x0,
                y0,
                dx,
                dy,
                cut_iz,
                None,
                [0.0, 1.0, 0.0],
                0.88,
                6,
            );
        }
        if c0 > 0 && c0 < cols - 1 {
            let wall_c0: Vec<(usize, usize)> = (r0..=r1).rev().map(|r| (r, c0)).collect();
            emit_vertical_wall_strip(
                &mut mesh,
                &wall_c0,
                dem,
                voxel_units,
                z_levels,
                unit_palette_rgba,
                cols,
                x0,
                y0,
                dx,
                dy,
                cut_iz,
                None,
                [1.0, 0.0, 0.0],
                0.91,
                6,
            );
        }
        if c1 > c0 && c1 < cols - 1 {
            let wall_c1: Vec<(usize, usize)> = (r0..=r1).map(|r| (r, c1)).collect();
            emit_vertical_wall_strip(
                &mut mesh,
                &wall_c1,
                dem,
                voxel_units,
                z_levels,
                unit_palette_rgba,
                cols,
                x0,
                y0,
                dx,
                dy,
                cut_iz,
                None,
                [-1.0, 0.0, 0.0],
                0.85,
                6,
            );
        }
    }

    // 4. Bottom base face (`z = z_bottom`, `face_group = 5`)
    let x_west = x0 + 0.5 * dx;
    let x_east = x0 + ((cols - 1) as f64 + 0.5) * dx;
    let y_north = y0 + 0.5 * dy;
    let y_south = y0 + ((rows - 1) as f64 + 0.5) * dy;
    mesh.push_quad(
        [x_west, y_north, z_bottom],
        [x_east, y_north, z_bottom],
        [x_east, y_south, z_bottom],
        [x_west, y_south, z_bottom],
        [0.0, 0.0, -1.0],
        [75, 78, 84, 255],
        -2,
        5,
    );

    // 5. 3D Fault plane surface (`face_group = 7`) clipped to `[z_bottom, z_dem]`
    if fault_surface_z.len() == rows * cols {
        let mut flt_idx = vec![u32::MAX; rows * cols];
        for r in 0..rows {
            let y = y0 + ((r as f64) + 0.5) * dy;
            for c in 0..cols {
                let idx2d = r * cols + c;
                let zf = fault_surface_z[idx2d];
                let z_top_cell = if is_in_cutaway(r, c) {
                    dem[idx2d].min(z_cut)
                } else {
                    dem[idx2d]
                };
                if zf.is_finite() && zf >= z_bottom - 20.0 && zf <= z_top_cell + 20.0 {
                    let x = x0 + ((c as f64) + 0.5) * dx;
                    let z_clamped = zf.clamp(z_bottom, z_top_cell);
                    flt_idx[idx2d] = mesh.push_vertex(
                        [x, y, z_clamped],
                        [0.0, 0.0, 1.0],
                        [210, 30, 30, 220],
                        -3,
                        7,
                    );
                }
            }
        }
        for r in 0..(rows - 1) {
            for c in 0..(cols - 1) {
                let i00 = flt_idx[r * cols + c];
                let i10 = flt_idx[(r + 1) * cols + c];
                let i11 = flt_idx[(r + 1) * cols + (c + 1)];
                let i01 = flt_idx[r * cols + (c + 1)];
                if i00 != u32::MAX && i10 != u32::MAX && i11 != u32::MAX && i01 != u32::MAX {
                    mesh.triangles
                        .extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
                }
            }
        }
    }

    mesh
}

/// Output of `slice_3d_block_fence`.
#[derive(Debug, Clone)]
pub struct FenceSliceResult {
    pub distances_m: Vec<f64>,
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub z_topo: Vec<f64>,
    /// Flattened `(nz, n_samples)` `i32` formation IDs along the fence slice.
    pub section_units: Vec<i32>,
    /// 3D vertical ribbon triangle mesh for rendering the fence panel in 3D space.
    pub mesh: Mesh3DData,
}

/// Slice an arbitrary vertical 3D fence cross-section ribbon from `(xa, ya)` to `(xb, yb)`
/// through `voxel_units` `(rows, cols, nz)`.
#[allow(clippy::too_many_arguments)]
pub fn slice_3d_block_fence(
    dem: &[f64],
    voxel_units: &[i32],
    z_levels: &[f64],
    unit_palette_rgba: &[u8],
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
) -> FenceSliceResult {
    let ns = n_samples.max(2);
    let nz = z_levels.len();
    let total_len = ((xb - xa).powi(2) + (yb - ya).powi(2)).sqrt();
    let nx = if total_len > 1e-9 {
        -(yb - ya) / total_len
    } else {
        0.0
    };
    let ny = if total_len > 1e-9 {
        (xb - xa) / total_len
    } else {
        1.0
    };
    let normal = [nx, ny, 0.0];

    let mut distances_m = Vec::with_capacity(ns);
    let mut xs = Vec::with_capacity(ns);
    let mut ys = Vec::with_capacity(ns);
    let mut z_topo = Vec::with_capacity(ns);
    let mut rc_indices = Vec::with_capacity(ns);

    for i in 0..ns {
        let t = (i as f64) / ((ns - 1) as f64);
        let x = xa + t * (xb - xa);
        let y = ya + t * (yb - ya);
        distances_m.push(t * total_len);
        xs.push(x);
        ys.push(y);

        let cf = ((x - x0) / dx - 0.5).round() as isize;
        let rf = ((y - y0) / dy - 0.5).round() as isize;
        let c = cf.clamp(0, cols as isize - 1) as usize;
        let r = rf.clamp(0, rows as isize - 1) as usize;
        rc_indices.push((r, c));
        z_topo.push(dem[r * cols + c]);
    }

    let mut section_units = vec![-1i32; nz * ns];
    for iz in 0..nz {
        for (s_idx, &(r, c)) in rc_indices.iter().enumerate() {
            section_units[iz * ns + s_idx] = voxel_units[(r * cols + c) * nz + iz];
        }
    }

    let mut mesh = Mesh3DData::new();
    for seg in 0..(ns - 1) {
        let (r0, c0) = rc_indices[seg];
        let idx0 = r0 * cols + c0;
        let x_seg0 = xs[seg];
        let y_seg0 = ys[seg];
        let x_seg1 = xs[seg + 1];
        let y_seg1 = ys[seg + 1];
        let z_cap0 = z_topo[seg];
        let z_cap1 = z_topo[seg + 1];
        if !z_cap0.is_finite() || !z_cap1.is_finite() {
            continue;
        }

        for iz in 0..(nz.saturating_sub(1)) {
            let z_lo = z_levels[iz];
            let z_hi = z_levels[iz + 1];
            if z_lo >= z_cap0 && z_lo >= z_cap1 {
                break;
            }
            let za0 = z_lo.min(z_cap0);
            let zb0 = z_lo.min(z_cap1);
            let za1 = z_hi.min(z_cap0);
            let zb1 = z_hi.min(z_cap1);
            if (za1 - za0) < 1e-3 && (zb1 - zb0) < 1e-3 {
                continue;
            }

            let mut uid = voxel_units[idx0 * nz + iz];
            if uid < 0 && iz > 0 {
                uid = voxel_units[idx0 * nz + (iz - 1)];
            }
            if uid < 0 {
                uid = 0;
            }
            let rgba = lookup_unit_color(unit_palette_rgba, uid, 0.92);
            mesh.push_quad(
                [x_seg0, y_seg0, za0],
                [x_seg1, y_seg1, zb0],
                [x_seg1, y_seg1, zb1],
                [x_seg0, y_seg0, za1],
                normal,
                rgba,
                uid,
                6,
            );
        }
    }

    FenceSliceResult {
        distances_m,
        xs,
        ys,
        z_topo,
        section_units,
        mesh,
    }
}

/// Export a 3D triangle mesh (`vertices: (N, 3)`, `normals: (N, 3)`, `triangles: (M, 3)`,
/// `colors_rgba: (N, 4)`) to a standard binary glTF 2.0 `.glb` file in pure Rust.
pub fn export_mesh_glb(
    path: &str,
    vertices: &[f64],
    normals: &[f64],
    triangles: &[u32],
    colors_rgba: &[u8],
) -> std::io::Result<usize> {
    let n_verts = vertices.len() / 3;
    let n_idx = triangles.len();

    let mut min_xyz = [f32::INFINITY; 3];
    let mut max_xyz = [f32::NEG_INFINITY; 3];

    let mut pos_bytes = Vec::with_capacity(n_verts * 12);
    for i in 0..n_verts {
        // Center/convert coordinates into glTF Y-up convention or keep metric coordinates in float32
        let x = vertices[i * 3] as f32;
        let y = vertices[i * 3 + 1] as f32;
        let z = vertices[i * 3 + 2] as f32;
        min_xyz[0] = min_xyz[0].min(x);
        min_xyz[1] = min_xyz[1].min(y);
        min_xyz[2] = min_xyz[2].min(z);
        max_xyz[0] = max_xyz[0].max(x);
        max_xyz[1] = max_xyz[1].max(y);
        max_xyz[2] = max_xyz[2].max(z);
        pos_bytes.extend_from_slice(&x.to_le_bytes());
        pos_bytes.extend_from_slice(&y.to_le_bytes());
        pos_bytes.extend_from_slice(&z.to_le_bytes());
    }
    if n_verts == 0 {
        min_xyz = [0.0; 3];
        max_xyz = [0.0; 3];
    }

    let mut norm_bytes = Vec::with_capacity(n_verts * 12);
    for i in 0..n_verts {
        let nx = normals.get(i * 3).copied().unwrap_or(0.0) as f32;
        let ny = normals.get(i * 3 + 1).copied().unwrap_or(0.0) as f32;
        let nz = normals.get(i * 3 + 2).copied().unwrap_or(1.0) as f32;
        norm_bytes.extend_from_slice(&nx.to_le_bytes());
        norm_bytes.extend_from_slice(&ny.to_le_bytes());
        norm_bytes.extend_from_slice(&nz.to_le_bytes());
    }

    let mut col_bytes = Vec::with_capacity(n_verts * 4);
    for i in 0..n_verts {
        let r = colors_rgba.get(i * 4).copied().unwrap_or(200);
        let g = colors_rgba.get(i * 4 + 1).copied().unwrap_or(200);
        let b = colors_rgba.get(i * 4 + 2).copied().unwrap_or(200);
        let a = colors_rgba.get(i * 4 + 3).copied().unwrap_or(255);
        col_bytes.extend_from_slice(&[r, g, b, a]);
    }

    let mut idx_bytes = Vec::with_capacity(n_idx * 4);
    for &idx in triangles {
        idx_bytes.extend_from_slice(&idx.to_le_bytes());
    }

    let pos_len = pos_bytes.len();
    let norm_offset = pos_len;
    let norm_len = norm_bytes.len();
    let col_offset = norm_offset + norm_len;
    let col_len = col_bytes.len();
    let idx_offset = col_offset + col_len;
    let idx_len = idx_bytes.len();

    let mut bin_chunk = Vec::with_capacity(idx_offset + idx_len + 4);
    bin_chunk.extend_from_slice(&pos_bytes);
    bin_chunk.extend_from_slice(&norm_bytes);
    bin_chunk.extend_from_slice(&col_bytes);
    bin_chunk.extend_from_slice(&idx_bytes);
    while bin_chunk.len() % 4 != 0 {
        bin_chunk.push(0u8);
    }

    let json_str = format!(
        concat!(
            r#"{{"asset":{{"generator":"pyGeoMapper Rust mesh3d","version":"2.0"}},"#,
            r#""scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0,"name":"GeologicalBlock"}}],"#,
            r#""meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"NORMAL":1,"COLOR_0":2}},"indices":3,"mode":4}}]}}],"#,
            r#""accessors":["#,
            r#"{{"bufferView":0,"componentType":5126,"count":{n_verts},"type":"VEC3","min":[{minx:.3},{miny:.3},{minz:.3}],"max":[{maxx:.3},{maxy:.3},{maxz:.3}]}},"#,
            r#"{{"bufferView":1,"componentType":5126,"count":{n_verts},"type":"VEC3"}},"#,
            r#"{{"bufferView":2,"componentType":5121,"normalized":true,"count":{n_verts},"type":"VEC4"}},"#,
            r#"{{"bufferView":3,"componentType":5125,"count":{n_idx},"type":"SCALAR"}}"#,
            r#"],"bufferViews":["#,
            r#"{{"buffer":0,"byteOffset":0,"byteLength":{pos_len},"target":34962}},"#,
            r#"{{"buffer":0,"byteOffset":{norm_offset},"byteLength":{norm_len},"target":34962}},"#,
            r#"{{"buffer":0,"byteOffset":{col_offset},"byteLength":{col_len},"target":34962}},"#,
            r#"{{"buffer":0,"byteOffset":{idx_offset},"byteLength":{idx_len},"target":34963}}"#,
            r#"],"buffers":[{{"byteLength":{bin_len}}}]}}"#
        ),
        n_verts = n_verts,
        n_idx = n_idx,
        minx = min_xyz[0],
        miny = min_xyz[1],
        minz = min_xyz[2],
        maxx = max_xyz[0],
        maxy = max_xyz[1],
        maxz = max_xyz[2],
        pos_len = pos_len,
        norm_offset = norm_offset,
        norm_len = norm_len,
        col_offset = col_offset,
        col_len = col_len,
        idx_offset = idx_offset,
        idx_len = idx_len,
        bin_len = bin_chunk.len(),
    );

    let mut json_bytes = json_str.into_bytes();
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(0x20); // space padding per glTF 2.0 spec
    }

    let total_length = 12 + 8 + json_bytes.len() + 8 + bin_chunk.len();
    let mut out = BufWriter::new(File::create(path)?);
    // 12-byte GLB header
    out.write_all(&0x46546C67u32.to_le_bytes())?; // "glTF"
    out.write_all(&2u32.to_le_bytes())?; // version 2
    out.write_all(&(total_length as u32).to_le_bytes())?;
    // Chunk 0: JSON
    out.write_all(&(json_bytes.len() as u32).to_le_bytes())?;
    out.write_all(&0x4E4F534Au32.to_le_bytes())?; // "JSON"
    out.write_all(&json_bytes)?;
    // Chunk 1: BIN
    out.write_all(&(bin_chunk.len() as u32).to_le_bytes())?;
    out.write_all(&0x004E4942u32.to_le_bytes())?; // "BIN\0"
    out.write_all(&bin_chunk)?;
    out.flush()?;

    Ok(total_length)
}

/// Export a 3D triangle mesh to an ASCII Stanford `.ply` file with per-vertex normals and RGBA colors.
pub fn export_mesh_ply(
    path: &str,
    vertices: &[f64],
    normals: &[f64],
    triangles: &[u32],
    colors_rgba: &[u8],
) -> std::io::Result<usize> {
    let n_verts = vertices.len() / 3;
    let n_tris = triangles.len() / 3;
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "ply")?;
    writeln!(out, "format ascii 1.0")?;
    writeln!(out, "comment Generated by pyGeoMapper Rust mesh3d")?;
    writeln!(out, "element vertex {n_verts}")?;
    writeln!(out, "property float x")?;
    writeln!(out, "property float y")?;
    writeln!(out, "property float z")?;
    writeln!(out, "property float nx")?;
    writeln!(out, "property float ny")?;
    writeln!(out, "property float nz")?;
    writeln!(out, "property uchar red")?;
    writeln!(out, "property uchar green")?;
    writeln!(out, "property uchar blue")?;
    writeln!(out, "property uchar alpha")?;
    writeln!(out, "element face {n_tris}")?;
    writeln!(out, "property list uchar uint vertex_indices")?;
    writeln!(out, "end_header")?;

    for i in 0..n_verts {
        let x = vertices[i * 3];
        let y = vertices[i * 3 + 1];
        let z = vertices[i * 3 + 2];
        let nx = normals.get(i * 3).copied().unwrap_or(0.0);
        let ny = normals.get(i * 3 + 1).copied().unwrap_or(0.0);
        let nz = normals.get(i * 3 + 2).copied().unwrap_or(1.0);
        let r = colors_rgba.get(i * 4).copied().unwrap_or(200);
        let g = colors_rgba.get(i * 4 + 1).copied().unwrap_or(200);
        let b = colors_rgba.get(i * 4 + 2).copied().unwrap_or(200);
        let a = colors_rgba.get(i * 4 + 3).copied().unwrap_or(255);
        writeln!(out, "{x:.3} {y:.3} {z:.3} {nx:.4} {ny:.4} {nz:.4} {r} {g} {b} {a}")?;
    }

    for t in 0..n_tris {
        let i0 = triangles[t * 3];
        let i1 = triangles[t * 3 + 1];
        let i2 = triangles[t * 3 + 2];
        writeln!(out, "3 {i0} {i1} {i2}")?;
    }
    out.flush()?;
    Ok(n_verts + n_tris)
}

/// Export a 3D triangle mesh to Wavefront `.obj` format (including vertex RGB colors and normals).
pub fn export_mesh_obj(
    path: &str,
    vertices: &[f64],
    normals: &[f64],
    triangles: &[u32],
    colors_rgba: &[u8],
) -> std::io::Result<usize> {
    let n_verts = vertices.len() / 3;
    let n_tris = triangles.len() / 3;
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(out, "# Generated by pyGeoMapper Rust mesh3d")?;
    for i in 0..n_verts {
        let x = vertices[i * 3];
        let y = vertices[i * 3 + 1];
        let z = vertices[i * 3 + 2];
        let r = (colors_rgba.get(i * 4).copied().unwrap_or(200) as f64) / 255.0;
        let g = (colors_rgba.get(i * 4 + 1).copied().unwrap_or(200) as f64) / 255.0;
        let b = (colors_rgba.get(i * 4 + 2).copied().unwrap_or(200) as f64) / 255.0;
        writeln!(out, "v {x:.3} {y:.3} {z:.3} {r:.4} {g:.4} {b:.4}")?;
    }
    for i in 0..n_verts {
        let nx = normals.get(i * 3).copied().unwrap_or(0.0);
        let ny = normals.get(i * 3 + 1).copied().unwrap_or(0.0);
        let nz = normals.get(i * 3 + 2).copied().unwrap_or(1.0);
        writeln!(out, "vn {nx:.4} {ny:.4} {nz:.4}")?;
    }
    for t in 0..n_tris {
        let i0 = triangles[t * 3] + 1;
        let i1 = triangles[t * 3 + 1] + 1;
        let i2 = triangles[t * 3 + 2] + 1;
        writeln!(out, "f {i0}//{i0} {i1}//{i1} {i2}//{i2}")?;
    }
    out.flush()?;
    Ok(n_verts + n_tris)
}
