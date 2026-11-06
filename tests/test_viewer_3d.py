# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import json
from pathlib import Path
import struct

import matplotlib.pyplot as plt
import numpy as np

from pygeomapper import _core, raster_io, viewer_3d

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "example_data"


def test_rust_3d_block_volume_and_fault_offset():
    """Verify Rust 3D voxel volume evaluation, near-surface bedrock lock, and 3D fault throw."""
    rows, cols, nz = 24, 28, 20
    x0, y0, dx, dy = 500000.0, 4900000.0, 20.0, -20.0
    xs = x0 + (np.arange(cols) + 0.5) * dx
    ys = y0 + (np.arange(rows) + 0.5) * dy
    xx, yy = np.meshgrid(xs, ys)

    # Gently dipping DEM and synthetic 3-formation outcrop pattern
    dem = 1200.0 + 0.08 * (xx - x0)
    surf_units = np.zeros((rows, cols), dtype=np.int32)
    surf_units[:, cols // 3 : (2 * cols) // 3] = 1
    surf_units[:, (2 * cols) // 3 :] = 2

    h_struct = 1150.0 - 0.12 * (xx - x0)
    thresholds = np.array([60.0, 140.0], dtype=np.float64)
    bedrock_ids = np.array([0, 1, 2], dtype=np.int32)

    # Dipping reverse fault with +55 m hanging-wall throw
    fault_params = (float(np.mean(xs)), float(np.mean(ys)), 1180.0, 0.0, 60.0, 55.0, 0.0)
    vol = _core.build_3d_block_volume_rs(
        np.ascontiguousarray(dem, dtype=np.float64),
        np.ascontiguousarray(surf_units, dtype=np.int32),
        np.ascontiguousarray(h_struct, dtype=np.float64),
        thresholds,
        bedrock_ids,
        x0,
        y0,
        dx,
        dy,
        900.0,
        1300.0,
        nz,
        -1,
        0.0,
        15.0,
        fault_params,
    )

    vox = vol["voxel_units"]
    flt_off = vol["fault_offset_volume"]
    assert vox.shape == (rows, cols, nz)
    assert vol["horizon_elevations"].shape == (2, rows, cols)
    # Voxels at z=900 (index 0) are below DEM everywhere and must have valid formation IDs (0, 1, or 2)
    assert np.all(vox[:, :, 0] >= 0)
    # Voxels at z=1300 (index nz-1) are above DEM everywhere and must be -1 (air)
    assert np.all(vox[:, :, -1] == -1)
    # Verify hanging-wall fault throw (+55 m) is present in fault_offset_volume
    assert np.isclose(np.max(flt_off), 55.0)
    assert np.isclose(np.min(flt_off), 0.0)


def test_sheep_mountain_3d_block_and_cutaway_meshes():
    """Verify full 3D block diagram and cutaway mesh topology, normals, and face groups on Sheep Mountain."""
    dem = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_dem_3dep.tif")
    geol = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_geol_units.tif")
    units_meta = json.loads((DATA_DIR / "sheep_mtn_units.json").read_text(encoding="utf-8"))

    xs = dem.x_coords()
    ys = dem.y_coords()
    fault_spec = {
        "name": "Sheep Mountain Forelimb Thrust",
        "x0": float(xs.min() + 0.34 * (xs.max() - xs.min())),
        "y0": float(ys.min() + 0.34 * (ys.max() - ys.min())),
        "z0": 1245.0,
        "strike_deg": 140.0,
        "dip_deg": 58.0,
        "throw_m": 75.0,
    }

    block = viewer_3d.build_3d_block_diagram(
        dem=dem,
        geol_labels=geol,
        units_metadata=units_meta,
        z_bottom=750.0,
        nz=48,
        fault_spec=fault_spec,
        cutaway_fraction=(0.45, 0.45, 0.35),
    )

    mesh = block["mesh"]
    cut_mesh = block["cutaway_mesh"]
    assert mesh["n_vertices"] > 80_000
    assert mesh["n_triangles"] > 70_000
    assert cut_mesh["n_vertices"] > mesh["n_vertices"]

    # Check unit normal lengths == 1.0
    norms = mesh["normals"]
    n_lengths = np.linalg.norm(norms, axis=1)
    assert np.allclose(n_lengths, 1.0, atol=1e-5)

    # Check triangle vertex indices are within valid bounds [0, n_vertices)
    tris = mesh["triangles"]
    assert int(tris.max()) < mesh["n_vertices"]

    # Verify face groups present: 0 (top), 1..4 (N/S/E/W walls), 5 (base), 7 (fault plane)
    fg = set(np.unique(mesh["face_groups"]).tolist())
    for expected_fg in (0, 1, 2, 3, 4, 5, 7):
        assert expected_fg in fg
    # Cutaway mesh must additionally include face_group 6 (internal cutaway walls & floor)
    assert 6 in set(np.unique(cut_mesh["face_groups"]).tolist())

    # Verify all 7 formations (0..6) are represented in the 3D subsurface voxel volume
    stats = {s["id"]: s["voxel_count"] for s in block["unit_volume_stats"]}
    for uid in range(7):
        assert stats[uid] > 0


def test_3d_fence_diagram_slicing():
    """Verify intersecting 3D fence cross-section slicing in Rust."""
    dem = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_dem_3dep.tif")
    geol = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_geol_units.tif")
    units_meta = json.loads((DATA_DIR / "sheep_mtn_units.json").read_text(encoding="utf-8"))

    block = viewer_3d.build_3d_block_diagram(
        dem=dem, geol_labels=geol, units_metadata=units_meta, z_bottom=750.0, nz=32
    )
    fences = viewer_3d.slice_3d_fence_diagram(block, n_samples=100)
    assert len(fences) == 3
    for panel in fences:
        assert len(panel["distance_m"]) == 100
        assert panel["section_units"].shape == (32, 100)
        assert panel["mesh"]["n_triangles"] > 1000


def test_glb_ply_obj_and_interactive_html_export(tmp_path: Path):
    """Verify binary glTF 2.0 (.glb), Stanford .ply, Wavefront .obj, and offline HTML5 3D viewer export."""
    dem = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_dem_3dep.tif")
    geol = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_geol_units.tif")
    units_meta = json.loads((DATA_DIR / "sheep_mtn_units.json").read_text(encoding="utf-8"))

    block = viewer_3d.build_3d_block_diagram(
        dem=dem, geol_labels=geol, units_metadata=units_meta, z_bottom=800.0, nz=24
    )

    glb_path = tmp_path / "block.glb"
    ply_path = tmp_path / "cutaway.ply"
    obj_path = tmp_path / "block.obj"
    html_path = tmp_path / "viewer.html"

    r_glb = viewer_3d.export_3d_block_mesh(block, glb_path)
    r_ply = viewer_3d.export_3d_block_mesh(block, ply_path, use_cutaway=True)
    r_obj = viewer_3d.export_3d_block_mesh(block, obj_path)

    # Check GLB magic header ('glTF', version 2, exact byte length)
    raw_glb = glb_path.read_bytes()
    magic, version, total_len = struct.unpack("<III", raw_glb[:12])
    assert magic == 0x46546C67  # 'glTF'
    assert version == 2
    assert total_len == len(raw_glb) == r_glb["file_size_bytes"]

    assert ply_path.read_text(encoding="utf-8")[:15].startswith("ply\nformat asci")
    assert r_ply["n_triangles"] > 10_000
    assert obj_path.read_text(encoding="utf-8").startswith("# Generated by pyGeoMapper")
    assert r_obj["n_vertices"] > 10_000

    # Check interactive HTML viewer
    r_html = viewer_3d.export_interactive_block_html(
        block, html_path, grid_stride=4, title="Test 3D Block Viewer"
    )
    html_txt = html_path.read_text(encoding="utf-8")
    assert "Test 3D Block Viewer" in html_txt
    assert "glCanvas" in html_txt
    assert r_html["file_size_bytes"] > 50_000


def test_3d_renderers_and_forge3d_adapter():
    """Verify static 3D block, cutaway, fence plotters and forge3d adapter."""
    dem = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_dem_3dep.tif")
    geol = raster_io.load_geotiff(DATA_DIR / "sheep_mtn_geol_units.tif")
    units_meta = json.loads((DATA_DIR / "sheep_mtn_units.json").read_text(encoding="utf-8"))

    forge_res = viewer_3d.render_with_forge3d(dem)
    assert forge_res["draped_rgb"].shape == (dem.rows, dem.cols, 3)

    block = viewer_3d.build_3d_block_diagram(
        dem=dem, geol_labels=geol, units_metadata=units_meta, z_bottom=800.0, nz=20
    )
    fig1, _ = viewer_3d.plot_3d_block_diagram(block, top_stride=4, wall_stride=2)
    plt.close(fig1)
    fig2, _ = viewer_3d.plot_3d_cutaway_block(block, face_stride=3)
    plt.close(fig2)
    fig3, _ = viewer_3d.plot_3d_fence_diagram(block)
    plt.close(fig3)

    # Fault trace on the top surface and front walls when show_fault_plane=True (NE-dipping, strike 320)
    xs, ys = dem.x_coords(), dem.y_coords()
    fault = {"x0": float(xs.mean()), "y0": float(ys.mean()), "z0": 1300.0, "strike_deg": 320.0, "dip_deg": 58.0,
             "throw_m": 75.0}
    block_f = viewer_3d.build_3d_block_diagram(
        dem=dem, geol_labels=geol, units_metadata=units_meta, z_bottom=800.0, nz=20, fault_spec=fault
    )
    fig4, ax4 = viewer_3d.plot_3d_block_diagram(block_f, top_stride=4, wall_stride=2, show_fault_plane=True)
    red = [ln for ln in ax4.get_lines() if ln.get_color() == "#d62728"]
    assert red, "fault trace not drawn"
    plt.close(fig4)


def test_notebook_4_executed_cleanly():
    """Saved notebook JSON only: every code cell has an execution count and output, and the figures exist.
    This does NOT re-run the notebook; `outputs/` (HTML viewer, meshes) is untracked, so run notebook 4 locally first."""
    nb_path = REPO_ROOT / "notebooks" / "pyGeoMapper_4_3d_block_diagram_and_viewer.ipynb"
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 6
    for idx, c in enumerate(code_cells):
        assert c.get("execution_count") is not None, f"Code cell {idx} was not executed"
        assert len(c.get("outputs", [])) > 0, f"Code cell {idx} has no outputs"

    for fig_name in (
        "3d_block_diagram.png",
        "3d_cutaway_and_fence.png",
        "surface_drape_comparison.png",
    ):
        assert (REPO_ROOT / "image" / fig_name).exists()
    assert (REPO_ROOT / "outputs" / "sheep_mtn_3d_block_viewer.html").exists()
