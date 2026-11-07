# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
from __future__ import annotations

import json
from pathlib import Path
import struct

import matplotlib.pyplot as plt
import numpy as np

from pygeomapper import _core, cross_section, raster_io, viewer_3d

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

    start_xy, end_xy = (726900.0, 4942800.0), (729700.0, 4944600.0)
    tip = cross_section.locate_fault_tip_along_section(dem, start_xy, end_xy)
    fault_spec = cross_section.reconstruct_planar_fault(
        dem,
        strike_deg=140.0,
        dip_deg=50.0,
        throw_m=float(np.clip(0.5 * tip["section_relief_m"], 80.0, 200.0)),
        name="Sheep Mountain basement thrust",
        tip_section=(start_xy, end_xy, tip["distance_m"]),
        z0=tip["z"],
        published={"dip_deg": 50.0, "strike_deg": 140.0},
        inferred={"throw_m": "from DEM relief", "tip": "forelimb toe"},
    )
    assert fault_spec["dip_deg"] == 50.0
    assert fault_spec["strike_deg"] == 140.0

    block = viewer_3d.build_3d_block_diagram(
        dem=dem,
        geol_labels=geol,
        units_metadata=units_meta,
        z_bottom=750.0,
        nz=48,
        fault_spec=fault_spec,
        include_fault=True,
        cutaway_fraction=(0.45, 0.45, 0.35),
    )
    block_plain = viewer_3d.build_3d_block_diagram(
        dem=dem,
        geol_labels=geol,
        units_metadata=units_meta,
        z_bottom=750.0,
        nz=32,
        fault_spec=fault_spec,
        include_fault=False,
    )
    assert block_plain["fault_spec"] is None
    assert np.allclose(block_plain["fault_offset_volume"], 0.0)

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

    # SW-dipping basement thrust (strike 140° RHR); show_fault_plane draws surface/wall traces on solid block
    start_xy, end_xy = (726900.0, 4942800.0), (729700.0, 4944600.0)
    tip = cross_section.locate_fault_tip_along_section(dem, start_xy, end_xy)
    fault = cross_section.reconstruct_planar_fault(
        dem,
        strike_deg=140.0,
        dip_deg=50.0,
        throw_m=120.0,
        tip_section=(start_xy, end_xy, tip["distance_m"]),
        z0=tip["z"],
    )
    block_f = viewer_3d.build_3d_block_diagram(
        dem=dem, geol_labels=geol, units_metadata=units_meta, z_bottom=800.0, nz=20, fault_spec=fault
    )
    fig4, ax4 = viewer_3d.plot_3d_block_diagram(block_f, top_stride=4, wall_stride=2, show_fault_plane=True)
    red = [ln for ln in ax4.get_lines() if ln.get_color() == "#d62728"]
    assert red, "fault trace not drawn"
    plt.close(fig4)

    b0, b1 = (728390.0, 4941805.0), (728250.0, 4944795.0)
    fig5, axes5 = viewer_3d.plot_3d_block_and_section(
        block_f,
        start_xy,
        end_xy,
        sections=[(start_xy, end_xy, "A", "A'"), (b0, b1, "B", "B'")],
        n_samples=80,
        top_stride=2,
        wall_stride=2,
        azimuth_deg=-78.0,
        elevation_deg=28.0,
    )
    # Two panels only: left opaque block | right one box containing both A–A′ and B–B′ fences.
    assert len(axes5) == 2
    # The solid left block is a single DEM/walls shell, never interior fences.
    assert len(axes5[0].collections) == 1
    assert len(fig5.legends) == 1
    assert len(fig5.legends[0].get_texts()) == 8
    assert fig5.legends[0]._ncols == 4
    assert all(":" not in txt.get_text() for txt in fig5.legends[0].get_texts())
    # Left opaque block should carry section marks (black) and a red fault trace with ticks.
    assert any(ln.get_color() in ("#111111", "k", "black") for ln in axes5[0].get_lines())
    assert any(ln.get_color() == "#d62728" for ln in axes5[0].get_lines())
    # Right panel labels both section ends (A/A' and B/B').
    right_txt = " ".join(t.get_text() for t in axes5[1].texts)
    assert "A" in right_txt and "B" in right_txt
    # Both panels share the same camera.
    assert axes5[0].azim == axes5[1].azim == -78.0
    assert axes5[0].elev == axes5[1].elev == 28.0
    # Both opaque fences share one per-triangle depth sort, plus transparent fault.
    assert len(axes5[1].collections) == 2
    intersections = [ln for ln in axes5[1].get_lines()
                     if ln.get_color() == "#000000" and 0.8 <= ln.get_linewidth() <= 1.2]
    assert len(intersections) == 1
    ix, iy, iz = intersections[0].get_data_3d()
    assert ix[0] == ix[1] and iy[0] == iy[1] and iz[1] > iz[0]
    plt.close(fig5)


def test_notebook_4_executed_cleanly():
    """Saved notebook JSON only: every code cell has an execution count and output, and the figures exist.
    This does NOT re-run the notebook; HTML viewer and mesh exports in `outputs/` are untracked."""
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
        "3d_block_diagram_rotate.gif",
        "3d_cutaway_and_fence.png",
        "surface_drape_comparison.png",
    ):
        assert (REPO_ROOT / "image" / fig_name).exists()
    # The HTML viewer is an untracked local export, not a shipped test fixture.
    assert "export_interactive_block_html" in "\n".join("".join(c["source"]) for c in code_cells)


def _small_control_block():
    """Small real Rust-built block for inexpensive plotting API regressions."""
    rows, cols = 12, 14
    xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
    transform = (500000.0, 20.0, 0.0, 4900000.0, 0.0, -20.0)
    dem = viewer_3d.GeoGrid(1200.0 + 2.0 * xx + yy, transform=transform)
    geol = viewer_3d.GeoGrid((xx >= cols // 2).astype(np.int32), transform=transform)
    start, end = (500020.0, 4899790.0), (500250.0, 4899970.0)
    fault = cross_section.reconstruct_planar_fault(
        dem, strike_deg=140.0, dip_deg=50.0, throw_m=30.0,
        tip_xy=(500140.0, 4899880.0), z0=1210.0,
    )
    block = viewer_3d.build_3d_block_diagram(
        dem, geol, [{"id": 0, "code": "U0", "color": "#aabbcc"},
                    {"id": 1, "code": "U1", "color": "#eebb44"}],
        z_bottom=950.0, nz=12, fault_spec=fault, include_fault=True,
    )
    return block, start, end


def test_reviewed_defaults_and_surface_contact_controls():
    block, start, end = _small_control_block()
    # Unit colors must not have a second, baked-in pixel-width black outline.
    rgb = block["surface_drape_rgba"][..., :3]
    assert not np.any(np.all(rgb < 40, axis=-1))
    fig, axes = viewer_3d.plot_3d_block_and_section(block, start, end, n_samples=20)
    try:
        assert all((ax.azim, ax.elev, ax.roll) == (-33.0, 28.0, 0.0) for ax in axes)
        contacts = [ln for ln in axes[0].lines if ln.get_linewidth() == 0.4]
        assert contacts
        assert np.allclose(axes[0].collections[0].get_facecolor()[:, 3], 1.0)
        assert np.allclose(axes[1].collections[0].get_facecolor()[:, 3], 1.0)
        assert np.allclose(axes[1].collections[1].get_facecolor()[:, 3], 0.10)
    finally:
        plt.close(fig)

    # 0 suppresses extra surface contact polylines; independent fill/fault controls.
    fig, axes = viewer_3d.plot_3d_block_and_section(
        block, start, end, n_samples=20, unit_boundary_linewidth=0,
        fill_alpha=0.35, section_alpha=0.6, fault_alpha=0.04, show_fault_plane=False,
    )
    try:
        assert not any(ln.get_linewidth() == 0.4 for ln in axes[0].lines)
        assert not any(ln.get_color() == "#d62728" for ln in axes[0].lines)
        assert len(axes[1].collections) == 1
        assert np.allclose(axes[0].collections[0].get_facecolor()[:, 3], 0.35)
        assert np.allclose(axes[1].collections[0].get_facecolor()[:, 3], 0.6)
        assert "Thrust" not in [t.get_text() for t in fig.legends[0].get_texts()]
    finally:
        plt.close(fig)
    fig, ax = viewer_3d.plot_3d_block_diagram(
        block, unit_boundary_linewidth=0.85, show_attitudes=False, fill_alpha=0,
    )
    try:
        assert any(ln.get_linewidth() == 0.85 for ln in ax.lines)
        assert np.allclose(ax.collections[0].get_facecolor()[:, 3], 0.0)
    finally:
        plt.close(fig)


def test_independent_section_and_fault_alpha_controls():
    block, start, end = _small_control_block()
    panels = viewer_3d.slice_3d_fence_diagram(block, transects=[(start, end)], n_samples=20)
    for section_alpha, fault_alpha in [(0.6, 0.04), (0.0, 0.0)]:
        fig, ax = viewer_3d.plot_3d_fence_diagram(
            block, fence_panels=panels, section_alpha=section_alpha,
            fault_alpha=fault_alpha, show_wireframe_surface=False,
        )
        try:
            assert np.allclose(ax.collections[0].get_facecolor()[:, 3], section_alpha)
            assert np.allclose(ax.collections[1].get_facecolor()[:, 3], fault_alpha)
        finally:
            plt.close(fig)


def test_look_toward_camera_conversion_and_legacy_aliases():
    block, start, end = _small_control_block()
    fig, old = viewer_3d.plot_3d_block_and_section(
        block, start, end, n_samples=20, azimuth_deg=-33, elevation_deg=28,
    )
    fig2, new = viewer_3d.plot_3d_block_and_section(
        block, start, end, n_samples=20, azimuth_deg=70, elevation_deg=5,
        block_azimuth_deg=80, section_azimuth_deg=90, section_elevation_deg=6,
        look_azimuth_deg=147, look_elevation_deg=28, roll_deg=0,
    )
    try:
        for a, b in zip(old, new):
            assert np.allclose(a.get_proj(), b.get_proj())
            assert (b.azim, b.elev, b.roll) == (-33.0, 28.0, 0.0)
    finally:
        plt.close(fig)
        plt.close(fig2)

    fig, axes = viewer_3d.plot_3d_block_and_section(
        block, start, end, n_samples=20, look_azimuth_deg=90,
        look_elevation_deg=35, roll_deg=12,
    )
    try:
        assert all((ax.azim, ax.elev, ax.roll) == (-90.0, 35.0, 12.0) for ax in axes)
    finally:
        plt.close(fig)

    # Horizontal look-toward 90 means camera is at -90, looking along +y.
    # Check the installed Matplotlib view vector as well as stored view_init angles.
    for plot in (viewer_3d.plot_3d_block_diagram, viewer_3d.plot_3d_fence_diagram):
        fig, ax = plot(block, azimuth_deg=33, elevation_deg=11,
                       look_azimuth_deg=90, look_elevation_deg=35, roll_deg=12)
        try:
            assert (ax.azim, ax.elev, ax.roll) == (-90.0, 35.0, 12.0)
            ax.get_proj()
            look = -np.asarray(ax._view_w)
            expected = [0.0, np.cos(np.deg2rad(35)), -np.sin(np.deg2rad(35))]
            assert np.allclose(look, expected, atol=1e-12)
        finally:
            plt.close(fig)
    # New names override only their corresponding old angle; wrapping is periodic.
    assert viewer_3d._resolve_3d_view(20, 11, 507, None, 7) == (-33.0, 11.0, 7.0)
    assert viewer_3d._resolve_3d_view(20, 11, None, 42, 7) == (20.0, 42.0, 7.0)


def test_animation_rotates_both_cameras_and_keeps_timing(tmp_path, monkeypatch):
    from PIL import Image
    from mpl_toolkits.mplot3d import Axes3D

    block, start, end = _small_control_block()
    original = Axes3D.view_init
    views = {}

    def record(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        views.setdefault(self, []).append((self.azim, self.elev, self.roll))
        return result

    monkeypatch.setattr(Axes3D, "view_init", record)
    path = viewer_3d.animate_3d_block_and_section(
        block, start, end, tmp_path / "nested" / "rotation.gif",
        frames=4, fps=3, n_samples=20, azimuth_deg=-33, elevation_deg=28,
    )
    assert path.is_file()
    assert len(views) == 2
    expected = [(-33 + angle, 28, 0) for angle in (0, 90, 180, 270)]
    for history in views.values():
        assert history[-4:] == expected
    with Image.open(path) as image:
        assert image.n_frames == 4
        assert image.width <= 1400
        assert image.info["loop"] == 0
        for i in range(image.n_frames):
            image.seek(i)
            assert image.info["duration"] == 330
