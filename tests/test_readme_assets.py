# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
README / packaging smoke tests (fast; need neither the Rust core nor example data):
- every image referenced by `README.md` exists in `image/` and stays within the size budget
- `README-pypi.md` is a text-only package overview (no figures or usage examples)
- no machine-specific absolute paths in shipped text files (code, notebooks, docs, metadata)
- the version string agrees across `pyproject.toml`, `Cargo.toml`, `CITATION.cff`, `pygeomapper/__init__.py`
  and the PyPI README citation
"""

from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent.parent
README_IMAGE_MAX_BYTES = 800_000
# Multi-panel full-resolution board and 24-second animation have separate budgets.
README_ASSET_BUDGETS = {
    "image/slumgullion_earthflow.png": 1_500_000,
    "image/3d_block_diagram_rotate.gif": 16_000_000,
    "image/holiday_farm_fire.png": 1_600_000,
    "image/yolo_bypass_flood.png": 1_300_000,
}
RAW_PREFIX = "https://raw.githubusercontent.com/GeoLarryLai/pyGeoMapper/main/"
TEXT_SUFFIXES = {".py", ".rs", ".md", ".ipynb", ".toml", ".yml", ".yaml", ".cff", ".txt", ".json"}
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "target",
    "outputs",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
}
# Built from pieces so this file does not match itself
ABSOLUTE_PATH_MARKERS = ["/" + "Users" + "/", "/" + "home" + "/", "/" + "workspace" + "/"]


def _readme_image_refs(readme: Path) -> list[str]:
    text = readme.read_text(encoding="utf-8")
    refs = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text) + re.findall(r"<img[^>]+src=\"([^\"]+)\"", text)
    out = []
    for ref in refs:
        if ref.startswith(RAW_PREFIX):
            ref = ref[len(RAW_PREFIX):]
        if ref.startswith("image/"):
            out.append(ref)
    return out


def test_readme_images_exist_and_fit_budget() -> None:
    refs = _readme_image_refs(REPO_ROOT / "README.md")
    assert refs, "README.md references no image/ files"
    for ref in refs:
        p = REPO_ROOT / ref
        assert p.is_file(), f"README.md -> missing {ref}"
        assert p.stat().st_size <= README_ASSET_BUDGETS.get(ref, README_IMAGE_MAX_BYTES), f"README.md -> {ref} is {p.stat().st_size / 1e3:.0f} KB"


def test_pypi_readme_is_overview_only() -> None:
    text = (REPO_ROOT / "README-pypi.md").read_text(encoding="utf-8")
    assert not _readme_image_refs(REPO_ROOT / "README-pypi.md")
    assert "![" not in text
    assert "<img" not in text.lower()
    assert "```python" not in text


def test_readme_gallery_uses_restored_science_figures() -> None:
    """GitHub README must show LiCSAR, Sheep Mountain CWT board, Oso Δcomplexity, and dual-section 3D."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "image/case1a_myanmar_earthquake.jpg",
        "image/cwt_multiscale.png",
        "image/oso_landslide_complexity.png",
        "image/3d_block_diagram_rotate.gif",
        "image/case1d_jakobshavn_glacier.jpg",
        "image/slumgullion_earthflow.png",
        "image/cross_section.png",
        "image/holiday_farm_fire.png",
        "image/yolo_bypass_flood.png",
    ):
        assert required in text, f"README missing {required}"
    assert "image/3d_block_diagram.png" not in text
    assert (REPO_ROOT / "image" / "3d_block_diagram.png").is_file()
    assert "image/myanmar_insar_3d.jpg" not in text
    assert "image/oso_cwt_change.png" not in text
    assert "image/oso_landslide_dod.png" not in text
    assert "SW-dipping" in text or "basement thrust" in text
    assert "NE-dipping forelimb reverse fault" not in text
    assert "Δ complexity" in text or "delta complexity" in text.lower()
    assert (REPO_ROOT / "image" / "oso_landslide_complexity.png").is_file()
    assert not (REPO_ROOT / "image" / "oso_cwt_change.png").exists()
    assert not (REPO_ROOT / "image" / "oso_landslide_dod.png").exists()


def test_no_machine_absolute_paths_in_shipped_files() -> None:
    offenders = []
    for p in REPO_ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in TEXT_SUFFIXES or SKIP_DIRS.intersection(p.relative_to(REPO_ROOT).parts):
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        hits = [m for m in ABSOLUTE_PATH_MARKERS if m in text]
        if hits:
            offenders.append(f"{p.relative_to(REPO_ROOT)}: {hits}")
    assert not offenders, "absolute machine paths found:\n" + "\n".join(offenders)


def _first(pattern: str, path: Path) -> str:
    m = re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    assert m, f"no version found in {path.name}"
    return m.group(1)


def test_version_strings_agree() -> None:
    versions = {
        "pyproject.toml": _first(r'^version\s*=\s*"([^"]+)"', REPO_ROOT / "pyproject.toml"),
        "Cargo.toml": _first(r'^version\s*=\s*"([^"]+)"', REPO_ROOT / "Cargo.toml"),
        "CITATION.cff": _first(r'^version:\s*"?([^"\s]+)"?', REPO_ROOT / "CITATION.cff"),
        "pygeomapper/__init__.py": _first(r'^__version__\s*=\s*"([^"]+)"', REPO_ROOT / "pygeomapper" / "__init__.py"),
    }
    pypi_cite = re.findall(r"\(version ([0-9][^)]*)\)", (REPO_ROOT / "README-pypi.md").read_text(encoding="utf-8"))
    versions.update({f"README-pypi.md citation #{i + 1}": v for i, v in enumerate(pypi_cite)})
    assert len(set(versions.values())) == 1, f"version drift: {versions}"


def test_release_030_public_apis() -> None:
    import pygeomapper as pgm

    assert pgm.__version__ == "0.3.0"
    assert pgm.blend_dem is pgm.raster_io.blend_dem
    assert pgm.compute_los_from_unwrapped_phase is pgm.surface_change.compute_los_from_unwrapped_phase
    assert pgm.animate_3d_block_and_section is pgm.viewer_3d.animate_3d_block_and_section


def test_readme_rotation_keeps_slow_reviewed_timing() -> None:
    from PIL import Image

    with Image.open(REPO_ROOT / "image" / "3d_block_diagram_rotate.gif") as image:
        assert image.size == (1400, 643)
        assert image.n_frames == 72
        assert image.info["loop"] == 0
        for frame in range(image.n_frames):
            image.seek(frame)
            assert image.info["duration"] == 330


def test_readme_gallery_order_and_settled_stories() -> None:
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    expected = [
        "image/cwt_multiscale.png",
        "image/auto_mapping_validation.png",
        "image/cross_section.png",
        "image/3d_block_diagram_rotate.gif",
        "image/dod_and_spectral_change.png",
        "image/oso_landslide_complexity.png",
        "image/holiday_farm_fire.png",
        "image/yolo_bypass_flood.png",
        "image/case1a_myanmar_earthquake.jpg",
        "image/slumgullion_earthflow.png",
        "image/case1d_jakobshavn_glacier.jpg",
    ]
    positions = [text.index(ref) for ref in expected]
    assert positions == sorted(positions)
    assert "2024−2015" in text
    assert "red = lowering, blue = raising" in text
    assert "image/case1e_slumgullion_landslide.jpg" not in text
    assert "jakobshavn_dod" not in text.lower()


def test_release_notebooks_match_shipped_figures() -> None:
    import json

    sources = {}
    for number in (4, 5):
        path = next((REPO_ROOT / "notebooks").glob(f"pyGeoMapper_{number}_*.ipynb"))
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        for i, cell in enumerate(cells):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"{path.name}:cell{i}", "exec")
        sources[number] = "\n".join("".join(cell["source"]) for cell in cells)
    assert "animate_3d_block_and_section" in sources[4]
    assert "frames=72, fps=3" in sources[4]
    assert "azimuth_deg=-33.0" in sources[4]
    assert "readme_3d_kwargs" in sources[4]
    assert "slumgullion_earthflow.png" in sources[5]
    assert "compute_los_from_unwrapped_phase" in sources[5]
    assert "wavelength_m=0.238403545" in sources[5]
    for case in (2, 3, 4, 5):
        assert f"## Case {case}" in sources[5]


def test_regeneration_can_copy_reviewed_assets_without_reencoding(tmp_path, monkeypatch) -> None:
    from scripts import regenerate_readme_science_figures as figures

    source, dest = tmp_path / "reviewed", tmp_path / "image"
    source.mkdir()
    dest.mkdir()
    for i, name in enumerate(figures.APPROVED_ASSETS.values()):
        (source / name).write_bytes(bytes([i]) * (100 + i))
    monkeypatch.setattr(figures, "IMG", dest)
    outputs = figures.copy_approved_assets(source)
    assert len(outputs) == 4
    for target_name, source_name in figures.APPROVED_ASSETS.items():
        assert (dest / target_name).read_bytes() == (source / source_name).read_bytes()
