# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
README / packaging smoke tests (fast; need neither the Rust core nor example data):
- every image referenced by `README.md` / `README-pypi.md` exists in `image/` and stays within the size budget
- no machine-specific absolute paths in shipped text files (code, notebooks, docs, metadata)
- the version string agrees across `pyproject.toml`, `Cargo.toml`, `CITATION.cff`, `pygeomapper/__init__.py`
  and the PyPI README citation
"""

from __future__ import annotations

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent.parent
README_IMAGE_MAX_BYTES = 800_000
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
    for name in ("README.md", "README-pypi.md"):
        refs = _readme_image_refs(REPO_ROOT / name)
        assert refs, f"{name} references no image/ files"
        for ref in refs:
            p = REPO_ROOT / ref
            assert p.is_file(), f"{name} -> missing {ref}"
            assert p.stat().st_size <= README_IMAGE_MAX_BYTES, f"{name} -> {ref} is {p.stat().st_size / 1e3:.0f} KB"


def test_pypi_readme_images_are_subset_of_github_readme() -> None:
    github = set(_readme_image_refs(REPO_ROOT / "README.md"))
    pypi = set(_readme_image_refs(REPO_ROOT / "README-pypi.md"))
    assert pypi <= github, f"PyPI README shows images the GitHub README does not: {sorted(pypi - github)}"


def test_readme_gallery_uses_restored_science_figures() -> None:
    """GitHub README must show the 4-panel LiCSAR board, six-panel CWT board, and solid+section 3D pair."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for required in (
        "image/case1a_myanmar_earthquake.jpg",
        "image/cwt_multiscale.png",
        "image/3d_block_diagram.png",
        "image/cross_section.png",
    ):
        assert required in text, f"README missing {required}"
    assert "image/myanmar_insar_3d.jpg" not in text
    assert "image/oso_cwt_change.png" not in text
    assert "SW-dipping" in text or "basement thrust" in text
    assert "NE-dipping forelimb reverse fault" not in text


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
