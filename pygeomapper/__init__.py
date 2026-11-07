# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
pyGeoMapper: Geoscience remote sensing, bi-temporal surface change detection,
3D terrain visualization, and automated geological mapping toolkit (Python + Rust).

Note: This geoscience package is distinct from the unrelated The-Magicians-Code/PyGeoMapper repository.
Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later).
"""

from . import (
    auto_mapping,
    cross_section,
    insar3d,
    plotting,
    raster_io,
    spectral_indices,
    surface_change,
    terrain_analysis,
    viewer_3d,
)
from .raster_io import GeoGrid, RasterStack, blend_dem
from .surface_change import compute_los_from_unwrapped_phase
from .viewer_3d import animate_3d_block_and_section

try:
    from . import _core

    HAS_RUST_CORE = True
except ImportError:  # pragma: no cover
    _core = None
    HAS_RUST_CORE = False

__version__ = "0.3.0"

__all__ = [
    "__version__",
    "HAS_RUST_CORE",
    "GeoGrid",
    "RasterStack",
    "blend_dem",
    "compute_los_from_unwrapped_phase",
    "animate_3d_block_and_section",
    "raster_io",
    "spectral_indices",
    "surface_change",
    "terrain_analysis",
    "auto_mapping",
    "cross_section",
    "viewer_3d",
    "insar3d",
    "plotting",
]
