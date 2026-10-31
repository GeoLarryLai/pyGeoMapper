<p align="center"><img src="https://raw.githubusercontent.com/GeoLarryLai/pyGeoMapper/main/image/logo.png" alt="pyGeoMapper" width="220"></p>

# pyGeoMapper

[![PyPI version](https://img.shields.io/pypi/v/pygeomapper.svg)](https://pypi.org/project/pygeomapper/) [![pepy downloads](https://static.pepy.tech/badge/pygeomapper)](https://pepy.tech/project/pygeomapper) [![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](https://github.com/GeoLarryLai/pyGeoMapper/blob/main/LICENSE)

![DEM of Difference and spectral change](https://raw.githubusercontent.com/GeoLarryLai/pyGeoMapper/main/image/dod_and_spectral_change.png)
*A DEM of Difference and a spectral change map, the two basic ways the package spots change on the ground.*

> [!WARNING]
> **pyGeoMapper** is under active development, and many functions are still experimental and not fully validated. Please use with caution.

**pyGeoMapper** maps what changed on the ground, and what it's made of, using satellite images and elevation models (DEMs). I wanted one place to go from raw rasters to something a geologist can actually read: change maps after a landslide, flood, wildfire, earthquake, volcano or glacier event, unit maps scored against a real geologic map, and cross-sections and 3D block diagrams you can open in a browser. It's Python with a Rust core. (Not related to `The-Magicians-Code/PyGeoMapper`.)

## What it can do

- Measure surface change: DEM of Difference with co-registration, spectral change, and InSAR/GNSS 3D displacement
- Compute spectral indices (NDVI, NBR, NDWI, mineral ratios) and terrain metrics (slope, curvature, roughness, multiscale CWT complexity)
- Make unsupervised and supervised geologic unit maps, with spatial-block holdout accuracy
- Draw cross-sections and 3D block, cutaway and fence diagrams, and view them in an offline WebGL viewer
- Five executed notebooks walk through every module with public data

![3D geological block diagram](https://raw.githubusercontent.com/GeoLarryLai/pyGeoMapper/main/image/3d_block_diagram.png)
*A 3D block diagram of Sheep Mountain Anticline.*

## Install

```bash
pip install pygeomapper
# from source (Python 3.10+, Rust 1.75+):
git clone https://github.com/GeoLarryLai/pyGeoMapper.git && cd pyGeoMapper
pip install maturin && maturin develop --release
```

Optional extras: `stac`, `ml`, `viz3d`, `spectral`, `dem`, `test`.

## Quick start

```python
from pygeomapper import raster_io, surface_change, terrain_analysis

dem1 = raster_io.load_geotiff("example_data/oso_dem_pre_cop30.tif")
dem2 = raster_io.load_geotiff("example_data/oso_dem_post_3dep.tif")
dod = surface_change.compute_dod(dem1, dem2)                        # elevation change between the two DEMs
cwt = terrain_analysis.compute_multiscale_complexity(dem2)          # terrain roughness at several scales
```

## Key features

- **Raster I/O (`raster_io`):** CRS-aware GeoTIFF/COG and multi-band stacks, STAC search, AlphaEarth embeddings.
- **Spectral indices (`spectral_indices`):** NDVI, NBR, NDWI, NDMI, BSI and mineral ratios in parallel Rust.
- **Surface change (`surface_change`):** DEM of Difference with co-registration and LoD volumes, spectral and
  embedding change, InSAR/GNSS 3D displacement, debris-flow susceptibility and multi-hazard fingerprints.
- **Terrain (`terrain_analysis`):** slope, aspect, hillshade, curvature, roughness and multiscale 2D CWT complexity.
- **Mapping (`auto_mapping`):** unsupervised and supervised unit mapping with spatial-block holdout validation.
- **Cross-sections (`cross_section`):** swath profiles, 3-point bed attitudes and geological cross-sections.
- **3D (`viewer_3d`):** 3D block, cutaway and fence diagrams, GLB/PLY/OBJ export and an offline WebGL viewer.

## Examples

Five executed notebooks in `notebooks/` of the [repository](https://github.com/GeoLarryLai/pyGeoMapper) walk through every module with public data:

1. `pyGeoMapper_1_spectral_and_surface_change.ipynb`: spectral indices and surface change (Oso landslide)
2. `pyGeoMapper_2_terrain_complexity_and_cwt.ipynb`: terrain derivatives and multiscale complexity
3. `pyGeoMapper_3_auto_mapping_and_cross_section.ipynb`: automated unit mapping and a cross-section
4. `pyGeoMapper_4_3d_block_diagram_and_viewer.ipynb`: 3D block diagrams and the offline viewer
5. `pyGeoMapper_5_multi_hazard_change_showcase.ipynb`: earthquake, volcano, glacier, landslide, flood and wildfire cases

## How to cite

> Lai, L. S.-H. (2026). pyGeoMapper (version 0.1.0) [Computer software]. https://github.com/GeoLarryLai/pyGeoMapper

Full metadata is in `CITATION.cff` in the repository. Methods: [Horn (1981)](https://doi.org/10.1109/PROC.1981.11918),
[Zevenbergen & Thorne (1987)](https://doi.org/10.1002/esp.3290120107),
[Nuth & Kääb (2011)](https://doi.org/10.5194/tc-5-271-2011),
[Montero et al. (2023)](https://doi.org/10.1038/s41597-023-02096-0),
[Lai et al. (2025)](https://doi.org/10.5194/esurf-13-417-2025),
[Brown et al. (2025)](https://arxiv.org/abs/2507.22291),
[Okada (1985)](https://doi.org/10.1785/BSSA0750041135),
[Staley et al. (2017)](https://doi.org/10.1016/j.geomorph.2016.10.019),
[Horton et al. (2017)](https://doi.org/10.3133/ds1052). Data sources: `example_data/SOURCES.txt`, `NOTICE`.

Third-party notices and data credits: [`NOTICE`](https://github.com/GeoLarryLai/pyGeoMapper/blob/main/NOTICE).
