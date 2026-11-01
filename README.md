<p align="center"><img src="image/logo.png" alt="pyGeoMapper" width="220"></p>
<p align="center">
<a href="https://doi.org/10.5281/zenodo.23832020"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23832020-blue" alt="DOI"></a> <a href="https://github.com/GeoLarryLai/pyGeoMapper/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-lightgrey" alt="License"></a>
<br>
<a href="https://pypi.org/project/pygeomapper/"><img src="https://img.shields.io/badge/python-3.10%2B-lightgrey" alt="Python versions"></a> <a href="https://pypi.org/project/pygeomapper/"><img src="https://img.shields.io/pypi/v/pygeomapper?color=orange" alt="PyPI version"></a> <a href="https://pepy.tech/projects/pygeomapper"><img src="https://img.shields.io/pepy/dt/pygeomapper?label=pypi%20downloads&color=orange" alt="PyPI downloads"></a> <a href="https://anaconda.org/conda-forge/pygeomapper"><img src="https://img.shields.io/conda/vn/conda-forge/pygeomapper?label=conda-forge&color=green" alt="conda-forge version"></a> <a href="https://anaconda.org/conda-forge/pygeomapper"><img src="https://img.shields.io/conda/dn/conda-forge/pygeomapper?label=conda%20downloads&color=green" alt="conda downloads"></a>
</p>

# pyGeoMapper

> [!WARNING]
> **pyGeoMapper** is under active development, and many functions are still experimental and not fully validated. Please use with caution.

**pyGeoMapper** maps what changed on the ground, and what it's made of, using satellite images and elevation models (DEMs). I wanted one place to go from raw rasters to something a geologist can actually read: change maps after a landslide, flood, wildfire, earthquake, volcano or glacier event, unit maps scored against a real geologic map, and cross-sections and 3D block diagrams you can open in a browser. It's Python with a Rust core. (Not related to `The-Magicians-Code/PyGeoMapper`.)

## Installation

```bash
pip install pygeomapper
# from source (Python 3.10+, Rust 1.75+):
git clone https://github.com/GeoLarryLai/pyGeoMapper.git && cd pyGeoMapper
pip install maturin && maturin develop --release
```

## Citation

Lai, L.S.-H. (2026) pyGeoMapper. *Zenodo*, [https://doi.org/10.5281/zenodo.23832020](https://doi.org/10.5281/zenodo.23832020)

Methods: [Horn (1981)](https://doi.org/10.1109/PROC.1981.11918),
[Zevenbergen & Thorne (1987)](https://doi.org/10.1002/esp.3290120107),
[Nuth & Kääb (2011)](https://doi.org/10.5194/tc-5-271-2011),
[Montero et al. (2023)](https://doi.org/10.1038/s41597-023-02096-0),
[Lai et al. (2025)](https://doi.org/10.5194/esurf-13-417-2025),
[Brown et al. (2025)](https://arxiv.org/abs/2507.22291),
[Okada (1985)](https://doi.org/10.1785/BSSA0750041135),
[Staley et al. (2017)](https://doi.org/10.1016/j.geomorph.2016.10.019),
[Horton et al. (2017)](https://doi.org/10.3133/ds1052). Data sources: `example_data/SOURCES.txt`, `NOTICE`.

![DEM of Difference and spectral change](image/dod_and_spectral_change.png)
*A DEM of Difference and a spectral change map, the two basic ways the package spots change on the ground.*

## What it can do

- Measure surface change: DEM of Difference with co-registration, spectral change, and InSAR/GNSS 3D displacement
- Compute spectral indices (NDVI, NBR, NDWI, mineral ratios) and terrain metrics (slope, curvature, roughness, multiscale CWT complexity)
- Make unsupervised and supervised geologic unit maps, with spatial-block holdout accuracy
- Draw cross-sections and 3D block, cutaway and fence diagrams, and view them in an offline WebGL viewer
- Five executed notebooks walk through every module with public data

## Examples

![Earthquake displacement](image/readme_myanmar_earthquake.jpg)
*Ground displacement and offsets from the 2025 Mw 7.7 Myanmar earthquake, from Sentinel-1 (COMET LiCSAR).*

![Auto-mapping validation](image/auto_mapping_validation.png)
*Units auto-mapped at Sheep Mountain Anticline compared with the USGS map and scored on a held-out block.*

![Cross-section](image/cross_section.png)
*A Sheep Mountain Anticline cross-section with bed attitudes from 3-point solutions.*

![3D geological block diagram](image/3d_block_diagram.png)
*A 3D block diagram of the same area.*

Five executed notebooks in `notebooks/` walk through every module with public data (spectral and surface change,
terrain complexity, auto-mapping and cross-sections, 3D, and a multi-hazard showcase).

## Quick start

```python
from pygeomapper import raster_io, surface_change, terrain_analysis

dem1 = raster_io.load_geotiff("example_data/oso_dem_pre_cop30.tif")
dem2 = raster_io.load_geotiff("example_data/oso_dem_post_3dep.tif")
dod = surface_change.compute_dod(dem1, dem2)                        # elevation change between the two DEMs
cwt = terrain_analysis.compute_multiscale_complexity(dem2)          # terrain roughness at several scales
```

Third-party notices and data credits: [`NOTICE`](NOTICE).
