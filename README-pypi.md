# pyGeoMapper

**pyGeoMapper** is under active development, and many functions are still experimental and not fully validated. Please use with caution.

**pyGeoMapper** maps what changed on the ground, and what it's made of, using satellite images and elevation models (DEMs). I wanted one place to go from raw rasters to something a geologist can actually read: change maps after a landslide, flood, wildfire, earthquake, volcano or glacier event, unit maps scored against a real geologic map, and cross-sections and 3D block diagrams you can open in a browser. It's Python with a Rust core. (Not related to `The-Magicians-Code/PyGeoMapper`.)

## What it can do

- Automated geologic mapping with supervised machine learning, trained from geomorphic surface features and surface-geology constraints, then reconstructed as cross-sections and a 3D structural model
- Toolkits for detecting surface change at different temporal and spatial scales, using a range of remote-sensing data types and methods

## Data types

pyGeoMapper can read, analyze, and visualize:

- **Elevation:** DEMs and lidar (Copernicus GLO-30, USGS 3DEP lidar); terrain derivatives, CWT complexity, DEM of Difference
- **Optical imagery:** multispectral Sentinel-2 and Landsat, NAIP aerial RGB; spectral indices (NDVI, NBR, NDWI, BSI, clay and iron ratios) and their change; optical feature-tracking velocity
- **Satellite embeddings:** 64-band Google AlphaEarth annual embeddings for classification and embedding change
- **InSAR:** Sentinel-1 LiCSAR displacement, offsets, and wrapped phase; UAVSAR unwrapped phase to LOS; Okada fault models
- **Ice velocity:** NASA ITS_LIVE velocity grids and speed change
- **Geology:** geologic map polygons as training and validation labels, bed attitudes (strike/dip), and fault traces for cross-sections and 3D block models

Install with `pip install pygeomapper` (Python 3.10+). Figures, tutorials, and notebooks are on the [GitHub repository](https://github.com/GeoLarryLai/pyGeoMapper).

## Citation

Lai, L. S.-H. (2026). pyGeoMapper (version 0.3.0) [Computer software]. [https://doi.org/10.5281/zenodo.23832020](https://doi.org/10.5281/zenodo.23832020)

License: AGPL-3.0-or-later. Third-party notices and data credits are in `NOTICE` in the repository.
