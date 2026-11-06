# Changelog

## 0.2.0 (unreleased)

Polish release: cleaner README figures and notebooks, a lighter clone, and InSAR figure upgrades.

- README: six captioned result figures (Oso DoD, Myanmar 2025 InSAR in 3D, Oso roughness change, auto-mapping, cross-section, 3D block); PyPI citation synced to 0.2.0.
- New `pygeomapper.plotting`: one figure-export path (no overlapping titles or `1e6` tick labels, capped file size) and `drape_fringes` for wrapped phase on imagery. `pygeomapper.insar3d` is now public.
- One DEM-of-Difference color rule in every notebook: red = erosion, blue = deposit.
- Cross-sections: bedding contacts now break along the dipping fault plane like the unit colors, and horizons stay in stratigraphic order. The Sheep Mountain forelimb fault dips NE in notebooks 3 and 4.
- 3D block diagrams: `show_fault_plane=True` draws the fault trace on the top surface and the front walls.
- Notebooks: fixed broken setup cells (notebooks 2–4), the Colab link, section text and figure captions; figures are saved to `image/` only.
- Example data: Myanmar rasters downsampled (117 MB → 19 MB, `scripts/make_myanmar_demo.py`); new pre-event Sentinel-2 mosaic (`scripts/fetch_myanmar_s2_rgb.py`).
- Packaging: no pip extras; `pip install pygeomapper` still includes `forge3d`. Conda users without it can `pip install forge3d`; otherwise `viewer_3d` falls back to its built-in renderer.
- Data: notebook 5 Cases 3-5 (flood, wildfire, agri-urban) now use real Sentinel-2 scenes of the February 2019 Yolo Bypass flood (CA), the 2020 Holiday Farm Fire (OR) and Boise Valley farmland (ID), with one Copernicus GLO-30 DEM per case (`scripts/fetch_surface_change_cases.py`).
- Tests: README image/size, absolute-path and version-sync checks; hazard tests skip when the data are absent; notebook checks say they read saved outputs only.
