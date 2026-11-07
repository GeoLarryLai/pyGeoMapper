# Changelog

## 0.3.0 (2026-11-07)

- Reorganize the README into two sections: landscape features and geologic mapping, then surface change.
- README section intros: automated geologic mapping from geomorphic features, surface-geology constraints, and 3D structural reconstruction; then surface-change toolkits at different temporal and spatial scales.
- README gallery keeps the rotating 3D block GIF and omits the static 3D still (file retained in the repository).
- README auto-mapping four-panel from 64-band AlphaEarth embeddings, with a northwest–southeast diagonal 50/50 holdout table (engineered 29 bands 69.1% / macro-F1 0.64; AlphaEarth 64 91.0% / 0.89; combined 93 89.5% / 0.87).
- Jakobshavn two-panel ITS_LIVE figure: 2018 speed with flow vectors, and the 2018−2016 change in speed.
- Slumgullion 2×2 board: NAIP optical velocity, package-computed UAVSAR LOS and rewrapped phase, and 2024−2015 elevation change.
- Sheep Mountain opaque 3D geologic block with matching faulted sections, plus a 72-frame rotation at 3 fps.
- Cleaned Holiday Farm Fire (Oregon) and Yolo Bypass flood (California) README figures.
- Add `raster_io.blend_dem` for aligned, optionally feathered DEM gap filling with an optional median vertical seam shift.
- Add `surface_change.compute_los_from_unwrapped_phase`, the unwrapped inverse of `compute_insar_wrapped_phase`; retain the input atmospheric ramp.
- Add `viewer_3d.animate_3d_block_and_section` to rotate both subplots together.
- Update notebooks 3–5 so they do not overwrite the accepted README figures with older tutorial boards.
- Add DEM blend, phase/LOS round-trip, animation, and README asset regression tests.
- PyPI README is a short text overview only (no figures or usage examples); the full gallery stays on GitHub.

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
