# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""
CRS-aware GeoTIFF/COG, multi-band optical/hyperspectral, and 64-band satellite embedding I/O.

Every raster read or transformed through `raster_io` retains its Coordinate Reference System
(EPSG code / WKT), affine geotransform, spatial bounds, physical units, and nodata sentinel.
Adapts `rasterio`, `pyproj`, and `pystac-client` with `tifffile`/`scipy` fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject

    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False


def _normalize_crs_string(crs_obj: Any) -> str:
    """Normalize a rasterio/pyproj CRS object or string into 'EPSG:XXXX' when possible."""
    if crs_obj is None:
        return "EPSG:4326"
    if isinstance(crs_obj, str):
        s = crs_obj.strip()
        return s if s else "EPSG:4326"
    if hasattr(crs_obj, "to_epsg"):
        epsg = crs_obj.to_epsg()
        if epsg is not None:
            return f"EPSG:{epsg}"
    if hasattr(crs_obj, "to_string"):
        return str(crs_obj.to_string())
    return str(crs_obj)


@dataclass
class GeoGrid:
    """
    2D georeferenced raster grid preserving CRS, affine transform, and physical units.

    Parameters
    ----------
    data : np.ndarray
        2D float64 or numeric array of shape `(rows, cols)`.
    crs : str
        Coordinate Reference System string (e.g., `'EPSG:32610'`).
    transform : tuple[float, float, float, float, float, float]
        GDAL-style 6-tuple `(x_origin, dx, rx, y_origin, ry, dy)` where `(x_origin, y_origin)`
        is the top-left corner of pixel `(0, 0)`, `dx > 0` is easting pixel width,
        and `dy < 0` is northing pixel height for north-up rasters.
    nodata : float
        Sentinel value for invalid pixels (default `np.nan`).
    units : str
        Physical units of the grid values (e.g. `'m'`, `'reflectance'`, `'dimensionless'`).
    name : str
        Descriptive layer name.
    metadata : dict
        Additional provenance or sensor metadata.
    """

    data: np.ndarray
    crs: str = "EPSG:32610"
    transform: tuple[float, float, float, float, float, float] = (
        0.0,
        10.0,
        0.0,
        0.0,
        0.0,
        -10.0,
    )
    nodata: float = np.nan
    units: str = "dimensionless"
    name: str = "layer"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.data)
        if arr.ndim != 2:
            raise ValueError(f"GeoGrid expects a 2D array, got shape {arr.shape}")
        self.data = np.ascontiguousarray(arr, dtype=np.float64)
        self.crs = _normalize_crs_string(self.crs)
        self.transform = tuple(float(v) for v in self.transform)  # type: ignore[assignment]

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def rows(self) -> int:
        return int(self.data.shape[0])

    @property
    def cols(self) -> int:
        return int(self.data.shape[1])

    @property
    def dx(self) -> float:
        return float(self.transform[1])

    @property
    def dy(self) -> float:
        return float(self.transform[5])

    @property
    def cell_area_m2(self) -> float:
        return abs(self.dx * self.dy)

    @property
    def x0(self) -> float:
        """Center X coordinate of pixel (0, 0)."""
        return float(self.transform[0] + 0.5 * self.dx)

    @property
    def y0(self) -> float:
        """Center Y coordinate of pixel (0, 0)."""
        return float(self.transform[3] + 0.5 * self.dy)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Return `(xmin, ymin, xmax, ymax)` bounding box in CRS coordinates."""
        x_ul, dx, _, y_ul, _, dy = self.transform
        x_lr = x_ul + self.cols * dx
        y_lr = y_ul + self.rows * dy
        xmin, xmax = (min(x_ul, x_lr), max(x_ul, x_lr))
        ymin, ymax = (min(y_ul, y_lr), max(y_ul, y_lr))
        return (xmin, ymin, xmax, ymax)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """Return `(xmin, xmax, ymin, ymax)` suitable for `matplotlib.pyplot.imshow(..., extent=grid.extent)`."""
        xmin, ymin, xmax, ymax = self.bounds
        return (xmin, xmax, ymin, ymax)

    def x_coords(self) -> np.ndarray:
        """1D array of pixel center X coordinates of length `cols`."""
        return self.x0 + np.arange(self.cols, dtype=np.float64) * self.dx

    def y_coords(self) -> np.ndarray:
        """1D array of pixel center Y coordinates of length `rows`."""
        return self.y0 + np.arange(self.rows, dtype=np.float64) * self.dy

    def valid_mask(self) -> np.ndarray:
        """Boolean mask of finite, non-nodata pixels."""
        m = np.isfinite(self.data)
        if not np.isnan(self.nodata):
            m &= np.abs(self.data - self.nodata) > 1e-9
        return m

    def with_data(
        self,
        new_data: np.ndarray,
        name: str | None = None,
        units: str | None = None,
        nodata: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GeoGrid:
        """Create a new GeoGrid sharing this grid's CRS and affine transform."""
        meta = dict(self.metadata)
        if metadata:
            meta.update(metadata)
        return GeoGrid(
            data=new_data,
            crs=self.crs,
            transform=self.transform,
            nodata=self.nodata if nodata is None else nodata,
            units=self.units if units is None else units,
            name=self.name if name is None else name,
            metadata=meta,
        )


@dataclass
class RasterStack:
    """
    3D georeferenced multi-band or embedding raster stack `(rows, cols, bands)`.

    Designed for both optical/hyperspectral cubes (e.g. Sentinel-2, EMIT) and
    high-dimensional geospatial embeddings (e.g. 64-band AlphaEarth Foundations `A00..A63`).
    """

    data: np.ndarray
    band_names: list[str]
    crs: str = "EPSG:32610"
    transform: tuple[float, float, float, float, float, float] = (
        0.0,
        10.0,
        0.0,
        0.0,
        0.0,
        -10.0,
    )
    nodata: float = np.nan
    units: str = "reflectance"
    name: str = "stack"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr = np.asarray(self.data)
        if arr.ndim != 3:
            raise ValueError(
                f"RasterStack expects a 3D array (rows, cols, bands), got {arr.shape}"
            )
        self.data = np.ascontiguousarray(arr, dtype=np.float64)
        self.crs = _normalize_crs_string(self.crs)
        self.transform = tuple(float(v) for v in self.transform)  # type: ignore[assignment]
        if len(self.band_names) != self.data.shape[2]:
            raise ValueError(
                f"band_names length ({len(self.band_names)}) != band count ({self.data.shape[2]})"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.data.shape  # type: ignore[return-value]

    @property
    def rows(self) -> int:
        return int(self.data.shape[0])

    @property
    def cols(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_bands(self) -> int:
        return int(self.data.shape[2])

    @property
    def nbands(self) -> int:
        return int(self.data.shape[2])

    @property
    def dx(self) -> float:
        return float(self.transform[1])

    @property
    def dy(self) -> float:
        return float(self.transform[5])

    @property
    def cell_area_m2(self) -> float:
        return abs(self.dx * self.dy)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        x_ul, dx, _, y_ul, _, dy = self.transform
        x_lr = x_ul + self.cols * dx
        y_lr = y_ul + self.rows * dy
        return (min(x_ul, x_lr), min(y_ul, y_lr), max(x_ul, x_lr), max(y_ul, y_lr))

    @property
    def extent(self) -> tuple[float, float, float, float]:
        xmin, ymin, xmax, ymax = self.bounds
        return (xmin, xmax, ymin, ymax)

    def band(self, key: str | int) -> GeoGrid:
        """Extract a single band by name (case-insensitive) or 0-based index as a `GeoGrid`."""
        if isinstance(key, int):
            idx = key
            bname = self.band_names[idx]
        else:
            key_lower = key.lower()
            idx = -1
            for i, bn in enumerate(self.band_names):
                if bn.lower() == key_lower:
                    idx = i
                    break
            if idx < 0:
                raise KeyError(
                    f"Band '{key}' not found in RasterStack bands: {self.band_names}"
                )
            bname = self.band_names[idx]
        return GeoGrid(
            data=self.data[:, :, idx],
            crs=self.crs,
            transform=self.transform,
            nodata=self.nodata,
            units=self.units,
            name=bname,
            metadata=dict(self.metadata),
        )

    def to_unit_norm(self) -> RasterStack:
        """Normalize each pixel's D-dimensional vector to unit L2 length (for unit-sphere embeddings)."""
        norms = np.linalg.norm(self.data, axis=2, keepdims=True)
        safe = np.where(norms > 1e-12, self.data / norms, np.nan)
        return RasterStack(
            data=safe,
            band_names=list(self.band_names),
            crs=self.crs,
            transform=self.transform,
            nodata=self.nodata,
            units="unit_norm",
            name=f"{self.name}_unit_norm",
            metadata=dict(self.metadata),
        )

    @classmethod
    def from_grids(
        cls,
        grids: Sequence[GeoGrid] | dict[str, GeoGrid],
        name: str = "stack",
        units: str | None = None,
    ) -> RasterStack:
        """Stack a sequence or dict of co-registered `GeoGrid` layers into a `RasterStack`."""
        if isinstance(grids, dict):
            names = list(grids.keys())
            grid_list = list(grids.values())
        else:
            grid_list = list(grids)
            names = [g.name for g in grid_list]
        if not grid_list:
            raise ValueError("from_grids requires at least one GeoGrid")
        ref = grid_list[0]
        for g in grid_list[1:]:
            if g.shape != ref.shape:
                raise ValueError(
                    f"Grid shape mismatch: {g.name} {g.shape} != {ref.name} {ref.shape}"
                )
        cube = np.stack([g.data for g in grid_list], axis=2)
        return cls(
            data=cube,
            band_names=names,
            crs=ref.crs,
            transform=ref.transform,
            nodata=ref.nodata,
            units=ref.units if units is None else units,
            name=name,
            metadata=dict(ref.metadata),
        )


def load_geotiff(
    path: str | Path,
    band: int = 1,
    units: str = "m",
    name: str | None = None,
    scale_factor: float = 1.0,
    offset: float = 0.0,
) -> GeoGrid:
    """
    Load a single band from a GeoTIFF/COG file into a CRS-aware `GeoGrid`.

    Invalid / nodata pixels are converted to `np.nan` in the returned float64 array.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {p}")

    if HAS_RASTERIO:
        with rasterio.open(p) as src:
            raw = src.read(band).astype(np.float64)
            nodata = src.nodata
            if nodata is not None and not np.isnan(nodata):
                raw[np.isclose(raw, float(nodata), rtol=0.0, atol=1e-6)] = np.nan
            if scale_factor != 1.0 or offset != 0.0:
                raw = raw * scale_factor + offset
            t = src.transform
            transform = (float(t.c), float(t.a), float(t.b), float(t.f), float(t.d), float(t.e))
            crs_str = _normalize_crs_string(src.crs)
            tags = dict(src.tags())
            unit_str = tags.get("units", units)
            layer_name = name or tags.get("name") or p.stem
            return GeoGrid(
                data=raw,
                crs=crs_str,
                transform=transform,
                nodata=np.nan,
                units=unit_str,
                name=layer_name,
                metadata={"source_path": str(p), "tags": tags},
            )

    import tifffile

    raw = tifffile.imread(p).astype(np.float64)
    if raw.ndim == 3:
        raw = raw[band - 1] if raw.shape[0] < raw.shape[2] else raw[:, :, band - 1]
    if scale_factor != 1.0 or offset != 0.0:
        raw = raw * scale_factor + offset
    return GeoGrid(
        data=raw,
        crs="EPSG:32610",
        transform=(0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
        nodata=np.nan,
        units=units,
        name=name or p.stem,
        metadata={"source_path": str(p)},
    )


def load_multiband_geotiff(
    path: str | Path,
    band_names: Sequence[str] | None = None,
    scale_factor: float = 1.0,
    offset: float = 0.0,
    dequantize_alphaearth: bool = False,
    units: str = "reflectance",
    name: str | None = None,
) -> RasterStack:
    """
    Load a multi-band GeoTIFF (optical stack, hyperspectral cube, or 64-band AlphaEarth embedding)
    into a CRS-aware `RasterStack` shaped `(rows, cols, bands)`.

    Parameters
    ----------
    dequantize_alphaearth : bool
        If True and input is quantized `int8` in `[-127, 127]` (Google AlphaEarth convention),
        applies `(x / 127.0)` dequantization and normalizes each pixel to unit L2 length,
        treating `-128` as nodata.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GeoTIFF not found: {p}")

    if HAS_RASTERIO:
        with rasterio.open(p) as src:
            arr_b_r_c = src.read()  # (bands, rows, cols)
            src_dtype = src.dtypes[0]
            nodata = src.nodata
            raw = np.moveaxis(arr_b_r_c, 0, -1).astype(np.float64)  # (rows, cols, bands)

            if dequantize_alphaearth and "int8" in str(src_dtype):
                invalid = arr_b_r_c[0] == -128
                raw = raw / 127.0
                norms = np.linalg.norm(raw, axis=2, keepdims=True)
                raw = np.where(norms > 1e-12, raw / norms, np.nan)
                raw[invalid, :] = np.nan
            else:
                if nodata is not None and not np.isnan(nodata):
                    raw[np.isclose(raw, float(nodata), rtol=0.0, atol=1e-6)] = np.nan
                if scale_factor != 1.0 or offset != 0.0:
                    raw = raw * scale_factor + offset

            t = src.transform
            transform = (float(t.c), float(t.a), float(t.b), float(t.f), float(t.d), float(t.e))
            crs_str = _normalize_crs_string(src.crs)
            tags = dict(src.tags())

            if band_names is not None:
                bnames = list(band_names)
            elif any(src.descriptions):
                bnames = [
                    desc if desc else f"B{i + 1:02d}"
                    for i, desc in enumerate(src.descriptions)
                ]
            elif raw.shape[2] == 64:
                bnames = [f"A{i:02d}" for i in range(64)]
            else:
                bnames = [f"B{i + 1:02d}" for i in range(raw.shape[2])]

            return RasterStack(
                data=raw,
                band_names=bnames,
                crs=crs_str,
                transform=transform,
                nodata=np.nan,
                units=tags.get("units", units),
                name=name or p.stem,
                metadata={"source_path": str(p), "tags": tags},
            )

    import tifffile

    raw = tifffile.imread(p).astype(np.float64)
    if raw.ndim == 2:
        raw = raw[:, :, np.newaxis]
    elif raw.shape[0] < raw.shape[1] and raw.shape[0] < raw.shape[2]:
        raw = np.moveaxis(raw, 0, -1)
    bnames = (
        list(band_names)
        if band_names is not None
        else [f"B{i + 1:02d}" for i in range(raw.shape[2])]
    )
    return RasterStack(
        data=raw * scale_factor + offset,
        band_names=bnames,
        crs="EPSG:32610",
        transform=(0.0, 10.0, 0.0, 0.0, 0.0, -10.0),
        nodata=np.nan,
        units=units,
        name=name or p.stem,
        metadata={"source_path": str(p)},
    )


def save_geotiff(
    grid_or_stack: GeoGrid | RasterStack,
    path: str | Path,
    dtype: str = "float32",
    compress: str = "deflate",
) -> Path:
    """Save a `GeoGrid` or `RasterStack` to a compressed, CRS-tagged GeoTIFF."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x_ul, dx, rx, y_ul, ry, dy = grid_or_stack.transform

    if isinstance(grid_or_stack, GeoGrid):
        bands_data = grid_or_stack.data[np.newaxis, :, :]
        descriptions = [grid_or_stack.name]
    else:
        bands_data = np.moveaxis(grid_or_stack.data, -1, 0)
        descriptions = list(grid_or_stack.band_names)

    count, rows, cols = bands_data.shape
    np_dtype = np.dtype(dtype)

    if HAS_RASTERIO:
        transform = Affine(dx, rx, x_ul, ry, dy, y_ul)
        nodata_val: float | int = -9999.0 if np.issubdtype(np_dtype, np.floating) else -128
        out_arr = np.where(np.isfinite(bands_data), bands_data, nodata_val).astype(np_dtype)
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            height=rows,
            width=cols,
            count=count,
            dtype=np_dtype,
            crs=grid_or_stack.crs,
            transform=transform,
            nodata=nodata_val,
            compress=compress,
        ) as dst:
            dst.write(out_arr)
            for idx, desc in enumerate(descriptions, start=1):
                dst.set_band_description(idx, str(desc))
            dst.update_tags(units=grid_or_stack.units, name=grid_or_stack.name)
        return p

    import tifffile

    tifffile.imwrite(p, bands_data.astype(np_dtype))
    return p


def align_to_reference(
    source: GeoGrid | RasterStack,
    reference: GeoGrid | RasterStack,
    resampling: str = "bilinear",
) -> GeoGrid | RasterStack:
    """
    Reproject and resample `source` onto the exact CRS, affine grid, and pixel dimensions of `reference`.
    """
    if (
        source.crs == reference.crs
        and np.allclose(source.transform, reference.transform, atol=1e-6)
        and source.rows == reference.rows
        and source.cols == reference.cols
    ):
        return source

    if not HAS_RASTERIO:
        raise RuntimeError("align_to_reference requires rasterio to reproject between grids")

    resamp_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    resamp_enum = resamp_map.get(resampling.lower(), Resampling.bilinear)

    sx_ul, sdx, srx, sy_ul, sry, sdy = source.transform
    rx_ul, rdx, rrx, ry_ul, rry, rdy = reference.transform
    src_aff = Affine(sdx, srx, sx_ul, sry, sdy, sy_ul)
    dst_aff = Affine(rdx, rrx, rx_ul, rry, rdy, ry_ul)

    if isinstance(source, GeoGrid):
        dst_arr = np.full((reference.rows, reference.cols), np.nan, dtype=np.float64)
        reproject(
            source=source.data,
            destination=dst_arr,
            src_transform=src_aff,
            src_crs=source.crs,
            dst_transform=dst_aff,
            dst_crs=reference.crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resamp_enum,
        )
        return GeoGrid(
            data=dst_arr,
            crs=reference.crs,
            transform=reference.transform,
            nodata=np.nan,
            units=source.units,
            name=source.name,
            metadata=dict(source.metadata),
        )

    dst_cube = np.full(
        (reference.rows, reference.cols, source.n_bands), np.nan, dtype=np.float64
    )
    for b in range(source.n_bands):
        reproject(
            source=source.data[:, :, b],
            destination=dst_cube[:, :, b],
            src_transform=src_aff,
            src_crs=source.crs,
            dst_transform=dst_aff,
            dst_crs=reference.crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=resamp_enum,
        )
    return RasterStack(
        data=dst_cube,
        band_names=list(source.band_names),
        crs=reference.crs,
        transform=reference.transform,
        nodata=np.nan,
        units=source.units,
        name=source.name,
        metadata=dict(source.metadata),
    )


def search_stac_scenes(
    bbox_lonlat: tuple[float, float, float, float],
    datetime_range: str,
    collections: Iterable[str] = ("sentinel-2-l2a",),
    max_cloud_cover: float = 15.0,
    stac_url: str = "https://earth-search.aws.element84.com/v1",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Search a public STAC API (default: AWS Element84 Earth Search v1) for scenes covering
    `bbox_lonlat = (min_lon, min_lat, max_lon, max_lat)` and `datetime_range`.
    """
    from pystac_client import Client

    client = Client.open(stac_url)
    query: dict[str, Any] = {}
    if max_cloud_cover is not None and "sentinel-2-l2a" in list(collections):
        query["eo:cloud_cover"] = {"lt": float(max_cloud_cover)}

    search = client.search(
        collections=list(collections),
        bbox=list(bbox_lonlat),
        datetime=datetime_range,
        query=query if query else None,
        max_items=limit,
    )
    results = []
    for item in search.items():
        assets = {k: v.href for k, v in item.assets.items()}
        results.append(
            {
                "id": item.id,
                "datetime": item.datetime.isoformat() if item.datetime else None,
                "cloud_cover": item.properties.get("eo:cloud_cover"),
                "epsg": item.properties.get("proj:epsg"),
                "bbox": item.bbox,
                "assets": assets,
            }
        )
    return results


def fetch_alphaearth_gee(
    bbox_lonlat: tuple[float, float, float, float],
    year: int,
    scale_m: float = 10.0,
    project: str | None = None,
    cache_tif: str | Path | None = None,
) -> RasterStack:
    """
    Fetch or load a 64-band Google DeepMind AlphaEarth Foundations annual satellite embedding
    (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, bands `A00..A63`, CC-BY-4.0).

    If `cache_tif` exists locally, loads directly from disk without requiring Earth Engine authentication.
    Otherwise uses `earthengine-api` (`ee`) to download the cropped 64-band tile for `year`.
    """
    if cache_tif is not None and Path(cache_tif).exists():
        return load_multiband_geotiff(
            cache_tif,
            dequantize_alphaearth=True,
            units="unit_norm",
            name=f"alphaearth_{year}",
        )

    try:
        import ee
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "earthengine-api ('ee') is not installed. Pass an existing cache_tif or install earthengine-api."
        ) from exc

    if project:
        ee.Initialize(project=project)
    else:
        ee.Initialize()

    min_lon, min_lat, max_lon, max_lat = bbox_lonlat
    region = ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat])
    col = (
        ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
        .filterBounds(region)
        .filterDate(f"{year}-01-01", f"{year}-12-31")
    )
    img = col.mosaic().clip(region)
    band_names = [f"A{i:02d}" for i in range(64)]
    sample = img.select(band_names).sampleRectangle(region=region, defaultValue=0.0)
    props = sample.getInfo()["properties"]
    arrays = [np.asarray(props[b], dtype=np.float64) for b in band_names]
    cube = np.stack(arrays, axis=2)
    stack = RasterStack(
        data=cube,
        band_names=band_names,
        crs="EPSG:4326",
        transform=(min_lon, scale_m / 111320.0, 0.0, max_lat, 0.0, -scale_m / 111320.0),
        units="unit_norm",
        name=f"alphaearth_{year}",
        metadata={"collection": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL", "year": year},
    ).to_unit_norm()
    if cache_tif is not None:
        save_geotiff(stack, cache_tif)
    return stack


COPERNICUS_DEM_BUCKETS = {
    30: ("https://copernicus-dem-30m.s3.amazonaws.com", "10"),
    90: ("https://copernicus-dem-90m.s3.amazonaws.com", "30"),
}


def copernicus_dem_tile_urls(
    bounds: tuple[float, float, float, float], resolution_m: int = 30
) -> list[str]:
    """
    Return public AWS Open Data URLs of Copernicus GLO-30 / GLO-90 DSM COG tiles (1 x 1 degree)
    intersecting `bounds = (lon_min, lat_min, lon_max, lat_max)` in EPSG:4326.
    """
    import math

    if resolution_m not in COPERNICUS_DEM_BUCKETS:
        raise ValueError("resolution_m must be 30 (GLO-30) or 90 (GLO-90)")
    base, code = COPERNICUS_DEM_BUCKETS[resolution_m]
    lon_min, lat_min, lon_max, lat_max = bounds
    urls = []
    for lat in range(math.floor(lat_min), math.ceil(lat_max)):
        for lon in range(math.floor(lon_min), math.ceil(lon_max)):
            ns = f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00"
            ew = f"{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00"
            stem = f"Copernicus_DSM_COG_{code}_{ns}_{ew}_DEM"
            urls.append(f"{base}/{stem}/{stem}.tif")
    return urls


def fetch_copernicus_dem(
    bounds: tuple[float, float, float, float],
    resolution_m: int = 30,
    out_path: str | Path | None = None,
    dst_crs: str | None = None,
    dst_res_m: float | None = None,
) -> GeoGrid:
    """
    Fetch full-resolution Copernicus DEM (GLO-30 at 1 arc-second or GLO-90 at 3 arc-seconds) on demand
    from the public AWS Open Data buckets (`copernicus-dem-30m` / `copernicus-dem-90m`, no credentials),
    mosaic and crop to `bounds = (lon_min, lat_min, lon_max, lat_max)`.

    Optionally reprojects to `dst_crs` at `dst_res_m`, and writes a DEFLATE-compressed int16 COG
    to `out_path`. Ocean / missing tiles are skipped (Copernicus omits all-water tiles).
    Elevations are EGM2008 orthometric heights (m) of the DSM surface.
    """
    if not HAS_RASTERIO:  # pragma: no cover
        raise ImportError("fetch_copernicus_dem requires rasterio")
    from rasterio.merge import merge
    from rasterio.warp import calculate_default_transform

    datasets = []
    for url in copernicus_dem_tile_urls(bounds, resolution_m):
        try:
            datasets.append(rasterio.open(f"/vsicurl/{url}"))
        except Exception:
            continue  # missing (all-ocean) tile
    if not datasets:
        raise RuntimeError(f"No Copernicus DEM tiles found for bounds {bounds}")
    try:
        mosaic, mtrans = merge(datasets, bounds=bounds, nodata=-32768)
        src_crs = datasets[0].crs
    finally:
        for d in datasets:
            d.close()
    arr = mosaic[0].astype(np.float32)
    arr[arr <= -1000] = np.nan
    crs_out = src_crs
    trans_out = mtrans
    if dst_crs is not None:
        h, w = arr.shape
        left, top = mtrans.c, mtrans.f
        right, bottom = left + w * mtrans.a, top + h * mtrans.e
        dt, dw, dh = calculate_default_transform(
            src_crs, dst_crs, w, h, left, bottom, right, top, resolution=dst_res_m
        )
        dst = np.full((dh, dw), np.nan, dtype=np.float32)
        reproject(arr, dst, src_transform=mtrans, src_crs=src_crs, dst_transform=dt,
                  dst_crs=dst_crs, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        arr, crs_out, trans_out = dst, CRS.from_user_input(dst_crs), dt

    if out_path is not None:
        out = np.where(np.isfinite(arr), np.round(arr), -32768).astype(np.int16)
        with rasterio.open(
            out_path, "w", driver="COG", height=out.shape[0], width=out.shape[1], count=1,
            dtype="int16", crs=crs_out, transform=trans_out, nodata=-32768,
            compress="DEFLATE", predictor=2, level=9, blocksize=512,
        ) as dst_ds:
            dst_ds.write(out, 1)
            dst_ds.update_tags(units="m", name="copernicus_dem",
                               source=f"Copernicus GLO-{resolution_m} DSM (AWS Open Data)",
                               lon_min=str(bounds[0]), lat_min=str(bounds[1]),
                               lon_max=str(bounds[2]), lat_max=str(bounds[3]))
    t = trans_out
    return GeoGrid(
        data=arr.astype(np.float64), crs=_normalize_crs_string(crs_out),
        transform=(float(t.c), float(t.a), float(t.b), float(t.f), float(t.d), float(t.e)),
        nodata=np.nan, units="m", name=f"copernicus_glo{resolution_m}",
        metadata={"source": f"Copernicus GLO-{resolution_m}", "bounds": bounds},
    )
