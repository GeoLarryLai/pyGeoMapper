# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Fast synthetic tests for mixed-source topographic DEMs."""

import numpy as np
import pytest

from pygeomapper import raster_io


def grid(data, spacing=1, nodata=np.nan):
    return raster_io.GeoGrid(
        data, crs="EPSG:3413", transform=(0, spacing, 0, 40, 0, -spacing),
        nodata=nodata, name="synthetic", units="m",
    )


def test_blend_fills_holes_matches_heights_and_reference():
    pytest.importorskip("rasterio")
    base = grid(np.full((20, 20), 90.0), spacing=2)
    values = np.full((40, 40), np.nan)
    values[4:36, 4:36] = 100
    values[19:21, 19:21] = 123  # Local relief must survive unchanged.
    detail = grid(values)
    result = raster_io.blend_dem(base, detail)
    assert result.shape == detail.shape
    assert result.transform == detail.transform
    assert result.crs == detail.crs
    assert result.units == "m" and np.isnan(result.nodata)
    assert np.isfinite(result.data).all()
    assert result.data[0, 0] == pytest.approx(100)
    np.testing.assert_array_equal(result.data[19:21, 19:21], values[19:21, 19:21])
    assert result.metadata["blend_dem"]["base_vertical_shift"] == pytest.approx(10)
    # The plane is continuous across the edge after shifting only the base.
    assert np.max(np.abs(result.data[3:7, 4:36] - 100)) < 1e-10
    assert np.isnan(detail.data[0, 0])
    np.testing.assert_array_equal(base.data, 90)


def test_feather_is_local_and_optional():
    base = grid(np.zeros((40, 40)))
    values = np.full((40, 40), np.nan)
    values[4:36, 4:36] = 10
    detail = grid(values)
    smooth = raster_io.blend_dem(base, detail, feather_pixels=2, match_vertical=False)
    np.testing.assert_allclose(smooth.data[4:8, 20], [0, 5, 10, 10])
    sharp = raster_io.blend_dem(base, detail, feather_pixels=0, match_vertical=False)
    np.testing.assert_array_equal(sharp.data[detail.valid_mask()], values[detail.valid_mask()])
    assert sharp.metadata["blend_dem"]["base_vertical_shift"] == 0


def test_nodata_and_detail_only_pixels():
    b = np.full((40, 40), 90.0)
    b[0, 0] = -9999
    b[5, 5] = -9999
    d = np.full((40, 40), -9999.0)
    d[4:36, 4:36] = 100
    result = raster_io.blend_dem(grid(b, nodata=-9999), grid(d, nodata=-9999))
    assert np.isnan(result.data[0, 0])  # Missing in both.
    assert result.data[5, 5] == 100  # Detail retained even at a base hole/edge.
    assert result.data[0, 1] == 100


def test_finite_base_nodata_not_interpolated():
    pytest.importorskip("rasterio")
    b = np.full((20, 20), 90.0)
    b[:10, :10] = -9999
    result = raster_io.blend_dem(
        grid(b, spacing=2, nodata=-9999), grid(np.full((40, 40), np.nan))
    )
    assert np.isnan(result.data[0, 0])
    np.testing.assert_allclose(result.data[np.isfinite(result.data)], 90)


def test_no_overlap_or_no_detail_holes():
    base = grid(np.full((40, 40), np.nan))
    detail = grid(np.full((40, 40), 100))
    result = raster_io.blend_dem(base, detail)
    np.testing.assert_array_equal(result.data, detail.data)
    assert result.metadata["blend_dem"]["overlap_pixels"] == 0
    assert result.metadata["blend_dem"]["base_vertical_shift"] == 0
    # A fully valid detail grid must not acquire artificial raster-edge feather.
    result = raster_io.blend_dem(detail, detail.with_data(detail.data + 10), match_vertical=False)
    np.testing.assert_array_equal(result.data, 110)


@pytest.mark.parametrize("width", [-1, np.nan, np.inf])
def test_invalid_feather_width(width):
    dem = grid(np.ones((2, 2)))
    with pytest.raises(ValueError, match="feather_pixels"):
        raster_io.blend_dem(dem, dem, feather_pixels=width)
