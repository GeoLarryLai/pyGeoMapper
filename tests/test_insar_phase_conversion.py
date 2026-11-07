# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic phase/LOS conversion checks; no survey I/O."""
import numpy as np
import pytest

from pygeomapper import surface_change
from pygeomapper.raster_io import GeoGrid


@pytest.mark.parametrize("wavelength", [0.05546576, 0.238403545])
@pytest.mark.parametrize("reference", [None, 0.0, 3.7])
def test_unwrapped_los_and_rewrapped_roundtrip(wavelength, reference):
    los = np.array([[-0.3, -0.01, 0.0], [0.01, 0.3, np.nan]])
    phase = los * (4 * np.pi / wavelength) + (reference or 0.0)
    original = phase.copy()
    result = surface_change.compute_los_from_unwrapped_phase(
        phase, wavelength, phase_reference=reference,
    )
    np.testing.assert_allclose(result, los, atol=1e-15)
    wrapped = surface_change.compute_insar_wrapped_phase(result, wavelength_m=wavelength)
    expected = ((phase - (reference or 0.0) + np.pi) % (2 * np.pi)) - np.pi
    np.testing.assert_allclose(wrapped, expected, atol=1e-13)
    np.testing.assert_array_equal(phase, original)


def test_cycle_scale_and_away_sign():
    wavelength = 0.238403545
    phase = np.array([[-2 * np.pi, 0.0, 2 * np.pi]])
    result = surface_change.compute_los_from_unwrapped_phase(phase, wavelength)
    np.testing.assert_allclose(result, [[-wavelength / 2, 0.0, wavelength / 2]])


def test_geogrid_metadata_nodata_and_no_mutation():
    grid = GeoGrid(np.array([[0.0, 2 * np.pi, -9999.0, np.nan, np.inf]]),
                   crs="EPSG:26913", transform=(200.0, 10.0, 0.0, 500.0, 0.0, -10.0),
                   units="rad", nodata=-9999.0, metadata={"sensor": "synthetic"})
    original = grid.data.copy()
    result = surface_change.compute_los_from_unwrapped_phase(grid, 0.238403545)
    assert isinstance(result, GeoGrid)
    assert result.crs == grid.crs
    assert result.transform == grid.transform
    assert result.metadata == grid.metadata
    assert result.units == "m"
    assert np.isnan(result.nodata)
    np.testing.assert_allclose(result.data[0, :2], [0.0, 0.238403545 / 2])
    assert np.isnan(result.data[0, 2:]).all()
    np.testing.assert_array_equal(grid.data, original)
    wrapped = surface_change.compute_insar_wrapped_phase(result, wavelength_m=0.238403545)
    assert isinstance(wrapped, GeoGrid)
    assert wrapped.units == "rad"
    assert wrapped.transform == grid.transform
    assert np.isnan(wrapped.data[0, 2:]).all()


@pytest.mark.parametrize("wavelength", [0, -0.2, np.nan, np.inf])
def test_invalid_wavelength(wavelength):
    with pytest.raises(ValueError, match="wavelength_m"):
        surface_change.compute_los_from_unwrapped_phase(np.zeros((1, 2)), wavelength)


@pytest.mark.parametrize("reference", [np.nan, np.inf, -np.inf])
def test_invalid_reference(reference):
    with pytest.raises(ValueError, match="phase_reference"):
        surface_change.compute_los_from_unwrapped_phase(
            np.zeros((1, 2)), 0.238403545, phase_reference=reference,
        )


def test_spatial_ramp_is_retained_with_only_constant_reference_removed():
    yy, xx = np.mgrid[:8, :11]
    phase = 20.0 + 0.7 * xx - 1.2 * yy
    result = surface_change.compute_los_from_unwrapped_phase(
        phase, wavelength_m=0.238403545, phase_reference=20.0,
    )
    scale = 0.238403545 / (4 * np.pi)
    np.testing.assert_allclose(np.diff(result, axis=1), 0.7 * scale)
    np.testing.assert_allclose(np.diff(result, axis=0), -1.2 * scale)
