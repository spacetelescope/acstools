import numpy as np
import pytest
from astropy import units
from astropy.tests.helper import assert_quantity_allclose

from acstools import polarization_tools


def test_tables():
    """Test that the tables are loaded correctly and are all present."""
    # Check if we can load the tables from the package data.
    tables = polarization_tools.PolarizerTables.from_package_data()
    assert hasattr(tables, 'wfc_transmission')
    assert hasattr(tables, 'hrc_transmission')
    assert hasattr(tables, 'wfc_efficiency')
    assert hasattr(tables, 'hrc_efficiency')


def test_theta():
    """Test that the calc_theta function returns a quantity object
    in degrees and that the value is as expected."""
    # Check the angle for when we have all U polarization.
    all_u = polarization_tools.calc_theta(0, 1, 'wfc', 38.2)
    assert_quantity_allclose(all_u, 45 * units.degree)

    # Also check the value for the all Q case.
    all_q = polarization_tools.calc_theta(1, 0, 'wfc', 38.2)
    assert_quantity_allclose(all_q, 0 * units.degree)


def test_fraction():
    """Test that the polarization fraction value is as expected."""

    fraction = polarization_tools.calc_fraction(1, 0.353553, 0.353553)
    np.testing.assert_allclose(fraction, 0.5, atol=1e-5)


def test_stokes():
    """Test that the Stokes computation is correct."""

    i, q, u = polarization_tools.calc_stokes(1, 1, 1, c0=1, c60=0.5, c120=1.5)
    assert i == 2.0
    assert q == 0.0
    np.testing.assert_allclose(u, -1.1547005383792517)


def test_pol_class():
    """The polarization class uses the other methods we already tested,
    but let's test that we get expected exceptions."""

    with pytest.raises(IndexError):
        polarization_tools.Polarization(1, 1, 1, 'F607W', 'WFC', 1)

    with pytest.raises(ValueError):
        polarization_tools.Polarization(1, 1, 1, 'F606W', 'WFB', 1)
