import pytest
from .. import polarization_tools
from astropy import units


def test_tables():
    """Test that the tables are loaded correctly and are all present."""
    # Check if we can load the tables from the package data.
    tables = polarization_tools.PolarizerTables.from_package_data()
    assert hasattr(tables, 'wfc_transmission')
    assert hasattr(tables, 'hrc_transmission')
    assert hasattr(tables, 'wfc_efficiency')
    assert hasattr(tables, 'hrc_efficiency')

    # Check if we can load the tables by specifying the path to a YAML file.
    _ = polarization_tools.PolarizerTables.from_yaml('acstools/data/polarizer_tables.yaml')


def test_theta():
    """Test that the calc_theta function returns a quantity object
    in degrees and that the value is as expected."""
    # Check the angle for when we have all U polarization.
    all_u = polarization_tools.calc_theta(0, 1, 'wfc', 38.2)
    assert type(all_u) is units.quantity.Quantity
    assert all_u.unit is units.degree
    assert all_u.value == 45

    # Also check the value for the all Q case.
    all_q = polarization_tools.calc_theta(1, 0, 'wfc', 38.2)
    assert all_q.value == 0


def test_fraction():
    """Test that the polarization fraction value is as expected."""

    fraction = polarization_tools.calc_fraction(1, 0.353553, 0.353553)
    assert fraction == pytest.approx(0.5, abs=1e-5)


def test_stokes():
    """Test that the Stokes computation is correct."""

    i, q, u = polarization_tools.calc_stokes(1, 1, 1, c0=1, c60=0.5, c120=1.5)
    assert i == 2.0
    assert q == 0.0
    assert u == pytest.approx(-1.1547005383792517)


def test_pol_class():
    """The polarization class uses the other methods we already tested,
    but let's test that we get expected exceptions."""

    with pytest.raises(IndexError):
        _ = polarization_tools.Polarization(1, 1, 1, 'F607W', 'WFC', 1)

    with pytest.raises(ValueError):
        _ = polarization_tools.Polarization(1, 1, 1, 'F606W', 'WFB', 1)
