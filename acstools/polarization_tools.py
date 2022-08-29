"""
Toolkit for analyzing HST/ACS WFC and HRC polarization data. For more information
about ACS polarization data analysis, see `Section 5.3 of the ACS Data Handbook <https://hst-docs.stsci.edu/acsdhb/chapter-5-acs-data-analysis/5-3-polarimetry>`_.

"""

import os
import numpy as np
import yaml

from astropy import units
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

__all__ = ['calc_theta', 'calc_fraction', 'calc_stokes', 'PolarizerTables', 'Polarization']


def calc_stokes(pol0, pol60, pol120, c0=1, c60=1, c120=1):
    """
    Calculate the Stokes parameters for ACS observations.

    Parameters
    ----------
    pol0 : float
        Measurement in the POL0 filter. Units: electrons or electrons/second.
    pol60 : float
        Measurement in the POL60 filter. Units: electrons or electrons/second.
    pol120 : float
        Measurement in the POL120 filter. Units: electrons or electrons/second.
    c0 : float (Default = 1; optional)
        Efficiency term for the POL0 filter. This comes from the ACS Data Handbook
        Table 5.6 in Section 5.3.4.
    c60 : float (Default = 1; optional)
        Efficiency term for the POL60 filter. This comes from the ACS Data Handbook
        Table 5.6 in Section 5.3.4.
    c120 : float (Default = 1; optional)
        Efficiency term for the POL120 filter. This comes from the ACS Data Handbook
        Table 5.6 in Section 5.3.4.

    Returns
    -------
    i, q, u : tuple of floats
        Stokes I, Q, and U parameters.
    """

    r0 = pol0 * c0
    r60 = pol60 * c60
    r120 = pol120 * c120

    i = (2 / 3) * (r0 + r60 + r120)
    q = (2 / 3) * ((2 * r0) - r60 - r120)
    u = (2 / np.sqrt(3)) * (r60 - r120)

    return i, q, u


def calc_fraction(i, q, u, transmission_correction=1):
    """
    Method for determining the fractional polarization.

    Parameters
    ----------
    i : float
        Stokes I parameter.
    q : float
        Stokes Q parameter.
    u : float
        Stokes U parameter.
    transmission_correction : float (Default = 1)
        Correction factor to account for the leak of photons with
        non-parallel electric field position angles. See Section 5.3.4
        of the ACS Data Handbook.

    Returns
    -------
    pol_fraction : float
        Polarization fraction.
    """

    pol_fraction = np.sqrt(q**2 + u**2) / i
    pol_fraction *= transmission_correction

    return pol_fraction


def calc_theta(q, u, detector, pav3):
    """
    Calculate the position angle of the electric field vector given Stokes Q and U.
    For ACS, it is also necessary to supply the position angle of the V3 axis of the
    telescope (PA_V3 in the primary header) and the name of the ACS detector used.

    Parameters
    ----------
    q : float
        Stokes Q parameter.
    u : float
        Stokes U parameter.
    detector : {'wfc', 'hrc'}
        Name of the ACS detector used for the observation. Must be either WFC or HRC.
    pav3 : float or `~astropy.units.Quantity`
        Position angle of the V3 axis in units of degrees. Found in ACS primary headers
        with keyword PA_V3.

    Returns
    -------
    theta : `~astropy.units.Quantity`
        Position angle of the electric field vector in degrees.
    """

    # If the user supplied an astropy Quantity object, strip off the units.
    # This will make subsequent math easier without attaching units to
    # everything. Make sure it's in degrees first.
    if isinstance(pav3, units.quantity.Quantity):
        pav3 = pav3.to_value(units.degree)

    if detector.lower() not in ['wfc', 'hrc']:
        raise ValueError('Detector must be either WFC or HRC.')

    # Add detector-dependent geometry correction from the ACS Data Handbook.
    # This is -38.2 degrees for WFC and -69.4 degrees for HRC.
    chi = -38.2 if detector.lower() == 'wfc' else -69.4
    theta = 0.5 * np.degrees(np.arctan2(u, q)) + pav3 + chi

    # Force the angle to be between 0 and 360 degrees. This result of
    # the above equation may not always fall in this range because of
    # the roll angle of the telescope and the camera geometry.
    theta = np.mod(theta, 360)

    return theta * units.degree


class PolarizerTables:
    """
    A class for holding all of the polarization tables (as astropy tables) in attributes.
    These attributes are:

    * wfc_transmission: Transmission and leak correction factors for computing ACS/WFC fractional polarization.
    * hrc_transmission: Transmission and leak correction factors for computing ACS/HRC fractional polarization.
    * wfc_efficiency: Efficiencies of the ACS/WFC polarizers for computing Stokes parameters.
    * hrc_efficiency: Efficiencies of the ACS/HRC polarizers for computing Stokes parameters.

    .. note::
        The default table contained within the acstools package uses average transmission leak correction terms
        for a source with a spectrum flat in wavelength space.

    Polarizer calibration information can be read from the default YAML file contained in the
    acstools package, or from a user-supplied YAML file using the class methods :meth:`from_yaml` and
    :meth:`from_package_data`. The YAML file format is:

    .. code-block:: yaml

        transmission:
            meta: dictionary of metadata
            detector:
                filter: list of ACS filters
                t_para: list of parallel transmissions for each filter
                t_perp: list of perpendicular transmissions for each filter
                correction: list of transmission leak correction factors for each filter
        efficiency:
            meta: dictionary of metadata
            detector:
                filter: list of ACS filters
                pol0: list of POL0 coefficients matching each filter
                pol60: list of POL60 coefficients matching each filter
                pol120: list of POL120 coefficients matching each filter

    The meta elements will pass a dictionary of metadata to the output tables. Any metadata
    can be included, but at minimum a description of the origin of the table values should
    be provided. Multiple detectors can be contained in a single YAML file. An example is
    shown below:

    .. code-block:: yaml

        transmission:
            meta: {'description': 'Descriptive message.'}
            wfc:
                filter: ['F475W', 'F606W']
                t_para: [0.42, 0.51]
                t_perp: [0.0, 0.0]
                correction: [1.0, 1.0]
            hrc:
                filter: ['F330W']
                t_para: [0.48]
                t_perp: [0.05]
                correction: [1.21]
        efficiency:
            meta: {'description': 'Descriptive message.'}
            wfc:
                filter: ['F475W', 'F606W']
                pol0: [1.43, 1.33]
                pol60: [1.47, 1.36]
                pol120: [1.42, 1.30]
            hrc:
                filter: ['F330W']
                pol0: [1.73]
                pol60: [1.53]
                pol120: [1.64]

    Parameters
    ----------
    input_dict : dict

    Examples
    --------

    To use the default values supplied in the acstools package (which come from the ACS
    Data Handbook section 5.3):

    >>> from acstools.polarization_tools import PolarizerTables
    >>> tables = PolarizerTables.from_package_data()
    >>> print(tables.wfc_efficiency)
    filter  pol0  pol60  pol120
    ------ ------ ------ ------
     F475W 1.4303 1.4717 1.4269
     F606W 1.3314 1.3607 1.3094
     F775W 0.9965 1.0255 1.0071

    To supply your own YAML file of the appropriate format:

    >>> from acstools.polarization_tools import PolarizerTables
    >>> tables = PolarizerTables.from_yaml('data/polarizer_tables.yaml')
    >>> print(tables.wfc_transmission)
    filter       t_para               t_perp             correction
    ------ ------------------ ---------------------- ------------------
    F475W 0.4239276798513098 0.00015240583841551956  1.000719276691027
    F606W 0.5156734594049419 5.5908749369641956e-05  1.000216861312415
    F775W 0.6040891283746557    0.07367364117759412 1.2777959654493372

    >>> print(tables.wfc_transmission.meta['description'])
    WFC filters use MJD corresponding to 2020-01-01. HRC filters use MJD corresponding to 2007-01-01.
    """
    def __init__(self, input_dict):

        self.data = input_dict

        self.wfc_transmission = Table(self.data['transmission']['wfc'],
                                      names=('filter', 't_para', 't_perp', 'correction'),
                                      meta=self.data['transmission']['meta'])

        self.hrc_transmission = Table(self.data['transmission']['hrc'],
                                      names=('filter', 't_para', 't_perp', 'correction'),
                                      meta=self.data['transmission']['meta'])

        self.wfc_efficiency = Table(self.data['efficiency']['wfc'], names=('filter', 'pol0', 'pol60', 'pol120'),
                                    meta=self.data['efficiency']['meta'])

        self.hrc_efficiency = Table(self.data['efficiency']['hrc'], names=('filter', 'pol0', 'pol60', 'pol120'),
                                    meta=self.data['efficiency']['meta'])

    @classmethod
    def from_yaml(cls, yaml_file):
        """
        Read in a YAML file containing polarizer calibration data.

        Parameters
        ----------
        yaml_file : str
            Path to the YAML file containing the polarizer calibration information.

        Returns
        -------
        pol_tables : `~acstools.polarizer_tools.PolarizerTables`
        """
        with open(yaml_file, 'r') as yf:
            input_dict = yaml.safe_load(yf)
        return cls(input_dict)

    @classmethod
    def from_package_data(cls):
        """
        Use the YAML file contained within the acstools package to retrieve the polarizer
        calibration data.

        Returns
        -------
        pol_tables : `~acstools.polarizer_tools.PolarizerTables`
        """
        filename = get_pkg_data_filename(os.path.join('data', 'polarizer_tables.yaml'))
        return cls.from_yaml(filename)


class Polarization:
    """
    Class for handling ACS polarization data. Input data for this class come from
    photometry of ACS polarization images. The methods associated with this class
    will transform the photometry into Stokes parameters and polarization properties,
    i.e., the polarization fraction and position angle.

    Parameters
    ----------
    pol0 : float
        Photometric measurement in POL0 filter. Units: electrons or electrons/second.
    pol60 : float
        Photometric measurement in POL60 filter. Units: electrons or electrons/second.
    pol120 : float
        Photometric measurement in POL120 filter. Units: electrons or electrons/second.
    filter_name : str
        Name of the filter crossed with the polarization filter, e.g., F606W.
    detector : {'wfc', 'hrc'}
        Name of the ACS detector used for the observation. Must be either WFC or HRC.
    pav3 : float or `~astropy.units.Quantity`
        Position angle of the HST V3 axis. This is stored in the ACS primary header under
        keyword PA_V3. Units: degrees.
    tables : `~acstools.polarization_tools.PolarizerTables`
        Object containing the polarization lookup tables containing the efficiency and
        transmission leak correction factors for the detectors and filters.

    Examples
    --------
    From an ACS/WFC F606W observation of Vela 1-81 (the polarized calibration standard
    star), we have count rates of 63684, 67420, and 63752 electrons/second in POL0V,
    POL60V, and POL120V, respectively. The PA_V3 keyword value in the image header is
    348.084 degrees. (Reference: Table 6; Cracraft & Sparks, 2007 (ACS ISR 2007-10)).
    In this simple case, we will use the polarization reference information contained
    in the acstools package for the calibration of the polarizers. We can use the
    Polarization class to determine the Stokes parameters and polarization properties
    as follows:

    >>> from acstools.polarization_tools import Polarization
    >>> vela_181 = Polarization(63684, 67420, 63752, 'F606W', 'WFC', 348.084)
    >>> vela_181.calc_stokes()
    >>> vela_181.calc_polarization()
    >>> print(f'I = {vela_181.stokes_i:.2f}, Q = {vela_181.stokes_q:.2f}, U = {vela_181.stokes_u:.2f}')
    I = 173336.09, Q = -3758.34, U = 9539.59

    >>> print(f'Polarization: {vela_181.polarization:.2%}, Angle: {vela_181.angle:.2f}')
    Polarization: 5.92%, Angle: 5.64 deg

    If we need to adjust the polarization calibration, we can do so by providing a
    different set of polarization tables using the `~acstools.polarization_tools.PolarizerTables`
    class. See the help text for that class for more information about input format.
    For the same source as above, we can explicitly provide the calibration tables
    (using the default tables in this example) as:

    >>> from acstools.polarization_tools import Polarization, PolarizerTables
    >>> vela_181 = Polarization(63684, 67420, 63752, 'F606W', 'WFC', 348.084,
    >>>                         tables=PolarizerTables.from_yaml('data/polarizer_tables.yaml'))
    >>> vela_181.calc_stokes()
    >>> vela_181.calc_polarization()
    >>> print(f'I = {vela_181.stokes_i:.2f}, Q = {vela_181.stokes_q:.2f}, U = {vela_181.stokes_u:.2f}')
    I = 173336.09, Q = -3758.34, U = 9539.59

    >>> print(f'Polarization: {vela_181.polarization:.2%}, Angle: {vela_181.angle:.2f}')
    Polarization: 5.92%, Angle: 5.64 deg
    """

    def __init__(self, pol0, pol60, pol120, filter_name, detector, pav3, tables=None):

        self.pol0 = pol0
        self.pol60 = pol60
        self.pol120 = pol120

        self.filter_name = filter_name.upper()
        self.detector = detector.lower()
        self.pav3 = pav3

        # Check if detector is a valid value.
        if self.detector not in ['wfc', 'hrc']:
            raise ValueError('Detector must be either WFC or HRC')

        self.stokes_i = None
        self.stokes_q = None
        self.stokes_u = None

        self.polarization = None
        self.angle = None

        # Get correction terms that we need from the polarization tables.
        tables = tables.data if tables else PolarizerTables.from_package_data().data

        if 'transmission' not in tables:
            raise KeyError('Missing polarization reference transmission table.')
        if 'efficiency' not in tables:
            raise KeyError('Missing polarization reference efficiency table.')

        try:
            leak_tab = Table(tables['transmission'][self.detector])
            eff_tab = Table(tables['efficiency'][self.detector])
        except KeyError:
            raise KeyError(f'Polarization reference tables may be missing information for detector '
                           f'{self.detector.upper()}.')

        try:
            self.transmission_correction = leak_tab[np.where(leak_tab['filter'] == self.filter_name)]['correction'][0]
        except IndexError:
            raise IndexError(f'No match found in input transmission leak correction table for detector '
                             f'{self.detector.upper()} and filter {self.filter_name}.')

        try:
            self.c0 = eff_tab[np.where(eff_tab['filter'] == filter_name.upper())]['pol0'][0]
            self.c60 = eff_tab[np.where(eff_tab['filter'] == filter_name.upper())]['pol60'][0]
            self.c120 = eff_tab[np.where(eff_tab['filter'] == filter_name.upper())]['pol120'][0]
        except IndexError:
            raise IndexError(f'No match found in input efficiency correction table for detector '
                             f'{self.detector.upper()} and filter {self.filter_name}.')

    def calc_stokes(self):
        """
        Calculate Stokes parameters using attributes set at initialization.
        """

        self.stokes_i, self.stokes_q, self.stokes_u = calc_stokes(self.pol0, self.pol60, self.pol120,
                                                                  c0=self.c0, c60=self.c60, c120=self.c120)

    def calc_polarization(self):
        """
        Calculate the polarization parameters (fractional polarization and position angle)
        using attributes set at initialization.
        """

        self.polarization = calc_fraction(self.stokes_i, self.stokes_q, self.stokes_u,
                                          transmission_correction=self.transmission_correction)

        self.angle = calc_theta(self.stokes_q, self.stokes_u, self.detector, self.pav3)
