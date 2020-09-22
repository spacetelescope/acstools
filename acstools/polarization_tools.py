#!/usr/bin/env python
"""
Toolkit for analyzing HST/ACS polarization data.

CHANGE LOG
----------
2020-09-22: Creation. (T. Desjardins)
"""

import numpy as np
import os
import yaml

from astropy.table import Table


class PolarizerTables:
    """
    A simple class for holding all of the polarization tables (as astropy tables) in attributes.
    """
    def __init__(self):
        with open(os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir, 'data',
                                                'polarizer_tables.yaml'))) as yf:
            self.table_data = yaml.load(yf, Loader=yaml.FullLoader)

        self.wfc_transmission = Table(self.table_data['transmission']['wfc'],
                                      names=('filter', 't_para', 't_perp', 'correction'))

        self.hrc_transmission = Table(self.table_data['transmission']['hrc'],
                                      names=('filter', 't_para', 't_perp', 'correction'))

        self.wfc_efficiency = Table(self.table_data['efficiency']['wfc'],
                                    names=('filter', 'pol0', 'pol60', 'pol120'))

        self.hrc_efficiency = Table(self.table_data['efficiency']['hrc'],
                                    names=('filter', 'pol0', 'pol60', 'pol120'))

class Polarization:
    """
    Class for handling ACS polarization data.

    Examples
    --------

    >>> from acstools.polarization_tools import Polarization
    >>> vela_181 = Polarization(63684, 67420, 63752, 'F606W', 'WFC', 348.084)
    >>> vela_181.calc_stokes()
    >>> vela_181.calc_polarization()
    >>> print(f'I = {vela_181.stokes_i:.2f}, Q = {vela_181.stokes_q:.2f}, U = {vela_181.stokes_u:.2f}')
    >>> print(f'Polarization: {vela_181.polarization * 100:.2f} %, Angle: {vela_181.angle:.2f} deg')
    """

    def __init__(self, pol0, pol60, pol120, filter_name, detector, pav3):
        """
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
        detector : str
            Name of the ACS detector used for the observation. Must be either WFC or HRC.
        pav3 : float
            Position angle of the HST V3 axis. This is stored in the ACS primary header under
            keyword PA_V3. Units: degrees.
        """

        self.pol0 = pol0
        self.pol60 = pol60
        self.pol120 = pol120

        self.filter_name = filter_name
        self.detector = detector
        self.pav3 = pav3

        # Check if detector is a valid value.
        if self.detector.lower() not in ['wfc', 'hrc']:
            raise ValueError('Detector must be either WFC or HRC')

        self.stokes_i = None
        self.stokes_q = None
        self.stokes_u = None

        self.polarization = None
        self.angle = None

        # Get correction terms that we need from the polarization tables.
        tables = PolarizerTables()
        leak_tab = Table(tables.table_data['transmission'][detector.lower()])
        self.transmission_correction = leak_tab[np.where(leak_tab['filter'] == filter_name.upper())]['correction'][0]

        eff_tab = Table(tables.table_data['efficiency'][detector.lower()])
        self.c0 = eff_tab[np.where(eff_tab['filter'] == filter_name.upper())]['pol0'][0]
        self.c60 = eff_tab[np.where(eff_tab['filter'] == filter_name.upper())]['pol60'][0]
        self.c120 = eff_tab[np.where(eff_tab['filter'] == filter_name.upper())]['pol120'][0]

    def calc_stokes(self):
        """
        Calculate Stokes parameters using attributes set at initialization.
        """

        self.stokes_i, self.stokes_q, self.stokes_u = self._calc_stokes([self.pol0, self.pol60, self.pol120],
                                                                        [self.c0, self.c60, self.c120])

    def _calc_stokes(self, pol_measurements, correction_terms):
        """
        Method for determining Stokes parameters for ACS observations.

        Parameters
        ----------
        pol_measurements : list or tuple
            Measurements in each of the three ACS polarization filters. Must be in order
            of [POL0, POL60, POL120]. Units: electrons or electrons/second.
        correction_terms : list or tuple
            Polarizer efficiency correction values. Must be in order of [POL0, POL60, POL120].
            These can be found in the tables provided by `acstools.acstools.polarization_tools.PolarizerTables()`
            or in Table 5.6 in Section 5.3 of the ACS Data Handbook.

        Returns
        -------
        i, q, u : tuple of floats
            Stokes I, Q, and U parameters.
        """

        r0 = pol_measurements[0] * correction_terms[0]
        r60 = pol_measurements[1] * correction_terms[1]
        r120 = pol_measurements[2] * correction_terms[2]

        i = (2 / 3) * (r0 + r60 + r120)
        q = (2 / 3) * ((2 * r0) - r60 - r120)
        u = (2 / np.sqrt(3)) * (r60 - r120)

        return i, q, u

    def calc_polarization(self):
        """
        Calculate the polarization parameters (fractional polarization and position angle) for
        using attributes set at initialization.
        """

        self.polarization = self._calc_fraction(self.stokes_i, self.stokes_q, self.stokes_u,
                                                transmission_correction=self.transmission_correction)

        self.angle = self._calc_theta(self.stokes_q, self.stokes_u, self.pav3)

    def _calc_fraction(self, i, q, u, transmission_correction=1):
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

    def _calc_theta(self, q, u, pav3):
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
        pav3 : float
            Position angle of the V3 axis in units of degrees. Found in ACS primary headers
            with keyword PA_V3.

        Returns
        -------
        theta : float
            Position angle of the electric field vector in degrees.
        """

        chi = -38.2 if self.detector.lower() == 'wfc' else -69.4
        theta = 0.5 * np.degrees(np.arctan2(u, q)) + pav3 + chi
        theta -= 360 if theta > 360 else 0

        return theta
