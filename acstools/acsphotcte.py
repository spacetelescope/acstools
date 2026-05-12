#!/usr/bin/env python
"""
The ACS Photometric CTE API is a programmatic interface for the
`ACS Photometric CTE Webtool <https://acsphotometriccte.stsci.edu>`_.
The API is a cloud-based service that employs a serverless approach on AWS
with API Gateway and Lambda to compute the photometric CTE corrections
using the model described in `ACS ISR 2022-06 <https://ui.adsabs.harvard.edu/abs/2022acs..rept....6C/abstract>`_.
The model corrects ACS/WFC aperture photometry extracted from FLT images for
CTE losses. It is only calibrated for photometry obtained after SM4 in May
2009. For pre-SM4 data, please see `ACS ISR 2009-01 <https://ui.adsabs.harvard.edu/abs/2009acs..rept....1C/abstract>`_, or use
pixel-based CTE-corrected files obtained from MAST. Currently, only 3 and 5
pixel aperture radii are supported. The model is designed to compute the
CTE losses as a function of the number of Y transfers, sky background,
source flux, and observation date.

Usage Guidelines
----------------
As this service is hosted on AWS, there are some guidelines that should be
followed when utilizing this tool. For each call to
:py:meth:`~acstools.acsphotcte.PhotCTEAPI.correct_photometry`, AWS will assemble the compute resources,
install the software, compute the CTE corrections, send the result back,
and then terminate all resources. Hence, users should try to process as many
sources as they can in a single function call. Testing has shown that the
optimal number is < 25,000, but the service can handle as many as 150,000
sources in a single request.

Examples
--------
In this example, we obtain the CTE-corrected FLT photometry for a list of 1000
artificial sources. For each parameter, we generate 1000 random values in the interval [0, 1) and
then scale the random data by a realistic value for stellar fluxes,
y-transfers, and local sky backgrounds. An arbitrary MJD is defined, and
we assume an aperture radius of 3 pixels.

>>> import numpy as np
>>> from acstools import acsphotcte
>>> n_sample = 1000
>>> ytransfers = 2048 * np.random.random(size=n_sample)
>>> fluxes = 20000*np.random.random(size=n_sample)
>>> magnitudes = -2.5 * np.log10(fluxes)
>>> print(magnitudes[:5])
[-10.75088228  -9.56321561 -10.10571966  -9.01893015 -10.48463087]
>>> local_skys = 80*np.random.random(size=n_sample)
>>> mjd = 59341.
>>> radius = 3.
>>> photctecalc = acsphotcte.PhotCTEAPI()
>>> cte_corrected_magnitudes = photctecalc.correct_photometry(
...    radius=radius,
...    ytransfers=ytransfers,
...    mjd=mjd,
...    local_skys=local_skys,
...    fluxes=fluxes
...)
>>> print(cte_corrected_magnitudes[:5])
[-10.85516545  -9.68284332 -10.11060704  -9.11828746 -10.4918177 ]
"""
from collections.abc import Iterable
import json
import logging

# LOCAL
from .utils_calib import SM4_MJD

import numpy as np
import requests

logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger('PhotCTEAPI')
LOG.setLevel(logging.INFO)

__taskname__ = "acsphotcte"
__author__ = "Nathan Miles"
__version__ = "1.0"
__vdate__ = "26-Mar-2020"

__all__ = ['PhotCTEAPI']


class PhotCTEAPI:
    """Convenience class for handling queries to ACS Photometric CTE API.
    """

    def __init__(self):

        # API Endpoint for the ACS Photometric CTE API
        self._api_url = ('https://acsphotometriccteapi.stsci.edu/'
                         'run-photcte-correction')

        # Public API ID for the ACS Photometric CTE API
        self._api_id = 'hdtenv1tu1'

        # Public API key for the ACS Photometric CTE API
        self._api_key = 'rwPplgrNMMa9duXXhL1tA39sOPviMiTd4vsKXXQh'

        # API credentials to pass in header of POST request
        self._api_credentials = {
            'x-api-id': self._api_id,
            'x-api-key': self._api_key
        }
        self._cte_corrections = None
        self._corrected_magnitudes = None

    @property
    def corrected_magnitudes(self):
        """CTE corrected magnitudes"""
        return self._corrected_magnitudes

    @corrected_magnitudes.setter
    def corrected_magnitudes(self, value):
        self._corrected_magnitudes = value

    @property
    def cte_corrections(self):
        """Computed CTE corrections"""
        return self._cte_corrections

    @cte_corrections.setter
    def cte_corrections(self, value):
        self._cte_corrections = value

    def _query(self, data):
        """ Submit a post request to the API

        Parameters
        ----------
        data : dict
            Dictionary containing the request payload

        Returns
        -------
        content :
            Body of the response object
        """

        response = requests.post(
            self._api_url,
            data=json.dumps(data),
            headers=self._api_credentials,
            timeout=100
        )

        status_code = response.status_code
        content = json.loads(response.content)
        if status_code > 400:
            content = (f"Status Code: {status_code}\n"
                       f"Message: {content['message']}")
        return content

    def _check_inputs(self, **inputs):
        """Check the data types of the user supplied inputs

        JSON is very finicky and the native python json module only works
        with native python types (int, float, list, str). Hence, before
        we serialize the inputs as a JSON object we need to compare all the
        input data types to their python counterparts.

        Parameters
        ----------
        inputs : dict
            Dictionary containing all of the user supplied inputs

        Returns
        -------
        inputs : dict, str, `None`
            If the inputs conform to what is expected, it returns the original
            dictionary with all of the inputs converted to their python
            counterparts for JSON serialization. If one of the inputs fails,
            it returns the corresponding key. If the iterable inputs are not
            all the same length, it returns `None`.
        """
        iterable_lengths = []
        for key in inputs.keys():
            # Check if the input is Iterable and not a list  (e.g. np.array)
            # We also need to make sure the input is not a str, since str are
            # Iterable in Python.
            if isinstance(inputs[key], Iterable) and \
                    not isinstance(inputs[key], (str, list)):
                # Convert the array to a python list
                inputs[key] = list(inputs[key])

                # Check if the first element of the input array is a float
                if not isinstance(inputs[key][0], float):
                    LOG.info('Converting elements..')
                    # Convert the array of elements to floats
                    inputs[key] = list(map(float, inputs[key]))
                # Save the length for comparision later
                iterable_lengths.append(len(inputs[key]))
            # Check if the input is a python float, not an np.float32
            # Handles the case of str inputs as well
            elif not isinstance(inputs[key], float):
                try:
                    inputs[key] = float(inputs[key])
                except TypeError as e:
                    # TypeError during the conversion, print the info and
                    # return the key it failed on
                    LOG.error(e)
                    return key
                else:
                    if key == 'radius' and inputs[key] not in [3., 5.]:
                        LOG.error('Submitted radii not supported...')
                        return key
        # the last step is to ensure all the iterable inputs are the same length
        nl = "\n"
        # First we check if there were any iterable inputs.
        # If there are, we make sure theyre all the same size.
        if iterable_lengths and len(set(iterable_lengths)) != 1:
            iterable_lengths = list(map(str, iterable_lengths))
            msg = (
                "Iterable inputs are not the same length.\n"
                f"Computed lengths:\n{nl.join(iterable_lengths)}"
            )
            LOG.error(msg)
            return
        # Check to see if the date occurs before SM4
        if inputs['mjd'] < SM4_MJD:
            LOG.error(f"Observation MJD must occur after SM4, {SM4_MJD}")
            return 'mjd'

        return inputs

    def correct_photometry(
            self,
            radius=None,
            ytransfers=None,
            mjd=None,
            local_skys=None,
            fluxes=None
    ):
        """Get the CTE corrected FLT photometry.

        Parameters
        ----------
        radius : {3, 5}
            Aperture radius used to extract photometry.

        ytransfers : float, list
            Number of row transfers. For chip 2 on WFC, this is the Y value.
            For chip 1, it is ``2048 - Y``.

        mjd : float
            MJD date of the observation.

        local_sky : float, list
            Local sky background for each source in units of electrons.

        fluxes : float, list
            Computed source flux, in electrons, for 3 or 5 pixel apertures.

        Returns
        -------
        corrected_magnitudes : `numpy.array`, `None`
            If the query is successful it returns the CTE corrected magnitudes
            for the sources. Otherwise, it returns `None`.
        """

        inputs = {
            'radius': radius,
            'ytransfers': ytransfers,
            'mjd': mjd,
            'local_skys': local_skys,
            'fluxes': fluxes
        }
        inputs = self._check_inputs(**inputs)

        if not isinstance(inputs, dict) and isinstance(inputs, str):
            LOG.error(f'Please check the following input: {inputs}')
            return
        elif inputs is None:
            LOG.error('Please check the iterable inputs')
            return

        # Submit the request and check the result
        content = self._query(inputs)
        # If the request fails, display the message and return None
        if isinstance(content, str):
            LOG.error(content)
            return

        # If successful, update the attribute with the computed corrections
        self.cte_corrections = np.array(content['deltamag'])

        # Use the corrections to compute the corrected fluxes
        self.corrected_magnitudes = \
            -2.5 * np.log10(np.array(fluxes)) - self.cte_corrections

        return self.corrected_magnitudes
