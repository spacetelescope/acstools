"""
This module contains a class, :class:`Query`, that was implemented to provide
users with means to programmatically query the
`ACS Zeropoints Calculator <https://acszeropoints.stsci.edu>`_ API.
This class works by submitting requests directly to the AWS API, which also
handles the requests from the above web address. It is only valid for ACS
specific instruments (HRC, SBC, or WFC).

The API can be used in two ways by specifying either a
``(date, detector, filter)`` combination or just a ``(date, detector)``
combination. In the first case, the query
will return the zeropoint information for the specific filter and detector at
specified date. In the second case, the query will return the zeropoint
information for all the filters for the desired detector at the specified date.
In either case, the result will be an ``astropy.table.QTable`` where each column
is an ``astropy.units.quantity.Quantity`` object with the appropriate units attached.

Examples
--------

Retrieve the zeropoint information for all of the WFC filters on 2016-04-01:

>>> from acstools import acszpt
>>> q = acszpt.Query(date="2016-04-01", detector="WFC")
>>> zpt_table = q.fetch()
>>> print(zpt_table)
Filter PHOTLAM             PHOTFLAM            STmag  VEGAmag  ABmag
       Angstrom erg / (Angstrom cm2 electron) mag(ST)   mag   mag(AB)
------ -------- ----------------------------- ------- ------- -------
 F435W   4329.9                    3.1858e-19  25.142  25.767  25.652
 F475W   4747.0                     1.845e-19  25.735  26.151  26.045
 F502N   5023.0                    5.2934e-18  22.091  22.367  22.278
 F550M   5581.5                    4.0186e-19   24.89  24.825  24.848
 F555W   5361.0                    1.9798e-19  25.658  25.714  25.704
 F606W   5921.9                    7.8774e-20  26.659  26.402  26.489
 F625W   6311.8                     1.198e-19  26.204  25.728  25.895
 F658N   6584.0                    1.9976e-18  23.149  22.378  22.748
 F660N   6599.4                    5.2219e-18  22.105  21.414    21.7
 F775W   7693.5                    1.0033e-19  26.396   25.26  25.658
 F814W   8045.5                     7.092e-20  26.773  25.506  25.937
F850LP   9031.5                    1.5313e-19  25.937  24.323  24.851
 F892N   8914.8                    1.5105e-18  23.452  21.892  22.394

Retrieve the zeropoint information for the WFC/F435W filter on 2016-04-01:

>>> from acstools import acszpt
>>> q = acszpt.Query(date="2016-04-01", detector="WFC", filt="F435W")
>>> zpt_table = q.fetch()
>>> print(zpt_table)
Filter PHOTLAM             PHOTFLAM            STmag  VEGAmag  ABmag
       Angstrom erg / (Angstrom cm2 electron) mag(ST)   mag   mag(AB)
------ -------- ----------------------------- ------- ------- -------
 F435W   4329.9                    3.1858e-19  25.142  25.767  25.652

Retrieve the zeropoint information for the WFC/F435W filter on multiple dates:

>>> from acstools import acszpt
>>> dates = ['2004-10-13', '2011-04-01', '2014-01-17', '2018-05-23']
>>> queries = []

>>> for date in dates:
...     q = acszpt.Query(date=date, detector='WFC', filt='F435W')
...     zpt_table = q.fetch()
...     # Each object has a zpt_table attribute, so we save the instance
...     queries.append(q)

>>> for q in queries:
...     print(q.date, q.zpt_table['PHOTFLAM'][0], q.zpt_table['STmag'][0])
2004-10-13 3.1111e-19 erg / (Angstrom cm2 electron) 25.168 mag(ST)
2011-04-01 3.1766e-19 erg / (Angstrom cm2 electron) 25.145 mag(ST)
2014-01-17 3.1817e-19 erg / (Angstrom cm2 electron) 25.143 mag(ST)
2018-05-23 3.1897e-19 erg / (Angstrom cm2 electron) 25.141 mag(ST)

>>> type(queries[0].zpt_table['PHOTFLAM'])
astropy.units.quantity.Quantity

"""
import datetime as dt
import json
import logging
import os
import requests

import astropy.units as u
from astropy.table import QTable

__taskname__ = "acszpt"
__author__   = "Gagandeep Anand, Jenna Ryon, Nathan Miles"
__version__  = "2.0"
__vdate__    = "10-Aug-2022"
__all__      = ['ACSZeropointQueryError', 'Query']

# Initialize the logger
logging.basicConfig()
LOG = logging.getLogger(f'{__taskname__}.Query')
LOG.setLevel(logging.INFO)


class ACSZeropointQueryError(Exception):
    """Class used for raising exceptions with API Gateway post requests.
    """
    pass


class Query:
    """Class used to interface with the ACS Zeropoints Calculator API.

    Parameters
    ----------
    date : str
        Input date in the following ISO format, YYYY-MM-DD.

    detector : {'HRC', 'SBC', 'WFC'}
        One of the three channels on ACS: HRC, SBC, or WFC.

    filt : str or `None`, optional
        One of valid filters for the chosen detector. If no filter is supplied,
        all of the filters for the chosen detector will be used:

            * HRC:
                 F220W, F250W, F330W,
                 F344N, F435W, F475W,
                 F502N, F550M, F555W,
                 F606W, F625W, F658N, F660N,
                 F775W, F814W, F850LP, F892N
            * WFC:
                 F435W, F475W,
                 F502N, F550M, F555W,
                 F606W, F625W, F658N, F660N,
                 F775W, F814W, F850LP, F892N
            * SBC:
                 F115LP, F122M, F125LP,
                 F140LP, F150LP, F165LP

    """

    def __init__(self, date, detector, filt=None):

        # Set the attributes
        self._date = date
        self._detector = detector.upper()
        if filt is None:
            self._filt = filt
        else:
            self._filt = filt.upper()

        # define valid detectors and filter combinations
        self._valid_detectors = ('HRC', 'SBC', 'WFC')
        self.valid_filters = {
            'WFC': ['F435W', 'F475W', 'F502N', 'F550M',
                    'F555W', 'F606W', 'F625W', 'F658N',
                    'F660N', 'F775W', 'F814W', 'F850LP', 'F892N'],

            'HRC': ['F220W', 'F250W', 'F330W', 'F344N',
                    'F435W', 'F475W', 'F502N', 'F550M',
                    'F555W', 'F606W', 'F625W', 'F658N',
                    'F660N', 'F775W', 'F814W', 'F850LP', 'F892N'],

            'SBC': ['F115LP', 'F122M', 'F125LP',
                    'F140LP', 'F150LP', 'F165LP']
        }

        self._zpt_table = None
        self._warnings = []

        # ACS Launch Date
        self._acs_installation_date = dt.datetime(2002, 3, 7)
        # end of data table in AWS S3 bucket
        self._end_table_date = dt.datetime(2029, 12, 31)
        self._msg_div = '-' * 79
        self._response = None
        self._failed = False

    @property
    def date(self):
        """The user supplied date. (str)"""
        return self._date

    @property
    def detector(self):
        """The user supplied detector. (str)"""
        return self._detector

    @property
    def filt(self):
        """The user supplied filter, if one was given. (str or `None`)"""
        return self._filt

    @property
    def zpt_table(self):
        """The results returned by the ACS Zeropoint Calculator. (`astropy.table.QTable`)"""
        return self._zpt_table

    def _check_inputs(self):
        """Check the inputs to ensure they are valid.

        Returns
        -------
        status : bool
            True if all inputs are valid, False if one is not.
        """

        valid_detector = True
        valid_filter = True
        valid_date = True

        # Determine the submitted detector is valid
        if self.detector not in self._valid_detectors:
            msg = (f'{self.detector} is not a valid detector option.\n'
                   'Please choose one of the following:\n'
                   f'{os.linesep.join(self._valid_detectors)}\n'
                   f'{self._msg_div}')
            LOG.error(msg)
            valid_detector = False

        # Determine if the submitted filter is valid
        if (self.filt is not None and valid_detector and
                self.filt not in self.valid_filters[self.detector]):
            msg = (f'{self.filt} is not a valid filter for {self.detector}\n'
                   'Please choose one of the following:\n'
                   f'{os.linesep.join(self.valid_filters[self.detector])}\n'
                   f'{self._msg_div}')
            LOG.error(msg)
            valid_filter = False

        # Determine if the submitted date is valid
        date_check = self._check_date()
        if date_check is not None:
            LOG.error(f'{date_check}\n{self._msg_div}')
            valid_date = False

        if not valid_detector or not valid_filter or not valid_date:
            return False

        return True

    def _check_date(self, fmt='%Y-%m-%d'):
        """For determining if the input date is valid.

        Parameters
        ----------
        fmt : str
            The format of the date string. The default is ``%Y-%m-%d``, which
            corresponds to ``YYYY-MM-DD``.

        Returns
        -------
        status : str or `None`
            If the date is valid, returns `None`. If the date is invalid,
            returns a message explaining the issue.
        """

        result = None
        try:
            dt_obj = dt.datetime.strptime(self.date, fmt)
        except ValueError:
            result = f'{self.date} does not match YYYY-MM-DD format'
        else:
            if dt_obj < self._acs_installation_date:
                result = ('The observation date cannot occur '
                          'before ACS was installed '
                          f'({self._acs_installation_date.strftime(fmt)})')
            if dt_obj > self._end_table_date:
                result = ('The observation date cannot be '
                          'after 2029 '
                          f'({self._end_table_date.strftime(fmt)})')
        finally:
            return result

    def fetch(self):
        """Function to query API on AWS APIGateway for zeropoints for single or all
        filters of a given ACS detector (HRC, SBC, or WFC) on a specified date.

        Returns
        -------
        table : `astropy.table.QTable` or `None`
            If the request was successful, returns an astropy Qtable; otherwise, `None`.
        """

        # check user input date, detector, and filter
        bool_inputs = self._check_inputs()

        if bool_inputs:

            # select between all filters or a single filter depending on if user has
            # specified a filter, and then generate a request body to send the API
            if self.filt is None:
                # URL to invoke all_filter API on AWS APIGateway
                invokeURL = "https://vtbopx9sf3.execute-api.us-east-1.amazonaws.com/main/all-filter-resource"  # noqa
                # Generate a request body to send the API
                body = '{"date": "%s", "detector": "%s"}' % (self.date, self.detector)
            else:
                # URL to invoke single_filter API on AWS APIGateway
                invokeURL = "https://vtbopx9sf3.execute-api.us-east-1.amazonaws.com/main/single-filter-resource"  # noqa
                # Generate a request body to send the API
                body = '{"date": "%s", "detector": "%s", "filter": "%s"}' % (self.date, self.detector, self.filt)

            # send request to APIGateway with try/except clauses
            help_desk_string = ("\nIf this error persists, please contact the HST Help Desk at \n"
                                "https://stsci.service-now.com/hst")

            try:
                response = requests.post(invokeURL, data=body, timeout=100)

            except requests.exceptions.HTTPError:
                raise ACSZeropointQueryError(f"HTTP Error to AWS API Gateway:{help_desk_string}") from None

            except requests.exceptions.ConnectionError:
                raise ACSZeropointQueryError(f"Error Connecting to AWS API Gateway:{help_desk_string}") from None

            except requests.exceptions.Timeout:
                raise ACSZeropointQueryError(f"Timeout Error to AWS API Gateway:{help_desk_string}") from None

            except requests.exceptions.RequestException:
                raise ACSZeropointQueryError(f"Request Exception Error to AWS API Gateway:{help_desk_string}") from None

            # and store the results
            APIoutput = json.loads(response.text)

            # define appropriate headers for the table
            headers = ['Detector', 'Filter', 'PHOTLAM', 'PHOTFLAM', 'STmag', 'VEGAmag', 'ABmag']

            # store each row of the results
            rows = APIoutput["rows"]

            # generate an astropy QTable from the results
            table = QTable(rows=rows, names=headers,
                           dtype=('S', 'S', 'float64', 'float64', 'float64', 'float64', 'float64'))

            # remove detector column (to match output of previous API version)
            table.remove_column('Detector')

            # set the appropriate units for each column
            table['Filter'].unit = u.dimensionless_unscaled
            table['PHOTLAM'].unit = u.angstrom
            table['PHOTFLAM'].unit = u.erg / (u.cm * u.cm * u.angstrom * u.electron)
            table['STmag'].unit = u.STmag
            table['VEGAmag'].unit = u.mag
            table['ABmag'].unit = u.ABmag

            self._zpt_table = table

        return self._zpt_table
