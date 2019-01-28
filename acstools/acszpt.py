"""
This module contains a class, :class:`Query`, that was implemented to provide
users with means to programmatically query the
`ACS Zeropoints Calculator <https://acszeropoints.stsci.edu>`_.
The API works by submitting requests to the
ACS Zeropoints Calculator referenced above and hence, it is only valid for ACS
specific instruments (HRC, SBC, or WFC).

The API can be used in two ways by specifying either a
``(date, detector, filter)`` combination or just a ``(date, detector)``
combination. In the first case, the query
will return the zeropoint information for the specific filter and detector at
specified date. In the second case, the query will return the zeropoint
information for all the filters for the desired detector at the specified date.
In either case, the result will be an Astropy table, where each column has the
correct units attached.

Examples
--------
Retrieve the zeropoint information for all the filters on 2016-04-01 for WFC:

>>> from acstools import acszpt
>>> date = '2016-04-01'
>>> detector = 'WFC'
>>> q = acszpt.Query(date=date, detector=detector)
>>> zpt_table = q.fetch()
>>> print(zpt_table)
FILTER PHOTPLAM        PHOTFLAM         STmag  VEGAmag  ABmag
       Angstrom erg / (Angstrom cm2 s) mag(ST)   mag   mag(AB)
 str6  float64         float64         float64 float64 float64
------ -------- ---------------------- ------- ------- -------
 F435W   4329.2              3.148e-19  25.155  25.763  25.665
 F475W   4746.2              1.827e-19  25.746  26.149  26.056
 F502N   5023.0              5.259e-18  22.098  22.365  22.285
 F550M   5581.5               3.99e-19  24.898  24.825  24.856
 F555W   5360.9              1.963e-19  25.667  25.713  25.713
 F606W   5922.0              7.811e-20  26.668  26.405  26.498
 F625W   6312.0              1.188e-19  26.213  25.735  25.904
 F658N   6584.0               1.97e-18  23.164  22.381  22.763
 F660N   6599.4              5.156e-18  22.119  21.428  21.714
 F775W   7693.2              9.954e-20  26.405  25.272  25.667
 F814W   8045.0              7.046e-20   26.78  25.517  25.944
F850LP   9033.2               1.52e-19  25.945  24.332  24.858
 F892N   8914.8              1.502e-18  23.458  21.905    22.4

Retrieve the zeropoint information for the F435W filter on 2016-04-01 for WFC:

>>> from acstools import acszpt
>>> date = '2016-04-01'
>>> detector = 'WFC'
>>> filt = 'F435W'
>>> q = acszpt.Query(date=date, detector=detector, filter=filt)
>>> zpt_table = q.fetch()
>>> print(zpt_table)
FILTER PHOTPLAM        PHOTFLAM         STmag  VEGAmag  ABmag
       Angstrom erg / (Angstrom cm2 s) mag(ST)   mag   mag(AB)
------ -------- ---------------------- ------- ------- -------
 F435W   4329.2              3.148e-19  25.155  25.763  25.665

"""
from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError

import datetime as dt
import logging
from collections import OrderedDict

import astropy.units as u
from astropy.table import Table
from bs4 import BeautifulSoup

__taskname__ = "acszpt"
__author__ = "Nathan Miles"
__version__ = "1.0"
__vdate__ = "22-Jan-2019"

__all__ = ['Query']

# Initialize the logger
logging.basicConfig()
LOG = logging.getLogger('{}.{}'.format(__taskname__, 'Query'))
LOG.setLevel(logging.INFO)


class Query(object):
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
        self._filt = filt

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

        # Set the private attributes
        if filt is None:
            self._url = ('https://acszeropoints.stsci.edu/results_all/?date={}'
                         '&detector={}'.format(self.date, self.detector))
        else:
            self._filt = filt.upper()
            self._url = ('https://acszeropoints.stsci.edu/'
                         'results_single/?date1={0}&detector={1}'
                         '&{1}_filter={2}'.format(self.date,
                                                  self.detector,
                                                  self.filt))

        self._zpts = OrderedDict()
        self._acs_installation_date = dt.datetime(2002, 3, 7)
        self._msg_div = '-' * 79
        self._valid_detectors = ['HRC', 'SBC', 'WFC']
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
        """The results returned by the ACS Zeropoint Calculator. (`astropy.table.Table`)"""
        return self._zpt_table

    def _check_inputs(self):
        """Check the inputs to ensure they are valid.

        Returns
        -------
        status : bool
            True if all inputs are valid, False if one is not.

        """
        # Determine the submitted detector is valid
        if self.detector not in self._valid_detectors:
            msg = ('{} is not a valid detector option.\n'
                   'Please choose one of the following:\n{}\n'
                   '{}'.format(self.detector,
                               '\n'.join(self._valid_detectors),
                               self._msg_div))
            LOG.error(msg)
            return False

        # Determine if the submitted filter is valid
        if (self.filt is not None and
                self.filt not in self.valid_filters[self.detector]):
            msg = ('{} is not a valid filter for {}\n'
                   'Please choose one of the following:\n{}\n'
                   '{}'.format(self.filt, self.detector,
                               '\n'.join(self.valid_filters[self.detector]),
                               self._msg_div))
            LOG.error(msg)
            return False

        # Determine if the submitted date is valid
        date_check = self._check_date()
        if date_check is not None:
            LOG.error('{}\n{}'.format(date_check, self._msg_div))
            return False

        return True

    def _check_date(self, fmt='%Y-%m-%d'):
        """Convenience method for determining if the input date is valid.

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
            result = '{} does not match YYYY-MM-DD format'.format(self.date)
        else:
            if dt_obj < self._acs_installation_date:
                result = ('The observation cannot occur '
                          'before ACS was installed ({})'.
                          format(self._acs_installation_date.strftime(fmt)))
        finally:
            return result

    def _submit_request(self):
        """Submit a request to the ACS Zeropoint Calculator.

        If an exception is raised during the request, an error message is
        given. Otherwise, the response is saved in the corresponding
        attribute.

        """
        try:
            self._response = urlopen(self._url)
        except URLError as e:
            msg = ('{}\n{}\nThe query failed! Please check your inputs. '
                   'If the error persists, submit a ticket to the '
                   'ACS Help Desk at hsthelp.stsci.edu with the error message '
                   'displayed above.'.format(str(e), self._msg_div))
            LOG.error(msg)
            self._failed = True
        else:
            self._failed = False

    def _parse_response(self):
        """Parse the HTML response returned from the request to find all
        tables on the webpage.

        Using ``beautifulsoup4``, find all the ``<tb> </tb>`` tags present in
        the response. Save
        the results in an :py:class:`OrderedDict` so when parsing occurs,
        we get stable results.

        """
        soup = BeautifulSoup(self._response.read(), 'html.parser')

        # Grab all elements in the table returned by the ZPT calc.
        td = soup.find_all('td')

        # the td variable is a list of all of the table elements. They are
        # ordered such that each block of 6 values corresponds to one row of
        # the html table. The first 6 elements in the list of all elements
        # correspond to the column names. The values after that correspond to
        # actual data for each row
        BLOCK_SIZE = 6
        for j in range(BLOCK_SIZE):
            for i, val in enumerate(td[j::BLOCK_SIZE]):
                if not i:
                    self._zpts[val.text.split('[')[0].strip()] = []
                else:
                    self._zpts[list(self._zpts.keys())[j]].append(val.text)

    def _format_results(self):
        """Format the results into an `astropy.table.Table` with corresponding
        units and assign it to the ``zpt_table`` attribute.

        """
        data_types = []
        data_units = []
        for i, key in enumerate(self._zpts.keys()):
            if key.lower() == 'filter':
                data_types.append(str)
                data_units.append(u.dimensionless_unscaled)

            elif key.lower() == 'photplam':
                data_types.append(float)
                data_units.append(u.angstrom)

            elif key.lower() == 'photflam':
                data_types.append(float)
                data_units.append(u.erg / u.cm**2 / u.second / u.angstrom)

            elif key.lower() == 'stmag':
                data_types.append(float)
                data_units.append(u.STmag)

            elif key.lower() == 'vegamag':
                data_types.append(float)
                data_units.append(u.mag)

            elif key.lower() == 'abmag':
                data_types.append(float)
                data_units.append(u.ABmag)

        tab = Table(list(self._zpts.values()),
                    names=list(self._zpts.keys()),
                    dtype=data_types)

        # Loop through each column and set the appropriate units
        for i, col in enumerate(tab.colnames):
            tab[col].unit = data_units[i]
        self._zpt_table = tab

    def fetch(self):
        """Submit the request to the ACS Zeropoints Calculator.

        This method will:

        * submit the request
        * parse the response
        * format the results into a table with the correct units

        Returns
        -------
        tab : `astropy.table.Table` or `None`
            If the request was successful, returns a table; otherwise, `None`.

        """
        LOG.info('Checking inputs...')
        valid_inputs = self._check_inputs()

        if valid_inputs:
            LOG.info('Submitting request to {}'.format(self._url))
            self._submit_request()
            if self._failed:
                return

            LOG.info('Parsing the response and formatting the results...')
            self._parse_response()
            self._format_results()
            return self.zpt_table

        LOG.error('Please fix the incorrect input(s)')
