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
In either case, the result will be an ``astropy.table.QTable`` where each column
is an ``astropy.units.quantity.Quantity`` object with the appropriate units attached.

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


Retrieve the zeropoint information for the F435W filter for WFC at multiple dates:

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
2004-10-13 3.074e-19 erg / (Angstrom cm2 s) 25.181 mag(ST)
2011-04-01 3.138e-19 erg / (Angstrom cm2 s) 25.158 mag(ST)
2014-01-17 3.144e-19 erg / (Angstrom cm2 s) 25.156 mag(ST)
2018-05-23 3.152e-19 erg / (Angstrom cm2 s) 25.154 mag(ST)
>>> type(queries[0].zpt_table['PHOTFLAM'])
astropy.units.quantity.Quantity
"""
import datetime as dt
import logging
from urllib.request import urlopen
from urllib.error import URLError

import astropy.units as u
from astropy.table import QTable
from bs4 import BeautifulSoup
import numpy as np

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
        # ACS Launch Date
        self._acs_installation_date = dt.datetime(2002, 3, 7)
        # The farthest date in future that the component and throughput files
        # are valid for. If input date is larger, extrapolation is not valid.
        self._extrapolation_date = dt.datetime(2021, 12, 31)
        self._msg_div = '-' * 79
        self._valid_detectors = ['HRC', 'SBC', 'WFC']
        self._response = None
        self._failed = False
        self._data_units = {
            'FILTER': u.dimensionless_unscaled,
            'PHOTPLAM': u.angstrom,
            'PHOTFLAM': u.erg / u.cm ** 2 / u.second / u.angstrom,
            'STmag': u.STmag,
            'VEGAmag': u.mag,
            'ABmag': u.ABmag
        }
        self._block_size = len(self._data_units)

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
            msg = ('{} is not a valid detector option.\n'
                   'Please choose one of the following:\n{}\n'
                   '{}'.format(self.detector,
                               '\n'.join(self._valid_detectors),
                               self._msg_div))
            LOG.error(msg)
            valid_detector = False

        # Determine if the submitted filter is valid
        if (self.filt is not None and valid_detector and
                self.filt not in self.valid_filters[self.detector]):
            msg = ('{} is not a valid filter for {}\n'
                   'Please choose one of the following:\n{}\n'
                   '{}'.format(self.filt, self.detector,
                               '\n'.join(self.valid_filters[self.detector]),
                               self._msg_div))
            LOG.error(msg)
            valid_filter = False

        # Determine if the submitted date is valid
        date_check = self._check_date()
        if date_check is not None:
            LOG.error('{}\n{}'.format(date_check, self._msg_div))
            valid_date = False

        if not valid_detector or not valid_filter or not valid_date:
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
                result = ('The observation date cannot occur '
                          'before ACS was installed ({})'
                          .format(self._acs_installation_date.strftime(fmt)))
            elif dt_obj > self._extrapolation_date:
                result = ('The observation date cannot occur after the '
                          'maximum allowable date, {}. Extrapolations of the '
                          'instrument throughput after this date lead to '
                          'high uncertainties and are therefore invalid.'
                          .format(self._extrapolation_date.strftime(fmt)))
        finally:
            return result

    def _submit_request(self):
        """Submit a request to the ACS Zeropoint Calculator.

        If an exception is raised during the request, an error message is
        given. Otherwise, the response is saved in the corresponding
        attribute.

        """
        if not self._url.startswith('http'):
            raise ValueError(f'Invalid URL {self._url}')
        try:
            self._response = urlopen(self._url)  # nosec
        except URLError as e:
            msg = ('{}\n{}\nThe query failed! Please check your inputs. '
                   'If the error persists, submit a ticket to the '
                   'ACS Help Desk at hsthelp.stsci.edu with the error message '
                   'displayed above.'.format(str(e), self._msg_div))
            LOG.error(msg)
            self._failed = True
        else:
            self._failed = False

    def _parse_and_format(self):
        """ Parse and format the results returned by the ACS Zeropoint Calculator.

        Using ``beautifulsoup4``, find all the ``<tb> </tb>`` tags present in
        the response. Format the results into an astropy.table.QTable with
        corresponding units and assign it to the zpt_table attribute.
        """

        soup = BeautifulSoup(self._response.read(), 'html.parser')

        # Grab all elements in the table returned by the ZPT calc.
        td = soup.find_all('td')

        # Remove the units attached to PHOTFLAM and PHOTPLAM column names.
        td = [val.text.split(' ')[0] for val in td]

        # Turn the single list into a 2-D numpy array
        data = np.reshape(td,
                          (int(len(td) / self._block_size), self._block_size))
        # Create the QTable, note that sometimes self._response will be empty
        # even though the return was successful; hence the try/except to catch
        # any potential index errors. Provide the user with a message and
        # set the zpt_table to None.
        try:
            tab = QTable(data[1:, :],
                         names=data[0],
                         dtype=[str, float, float, float, float, float])
        except IndexError as e:
            msg = ('{}\n{}\n There was an issue parsing the request. '
                   'Try resubmitting the query. If this issue persists, please '
                   'submit a ticket to the Help Desk at'
                   'https://stsci.service-now.com/hst'
                   .format(e, self._msg_div))
            LOG.info(msg)
            self._zpt_table = None
        else:
            # If and only if no exception was raised, attach the units to each
            # column of the QTable. Note we skip the FILTER column because
            # Quantity objects in astropy must be numerical (i.e. not str)
            for col in tab.colnames:
                if col.lower() == 'filter':
                    continue
                tab[col].unit = self._data_units[col]

            self._zpt_table = tab

    def fetch(self):
        """Submit the request to the ACS Zeropoints Calculator.

        This method will:

        * submit the request
        * parse the response
        * format the results into a table with the correct units

        Returns
        -------
        tab : `astropy.table.QTable` or `None`
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
            self._parse_and_format()
            return self.zpt_table

        LOG.error('Please fix the incorrect input(s)')
