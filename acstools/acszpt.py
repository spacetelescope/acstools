"""
The acszpt module contains a class, Query(), that was implemented to provide
users with a means to programmatically query the ACS Zeropoints Calculator hosted
at https://acszeropoints.stsci.edu. The API works by submitting requests to the
ACS Zeropoints Calculator referenced above and hence, it is only valid for ACS
specific instruments (HRC, SBC, or WFC).

The API can be used in two ways by specifying either a (date, detector, filter)
combination or just a (date, detector) combination. In the first case, the query
will return the zeropoint information for the specific filter and detector at
specified date. In the second case, the query will return the zeropoint
information for all the filters for the desired detector at the specified date.
In either case, the result will be an Astropy table where each column has the
correct units attached.

Examples
--------
Retrieve the zeropoint information for all the filters on 2016-04-01 for WFC:

>>> from acstools import acszpt
>>> date = '2016-04-01'
>>> detector='WFC'
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
>>> detector='WFC'
>>> filt = 'F435W'
>>> q = acszpt.Query(date=date, detector=detector, filter=filt)
>>> zpt_table = q.fetch()
>>> print(zpt_table)
FILTER PHOTPLAM        PHOTFLAM         STmag  VEGAmag  ABmag
       Angstrom erg / (Angstrom cm2 s) mag(ST)   mag   mag(AB)
------ -------- ---------------------- ------- ------- -------
 F435W   4329.2              3.148e-19  25.155  25.763  25.665
"""




from astropy.table import Table
import astropy.units as u
from collections import OrderedDict
import datetime as dt
import requests
from bs4 import BeautifulSoup


__taskname__ = "acszpt"
__version__ = "1.0"
__author__ = "Nathan Miles"
__vdate__ = "22-Jan-2019"
__all__ = ['acszpt']



class Query(object):
    """ Class used to interface with the ACS Zeropoints Calculator API.

    Parameters
    ----------
    date : str
        Input date in the following ISO format, YYYY-MM-DD
    detector : str
        One of the three channels on ACS: HRC, SBC, or WFC
    filt : str, optional
        One of valid filters for the chosen detector
            * HRC and WFC:
                 F435W, F475W,
                 F502N, F550M, F555W,
                 F606W, F625W, F658N, F660N,
                 F775W, F814W, F850LP, F892N
            * SBC:
                F115LP, F122M, F125LP,
                F140LP, F150LP, F165LP

    Attributes
    ----------
    date : str
        Input date in the following ISO format, YYYY-MM-DD
    detector : str
        One of the three channels on ACS: HRC, SBC, or WFC
    filt : str, optional
        One of valid filters for the chosen detector
    zpt_table : astropy.table.Table object
        Results returned from the ACS Zeropoint Calculator

    Methods
    -------
    fetch() :
        Submits the request to compute the zeropoints and returns an Astropy
        Table


    """

    def __init__(self, date, detector, filt=None):

        # Set the public attributes
        self.date = date
        self.detector = detector.upper()
        self.filt = filt

        self.valid_filters = {
            'WFC': ['F435W', 'F475W', 'F502N', 'F550M',
                    'F555W', 'F606W', 'F625W', 'F658N',
                    'F660N', 'F775W', 'F814W', 'F850LP','F892N'],

            'HRC': ['F220W', 'F250W', 'F330W', 'F344N',
                    'F435W', 'F475W', 'F502N', 'F550M',
                    'F555W', 'F606W', 'F625W', 'F658N',
                    'F660N', 'F775W', 'F814W', 'F850LP','F892N'],

            'SBC': ['F115LP', 'F122M', 'F125LP',
                   'F140LP', 'F150LP', 'F165LP']
        }
        self.zpt_table = None

        # Set the private attributes
        if filt is None:
            self._url = 'https://acszeropoints.stsci.edu/results_all/?date={}&' \
                        'detector={}'.format(self.date, self.detector)
        else:
            self.filt = filt.upper()

            self._url = 'https://acszeropoints.stsci.edu/results_single/' \
                        '?date1={0}&detector={1}&{1}_filter={2}'. \
                format(self.date, self.detector, self.filt)

        self._zpts = OrderedDict()
        self._response = None
        self._failed = False

    def _check_inputs(self):
        """ Check the input to ensure they are valid

        Returns
        -------

        """
        valid_date = True
        valid_detector = True
        valid_filter = True
        # Determine the submitted detector is valid
        if self.detector not in ['HRC','SBC','WFC']:
            print('Whoops! {} is not a valid detector option.'.
                  format(self.detector))
            print('Please choose one of the following:')
            print('HRC')
            print('SBC')
            print('WFC')
            print('-'*79)
            valid_detector = False

        # Determine if the submitted filter is valid
        if self.filt is None:
            pass
        elif self.filt not in self.valid_filters[self.detector]:
            valid_filter = False
            print('Whoops! {} is not a valid filter for {}'.
                  format(self.filt, self.detector))
            print('Please choose one of the following:')
            for filt in self.valid_filters[self.detector]:
                print(filt)
            print('-' * 79)

        # Determine if the submitted date is valid
        date_check = self._check_date()
        if isinstance(date_check, ValueError):
            valid_date = False
            print('Whoops! {}'.format(date_check))

        # Only continute if all the inputs are valid options
        if all([valid_date, valid_filter, valid_detector]):
            return True
        else:
            return False

    def _check_date(self, fmt='%Y-%m-%d'):
        """Convenience method for determing if the input date is valid
        """
        min_date = dt.datetime(2002, 3, 7)
        try:
            dt_obj = dt.datetime.strptime(self.date, fmt)
        except ValueError:
            return ValueError('Date format does not match YYYY-MM-DD')
        else:
            if min_date < dt_obj:
                result = None
            else:
                result = ValueError('The observation cannot occur before '
                                 'ACS was installed ({})'.
                                 format(min_date.strftime(fmt)))
            return result

    def _submit_request(self):
        """ Submit a request to the ACS Zeropoint Calculator

        If an exception is raised during the request an error message is
        printed. Otherwise, the response is saved in the corresponding
        attribute.
        """
        try:
            self._response = requests.get(self._url)
        except requests.exceptions.RequestException as e:
            print(e)
            print('-'*79)
            print('The query failed! Check your input args and if the error '
                  'persists submit a ticket to the ACS Help Desk at '
                  'hsthelp.stsci.edu with the error message displayed above.')
            self._failed = True

    def _parse_response(self):
        """ Parse the html response returned from the HTML request to find all
        tables on the webpage.

        Using bs4, find all the <tb> </tb> tags present in the response. Save
        the results in an OrderedDict() so when parsing occurs we get stable
        results.

        """
        soup = BeautifulSoup(
            self._response.content.decode(self._response.encoding),
            'html.parser'
        )
        # Grab all elements in the table returned by the ZPT calc.
        td = soup.find_all('td')
        for j in range(6):
            for i, val in enumerate(td[j::6]):
                if not i:
                    self._zpts[val.text.split('[')[0].strip()] = []
                else:
                    self._zpts[list(self._zpts.keys())[j]].append(val.text)

    def _format_results(self):
        """ Format the results into an astropy.table.Table with corresponding
        units

        Returns
        -------
        astropy.table.Table object with the zeropoints for the supplied date
        and detector
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

        self.zpt_table = tab

    def fetch(self):
        """ Submit the request and format the results

        Returns
        -------
        If the request is successful, returns an Astropy table with the correct
        units. If the request fails, returns None.
        """
        valid_inputs = self._check_inputs()
        if valid_inputs:
            self._submit_request()
            if self._failed:
                return None
            else:
                self._parse_response()
                self._format_results()
                return self.zpt_table
        else:
            print('Please fix the incorrect input(s)')
            return None


if __name__ == '__main__':
    date = '2014-03-01'
    detector = 'WFC'
    a = Query(date, detector)
    a.fetch()
