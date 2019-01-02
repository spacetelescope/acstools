#!/usr/bin/env python


from astropy.table import Table
import astropy.units as u
from collections import OrderedDict
import requests
from bs4 import BeautifulSoup
import sys


class Query(object):
    def __init__(self, url):
        self.url = url
        self.zpts = OrderedDict()
        self._response = None


    def submit_request(self):
        """ Submit a request to the ACS Zeropoint Calculator
        """
        try:
            self._response = requests.get(self.url)
        except requests.exceptions.RequestException as e:
            print(e)
            print('-'*79)
            print('The query timed out! If the error persists, submit a '
                  'ticket to the ACS Help Desk at hsthelp.stsci.edu with '
                  'the error message displayed above.')
            sys.exit()

    def parse_response(self):
        """ Parse the html response returned from the HTML request to find all
        tables on the webpage
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
                    self.zpts[val.text.split('[')[0].strip()] = []
                else:
                    # print(self.zpts.keys())
                    self.zpts[list(self.zpts.keys())[j]].append(val.text)

    def format_results(self):
        """ Format the results into an astropy.table.Table with corresponding
        units

        Returns
        -------
        astropy.table.Table object with the zeropoints for the supplied date
        and detector
        """
        data_types = []
        data_units = []
        for i, key in enumerate(self.zpts.keys()):
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

        tab = Table(list(self.zpts.values()),
                    names=list(self.zpts.keys()),
                    dtype=data_types)

        for i, col in enumerate(tab.colnames):
            tab[col].unit = data_units[i]

        return tab




if __name__ == '__main__':
    date = '2014-03-01'
    detector = 'WFC'
    url = 'http://127.0.0.1:5000/results_all/?date={}&detector={}'.\
        format(date, detector)
    a = Query(url)
    a.submit_request()
    a.parse_response()
    print(a.zpts)
    print(a.format_results())