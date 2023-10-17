"""

This module first contains two functions that were implemented to provide
users with means to programmatically query the `ACS/WFC
Focus-Diverse ePSF Generator <acspsf.stsci.edu>`_ API.
The first function (psf_retriever) provides the user with the ability
to perform downloads for a single image rootname, whereas the
second function (multi_psf_retriever) provides the ability to do so
for many rootnames. In all cases, the focus-diverse ePSFs will be
downloaded to a location specified by the user in the function call
(download_location).

Additionally, this module also contains a function (interp_epsf) that allows
users to interpolate the provided ePSF arrays to any arbitrary (x,y)
coordinates. This function can be called with "pixel_space" = True to
downsample the ePSF into detector space. Subpixel phase shifts can be applied by
setting subpixel_x and subpixel_y between 0.00 and 0.99.

We strongly recommend you read ACS ISR 2018-08 by A. Bellini et al. and
ACS ISR 2023-XY by G. Anand et al. (LINK WHEN LIVE) for more details 
on these ePSF models and tools before using them.

Examples
--------

Retrieve a single 101x101x90 ePSF FITS file.

>>> from focus_diverse_epsfs import psf_retriever
>>> download_location  = '/Users/username/download_folder/'
>>> retrieved_download = psf_retriever('jds408jsq', download_location)

Retrieve ePSFs based on flc images specified in an external text file (with one rootname per line).

>>> from focus_diverse_epsfs import multi_psf_retriever
>>> input_list = 'input_ipsoots.txt'
>>> download_location  = '/Users/username/download_folder/'
>>> retrieved_downloads = multi_psf_retriever(input_list, download_location, fromTextFile = True)

Retrieve ePSFs using rootnames obtained from an astroquery.

>>> from astroquery.mast import Observations
>>> from focus_diverse_epsfs import multi_psf_retriever
>>> download_location  = '/Users/username/download_folder/'
>>> obsTable = Observations.query_criteria(obs_collection = 'HST', proposal_id="13376", instrument_name = "ACS/WFC",
                                       provenance_name = "CALACS")

>>> dataProducts = Observations.get_product_list(obsTable)
>>> dataProducts = dataProducts[(dataProducts['productSubGroupDescription'] == 'FLC') &
             (dataProducts['type'] == 'S')]

>>> obs_rootnames = list(dataProducts['obs_id'])
>>> retrieved_downloads = multi_psf_retriever(obs_rootnames, download_location)

Interpolate a given ePSF to x,y = (2000,4048), which is near the middle of the detector along the x-axis,
and near the top of the WFC1 chip (and the detector overall).

>>> from focus_diverse_epsfs import interp_epsf
>>> x = 2000
>>> y = 4048
>>> P = interp_epsf(ePSFs, x, y)

Do the same as the previous example, but obtain the ePSF in detector space (instead of with 4x supersampling).
>>> from focus_diverse_epsfs import interp_epsf
>>> P = interp_epsf(ePSFs, x, y, pixel_space = True)

Do the same as the previous example, but now in detector space and with subpixel offsets.
>>> from focus_diverse_epsfs import interp_epsf
>>> P = interp_epsf(ePSFs, x, y, pixel_space = True, subpixel_x = 0.77, subpixel_y = 0.33)


More details for these examples are provided in a Jupyter notebook, which can be found at LINK.


"""

import logging

import dask
import numpy as np
import requests

from dask.diagnostics import ProgressBar
from requests.auth import HTTPBasicAuth
from scipy import ndimage
from urllib.request import urlopen, urlretrieve

__taskname__ = "focus_diverse_epsfs"
__author__   = "Gagandeep Anand, Yotam Cohen"
__version__  = "1.0"
__vdate__    = "02-Oct-2023"

# Initialize the logger
logging.basicConfig()
LOG = logging.getLogger(f'{__taskname__}.Query')
LOG.setLevel(logging.INFO)


def psf_retriever(ipsoot, download_location):
    """
    Function to query API on AWS API Gateway for the ePSF FITS file that
    corresponds to a given image rootname.

    Parameters
    -------
    ipsoot : string
        String of the image rootname/IPPPSSOOT.

    download_location : string
        String with the user's preferred download path/location.

    Returns
    -------

    Downloads the ePSF FITS file to download_location. The explicit return is
    the name of the downloaded FITS file.

    """

    # provide api URL and public key and ID
    api_url = (
        'https://8cclxcxse4.execute-api.us-east-1.amazonaws.com/main/psf-server-ops/')
    api_id = 'iwx1prnqog'
    api_key = 'T1fU5vycfM9KDlSLJoGBU6t0pzS0vjHKaSqmT6gU'

    # stitch together credentials
    auth = HTTPBasicAuth(api_id, api_key)

    # send up post request with ipsoot event
    myobj = {'ipsoot': ipsoot}
    result = requests.post(api_url, json=myobj, auth=auth)

    # grab url from result
    url = result.text[1:-1]
    remotefile = urlopen(url)

    # determine readable name for file
    content_disposition = remotefile.info()['Content-Disposition']
    desired_filename = (content_disposition.split(';')[1].split('=')[1].split('-')[0] +
                        '-' +
                        content_disposition.split(';')[1].split('=')[1].split('-')[1].split('/')[-1])

    # download file
    download_location = download_location + '/' if not download_location.endswith('/') else download_location
    urlretrieve(url, download_location + desired_filename)

    return desired_filename


def multi_psf_retriever(input_list, download_location, n_PROC=8, fromTextFile=False):
    """
    Function to batch query the API on AWS API Gateway for multiple ePSFs
    simultaneously.
   
    Parameters
    -------
    input_list : list (if fromTextFile=False, default) or text file name (if fromTextFile=True) 
        A list or text file with the image rootnames. See fromTextfile description for more 
        information.

    download_location : string
        String with the user's preferred download path/location.

    n_PROC : int
        Integer

    fromTextFile : Boolean
        Boolean. Should be set to True if input_list is a text file (with one ipsoot on each line).
        Alternatively, should be set to False if the input_list is a Python list. 

    Returns
    -------

    Downloads the ePSF FITS files to download_location. The explicit return is
    a list of the downloaded FITS files.

    """

    # select if either choosing to provide inputs via a text file or list
    if fromTextFile is True:
        with open(input_list) as file:
            lines = [line.rstrip() for line in file]

    if fromTextFile is False:
        lines = input_list

    # perform multiprocessing with dask
    download_location = download_location + '/' if not download_location.endswith('/') else download_location
    download_locations = [download_location] * len(lines)
    zipped_inputs = list(zip(lines, download_locations))

    dask_results = [dask.delayed(psf_retriever)(
        zipped_inputs[0], zipped_inputs[1]) for zipped_inputs in zipped_inputs]

    # run with progress bar
    with ProgressBar():
        results = dask.compute(
            *dask_results, num_workers=n_PROC, scheduler='processes')

    return results


def interp_epsf(ePSFs, x, y, pixel_space=False, subpixel_x=0, subpixel_y=0):
    """
    Function to perform further spatial interpolations given the input ePSF array from
    the previous retriever functions. The routine uses bi-linear interpolation for the
    integer pixel shifts, and bi-cubic interpolation for any specified subpixel phase shifts.

    Parameters
    -------
    ePSFs : numpy.ndarray
        Array with the ePSFs read in (e.g via astropy.fits.getdata).

    x : integer
        X-coordinate of the desired output ePSF. 

    y : integer
        Y-coordinate of the desired output ePSF. Please note that the range here is between 
        0 and 4096, i.e. WFC1 runs from 2048-4096.

    pixel_space : Boolean
        Boolean describing whether the user wants to further interpolate to a subpixel position.

    subpixel_x : Float (between 0.00 and 0.99)
        Float giving the desired subpixel-x coordinate.

    subpixel_y : Float (between 0.00 and 0.99)
        Float giving the desired subpixel-y coordinate.


    Returns
    -------
    An ePSF array with the specified interpolation parameters.

    """

    # input error checking
    if x not in range(0, 4096):
        msg = ("The x coordinate should be an integer between 0 and 4096. \n \
            Please double-check your inputs.")
        LOG.error(msg)
        return

    if y not in range(0, 4096):
        msg = ("The y coordinate should be an integer between 0 and 4096. \n \
            Please double-check your inputs.")
        LOG.error(msg)
        return

    if subpixel_x < 0.0 or subpixel_x > 0.99:
        msg = ("Subpixel shifts should be between 0 and 0.99. \n \
            Please double-check your inputs.")
        LOG.error(msg)
        return

    if subpixel_y < 0.0 or subpixel_y > 0.99:
        msg = ("Subpixel shifts should be between 0 and 0.99. \n \
            Please double-check your inputs.")
        LOG.error(msg)
        return

    if pixel_space not in [True, False]:
        msg = ("pixel_space should be True or False.\n \
            Please double-check your inputs.")
        LOG.error(msg)
        return

    if pixel_space is False and (subpixel_x != 0 or subpixel_y != 0):
        msg = ('Please set pixel_space = True to use the subpixel phase offsets.')
        LOG.error(msg)
        return

    # round subpixel_x and subpixel_y to second decimal
    subpixel_x = round(subpixel_x, 2)
    subpixel_y = round(subpixel_y, 2)

    # if user inputs coordinates for WFC1 (y > 2048), then adjust down and set chip = "WFC1",
    # otherwise assume chip = "WFC2"
    chip = "WFC2"

    if y > 2048:
        y = y-2048
        chip = "WFC1"

    # define tuple with user given coordinates
    userCoords = (x, y)

    # give positions of ePSFs based on Andrea's coordinate system
    acs_xCoords = np.array([0, 512, 1024, 1536, 2168, 2800, 3192, 3584, 4096])
    acs_yCoords = np.array([0, 512, 1024, 1536, 2048])

    # now, need to find closest four PSFs to any user-defined position

    # first, find coordinates of ePSFs surrounding userCoords
    xLow = acs_xCoords[acs_xCoords <= userCoords[0]].max()
    xHigh = acs_xCoords[acs_xCoords >= userCoords[0]].min()

    yLow = acs_yCoords[acs_yCoords <= userCoords[1]].max()
    yHigh = acs_yCoords[acs_yCoords >= userCoords[1]].min()

    # as well as 2d index of those ePSFs
    xLow_num = list(acs_xCoords).index(xLow)
    xHigh_num = list(acs_xCoords).index(xHigh)

    yLow_num = list(acs_yCoords).index(yLow)
    yHigh_num = list(acs_yCoords).index(yHigh)

    bot_left_psf_index = (xLow_num, yLow_num)
    bot_right_psf_index = (xHigh_num, yLow_num)

    top_left_psf_index = (xLow_num, yHigh_num)
    top_right_psf_index = (xHigh_num, yHigh_num)

    # convert above indices back to single dimensional retrievers
    # for (x = 1000, y = 2000, chip = "WFC2"),
    # should be 28, 29, 37, 38 (assuming index starts at 0)
    bot_left_single_index = bot_left_psf_index[0] + (bot_left_psf_index[1]*9)
    bot_right_single_index = bot_right_psf_index[0] + \
        (bot_right_psf_index[1]*9)
    top_left_single_index = top_left_psf_index[0] + (top_left_psf_index[1]*9)
    top_right_single_index = top_right_psf_index[0] + \
        (top_right_psf_index[1]*9)

    # do additional conversion if WFC1 (move up 45 indices each)
    if chip == "WFC1":
        bot_left_single_index = bot_left_single_index + 45
        bot_right_single_index = bot_right_single_index + 45
        top_left_single_index = top_left_single_index + 45
        top_right_single_index = top_right_single_index + 45

    # and now do bilinear interpolation
    # using notation from https://www.omnicalculator.com/math/bilinear-interpolation

    # case 1, where user coordinate is equivalent to a known ePSF
    if bot_left_single_index == bot_right_single_index == \
            top_left_single_index == top_right_single_index:
        c_Q11 = 0.25
        c_Q12 = 0.25
        c_Q21 = 0.25
        c_Q22 = 0.25

    # case 2, where point falls on a y-grid line, so interpolate along x-axis only
    elif bot_left_single_index == top_left_single_index:
        c_Q11 = ((xHigh-userCoords[0])/(xHigh-xLow))
        c_Q12 = 0
        c_Q21 = (1-c_Q11)
        c_Q22 = 0

    # case 3, where point falls on an x-grid line, so interpolate along y-axis only
    elif bot_left_single_index == bot_right_single_index:
        c_Q11 = ((yHigh-userCoords[1])/(yHigh-yLow))
        c_Q12 = (1-c_Q11)
        c_Q21 = 0
        c_Q22 = 0

    # else do regular bilinear interpolation
    else:
        c_Q11 = ((xHigh-userCoords[0])*(yHigh -
                                        userCoords[1]))/((xHigh-xLow)*(yHigh-yLow))
        c_Q21 = ((userCoords[0]-xLow)*(yHigh-userCoords[1])
                 )/((xHigh-xLow)*(yHigh-yLow))
        c_Q12 = ((xHigh-userCoords[0])*(userCoords[1]-yLow)
                 )/((xHigh-xLow)*(yHigh-yLow))
        c_Q22 = ((userCoords[0]-xLow)*(userCoords[1]-yLow)
                 )/((xHigh-xLow)*(yHigh-yLow))

    Q11 = ePSFs[bot_left_single_index]
    Q21 = ePSFs[bot_right_single_index]
    Q12 = ePSFs[top_left_single_index]
    Q22 = ePSFs[top_right_single_index]

    P = Q11*c_Q11 + Q21*c_Q21 + Q12*c_Q12 + Q22*c_Q22

    # if the user wants the ePSF in pixel space, return that with any subpixel offsets, instead of P.
    if pixel_space is True:

        # shift the ePSF by the specified subpixel amount (assumes no subpixel shift by default)
        P_sub = ndimage.shift(P, (subpixel_y*4, subpixel_x*4), order=3)

        # create blank array for downsampled ePSF
        P_sub_down = np.zeros((25, 25))

        # then downsample
        for i in range(25):
            for j in range(25):
                P_sub_down[i, j] = P_sub[4*i+2, 4*j+2]

        # re-center
        x_int_shift = 0
        y_int_shift = 0

        if subpixel_x >= 0.5:
            x_int_shift = -1

        if subpixel_y >= 0.5:
            y_int_shift = -1

        P_sub_down = ndimage.shift(P_sub_down, (y_int_shift, x_int_shift))

        # remove 1 pixel border
        P_sub_down = P_sub_down[1:-1, 1:-1]

        # and return downsampled version, with any shifts (if specified)
        return P_sub_down

    # otherwise by default return the interpolated 4x supersampled ePSF
    return P
