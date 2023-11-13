"""This module contains two functions that provide
means to programmatically query the `ACS/WFC
Focus-Diverse ePSF Generator <https://acspsf.stsci.edu>`_ API.

1. :func:`psf_retriever` provides the user with the ability
   to perform downloads for a single image rootname, whereas
2. :func:`multi_psf_retriever` provides the ability to do so
   for multiple rootnames.

In all cases, the focus-diverse ePSFs will be
downloaded to a location specified by the user.

Additionally, this module also contains a function, :func:`interp_epsf`, that allows
users to interpolate the provided ePSF arrays to any arbitrary pixel
coordinates, downsample the ePSF into detector space, or apply subpixel phase shifts.

We strongly recommend you read
`ACS ISR 2018-08 <https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/acs/documentation/instrument-science-reports-isrs/_documents/isr1808.pdf>`_
by A. Bellini et al. and
`ACS ISR 2023-06 <https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/acs/documentation/instrument-science-reports-isrs/_documents/isr2306.pdf>`_
by G. Anand et al. for more details
on these ePSF models and tools before using them.

Examples
--------

Define a folder location for downloaded data:

>>> download_location  = '/Users/username/download_folder/'

Retrieve a single 101x101x90 ePSF FITS file for the given rootname:

>>> from acstools.focus_diverse_epsfs import psf_retriever
>>> retrieved_download = psf_retriever('jds408jsq', download_location)

Retrieve ePSFs based on FLC images specified in an external text file
(with one rootname per line):

>>> from acstools.focus_diverse_epsfs import multi_psf_retriever
>>> retrieved_downloads = multi_psf_retriever('input_ipsoots.txt', download_location)

Retrieve ePSFs using rootnames from Proposal ID 13376 obtained from
`astroquery <https://astroquery.readthedocs.io/>`_:

>>> from astroquery.mast import Observations
>>> obsTable = Observations.query_criteria(
...     obs_collection = 'HST', proposal_id="13376", instrument_name = "ACS/WFC",
...     provenance_name = "CALACS")
>>> dataProducts = Observations.get_product_list(obsTable)
>>> dataProducts = dataProducts[
...     (dataProducts['productSubGroupDescription'] == 'FLC') &
...     (dataProducts['type'] == 'S')]
>>> obs_rootnames = list(dataProducts['obs_id'])
>>> retrieved_downloads = multi_psf_retriever(obs_rootnames, download_location)

Read the retrieved ePSFs from file into NumPy array:

>>> from astropy.io import fits
>>> ePSFs = fits.getdata(retrieved_download)

Interpolate a given ePSF to the given pixel location (0-indexed):

>>> from acstools.focus_diverse_epsfs import interp_epsf
>>> x = 2000  # near the middle of the detector along the X-axis
>>> y = 2000  # near the top of the WFC1 chip
>>> chip = "WFC1" # must specify we are interested in WFC1
>>> interpolated_epsf = interp_epsf(ePSFs, x, y, chip)

Similar to above but obtain the ePSF in detector space (instead of 4x supersampling):

>>> interpolated_epsf = interp_epsf(ePSFs, x, y, chip, pixel_space=True)

Similar to above but obtain the ePSF in detector space and with subpixel offsets:

>>> interpolated_epsf = interp_epsf(
...     ePSFs, x, y, chip, pixel_space=True, subpixel_x=0.77, subpixel_y=0.33)

"""  # noqa: E501
import logging
import os
import re
from pathlib import Path
from urllib.request import urlopen, urlretrieve

import numpy as np
import requests
from requests.auth import HTTPBasicAuth

__taskname__ = "focus_diverse_epsfs"
__author__ = "Gagandeep Anand, Yotam Cohen"
__version__ = "1.0"
__vdate__ = "02-Oct-2023"
__all__ = ["psf_retriever", "multi_psf_retriever", "interp_epsf"]

# Ex: "attachment; filename=jds408jsq-F606W/SM4/STDPBF_ACSWFC_F606W_SM4_F11.2.fits"
_re_patt = re.compile(r"attachment; filename=(\w{9})-\w*\/\w*\/(.*)")

# Initialize the logger
logging.basicConfig()
LOG = logging.getLogger(f'{__taskname__}.Query')
LOG.setLevel(logging.INFO)


def _validate_acs_ipsoot(line):
    return isinstance(line, str) and len(line) == 9 and line.startswith("j")


def psf_retriever(ipsoot, download_location, timeout=60):
    """Function to query API on AWS API Gateway for the ePSF FITS file that
    corresponds to a given image rootname.

    .. warning::

        There is no caching. Please check that you do not have the file
        already downloaded before running this function again.
        Any existing local file with the same name might be overwritten.

    Parameters
    ----------
    ipsoot : str
        Image rootname in the form of IPPPSSOOT.

    download_location : str
        Directory name where the file will be downloaded to.
        It must exist and you must have write permission to it.

    timeout : float
        Seconds before query timeout.

    Returns
    -------
    desired_filename : str or `None`
        The downloaded ePSF FITS file, if successful.

    """
    if not _validate_acs_ipsoot(ipsoot):
        LOG.error("Invalid ACS IPSOOT: %s" % ipsoot)
        return

    if not os.path.isdir(download_location):
        LOG.error("Invalid download location")
        return

    # provide api URL and public key and ID
    api_url = (
        'https://8cclxcxse4.execute-api.us-east-1.amazonaws.com/main/psf-server-ops/')
    api_id = 'iwx1prnqog'
    api_key = 'T1fU5vycfM9KDlSLJoGBU6t0pzS0vjHKaSqmT6gU'

    # stitch together credentials
    auth = HTTPBasicAuth(api_id, api_key)

    # send up post request with ipsoot event
    myobj = {'ipsoot': ipsoot}
    result = requests.post(api_url, json=myobj, auth=auth, timeout=timeout)

    if not result.ok:
        LOG.error("Query failed: %d %s" % (result.status_code, result.reason))
        return

    # grab url from result
    url = result.text[1:-1]
    if not url.startswith("http"):
        LOG.error("URL is not HTTP.")
        return
    with urlopen(url) as remotefile:  # nosec (already checked above)
        # determine readable name for file
        content_disposition = remotefile.info()['Content-Disposition']

    m = _re_patt.match(content_disposition)
    if not m:
        LOG.error("Query failed: No filename in Content-Disposition")
        return

    try:
        desired_filename = os.path.join(download_location, f"{m[1]}-{m[2]}")
    except Exception:
        LOG.error("Query failed: Error extracting filename from Content-Disposition")
        return

    # download file
    urlretrieve(url, filename=desired_filename)  # nosec (already checked above)

    return desired_filename


def multi_psf_retriever(input_list, download_location, num_workers=8):
    """Function to batch query the API on AWS API Gateway for multiple ePSFs
    simultaneously.

    .. note::

        This function requires an optional ``dask`` dependency to be installed.

    Parameters
    ----------
    input_list : list or str
        If a list is given, it must contain a list of rootnames.
        If a string is given, it must be a text file with one rootname per line.

    download_location : str
        Directory name where the file will be downloaded to.
        It must exist and you must have write permission to it.

    num_workers : int
        Max number of workers for ``dask.compute()``.

    Returns
    -------
    results : list of str
        List of downloaded ePSF FITS files.

    """
    import dask
    from dask.diagnostics import ProgressBar

    if not os.path.isdir(download_location):
        LOG.error("Invalid download location")
        return []

    if isinstance(input_list, str):  # Text file with one rootname per line
        orig_lines = Path(input_list).read_text().rsplit()
    elif not isinstance(input_list, (list, tuple)):
        LOG.error("Invalid input list")
        return []
    else:  # List of rootnames
        orig_lines = input_list

    # Should throw out things that are obviously wrong here to minimize network hit.
    lines = [line for line in orig_lines if _validate_acs_ipsoot(line)]
    n_lines = len(lines)

    if n_lines < 1:  # Nothing to do
        return []

    if n_lines == 1:  # No need for multiprocessing
        return [psf_retriever(lines[0], download_location)]

    if n_lines < num_workers:  # No need that many workers
        num_workers = n_lines

    # perform multiprocessing with dask
    dask_results = [dask.delayed(psf_retriever)(line, download_location) for line in lines]

    # run with progress bar
    with ProgressBar():
        results = dask.compute(
            *dask_results, num_workers=num_workers, scheduler='processes')

    return results


def interp_epsf(ePSFs, x, y, chip, pixel_space=False, subpixel_x=0, subpixel_y=0):
    """Function to perform further spatial interpolations given the input ePSF array.
    It uses bi-linear interpolation for the integer pixel shifts, and
    bi-cubic interpolation for any specified subpixel phase shifts.

    This function allows users to interpolate the provided ePSF arrays
    to any arbitrary ``(x, y)`` pixel coordinates. It can be called with
    ``pixel_space=True`` to downsample the ePSF into detector space.

    Subpixel phase shifts can be applied by setting ``subpixel_x`` and ``subpixel_y``
    between 0 and 0.99. Note that a 1 pixel border is removed from the subpixel
    phase shifted ePSF, such that the final dimensions are 23x23. Results from this
    subpixel phase shift routine may differ from other algorithmic
    implementations, typically at the level of <0.5% in the core of the ePSF.

    .. note::

        This function requires an optional ``scipy`` dependency to be installed
        for ``pixel_space=True``.

    .. note::

        This function requires users to specify the WFC chip (WFC1 or WFC2).

    Parameters
    ----------
    ePSFs : numpy.ndarray
        Array with the ePSFs.

    x : int
        X-coordinate (1-indexed) of the desired output ePSF. Please note that the range here is between
        1 and 4096, inclusive. The ePSF grid begins off the detector, at (0,0).

    y : int
        Y-coordinate (1-indexed) of the desired output ePSF. Please note that the range here is between
        1 and 2048, inclusive. The ePSF grid begins off the detector, at (0,0).

    chip : str
        String corresponding to which ACS/WFC detector the user is specifying the coordinates on,
        either "WFC1" or "WFC2".

    pixel_space : bool
        If `True`, downsample the ePSF into detector space.

    subpixel_x, subpixel_y : float
        The desired subpixel coordinate, between 0 and 0.99.
        ``pixel_space`` must be set to `True` to use this option.
        The defaults for both x and y are 0, and are relative to the central pixel of the ePSF.
        By default, ePSFs are centered on the central pixel.

    Returns
    -------
    P : numpy.ndarray or `None`
        An ePSF array with the specified interpolation parameters, if successful.

    See Also
    --------
    psf_retriever, multi_psf_retriever

    """
    valid_wfc_x_pixels = range(1, 4097)
    valid_wfc_y_pixels = range(1, 2049)

    if x not in valid_wfc_x_pixels:
        LOG.error("The X coordinate should be an integer between 1 and 4096.")
        return

    if y not in valid_wfc_y_pixels:
        LOG.error("The Y coordinate should be an integer between 1 and 2048.")
        return

    if subpixel_x < 0 or subpixel_x > 0.99 or subpixel_y < 0.0 or subpixel_y > 0.99:
        LOG.error("Subpixel shifts should be between 0 and 0.99")
        return

    if not isinstance(pixel_space, bool):
        LOG.error("pixel_space should be True or False.")
        return

    if not pixel_space and (subpixel_x != 0 or subpixel_y != 0):
        LOG.error("Please set pixel_space=True to use the subpixel phase offsets.")
        return

    # round subpixel_x and subpixel_y to second decimal
    subpixel_x = round(subpixel_x, 2)
    subpixel_y = round(subpixel_y, 2)

    # give positions of ePSFs based on Andrea and Jay's coordinate system
    # 0,0 is off the detector, but that is where the ePSF definitions begin
    acs_xCoords = np.array([0, 512, 1024, 1536, 2168, 2800, 3192, 3584, 4096])
    acs_yCoords = np.array([0, 512, 1024, 1536, 2048])

    # now, need to find closest four PSFs to any user-defined position

    # first, find coordinates of ePSFs surrounding userCoords
    xLow = acs_xCoords[acs_xCoords <= x].max()
    xHigh = acs_xCoords[acs_xCoords >= x].min()

    yLow = acs_yCoords[acs_yCoords <= y].max()
    yHigh = acs_yCoords[acs_yCoords >= y].min()

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
    bot_left_single_index = bot_left_psf_index[0] + (bot_left_psf_index[1] * 9)
    bot_right_single_index = bot_right_psf_index[0] + (bot_right_psf_index[1] * 9)
    top_left_single_index = top_left_psf_index[0] + (top_left_psf_index[1] * 9)
    top_right_single_index = top_right_psf_index[0] + (top_right_psf_index[1] * 9)

    # do additional conversion if WFC1 (move up 45 indices each)
    if chip == "WFC1":
        bot_left_single_index = bot_left_single_index + 45
        bot_right_single_index = bot_right_single_index + 45
        top_left_single_index = top_left_single_index + 45
        top_right_single_index = top_right_single_index + 45

    # and now do bilinear interpolation
    # using notation from https://www.omnicalculator.com/math/bilinear-interpolation

    # case 1, where user coordinate is equivalent to a known ePSF
    if (bot_left_single_index == bot_right_single_index ==
            top_left_single_index == top_right_single_index):
        c_Q11 = 0.25
        c_Q12 = 0.25
        c_Q21 = 0.25
        c_Q22 = 0.25

    # case 2, where point falls on a y-grid line, so interpolate along x-axis only
    elif bot_left_single_index == top_left_single_index:
        c_Q11 = ((xHigh - x) / (xHigh - xLow))
        c_Q12 = 0
        c_Q21 = (1 - c_Q11)
        c_Q22 = 0

    # case 3, where point falls on an x-grid line, so interpolate along y-axis only
    elif bot_left_single_index == bot_right_single_index:
        c_Q11 = ((yHigh - y) / (yHigh - yLow))
        c_Q12 = (1 - c_Q11)
        c_Q21 = 0
        c_Q22 = 0

    # else do regular bilinear interpolation
    else:
        dxh = (xHigh - x)
        dyh = (yHigh - y)
        dxl = (x - xLow)
        dyl = (y - yLow)
        denom = ((xHigh - xLow) * (yHigh - yLow))
        c_Q11 = (dxh * dyh) / denom
        c_Q21 = (dxl * dyh) / denom
        c_Q12 = (dxh * dyl) / denom
        c_Q22 = (dxl * dyl) / denom

    Q11 = ePSFs[bot_left_single_index]
    Q21 = ePSFs[bot_right_single_index]
    Q12 = ePSFs[top_left_single_index]
    Q22 = ePSFs[top_right_single_index]

    P = Q11 * c_Q11 + Q21 * c_Q21 + Q12 * c_Q12 + Q22 * c_Q22

    # if the user wants the ePSF in pixel space, return that with any subpixel offsets, instead of P.
    if pixel_space:
        from scipy import ndimage

        # shift the ePSF by the specified subpixel amount (assumes no subpixel shift by default)
        P_sub = ndimage.shift(P, (subpixel_y * 4, subpixel_x * 4), order=3)

        # create blank array for downsampled ePSF
        P_sub_down = np.zeros((25, 25))

        # then downsample
        for i in range(25):
            for j in range(25):
                P_sub_down[i, j] = P_sub[4 * i + 2, 4 * j + 2]

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
