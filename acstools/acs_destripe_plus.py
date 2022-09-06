#!/usr/bin/env python
"""
Fully calibrate post-SM4 ACS/WFC exposures using the standalone
:ref:`acsdestripe` tool to remove stripes between ACSCCD and ACSCTE
steps in CALACS.

This script runs CALACS (8.3.1 or higher only) and ``acs_destripe``
on ACS/WFC images. Input files must be RAW full-frame or supported subarray
ACS/WFC exposures taken after SM4. Resultant outputs are science-ready
FLT and FLC (if applicable) files.

This script is useful for when the built-in CALACS destriping algorithm
using overscans is insufficient or unavailable.

For more information, see
`ACS ISR 2011-05 <https://ui.adsabs.harvard.edu/abs/2011acs..rept....5G/abstract>`_.

Examples
--------

In Python:

>>> from acstools.acs_destripe_plus import destripe_plus
>>> destripe_plus(
...     'j12345678_raw.fits', suffix='strp', maxiter=15, sigrej=2.0,
...     scimask1='mymask_sci1.fits', scimask2='mymask_sci2.fits',
...     clobber=False, cte_correct=True)

From command line::

    % acs_destripe_plus [-h] [--suffix SUFFIX] [--stat STAT]
                        [--maxiter MAXITER] [--sigrej SIGREJ]
                        [--lower [LOWER]] [--upper [UPPER]]
                        [--binwidth BINWIDTH] [--sci1_mask SCI1_MASK]
                        [--sci2_mask SCI2_MASK] [--dqbits [DQBITS]]
                        [--rpt_clean RPT_CLEAN] [--atol [ATOL]] [--nocte]
                        [--keep_intermediate_files] [--clobber] [-q] [--version]
                        input

"""
#
# HISTORY:
# 16APR2014 Leonardo version 1.0
#          Based on instructions from Pey-Lian Lim
# 11SEP2014 Ogaz added capabilities for full frame processing
#           and stripe masking
# 29OCT2014 Ogaz clean up for posting final script for users
# 18NOV2014 Ogaz changed options/switches
# 12DEC2014 Lim incorporated script into ACSTOOLS
# 11MAR2015 Lim added parameters to be passed into acs_destripe
# 12MAR2015 (v0.3.0) Cara added capability to use DQ mask;
#           added support for multiple input files and wildcards in the file
#           names. Cara added weighted (by NPix) background computations
#           (especially important for vigneted filters).
#           See Tickets #1178 & #1180.
# 31MAR2015 (v0.4.0) Cara added repeated de-stripe iterations (to improve
#           corrections in the "RAW" space) and support for various
#           statistics modes. See Ticket #1183.
# 12JAN2016 (v0.4.1) Lim added new subarray modes that are allowed CTE corr.
# 01APR2020 (v0.4.3) Lim added capability to produce CRJ/CRC outputs.

# STDLIB
import logging
import os
import subprocess  # nosec

# ASTROPY
from astropy.time import Time

# THIRD-PARTY
import numpy as np

try:
    # This supports PIXVALUE
    from stsci.tools import stpyfits as fits
except ImportError:
    # Falls back to Astropy
    from astropy.io import fits

# LOCAL
from . import acs_destripe
from . import acs2d
from . import acsccd
from . import acscte
from . import acsrej
from .utils_calib import SM4_MJD

__taskname__ = 'acs_destripe_plus'
__version__ = '0.4.3'
__vdate__ = '01-Apr-2020'
__author__ = 'Leonardo Ubeda, Sara Ogaz (ACS Team), STScI'
__all__ = ['destripe_plus', 'crrej_plus']

SUBARRAY_LIST = [
    'WFC1-2K', 'WFC1-POL0UV', 'WFC1-POL0V', 'WFC1-POL60V',
    'WFC1-POL60UV', 'WFC1-POL120V', 'WFC1-POL120UV', 'WFC1-SMFL',
    'WFC1-IRAMPQ', 'WFC1-MRAMPQ', 'WFC2-2K', 'WFC2-ORAMPQ',
    'WFC2-SMFL', 'WFC2-POL0UV', 'WFC2-POL0V', 'WFC2-MRAMPQ',
    'WFC1A-512', 'WFC1A-1K', 'WFC1A-2K', 'WFC1B-512', 'WFC1B-1K', 'WFC1B-2K',
    'WFC2C-512', 'WFC2C-1K', 'WFC2C-2K', 'WFC2D-512', 'WFC2D-1K', 'WFC2D-2K']

logging.basicConfig()
LOG = logging.getLogger(__taskname__)
LOG.setLevel(logging.INFO)


def destripe_plus(inputfile, suffix='strp', stat='pmode1', maxiter=15,
                  sigrej=2.0, lower=None, upper=None, binwidth=0.3,
                  scimask1=None, scimask2=None,
                  dqbits=None, rpt_clean=0, atol=0.01,
                  cte_correct=True, keep_intermediate_files=False,
                  clobber=False, verbose=True):
    r"""Calibrate post-SM4 ACS/WFC exposure(s) and use
    standalone :ref:`acsdestripe`.

    This takes a RAW image and generates a FLT file containing
    its calibrated and destriped counterpart.
    If CTE correction is performed, FLC will also be present.

    Parameters
    ----------
    inputfile : str or list of str
        Input filenames in one of these formats:

            * a Python list of filenames
            * a partial filename with wildcards ('\*raw.fits')
            * filename of an ASN table ('j12345670_asn.fits')
            * an at-file (``@input``)

    suffix : str
        The string to use to add to each input file name to
        indicate an output product of ``acs_destripe``.
        This only affects the intermediate output file that will
        be automatically renamed to ``*blv_tmp.fits`` during the processing.

    stat : { 'pmode1', 'pmode2', 'mean', 'mode', 'median', 'midpt' } (Default = 'pmode1')
        Specifies the statistics to be used for computation of the
        background in image rows:

        * 'pmode1' - SEXTRACTOR-like mode estimate based on a
          modified `Pearson's rule <https://en.wikipedia.org/wiki/Nonparametric_skew#Pearson.27s_rule>`_:
          ``2.5*median-1.5*mean``;
        * 'pmode2' - mode estimate based on
          `Pearson's rule <https://en.wikipedia.org/wiki/Nonparametric_skew#Pearson.27s_rule>`_:
          ``3*median-2*mean``;
        * 'mean' - the mean of the distribution of the "good" pixels (after
          clipping, masking, etc.);
        * 'mode' - the mode of the distribution of the "good" pixels;
        * 'median' - the median of the distribution of the "good" pixels;
        * 'midpt' - estimate of the median of the distribution of the "good"
          pixels based on an algorithm similar to IRAF's `imagestats` task
          (``CDF(midpt)=1/2``).

        .. note::
            The midpoint and mode are computed in two passes through the
            image. In the first pass the standard deviation of the pixels
            is calculated and used with the *binwidth* parameter to compute
            the resolution of the data histogram. The midpoint is estimated
            by integrating the histogram and computing by interpolation
            the data value at which exactly half the pixels are below that
            data value and half are above it. The mode is computed by
            locating the maximum of the data histogram and fitting the peak
            by parabolic interpolation.

    maxiter : int
        This parameter controls the maximum number of iterations
        to perform when computing the statistics used to compute the
        row-by-row corrections.

    sigrej : float
        This parameters sets the sigma level for the rejection applied
        during each iteration of statistics computations for the
        row-by-row corrections.

    lower : float, None (Default = None)
        Lower limit of usable pixel values for computing the background.
        This value should be specified in the units of the input image(s).

    upper : float, None (Default = None)
        Upper limit of usable pixel values for computing the background.
        This value should be specified in the units of the input image(s).

    binwidth : float (Default = 0.1)
        Histogram's bin width, in sigma units, used to sample the
        distribution of pixel brightness values in order to compute the
        background statistics. This parameter is aplicable *only* to *stat*
        parameter values of `'mode'` or `'midpt'`.

    scimask1 : str or list of str
        Mask images for *calibrated* ``SCI,1``, one for each input file.
        Pixels with zero values will be masked out, in addition to clipping.

    scimask2 : str or list of str
        Mask images for *calibrated* ``SCI,2``, one for each input file.
        Pixels with zero values will be masked out, in addition to clipping.
        This is not used for subarrays.

    dqbits : int, str, None (Default = None)
        Integer sum of all the DQ bit values from the input image's DQ array
        that should be considered "good" when building masks for de-striping
        computations. For example, if pixels in the DQ array can be
        combinations of 1, 2, 4, and 8 flags and one wants to consider
        DQ "defects" having flags 2 and 4 as being acceptable for de-striping
        computations, then `dqbits` should be set to 2+4=6. Then a DQ pixel
        having values 2,4, or 6 will be considered a good pixel, while a DQ
        pixel with a value, e.g., 1+2=3, 4+8=12, etc. will be flagged
        as a "bad" pixel.

        Alternatively, one can enter a comma- or '+'-separated list of
        integer bit flags that should be added to obtain the final
        "good" bits. For example, both ``4,8`` and ``4+8`` are equivalent to
        setting `dqbits` to 12.

        | Set `dqbits` to 0 to make *all* non-zero pixels in the DQ
          mask to be considered "bad" pixels, and the corresponding image
          pixels not to be used for de-striping computations.

        | Default value (`None`) will turn off the use of image's DQ array
          for de-striping computations.

        | In order to reverse the meaning of the `dqbits`
          parameter from indicating values of the "good" DQ flags
          to indicating the "bad" DQ flags, prepend '~' to the string
          value. For example, in order not to use pixels with
          DQ flags 4 and 8 for sky computations and to consider
          as "good" all other pixels (regardless of their DQ flag),
          set `dqbits` to ``~4+8``, or ``~4,8``. To obtain the
          same effect with an `int` input value (except for 0),
          enter -(4+8+1)=-9. Following this convention,
          a `dqbits` string value of ``'~0'`` would be equivalent to
          setting ``dqbits=None``.

        .. note::
            DQ masks (if used), *will be* combined with user masks specified
            in the `scimask1` and `scimask2` parameters (if any).

    rpt_clean : int
        An integer indicating how many *additional* times stripe cleaning
        should be performed on the input image. Default = 0.

    atol : float, None
        The threshold for maximum absolute value of bias stripe correction
        below which repeated cleanings can stop. When `atol` is `None`
        cleaning will be repeated `rpt_clean` number of times.
        Default = 0.01 [e].

    cte_correct : bool
        Perform CTE correction.

    keep_intermediate_files : bool
        Keep de-striped BLV_TMP and BLC_TMP files around for CRREJ,
        if needed. Set to `True` if you want to run :func:`crrej_plus`.

    clobber : bool
        Specify whether or not to 'clobber' (delete then replace)
        previously generated products with the same names.

    verbose : bool
        Print informational messages. Default = True.

    Raises
    ------
    ImportError
        ``stsci.tools`` not found.

    IOError
        Input file does not exist.

    ValueError
        Invalid header values or CALACS version.

    """
    from astropy.nddata.bitmask import interpret_bit_flags

    # Optional package dependencies
    from stsci.tools import parseinput

    # process input file(s) and if we have multiple input files - recursively
    # call acs_destripe_plus for each input image:
    flist = parseinput.parseinput(inputfile)[0]

    if isinstance(scimask1, str):
        mlist1 = parseinput.parseinput(scimask1)[0]
    elif isinstance(scimask1, np.ndarray):
        mlist1 = [scimask1.copy()]
    elif scimask1 is None:
        mlist1 = []
    elif isinstance(scimask1, list):
        mlist1 = []
        for m in scimask1:
            if isinstance(m, np.ndarray):
                mlist1.append(m.copy())
            elif isinstance(m, str):
                mlist1 += parseinput.parseinput(m)[0]
            else:
                raise TypeError("'scimask1' must be a list of str or "
                                "numpy.ndarray values.")
    else:
        raise TypeError("'scimask1' must be either a str, or a "
                        "numpy.ndarray, or a list of the two type of "
                        "values.")

    if isinstance(scimask2, str):
        mlist2 = parseinput.parseinput(scimask2)[0]
    elif isinstance(scimask2, np.ndarray):
        mlist2 = [scimask2.copy()]
    elif scimask2 is None:
        mlist2 = []
    elif isinstance(scimask2, list):
        mlist2 = []
        for m in scimask2:
            if isinstance(m, np.ndarray):
                mlist2.append(m.copy())
            elif isinstance(m, str):
                mlist2 += parseinput.parseinput(m)[0]
            else:
                raise TypeError("'scimask2' must be a list of str or "
                                "numpy.ndarray values.")
    else:
        raise TypeError("'scimask2' must be either a str, or a "
                        "numpy.ndarray, or a list of the two type of "
                        "values.")

    n_input = len(flist)
    n_mask1 = len(mlist1)
    n_mask2 = len(mlist2)

    if n_input == 0:
        raise ValueError(
            'No input file(s) provided or the file(s) do not exist')

    if n_mask1 == 0:
        mlist1 = [None] * n_input
    elif n_mask1 != n_input:
        raise ValueError('Insufficient masks for [SCI,1]')

    if n_mask2 == 0:
        mlist2 = [None] * n_input
    elif n_mask2 != n_input:
        raise ValueError('Insufficient masks for [SCI,2]')

    if n_input > 1:
        for img, mf1, mf2 in zip(flist, mlist1, mlist2):
            destripe_plus(
                inputfile=img, suffix=suffix, stat=stat,
                lower=lower, upper=upper, binwidth=binwidth,
                maxiter=maxiter, sigrej=sigrej,
                scimask1=scimask1, scimask2=scimask2, dqbits=dqbits,
                cte_correct=cte_correct,
                keep_intermediate_files=keep_intermediate_files,
                clobber=clobber, verbose=verbose)
        return

    inputfile = flist[0]
    scimask1 = mlist1[0]
    scimask2 = mlist2[0]

    # verify that the RAW image exists in cwd
    cwddir = os.getcwd()
    if not os.path.exists(os.path.join(cwddir, inputfile)):
        raise IOError(f"{inputfile} does not exist.")

    # get image's primary header:
    header = fits.getheader(inputfile)

    # verify masks defined (or not) simultaneously:
    if (header['CCDAMP'] == 'ABCD' and
            ((scimask1 is not None and scimask2 is None) or
             (scimask1 is None and scimask2 is not None))):
        raise ValueError("Both 'scimask1' and 'scimask2' must be specified "
                         "or not specified together.")

    calacs_str = subprocess.check_output(['calacs.e', '--version']).split()[0]  # nosec # noqa
    calacs_ver = [int(x) for x in calacs_str.decode().split('.')]
    if calacs_ver < [8, 3, 1]:
        raise ValueError(f'CALACS {calacs_str} is incomptible. '
                         'Must be 8.3.1 or later.')

    # check date for post-SM4 and if supported subarray or full frame
    is_subarray = False
    ctecorr = header['PCTECORR']
    aperture = header['APERTURE']
    detector = header['DETECTOR']
    date_obs = Time(header['DATE-OBS']).mjd  # Convert to MJD for comparison

    # intermediate filenames
    blvtmp_name = inputfile.replace('raw', 'blv_tmp')
    blctmp_name = inputfile.replace('raw', 'blc_tmp')

    # output filenames
    tra_name = inputfile.replace('_raw.fits', '.tra')
    flt_name = inputfile.replace('raw', 'flt')
    flc_name = inputfile.replace('raw', 'flc')

    if detector != 'WFC':
        raise ValueError(f"{inputfile} is not a WFC image, please check the "
                         "'DETECTOR' keyword.")

    if date_obs < SM4_MJD:
        raise ValueError(f"{inputfile} is a pre-SM4 image.")

    if header['SUBARRAY'] and cte_correct:
        if aperture in SUBARRAY_LIST:
            is_subarray = True
        else:
            LOG.warning(f'Using non-supported subarray ({aperture}), '
                        'turning CTE correction off')
            cte_correct = False

    # delete files from previous CALACS runs
    if clobber:
        for tmpfilename in [blvtmp_name, blctmp_name, flt_name, flc_name,
                            tra_name]:
            if os.path.exists(tmpfilename):
                os.remove(tmpfilename)

    # run ACSCCD on RAW
    acsccd.acsccd(inputfile)

    # modify user mask with DQ masks if requested
    dqbits = interpret_bit_flags(dqbits)
    if dqbits is not None:
        # save 'tra' file in memory to trick the log file
        # not to save first acs2d log as this is done only
        # for the purpose of obtaining DQ masks.
        # WISH: it would have been nice is there was an easy way of obtaining
        #       just the DQ masks as if data were calibrated but without
        #       having to recalibrate them with acs2d.
        if os.path.isfile(tra_name):
            with open(tra_name) as fh:
                tra_lines = fh.readlines()
        else:
            tra_lines = None

        # apply flats, etc.
        acs2d.acs2d(blvtmp_name, verbose=False, quiet=True)

        # extract DQ arrays from the FLT image:
        dq1, dq2 = _read_DQ_arrays(flt_name)

        mask1 = _get_mask(scimask1, 1)
        scimask1 = acs_destripe._mergeUserMaskAndDQ(dq1, mask1, dqbits)

        mask2 = _get_mask(scimask2, 2)
        if dq2 is not None:
            scimask2 = acs_destripe._mergeUserMaskAndDQ(dq2, mask2, dqbits)
        elif mask2 is None:
            scimask2 = None

        # reconstruct trailer file:
        if tra_lines is not None:
            with open(tra_name, mode='w') as fh:
                fh.writelines(tra_lines)

        # delete temporary FLT image:
        if os.path.isfile(flt_name):
            os.remove(flt_name)

    # execute destriping (post-SM4 data only)
    acs_destripe.clean(
        blvtmp_name, suffix, stat=stat, maxiter=maxiter, sigrej=sigrej,
        lower=lower, upper=upper, binwidth=binwidth,
        mask1=scimask1, mask2=scimask2, dqbits=dqbits,
        rpt_clean=rpt_clean, atol=atol, clobber=clobber, verbose=verbose)
    blvtmpsfx = f'blv_tmp_{suffix}'
    os.rename(inputfile.replace('raw', blvtmpsfx), blvtmp_name)

    # update subarray header
    if is_subarray and cte_correct:
        fits.setval(blvtmp_name, 'PCTECORR', value='PERFORM')
        ctecorr = 'PERFORM'

    # perform CTE correction on destriped image
    if cte_correct:
        if ctecorr == 'PERFORM':
            acscte.acscte(blvtmp_name)
        else:
            LOG.warning(f"PCTECORR={ctecorr}, cannot run CTE correction")
            cte_correct = False

    # run ACS2D to get FLT and FLC images
    acs2d.acs2d(blvtmp_name)
    if cte_correct and os.path.isfile(blctmp_name):
        acs2d.acs2d(blctmp_name)

    # delete intermediate files
    if not keep_intermediate_files:
        os.remove(blvtmp_name)
        if cte_correct and os.path.isfile(blctmp_name):
            os.remove(blctmp_name)

    info_str = f'Done.\nFLT: {flt_name}\n'
    if cte_correct:
        info_str += f'FLC: {flc_name}\n'
    LOG.info(info_str)


def _read_DQ_arrays(flt_name):
    with fits.open(flt_name) as h:
        ampstring = h[0].header['CCDAMP']
        dq1 = h['dq', 1].data
        if (ampstring == 'ABCD'):
            dq2 = h['dq', 2].data
        else:
            dq2 = None
    return dq1, dq2


def _get_mask(scimask, n):
    if isinstance(scimask, str):
        if scimask.strip() == '':
            mask = None
        else:
            mask = fits.getdata(scimask)
    elif isinstance(scimask, np.ndarray):
        mask = scimask.copy()
    elif scimask is None:
        mask = None
    else:
        raise TypeError(f"'scimask{n}' must be either a str file name, "
                        "a numpy.ndarray, or None.")
    return mask


def crrej_plus(filelist, outroot, keep_intermediate_files=False, verbose=True):
    r"""Perform CRREJ and ACS2D on given BLV_TMP or BLC_TMP files.
    The purpose of this is primarily for combining destriped
    products from :func:`destripe_plus`.

    .. note::

        If you want to run this step, it is important to keep
        intermediate files when running :func:`destripe_plus`.

    Parameters
    ----------
    filelist : str or list of str
        List of BLV_TMP or BLC_TMP files to be combined and processed.
        Do not mix BLV and BLC in the same list, but rather run this
        once for BLV and once for BLC separately.
        Input filenames in one of these formats:

            * a Python list of filenames
            * a partial filename with wildcards ('\*blx_tmp.fits')
            * filename of an ASN table ('j12345670_asn.fits')
            * an at-file (``@input``)

    outroot : str
        Rootname for combined product, which will be named
        ``<outroot>_crj.fits`` (for BLV inputs) or
        ``<outroot>_crc.fits`` (for BLC inputs).

    keep_intermediate_files : bool
        Keep de-striped CRJ_TMP or CRC_TMP around after CRREJ.

    verbose : bool
        Print informational messages.

    Raises
    ------
    ValueError
        Ambiguous input list.

    Examples
    --------
    >>> from acstools.acs_destripe_plus import destripe_plus, crrej_plus

    First, run :func:`destripe_plus`. Remember to keep its intermediate files:

    >>> destripe_plus(..., keep_intermediate_files=True)

    Now, run this function, once on BLV only, and once again on BLC only:

    >>> rootname = 'j12345678'
    >>> crrej_plus('*_blv_tmp.fits', rootname)
    >>> crrej_plus('*_blc_tmp.fits', rootname)

    """
    # Optional package dependencies
    from stsci.tools import parseinput

    filelist = parseinput.parseinput(filelist)[0]

    is_blv = np.all(['blv_tmp.fits' in f for f in filelist])
    is_blc = np.all(['blc_tmp.fits' in f for f in filelist])

    if is_blv is is_blc:
        raise ValueError('Inputs must be all BLV_TMP or all BLC_TMP files')

    if is_blv:
        sfx = 'crj'
    else:  # is_blc
        sfx = 'crc'

    tmpname = f'{outroot}_{sfx}_tmp.fits'
    acsrej.acsrej(filelist, tmpname, verbose=verbose)
    acs2d.acs2d(tmpname, verbose=verbose)  # crx_tmp -> crx

    # Work around a bug in ACS2D that names crx_tmp -> crx_tmp_flt, see
    # https://github.com/spacetelescope/hstcal/pull/470
    wrongname = f'{outroot}_{sfx}_tmp_flt.fits'
    rightname = f'{outroot}_{sfx}.fits'
    if os.path.isfile(wrongname) and not os.path.exists(rightname):
        os.rename(wrongname, rightname)

        # Brute-force fixing of final output header.
        with fits.open(rightname, 'update') as pf:
            pf[0].header['FILENAME'] = rightname
            pf[0].header['ROOTNAME'] = outroot

            for hdu in pf[1:]:
                hdu.header['EXPNAME'] = outroot
                hdu.header['ROOTNAME'] = outroot

    # Delete intermediate file
    if not keep_intermediate_files and os.path.isfile(tmpname):
        os.remove(tmpname)

    # Delete extraneous trailer file generated by ACSREJ. Its contents are
    # copied to the trailer file generated by ACS2D after anyway.
    final_tra = f'{outroot}_{sfx}_tmp.tra'
    extra_tra = f'{outroot}.tra'
    if os.path.isfile(extra_tra) and os.path.isfile(final_tra):
        os.remove(extra_tra)

        # If fixing the HSTCAL bug changes TRA naming too, this needs to be
        # adjusted accordingly.
        os.rename(final_tra, f'{outroot}_{sfx}.tra')


#-----------------------------#
# Interfaces for command line #
#-----------------------------#

def main():
    """Command line driver."""
    import argparse

    # Parse input parameters
    parser = argparse.ArgumentParser(
        prog=__taskname__,
        description=(
            'Run CALACS and standalone acs_destripe script on given post-SM4 '
            'ACS/WFC RAW full-frame or supported subarray image.'))
    parser.add_argument(
        'arg0', metavar='input', type=str, help='Input file')
    parser.add_argument(
        '--suffix', type=str, default='strp',
        help='Output suffix for acs_destripe')
    parser.add_argument(
        '--stat', type=str, default='pmode1', help='Background statistics')
    parser.add_argument(
        '--maxiter', type=int, default=15, help='Max #iterations')
    parser.add_argument(
        '--sigrej', type=float, default=2.0, help='Rejection sigma')
    parser.add_argument(
        '--lower', nargs='?', type=float, default=None,
        help='Lower limit for "good" pixels.')
    parser.add_argument(
        '--upper', nargs='?', type=float, default=None,
        help='Upper limit for "good" pixels.')
    parser.add_argument(
        '--binwidth', type=float, default=0.1,
        help='Bin width for distribution sampling.')
    parser.add_argument(
        '--sci1_mask', nargs=1, type=str, default=None,
        help='Mask image for calibrated [SCI,1]')
    parser.add_argument(
        '--sci2_mask', nargs=1, type=str, default=None,
        help='Mask image for calibrated [SCI,2]')
    parser.add_argument(
        '--dqbits', nargs='?', type=str, default=None,
        help='DQ bits to be considered "good".')
    parser.add_argument(
        '--rpt_clean', type=int, default=0,
        help='Number of *repeated* bias de-stripes to perform.')
    parser.add_argument(
        '--atol', nargs='?', type=float, default=0.01,
        help='Absolute tolerance to stop *repeated* bias de-stripes.')
    parser.add_argument(
        '--nocte', action='store_true', help='Turn off CTE correction.')
    parser.add_argument(
        '--keep_intermediate_files', action='store_true',
        help='Keep intermediate BLV_TMP and BLC_TMP files')
    parser.add_argument(
        '--clobber', action='store_true', help='Clobber output')
    parser.add_argument(
        '-q', '--quiet', action="store_true",
        help='Do not print informational messages')
    parser.add_argument(
        '--version', action='version',
        version=f'{__taskname__} v{__version__} ({__vdate__})')
    options = parser.parse_args()

    if options.sci1_mask:
        mask1 = options.sci1_mask[0]
    else:
        mask1 = options.sci1_mask

    if options.sci2_mask:
        mask2 = options.sci2_mask[0]
    else:
        mask2 = options.sci2_mask

    destripe_plus(options.arg0, suffix=options.suffix,
                  maxiter=options.maxiter, sigrej=options.sigrej,
                  lower=options.lower, upper=options.upper,
                  binwidth=options.binwidth,
                  scimask1=mask1, scimask2=mask2, dqbits=options.dqbits,
                  rpt_clean=options.rpt_clean, atol=options.atol,
                  cte_correct=not options.nocte,
                  keep_intermediate_files=options.keep_intermediate_files,
                  clobber=options.clobber, verbose=not options.quiet)


if __name__ == '__main__':
    main()
