#!/usr/bin/env python
"""
Fully calibrate post-SM4 ACS/WFC exposures using the standalone
:ref:`acsdestripe` tool to remove stripes between ACSCCD and ACSCTE
steps in CALACS.

This script runs CALACS (8.3.1 or higher only) and ``acs_destripe``
on ACS/WFC images. Input files must be RAW full-frame or subarray
ACS/WFC exposures taken after SM4. Resultant outputs are science-ready
FLT and FLC (if applicable) files.

This script is useful for when built-in CALACS destriping algorithm
using overscans is insufficient or unavailable.

For more information, see
`Removal of Bias Striping Noise from Post-SM4 ACS WFC Images <http://www.stsci.edu/hst/acs/software/destripe/>`_.

Examples
--------

In Python without TEAL:

>>> from acstools import acs_destripe_plus
>>> acs_destripe_plus.destripe_plus(
...     'j12345678_raw.fits', suffix='strp', maxiter=15, sigrej=2.0,
...     scimask1='mymask_sci1.fits', scimask2='mymask_sci2.fits',
...     clobber=False, cte_correct=True)

In Python with TEAL:

>>> from acstools import acs_destripe_plus
>>> from stsci.tools import teal
>>> teal.teal('acs_destripe_plus')

In Pyraf::

    --> import acstools
    --> teal acs_destripe_plus

From command line::

    % acs_destripe_plus [-h] [--suffix SUFFIX] [--maxiter MAXITER]
                        [--sigrej SIGREJ] [--clobber] [--sci1_mask SCI1_MASK]
                        [--sci2_mask SCI2_MASK] [--nocte] [--version]
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
#
from __future__ import division, print_function

# STDLIB
import logging
import os
import subprocess

# ASTROPY
from astropy.io import fits
from astropy.time import Time

# LOCAL
from . import acs_destripe
from . import acs2d
from . import acsccd
from . import acscte


__taskname__ = 'acs_destripe_plus'
__version__ = '0.2.0'
__vdate__ = '11-Mar-2015'
__author__ = 'Leonardo Ubeda & Sara Ogaz, ACS Team, STScI'

SM4_DATE = Time('2008-01-01')
SUBARRAY_LIST = [
    'WFC1-2K', 'WFC1-POL0UV', 'WFC1-POL0V', 'WFC1-POL60V',
    'WFC1-POL60UV', 'WFC1-POL120V', 'WFC1-POL120UV', 'WFC1-SMFL',
    'WFC1-IRAMPQ', 'WFC1-MRAMPQ', 'WFC2-2K', 'WFC2-ORAMPQ',
    'WFC2-SMFL', 'WFC2-POL0UV', 'WFC2-POL0V', 'WFC2-MRAMPQ']

logging.basicConfig()
LOG = logging.getLogger(__taskname__)
LOG.setLevel(logging.INFO)


def destripe_plus(inputfile, suffix='strp', maxiter=15, sigrej=2.0,
                  clobber=False, scimask1=None, scimask2=None,
                  cte_correct=True):
    """Calibrate post-SM4 ACS/WFC exposure(s) and use
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
        indicate an output product. This string will be appended
        to the suffix in each input filename to create the
        new output filename. For example, setting `suffix='strp'`
        will create '\*_strp.fits' images.

    maxiter : int
        This parameter controls the maximum number of iterations
        to perform when computing the statistics used to compute the
        row-by-row corrections.

    sigrej : float
        This parameters sets the sigma level for the rejection applied
        during each iteration of statistics computations for the
        row-by-row corrections.

    clobber : bool
        Specify whether or not to 'clobber' (delete then replace)
        previously generated products with the same names.

    scimask1 : str or list of str
        Mask images for ``SCI,1``, one for each input file.
        Pixels with zero values will be masked out, in addition to clipping.

    scimask2 : str or list of str
        Mask images for ``SCI,2``, one for each input file.
        Pixels with zero values will be masked out, in addition to clipping.
        This is not used for subarrays.

    cte_correct : bool
        Perform CTE correction.

    Raises
    ------
    IOError
        Input file does not exist.

    ValueError
        Invalid header values or CALACS version.

    """
    # verify that the RAW image exists in cwd
    cwddir = os.getcwd()
    if not os.path.exists(os.path.join(cwddir, inputfile)):
        raise IOError("{0} does not exist.".format(inputfile))

    # verify CALACS is comptible
    calacs_str = subprocess.check_output(['calacs.e', '--version']).split()[0]
    calacs_ver = tuple(map(int, calacs_str.split('.')))
    if calacs_ver < (8, 3, 1):
        raise ValueError('CALACS {0} is incomptible. '
                         'Must be 8.3.1 or later.'.format(calacs_str))

    # check date for post-SM4 and if 2K subarray or full frame
    is_sub2K = False
    header = fits.getheader(inputfile)
    ctecorr = header['PCTECORR']
    aperture = header['APERTURE']
    detector = header['DETECTOR']
    date_obs = Time(header['DATE-OBS'])

    # intermediate filenames
    blvtmp_name = inputfile.replace('raw', 'blv_tmp')
    blctmp_name = inputfile.replace('raw', 'blc_tmp')

    if detector != 'WFC':
        raise ValueError("{0} is not a WFC image, please check the 'DETECTOR'"
                         " keyword.".format(inputfile))

    if date_obs < SM4_DATE:
        raise ValueError(
            "{0} is a pre-SM4 image.".format(inputfile))

    if header['SUBARRAY'] and cte_correct:
        if aperture in SUBARRAY_LIST:
            is_sub2K = True
        else:
            LOG.warning('Using non-2K subarray, turning CTE correction off')
            cte_correct = False

    # run ACSCCD on RAW subarray
    acsccd.acsccd(inputfile)

    # execute destriping of the subarray (post-SM4 data only)
    acs_destripe.clean(
        blvtmp_name, suffix, maxiter=maxiter, sigrej=sigrej, clobber=clobber,
        mask1=scimask1, mask2=scimask2)
    blvtmpsfx = 'blv_tmp_{0}'.format(suffix)
    os.rename(inputfile.replace('raw', blvtmpsfx), blvtmp_name)

    # update subarray header
    if is_sub2K and cte_correct:
        fits.setval(blvtmp_name, 'PCTECORR', value='PERFORM')
        ctecorr = 'PERFORM'

    # perform CTE correction on destriped image
    if cte_correct:
    	if ctecorr == 'PERFORM':
            acscte.acscte(blvtmp_name)
        else:
            LOG.warning(
                "PCTECORR={0}, cannot run CTE correction".format(ctecorr))
            cte_correct = False

    # run ACS2D to get FLT and FLC images
    acs2d.acs2d(blvtmp_name)
    if cte_correct:
        acs2d.acs2d(blctmp_name)

    # delete intermediate files
    os.remove(blvtmp_name)
    if cte_correct:
        os.remove(blctmp_name)

    info_str = 'Done.\nFLT: {0}\n'.format(inputfile.replace('raw', 'flt'))
    if cte_correct:
        info_str += 'FLC: {0}\n'.format(inputfile.replace('raw', 'flc'))
    LOG.info(info_str)


#-------------------------#
# Interfaces used by TEAL #
#-------------------------#

def getHelpAsString(fulldoc=True):
    """Returns documentation on :func:`destripe_plus`. Required by TEAL."""
    return destripe_plus.__doc__


def run(configobj=None):
    """TEAL interface for :func:`destripe_plus`."""
    destripe_plus(
        configobj['input'],
        suffix=configobj['suffix'],
        scimask1=configobj['scimask1'],
        scimask2=configobj['scimask2'],
        maxiter=configobj['maxiter'],
        sigrej=configobj['sigrej'],
        clobber=configobj['clobber'],
        cte_correct=configobj['cte_correct'])


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
            'ACS/WFC RAW full-frame or subarray image.'))
    parser.add_argument(
        'arg0', metavar='input', type=str, help='Input file')
    parser.add_argument(
        '--suffix', type=str, default='strp', help='Output suffix')
    parser.add_argument(
        '--maxiter', type=int, default=15, help='Max #iterations')
    parser.add_argument(
        '--sigrej', type=float, default=2.0, help='Rejection sigma')
    parser.add_argument(
        '--clobber', action='store_true', help='Clobber output')
    parser.add_argument(
        '--sci1_mask', nargs=1, type=str, default=None,
        help='Mask image for [SCI,1]')
    parser.add_argument(
        '--sci2_mask', nargs=1, type=str, default=None,
        help='Mask image for [SCI,2]')
    parser.add_argument(
        '--nocte', action='store_false', help='Turn off CTE correction.')
    parser.add_argument(
        '--version', action='version',
        version='{0} v{1} ({2})'.format(__taskname__, __version__, __vdate__))
    options = parser.parse_args()

    if options.sci1_mask:
        mask1 = options.sci1_mask[0]
    else:
        mask1 = options.sci1_mask

    if options.sci2_mask:
        mask2 = options.sci2_mask[0]
    else:
        mask2 = options.sci2_mask

    destripe_plus(options.arg0, suffix=options.suffix, clobber=options.clobber,
                  maxiter=options.maxiter, sigrej=options.sigrej,
                  scimask1=mask1, scimask2=mask2, cte_correct=options.nocte)


if __name__=='__main__':
    main()
