#!/usr/bin/env python
"""
Remove horizontal stripes from ACS WFC post-SM4 data.

For more information, see
`Removal of Bias Striping Noise from Post-SM4 ACS WFC Images <http://www.stsci.edu/hst/acs/software/destripe/>`_.

Examples
--------

In Python without TEAL:

>>> from acstools import acs_destripe
>>> acs_destripe.clean('uncorrected_flt.fits', 'csck',
...                    mask1='mymask_sci1.fits', mask2='mymask_sci2.fits',
...                    clobber=False, maxiter=15, sigrej=2.0)

In Python with TEAL:

>>> from acstools import acs_destripe
>>> from stsci.tools import teal
>>> teal.teal('acs_destripe')

In Pyraf::

    --> import acstools
    --> teal acs_destripe

From command line::

    % acs_destripe [-h] [-c] [--mask1 MASK1] [--mask2 MASK2] [--version]
                   input suffix [maxiter] [sigrej]

"""
from __future__ import print_function

# STDLIB
import logging
import os
import sys

# THIRD-PARTY
import numpy as np
from astropy.io import fits

# STSCI
from stsci.tools import parseinput, teal


__taskname__ = 'acs_destripe'
__version__ = '0.6.2'
__vdate__ = '11-Mar-2015'
__author__ = 'Norman Grogin, STScI, March 2012.'


MJD_SM4 = 54967

logging.basicConfig()
LOG = logging.getLogger(__taskname__)
LOG.setLevel(logging.INFO)


class StripeArray(object):
    """Class to handle data array to be destriped."""
    _IDLE_TIME = 3.0

    def __init__(self, image):
        self.hdulist = fits.open(image)
        self.ampstring = self.hdulist[0].header['CCDAMP']
        self.flatcorr = self.hdulist[0].header['FLATCORR']
        self.flshcorr = self.hdulist[0].header['FLSHCORR']
        self.darkcorr = self.hdulist[0].header['DARKCORR']

        exptime = self.hdulist[0].header['EXPTIME']
        self.flashdur = self.hdulist[0].header['FLASHDUR']
        self.darktime = exptime + self.flashdur

        if exptime > 0:  # Not BIAS
            self.darktime += self._IDLE_TIME

        self.configure_arrays()

    def configure_arrays(self):
        """Get the SCI and ERR data."""
        self.science = self.hdulist['sci',1].data
        self.err = self.hdulist['err',1].data
        if (self.ampstring == 'ABCD'):
            self.science = np.concatenate(
                (self.science, self.hdulist['sci',2].data[::-1,:]), axis=1)
            self.err = np.concatenate(
                (self.err, self.hdulist['err',2].data[::-1,:]), axis=1)
        self.ingest_dark()
        self.ingest_flash()
        self.ingest_flatfield()

    def _get_ref_section(self, refaxis1, refaxis2):
        """Get reference file section to use."""

        sizaxis1 = self.hdulist[1].header['SIZAXIS1']
        sizaxis2 = self.hdulist[1].header['SIZAXIS2']
        centera1 = self.hdulist[1].header['CENTERA1']
        centera2 = self.hdulist[1].header['CENTERA2']

        # configure the offset appropriate to left- or right-side of CCD
        if (self.ampstring[0] == 'A' or self.ampstring[0] == 'C'):
            xdelta = 13
        else:
            xdelta = 35

        if sizaxis1 == refaxis1:
            xlo = 0
            xhi = sizaxis1
        else:
            xlo = centera1 - xdelta - sizaxis1 // 2 - 1
            xhi = centera1 - xdelta + sizaxis1 // 2 - 1

        if sizaxis2 == refaxis2:
            ylo = 0
            yhi = sizaxis2
        else:
            ylo = centera2 - sizaxis2 // 2 - 1
            yhi = centera2 + sizaxis2 // 2 - 1

        return xlo, xhi, ylo, yhi

    def ingest_flatfield(self):
        """Process flatfield."""

        for ff in ['DFLTFILE', 'LFLTFILE']:
            vv = self.hdulist[0].header[ff]
            if vv != 'N/A':
                LOG.warning('{0}={1} is not accounted for!'.format(ff, vv))

        flatfile = self.hdulist[0].header['PFLTFILE']

        # if BIAS or DARK, set flatfield to unity
        if flatfile == 'N/A':
            self.invflat = np.ones_like(self.science)
            return

        hduflat = self.resolve_reffilename(flatfile)

        if (self.ampstring == 'ABCD'):
            self.invflat = np.concatenate(
                (1 / hduflat['sci',1].data,
                 1 / hduflat['sci',2].data[::-1,:]), axis=1)
        else:
            # complex algorithm to determine proper subarray of flatfield to use

            # which amp?
            if (self.ampstring == 'A' or self.ampstring == 'B' or
                    self.ampstring == 'AB'):
                self.invflat = 1 / hduflat['sci',2].data
            else:
                self.invflat = 1 / hduflat['sci',1].data

            # now, which section?
            flataxis1 = hduflat[1].header['NAXIS1']
            flataxis2 = hduflat[1].header['NAXIS2']

            xlo, xhi, ylo, yhi = self._get_ref_section(flataxis1, flataxis2)

            self.invflat = self.invflat[ylo:yhi,xlo:xhi]

        # apply the flatfield if necessary
        if self.flatcorr != 'COMPLETE':
            self.science = self.science * self.invflat
            self.err = self.err * self.invflat

    def ingest_flash(self):
        """Process post-flash."""

        flshfile = self.hdulist[0].header['FLSHFILE']
        flashsta = self.hdulist[0].header['FLASHSTA']

        # Set post-flash to zeros
        if flshfile == 'N/A' or self.flashdur <= 0:
            self.flash = np.zeros_like(self.science)
            return

        if flashsta != 'SUCCESSFUL':
            LOG.warning('Flash status is {0}'.format(flashsta))

        hduflash = self.resolve_reffilename(flshfile)

        if (self.ampstring == 'ABCD'):
            self.flash = np.concatenate(
                (hduflash['sci',1].data,
                 hduflash['sci',2].data[::-1,:]), axis=1)
        else:
            # complex algorithm to determine proper subarray of flash to use

            # which amp?
            if (self.ampstring == 'A' or self.ampstring == 'B' or
                    self.ampstring == 'AB'):
                self.flash = hduflash['sci',2].data
            else:
                self.flash = hduflash['sci',1].data

            # now, which section?
            flashaxis1 = hduflash[1].header['NAXIS1']
            flashaxis2 = hduflash[1].header['NAXIS2']

            xlo, xhi, ylo, yhi = self._get_ref_section(flashaxis1, flashaxis2)

            self.flash = self.flash[ylo:yhi,xlo:xhi]

        # Apply the flash subtraction if necessary.
        # Not applied to ERR, to be consistent with ingest_dark()
        if self.flshcorr != 'COMPLETE':
            self.science = self.science - self.flash * self.flashdur

    def ingest_dark(self):
        """Process dark."""

        if self.hdulist[0].header['PCTECORR'] == 'COMPLETE':
            darkfile = self.hdulist[0].header['DRKCFILE']
        else:
            darkfile = self.hdulist[0].header['DARKFILE']

        # if BIAS or DARK, set dark to zeros
        if darkfile == 'N/A':
            self.dark = np.zeros_like(self.science)
            return

        hdudark = self.resolve_reffilename(darkfile)

        if (self.ampstring == 'ABCD'):
            self.dark = np.concatenate(
                (hdudark['sci',1].data,
                 hdudark['sci',2].data[::-1,:]), axis=1)
        else:
            # complex algorithm to determine proper subarray of dark to use

            # which amp?
            if (self.ampstring == 'A' or self.ampstring == 'B' or
                    self.ampstring == 'AB'):
                self.dark = hdudark['sci',2].data
            else:
                self.dark = hdudark['sci',1].data

            # now, which section?
            darkaxis1 = hdudark[1].header['NAXIS1']
            darkaxis2 = hdudark[1].header['NAXIS2']

            xlo, xhi, ylo, yhi = self._get_ref_section(darkaxis1, darkaxis2)

            self.dark = self.dark[ylo:yhi,xlo:xhi]

        # Apply the dark subtraction if necessary.
        # Effect of DARK on ERR is insignificant for de-striping.
        if self.darkcorr != 'COMPLETE':
            self.science = self.science - self.dark * self.darktime

    def resolve_reffilename(self, reffile):
        """Resolve the filename into an absolute pathname (if necessary)."""
        refparts = reffile.partition('$')
        if refparts[1] == '$':
            refdir = os.getenv(refparts[0])
            if refdir is None:
                raise ValueError(
                    'Environment variable {0} undefined!'.format(refparts[0]))
            reffile = os.path.join(refdir, refparts[2])
        if not os.path.exists(reffile):
            raise IOError('{0} could not be found!'.format(reffile))

        return(fits.open(reffile))

    def write_corrected(self, output, clobber=False):
        """Write out the destriped data."""

        # un-apply the flatfield if necessary
        if self.flatcorr != 'COMPLETE':
            self.science = self.science / self.invflat
            self.err = self.err / self.invflat

        # un-apply the post-flash if necessary
        if self.flshcorr != 'COMPLETE':
            self.science = self.science + self.flash * self.flashdur

        # un-apply the dark if necessary
        if self.darkcorr != 'COMPLETE':
            self.science = self.science + self.dark * self.darktime

        # reverse the amp merge
        if (self.ampstring == 'ABCD'):
            tmp_1, tmp_2 = np.split(self.science, 2, axis=1)
            self.hdulist['sci',1].data = tmp_1.copy()
            self.hdulist['sci',2].data = tmp_2[::-1,:].copy()

            tmp_1, tmp_2 = np.split(self.err, 2, axis=1)
            self.hdulist['err',1].data = tmp_1.copy()
            self.hdulist['err',2].data = tmp_2[::-1,:].copy()
        else:
            self.hdulist['sci',1].data = self.science.copy()
            self.hdulist['err',1].data = self.err.copy()

        # Write the output
        self.hdulist.writeto(output, clobber=clobber)


def _read_mask(mask1, mask2):
    if mask1 is None and mask2 is None:
        mask = None
    elif mask2 is None:
        mask = fits.getdata(mask1)
    elif mask1 is None:
        mask = fits.getdata(mask2)
    else:
        dat1 = fits.getdata(mask1)
        dat2 = fits.getdata(mask2)
        mask = np.concatenate((dat1, dat2[::-1,:]), axis=1)

    # Mask must only have binary values
    if mask is not None:
        mask[mask != 0] = 1

    return mask


def clean(input, suffix, maxiter=15, sigrej=2.0, clobber=False,
          mask1=None, mask2=None):
    """Remove horizontal stripes from ACS WFC post-SM4 data.

    .. note::

        Does not work on RAW image.

        Uses the flatfield specified by the image header keyword PFLTFILE.
        If keyword value is 'N/A', as is the case with biases and darks,
        then unity flatfield is used.

        Uses post-flash image specified by the image header keyword FLSHFILE.
        If keyword value is 'N/A', then dummy post-flash with zeroes is used.

        Uses the dark image specified by the image header keyword DARKFILE.
        If keyword value is 'N/A', then dummy dark with zeroes is used.

    Parameters
    ----------
    input : str or list of str
        Input filenames in one of these formats:

            * a Python list of filenames
            * a partial filename with wildcards ('\*flt.fits')
            * filename of an ASN table ('j12345670_asn.fits')
            * an at-file (``@input``)

    suffix : str
        The string to use to add to each input file name to
        indicate an output product. This string will be appended
        to the suffix in each input filename to create the
        new output filename. For example, setting `suffix='csck'`
        will create '\*_csck.fits' images.

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

    mask1 : str or list of str
        Mask images for ``SCI,1``, one for each input file.
        Pixels with zero values will be masked out, in addition to clipping.

    mask2 : str or list of str
        Mask images for ``SCI,2``, one for each input file.
        Pixels with zero values will be masked out, in addition to clipping.
        This is not used for subarrays.

    """
    flist = parseinput.parseinput(input)[0]
    mlist1 = parseinput.parseinput(mask1)[0]
    mlist2 = parseinput.parseinput(mask2)[0]

    n_input = len(flist)
    n_mask1 = len(mlist1)
    n_mask2 = len(mlist2)

    if n_mask1 == 0:
        mlist1 = [None] * n_input
    elif n_mask1 != n_input:
        raise ValueError('Insufficient masks for [SCI,1]')

    if n_mask2 == 0:
        mlist2 = [None] * n_input
    elif n_mask2 != n_input:
        raise ValueError('Insufficient masks for [SCI,2]')

    for image, maskfile1, maskfile2 in zip(flist, mlist1, mlist2):
        # Skip processing pre-SM4 images
        if (fits.getval(image, 'EXPSTART') <= MJD_SM4):
            LOG.warn(image + ' is pre-SM4. Skipping...'%image)
            continue

        # Data must be in ELECTRONS
        if (fits.getval(image, 'BUNIT', ext=1) != 'ELECTRONS'):
            LOG.warn(image + ' is not in ELECTRONS. Skipping...')
            continue

        # Skip processing CTECORR-ed images
        if (fits.getval(image, 'PCTECORR') == 'COMPLETE'):
            LOG.warn(image + ' already has PCTECORR applied. Skipping...')
            continue

        # generate output filename for each input based on specification
        # of the output suffix
        output = image.replace('.fits', '_' + suffix + '.fits')
        LOG.info('Processing ' + image)
        maskdata = _read_mask(maskfile1, maskfile2)
        perform_correction(image, output, maxiter=maxiter, sigrej=sigrej,
                           clobber=clobber, mask=maskdata)
        LOG.info(output + ' created')


def perform_correction(image, output, maxiter=15, sigrej=2.0, clobber=False,
                       mask=None):
    """
    Clean each input image.

    Parameters
    ----------
    image : str
        Input image name.

    output : str
        Output image name.

    mask : `numpy.ndarray`
        Mask array.

    maxiter, sigrej, clobber
        See :func:`clean`.

    """

    # construct the frame to be cleaned, including the
    # associated data stuctures needed for cleaning
    frame = StripeArray(image)

    # Do the stripe cleaning
    clean_streak(frame, maxiter=maxiter, sigrej=sigrej, mask=mask)

    frame.write_corrected(output, clobber=clobber)


def clean_streak(image, maxiter=15, sigrej=2.0, mask=None):
    """
    Apply destriping algorithm to input array.

    Parameters
    ----------
    image : `StripeArray` object
        Arrays are modifed in-place.

    mask : `numpy.ndarray`
        Mask array. Pixels with zero values are masked out.

    maxiter, sigrej : see `clean`

    """
    if mask is not None and image.science.shape != mask.shape:
        raise ValueError('Mask shape does not match science data')

    # create the array to hold the stripe amplitudes
    corr = np.empty(image.science.shape[0])

    # loop over rows to fit the stripe amplitude
    for i in range(image.science.shape[0]):
        if mask is not None:
            mask_arr = mask[i]
        else:
            mask_arr = None

        # row-by-row iterative sigma-clipped mean; sigma, iters are adjustable
        SMean, SSig, SMedian, SMask = djs_iterstat(
            image.science[i], MaxIter=maxiter, SigRej=sigrej, Mask=mask_arr)

        # SExtractor-esque central value statistic; slightly sturdier against
        # skewness of pixel histogram due to faint source flux
        corr[i] = 2.5 * SMedian - 1.5 * SMean

    # preserve the original mean level of the image
    corr -= np.average(corr)

    # apply the correction row-by-row
    for i in range(image.science.shape[0]):
        # stripe is constant along the row, before flatfielding;
        # afterwards it has the shape of the inverse flatfield
        truecorr = corr[i] * image.invflat[i] / np.average(image.invflat[i])

        # correct the SCI extension
        image.science[i] -= truecorr

        # correct the ERR extension
        image.err[i] = np.sqrt(np.abs(image.err[i]**2 - truecorr))


def djs_iterstat(InputArr, MaxIter=10, SigRej=3.0, Max=None, Min=None,
                 Mask=None):
    """
    Iterative sigma-clipping.

    Parameters
    ----------
    InputArr : `numpy.ndarray`
        Input image array.

    MaxIter, SigRej : see `clean`

    Max, Min : float
        Max and min values for clipping.

    Mask : `numpy.ndarray`
        Mask array to indicate pixels to reject, in addition to clipping.
        Pixels where mask is zero will be rejected.
        If not given, all pixels will be used.

    Returns
    -------
    FMean, FSig, FMedian : float
        Mean, sigma, and median of final result.

    SaveMask : `numpy.ndarray`
        Image mask from the final iteration.

    """
    NGood    = InputArr.size
    ArrShape = InputArr.shape
    if NGood == 0:
        LOG.warn('djs_iterstat - No data points given')
        return 0, 0, 0, 0
    if NGood == 1:
        LOG.warn('djs_iterstat - Only one data point; cannot compute stats')
        return 0, 0, 0, 0
    if np.unique(InputArr).size == 1:
        LOG.warn('djs_iterstat - Only one value in data; cannot compute stats')
        return 0, 0, 0, 0

    # Determine Max and Min
    if Max is None:
        Max = InputArr.max()
    if Min is None:
        Min = InputArr.min()

    # Use all pixels if no mask is provided
    if Mask is None:
        Mask = np.ones(ArrShape, dtype=np.byte)

    # Reject those above Max and those below Min
    Mask[InputArr > Max] = 0
    Mask[InputArr < Min] = 0

    FMean = np.sum(1.0 * InputArr * Mask) / NGood
    FSig  = np.sqrt(np.sum((1.0 * InputArr - FMean)**2 * Mask) / (NGood - 1))

    NLast = -1
    Iter  =  0
    NGood = np.sum(Mask)
    if NGood < 2:
        LOG.warn('djs_iterstat - No good data points; cannot compute stats')
        return -1, -1, -1, -1

    while (Iter < MaxIter) and (NLast != NGood) and (NGood >= 2):
        LoVal = FMean - SigRej * FSig
        HiVal = FMean + SigRej * FSig
        NLast = NGood

        Mask[InputArr < LoVal] = 0
        Mask[InputArr > HiVal] = 0
        NGood = np.sum(Mask)

        if NGood >= 2:
            FMean = np.sum(1.0 * InputArr * Mask) / NGood
            FSig  = np.sqrt(np.sum(
                (1.0 * InputArr - FMean)**2 * Mask) / (NGood - 1))

        SaveMask = Mask.copy()

        Iter += 1

    if np.sum(SaveMask) > 2:
        FMedian = np.median(InputArr[SaveMask == 1])
    else:
        FMedian = FMean

    return FMean, FSig, FMedian, SaveMask


#-------------------------#
# Interfaces used by TEAL #
#-------------------------#


def getHelpAsString(fulldoc=True):
    """
    Returns documentation on the `clean` function. Required by TEAL.

    """
    return clean.__doc__


def run(configobj=None):
    """
    TEAL interface for the `clean` function.

    """
    clean(configobj['input'],
          configobj['suffix'],
          mask1=configobj['mask1'],
          mask2=configobj['mask2'],
          maxiter=configobj['maxiter'],
          sigrej=configobj['sigrej'],
          clobber=configobj['clobber'])


#-----------------------------#
# Interfaces for command line #
#-----------------------------#


def main():
    """Command line driver."""
    import argparse

    parser = argparse.ArgumentParser(
        prog=__taskname__,
        description='Remove horizontal stripes from ACS WFC post-SM4 data.')
    parser.add_argument(
        'arg0', metavar='input', type=str, help='Input file')
    parser.add_argument(
        'arg1', metavar='suffix', type=str, help='Output suffix')
    parser.add_argument(
        'maxiter', nargs='?', type=int, default=15, help='Max #iterations')
    parser.add_argument(
        'sigrej', nargs='?', type=float, default=2.0, help='Rejection sigma')
    parser.add_argument(
        '-c', '--clobber', action="store_true", help='Clobber output')
    parser.add_argument(
        '--mask1', nargs=1, type=str, default=None,
        help='Mask image for [SCI,1]')
    parser.add_argument(
        '--mask2', nargs=1, type=str, default=None,
        help='Mask image for [SCI,2]')
    parser.add_argument(
        '--version', action="version",
        version='{0} v{1} ({2})'.format(__taskname__, __version__, __vdate__))
    args = parser.parse_args()

    if args.mask1:
        mask1 = args.mask1[0]
    else:
        mask1 = args.mask1

    if args.mask2:
        mask2 = args.mask2[0]
    else:
        mask2 = args.mask2

    clean(args.arg0, args.arg1, clobber=args.clobber, maxiter=args.maxiter,
          sigrej=args.sigrej, mask1=mask1, mask2=mask2)


if __name__ == '__main__':
    main()
