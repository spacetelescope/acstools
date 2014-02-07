#!/usr/bin/env python
"""
Remove horizontal stripes from ACS WFC post-SM4 data.

Examples
--------

In Python without TEAL:

>>> from acstools import acs_destripe
>>> acs_destripe.clean('uncorrected_flt.fits', 'csck', clobber=False,
...                    maxiter=15, sigrej=2.0)

In Python with TEAL:

>>> from acstools import acs_destripe
>>> from stsci.tools import teal
>>> teal.teal('acs_destripe')

In Pyraf::

    --> import acstools
    --> teal acs_destripe

From command line::

    % ./acs_destripe [-h][-c] input output [maxiter # [sigrej #]]

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
__version__ = '0.4.1'
__vdate__ = '05-Feb-2014'
__author__ = 'Norman Grogin, STScI, March 2012.'


MJD_SM4 = 54967

logging.basicConfig()
LOG = logging.getLogger(__taskname__)
LOG.setLevel(logging.INFO)


class StripeArray(object):
    """Class to handle data array to be destriped."""

    def __init__(self, image):
        self.hdulist = fits.open(image)
        self.ampstring = self.hdulist[0].header['CCDAMP']
        self.flatcorr = self.hdulist[0].header['FLATCORR']
        self.darkcorr = self.hdulist[0].header['DARKCORR']

        self.darktime = self.hdulist[0].header['EXPTIME'] + self.hdulist[0].header['FLASHDUR']
        if self.hdulist[0].header['EXPTIME'] > 0:  # Not BIAS
            self.darktime += 3.0

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

        flatfile = self.hdulist[0].header['PFLTFILE']

        # if BIAS or DARK, set flatfield to unity
        if flatfile == 'N/A':
            self.invflat = np.ones(self.science.shape)
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

    def ingest_dark(self):
        """Process dark."""

        if self.hdulist[0].header['PCTECORR'] == 'COMPLETE':
            darkfile = self.hdulist[0].header['DRKCFILE']
        else:
            darkfile = self.hdulist[0].header['DARKFILE']

        # if BIAS or DARK, set dark to zeros
        if darkfile == 'N/A':
            self.dark = np.zeros(self.science.shape)
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

            # now, which section?\
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


def clean(input, suffix, maxiter=15, sigrej=2.0, clobber=False):
    """Remove horizontal stripes from ACS WFC post-SM4 data.

    .. note::

        Input data must be an ACS/WFC FLT image, with 2 SCI extensions.
        Does not work on RAW image.

        Uses the flatfield specified by the image header keyword PFLTFILE.
        If keyword value is 'N/A', as is the case with biases and darks,
        then unity flatfield is used.

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
        to the _flt suffix in each input filename to create the
        new output filename. For example, setting `suffix='csck'`
        will create '\*_flt_csck.fits' images.

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

    """
    flist, alist = parseinput.parseinput(input)

    for image in flist:
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
        output = image.replace('_flt', '_flt_' + suffix)
        LOG.info('Processing ' + image)
        perform_correction(image, output, maxiter=maxiter, sigrej=sigrej,
                           clobber=clobber)
        LOG.info(output + ' created')


def perform_correction(image, output, maxiter=15, sigrej=2.0, clobber=False):
    """
    Called by `clean` for each input image.

    Parameters
    ----------
    image : str
        Input image name.

    output : str
        Output image name.

    maxiter, sigrej, clobber : see `clean`

    """

    # construct the frame to be cleaned, including the
    # associated data stuctures needed for cleaning
    frame = StripeArray(image)

    # Do the stripe cleaning
    clean_streak(frame, maxiter=maxiter, sigrej=sigrej)

    frame.write_corrected(output, clobber=clobber)


def clean_streak(image, maxiter=15, sigrej=2.0):
    """
    Apply destriping algorithm to input array.

    Parameters
    ----------
    image : `StripeArray` object
        Arrays are modifed in-place.

    maxiter, sigrej : see `clean`

    """

    # create the array to hold the stripe amplitudes
    corr = np.empty(image.science.shape[0])

    # loop over rows to fit the stripe amplitude
    for i in range(image.science.shape[0]):
        # row-by-row iterative sigma-clipped mean; sigma, iters are adjustable
        SMean, SSig, SMedian, SMask = djs_iterstat(
            image.science[i], MaxIter=maxiter, SigRej=sigrej)

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
                 RejVal=None):
    """
    Iterative sigma-clipping.

    Parameters
    ----------
    InputArr : `numpy.ndarray`
        Input image array.

    MaxIter, SigRej : see `clean`

    Max, Min : float
        Max and min values for clipping.

    RejVal : float
        Array value to reject, in addition to clipping.

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

    Mask = np.zeros(ArrShape, dtype=np.byte) + 1

    # Reject those above Max and those below Min
    Mask[InputArr > Max] = 0
    Mask[InputArr < Min] = 0
    if RejVal is not None:
        Mask[InputArr == RejVal] = 0
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
          maxiter=configobj['maxiter'],
          sigrej=configobj['sigrej'],
          clobber=configobj['clobber'])


#-----------------------------#
# Interfaces for command line #
#-----------------------------#


def main():
    """Command line driver."""
    import getopt

    usg_str = 'USAGE: acs_destripe [-h][-c] input output [maxiter # [sigrej #]]'

    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'hc')
    except getopt.error as e:
      print(str(e))
      print(usg_str)
      print('\t', __version__)

    # initialize default values
    help = False
    clobber = False
    maxiter = 15
    sigrej = 2.0

    # read options
    for opt, value in optlist:
        if opt == '-h':
            help = True
        elif opt == '-c':
            clobber = True

    if help:
        print(usg_str)
        print('\t', __version__, '(', __vdate__, ')')
        return

    if len(args) < 2:
        LOG.error(usg_str)
        return

    if len(args) > 2:
        # User provided parameters for maxiter, and possibly sigrej
        maxiter = int(args[2])
        if len(args) == 4:
            sigrej = float(args[3])

    clean(args[0], args[1], clobber=clobber, maxiter=maxiter, sigrej=sigrej)


if __name__ == '__main__':
    main()
