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

    % acs_destripe [-h] [-c] [--mask1 MASK1] [--mask2 MASK2]
                    [--dqbits [DQBITS]] [--version]
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
from stsci.tools import parseinput, teal, bitmask


__taskname__ = 'acs_destripe'
__version__ = '0.7.2'
__vdate__ = '24-Mar-2015'
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
        self.dq = self.hdulist['dq',1].data
        if (self.ampstring == 'ABCD'):
            self.science = np.concatenate(
                (self.science, self.hdulist['sci',2].data[::-1,:]), axis=1)
            self.err = np.concatenate(
                (self.err, self.hdulist['err',2].data[::-1,:]), axis=1)
            self.dq = np.concatenate(
                (self.dq, self.hdulist['dq',2].data[::-1,:]), axis=1)
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
    if isinstance(mask1, str):
        mask1 = fits.getdata(mask1)
    if isinstance(mask2, str):
        mask2 = fits.getdata(mask2)

    if mask1 is None and mask2 is None:
        return None
    elif mask2 is None:
        mask = mask1
    elif mask1 is None:
        mask = mask2
    else:
        dat1 = mask1
        dat2 = mask2
        mask = np.concatenate((dat1, dat2[::-1,:]), axis=1)

    # Mask must only have binary values
    if mask is not None:
        mask[mask != 0] = 1

    return mask


def clean(input, suffix, maxiter=15, sigrej=2.0, clobber=False,
          mask1=None, mask2=None, dqbits=None):
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

    mask1 : str, numpy.ndarray, None, or list of these types
        Mask images for ``SCI,1``, one for each input file.
        Pixels with zero values will be masked out, in addition to clipping.

    mask2 : str, numpy.ndarray, None, or list of these types
        Mask images for ``SCI,2``, one for each input file.
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
            in the `mask1` and `mask2` parameters (if any).

    """
    flist = parseinput.parseinput(input)[0]

    if isinstance(mask1, str):
        mlist1 = parseinput.parseinput(mask1)[0]
    elif isinstance(mask1, np.ndarray):
        mlist1 = [ mask1.copy() ]
    elif mask1 is None:
        mlist1 = []
    elif isinstance(mask1, list):
        mlist1 = []
        for m in mask1:
            if isinstance(m, np.ndarray):
                mlist1.append(m.copy())
            elif isinstance(m, str):
                mlist1 += parseinput.parseinput(m)[0]
            else:
                raise TypeError("'mask1' must be a list of str or "
                                "numpy.ndarray values.")
    else:
        raise TypeError("'mask1' must be either a str, or a "
                        "numpy.ndarray, or a list of the two type of "
                        "values.")

    if isinstance(mask2, str):
        mlist2 = parseinput.parseinput(mask2)[0]
    elif isinstance(mask2, np.ndarray):
        mlist2 = [ mask2.copy() ]
    elif mask2 is None:
        mlist2 = []
    elif isinstance(mask2, list):
        mlist2 = []
        for m in mask2:
            if isinstance(m, np.ndarray):
                mlist2.append(m.copy())
            elif isinstance(m, str):
                mlist2 += parseinput.parseinput(m)[0]
            else:
                raise TypeError("'mask2' must be a list of str or "
                                "numpy.ndarray values.")
    else:
        raise TypeError("'mask2' must be either a str or a "
                        "numpy.ndarray, or a list of the two type of "
                        "values.")

    n_input = len(flist)
    n_mask1 = len(mlist1)
    n_mask2 = len(mlist2)

    if n_input == 0:
        raise ValueError("No input file(s) provided or the file(s) do not exist")

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

        # verify masks defined (or not) simultaneously:
        if fits.getval(image, 'CCDAMP') == 'ABCD' and \
           ((mask1 is not None and mask2 is None) or \
            (mask1 is None and mask2 is not None)):
            raise ValueError("Both 'mask1' and 'mask2' must be specified "
                             "or not specified together.")

        maskdata = _read_mask(maskfile1, maskfile2)
        perform_correction(image, output, maxiter=maxiter, sigrej=sigrej,
                           clobber=clobber, mask=maskdata, dqbits=dqbits)
        LOG.info(output + ' created')


def perform_correction(image, output, maxiter=15, sigrej=2.0, clobber=False,
                       mask=None, dqbits=None):
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

    dqbits : int, str, or None
        Data quality bits to be considered as "good" (or "bad").
        See :func:`clean` for more details.

    """

    # construct the frame to be cleaned, including the
    # associated data stuctures needed for cleaning
    frame = StripeArray(image)

    # combine user mask with image's DQ array:
    mask = _mergeUserMaskAndDQ(frame.dq, mask, dqbits)

    # Do the stripe cleaning
    Success, NUpdRows, NMaxIter, STDDEVCorr, MaxCorr = clean_streak(
        frame, maxiter=maxiter, sigrej=sigrej, mask=mask
    )

    if (STDDEVCorr > 0.9):
        LOG.warn('perform_correction - STDDEV of applied destripe '
                 'corrections {} exceeds known bias striping '
                 'STDDEV of 0.9e (see ISR ACS 2011-05).'.format(STDDEVCorr))
        LOG.warn('perform_correction - Maximum applied correction: {}.'.format(MaxCorr))

    frame.write_corrected(output, clobber=clobber)


def _mergeUserMaskAndDQ(dq, mask, dqbits):
    dqbits = bitmask.interpret_bits_value(dqbits)
    if dqbits is None:
        if mask is None:
            return None
        else:
            return mask.copy()

    if dq is None:
        raise ValueError("DQ array is None while 'dqbits' is not None.")

    dqmask = bitmask.bitmask2mask(bitmask=dq, ignore_bits=dqbits,
                                  good_mask_value=1, dtype=np.uint8)
    if mask is None:
        return dqmask

    # merge user mask with DQ mask:
    dqmask[mask != 0] = 1

    return mask


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

    Returns
    -------
    Success : bool
        Indicates successful execution.

    NUpdRows : int
        Number of updated rows in the image.

    NMaxIter : int
        Maximum number of clipping iterations performed on image rows.

    STDDEVCorr, MaxCorr : float
        Standard deviation of corrections and maximum correction applied
        to the non-flat-field-corrected (i.e., RAW) image rows.

    """
    if mask is not None and image.science.shape != mask.shape:
        raise ValueError('Mask shape does not match science data shape')

    # create the array to hold the stripe amplitudes
    corr = np.zeros(image.science.shape[0], dtype=np.float64)
    corr_scale = np.zeros(image.science.shape[0], dtype=np.float64)
    npix = np.zeros(image.science.shape[0], dtype=np.int)
    sigcorr2 = np.zeros(image.science.shape[0], dtype=np.float64)
    tcorr = 0.0
    tnpix = 0
    tnpix2 = 0
    NMaxIter = 0
    NUpdRows = 0
    MaxCorr = 0.0

    # loop over rows to fit the stripe amplitude
    mask_arr = None
    for i in xrange(image.science.shape[0]):
        if mask is not None:
            mask_arr = mask[i]

        # row-by-row iterative sigma-clipped mean; sigma, iters are adjustable
        SMean, SSig, SMedian, NPix, NIter, BMask = djs_iterstat(
            image.science[i], MaxIter=maxiter, SigRej=sigrej,
            Mask=mask_arr, lineno=i+1
        )

        # SExtractor-esque central value statistic; slightly sturdier against
        # skewness of pixel histogram due to faint source flux
        npix[i] = NPix
        corr[i] = 2.5 * SMedian - 1.5 * SMean
        if NPix > 0:
            sigcorr2[i] = np.sum((image.err[i][BMask])**2)/NPix**2
            corr_scale[i] = 1.0 / np.average(image.invflat[i][BMask])
        tnpix += NPix
        tnpix2 += NPix * NPix
        tcorr += corr[i] * NPix
        if NIter > NMaxIter:
            NMaxIter = NIter

    if tnpix <= 0:
        LOG.warn('clean_streak - No good data points; cannot de-stripe.')
        return False, 0, 0, 0.0, 0.0

    if NMaxIter >= maxiter:
        LOG.warn('clean_streak - Maximum number of clipping iterations '
                 'specified by the user ({}) has been reached.'.format(maxiter))

    # preserve the original mean level of the image
    wmean = tcorr / tnpix
    corr[npix > 0] -= wmean

    # convert corrections to the "RAW" space:
    corr *= corr_scale

    # weighted sample variance (in "RAW" space):
    wrmean = np.sum(npix * corr) / tnpix
    wvar = np.sum(npix * (corr-wrmean)**2) / tnpix
    uwvar = wvar / (1.0 - float(tnpix2)/float(tnpix)**2)
    STDDEVCorr = float(np.sqrt(uwvar))

    # apply the correction row-by-row
    for i in xrange(image.science.shape[0]):
        if npix[i] < 1:
            continue
        NUpdRows += 1

        if np.abs(corr[i]) > MaxCorr:
            MaxCorr = np.abs(corr[i])

        ffdark = image.darktime * image.dark[i] * image.invflat[i]
        t1 = np.maximum(image.science[i] + ffdark, 0.0)

        # stripe is constant along the row, before flatfielding;
        # afterwards it has the shape of the inverse flatfield
        truecorr = corr[i] * image.invflat[i]
        truecorr_sig2 = sigcorr2[i] * (corr_scale[i] * image.invflat[i])**2

        # correct the SCI extension
        image.science[i] -= truecorr

        t2 = np.maximum(image.science[i] + ffdark, 0.0)

        T = (t1 - t2) * image.invflat[i]

        # correct the ERR extension
        # NOTE: np.abs() in the err array recomputation is used for safety only
        #       and, in principle, assuming no errors have been made in
        #       the derivation of the formula, np.abs() should not be necessary.
        # NOTE: for debugging purposes, one may want to uncomment next line:
        #assert( np.all(image.err[i]**2 + truecorr_sig2[i] - T >= 0.0))
        image.err[i] = np.sqrt(np.abs(image.err[i]**2 + truecorr_sig2 - T)).astype(image.science.dtype)

    return True, NUpdRows, NMaxIter, STDDEVCorr, MaxCorr


def _write_row_number(lineno, offset=1, pad=1):
    if lineno is None:
        return ''
    return (pad*' ' + '(row #{:d})'.format(lineno+offset))


def djs_iterstat(InputArr, MaxIter=10, SigRej=3.0,
                 Max=None, Min=None, Mask=None, lineno=None):
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

    lineno : int or None
        Line number to be used in log and/or warning messages.

    Returns
    -------
    FMean, FSig, FMedian, NPix : float
        Mean, sigma, and median of final result.

    NIter : int
        Number of performed clipping iterations

    BMask : `numpy.ndarray`
        Logical image mask from the final iteration.

    """
    NGood    = InputArr.size
    ArrShape = InputArr.shape
    if NGood == 0:
        imrow =  _write_row_number(lineno=lineno, offset=0, pad=1)
        LOG.warn('djs_iterstat - No data points given' + imrow)
        return 0, 0, 0, 0, 0, None
    if NGood == 1:
        imrow =  _write_row_number(lineno=lineno, offset=0, pad=1)
        LOG.warn('djs_iterstat - Only one data point; cannot compute stats' + imrow)
        return 0, 0, 0, 0, 0, None
    if np.unique(InputArr).size == 1:
        imrow =  _write_row_number(lineno=lineno, offset=0, pad=1)
        LOG.warn('djs_iterstat - Only one value in data; cannot compute stats' + imrow)
        return 0, 0, 0, 0, 0, None

    # Determine Max and Min
    if Max is None:
        Max = InputArr.max()
    if Min is None:
        Min = InputArr.min()

    # Use all pixels if no mask is provided
    if Mask is None:
        Mask = np.ones(ArrShape, dtype=np.byte)
    else:
        Mask = Mask.copy()

    # Reject those above Max and those below Min
    Mask[InputArr > Max] = 0
    Mask[InputArr < Min] = 0

    FMean = np.sum(1.0 * InputArr * Mask) / NGood
    FSig  = np.sqrt(np.sum((1.0 * InputArr - FMean)**2 * Mask) / (NGood - 1))

    NLast = -1
    Iter  =  0
    NGood = np.sum(Mask)
    if NGood < 2:
        imrow =  _write_row_number(lineno=lineno, offset=0, pad=1)
        LOG.warn('djs_iterstat - No good data points; cannot compute stats' + imrow)
        return 0, 0, 0, 0, 0, None

    SaveMask = Mask.copy()
    if Iter >= MaxIter: # to support MaxIter=0
        NLast = NGood

    while (Iter < MaxIter) and (NLast != NGood) and (NGood >= 2):
        LoVal = FMean - SigRej * FSig
        HiVal = FMean + SigRej * FSig

        Mask[InputArr < LoVal] = 0
        Mask[InputArr > HiVal] = 0
        NLast = NGood
        npix = np.sum(Mask)

        if npix >= 2:
            FMean = np.sum(1.0 * InputArr * Mask) / npix
            FSig  = np.sqrt(np.sum(
                (1.0 * InputArr - FMean)**2 * Mask) / (npix - 1))
            SaveMask = Mask.copy() # last mask used for computation of mean
            NGood = npix
            Iter += 1
        else:
            break

    logical_mask = SaveMask.astype(np.bool)

    if NLast > 1:
        FMedian = np.median(InputArr[logical_mask])
    else:
        FMedian = FMean

    return FMean, FSig, FMedian, NLast, Iter, logical_mask


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
          dqbits=configobj['dqbits'],
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
        '--dqbits', nargs='?', type=str, default=None,
        help='DQ bits to be considered "good".')
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
          sigrej=args.sigrej, mask1=mask1, mask2=mask2, dqbits=args.dqbits)


if __name__ == '__main__':
    main()
