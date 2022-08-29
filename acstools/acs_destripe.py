#!/usr/bin/env python
"""
Remove horizontal stripes from ACS WFC post-SM4 data.

For more information, see
`ACS ISR 2011-05 <https://ui.adsabs.harvard.edu/abs/2011acs..rept....5G/abstract>`_.

.. note::

    * Does not work on RAW images.

    * Uses the flatfield specified by the image header keyword PFLTFILE.
      If keyword value is 'N/A', as is the case with biases and darks,
      then unity flatfield is used.

    * Uses post-flash image specified by the image header keyword FLSHFILE.
      If keyword value is 'N/A', then dummy post-flash with zeroes is used.

    * Uses the dark image specified by the image header keyword DARKFILE.
      If keyword value is 'N/A', then dummy dark with zeroes is used.

Examples
--------

In Python:

>>> from acstools import acs_destripe
>>> acs_destripe.clean('uncorrected_flt.fits', 'csck',
...                    mask1='mymask_sci1.fits', mask2='mymask_sci2.fits',
...                    clobber=False, maxiter=15, sigrej=2.0)

From command line::

    % acs_destripe [-h] [--stat STAT] [--lower [LOWER]] [--upper [UPPER]]
                   [--binwidth BINWIDTH] [--mask1 MASK1] [--mask2 MASK2]
                   [--dqbits [DQBITS]] [--rpt_clean RPT_CLEAN]
                   [--atol [ATOL]] [-c] [-q] [--version]
                   input suffix [maxiter] [sigrej]

"""
# STDLIB
import logging

# THIRD-PARTY
import numpy as np

try:
    # This supports PIXVALUE
    from stsci.tools import stpyfits as fits
except ImportError:
    # Falls back to Astropy
    from astropy.io import fits

# LOCAL
from .utils_calib import extract_dark, extract_flash, extract_flatfield, SM4_MJD

__taskname__ = 'acs_destripe'
__version__ = '0.8.2'
__vdate__ = '22-Sep-2016'
__author__ = 'Norman Grogin, STScI, March 2012.'
__all__ = ['clean']

#
# HISTORY:
# .........
# 12MAR2015 (v0.6.3) Cara added capability to use DQ mask;
#           added support for multiple input files and wildcards in the file
#           names. See Ticket #1178.
# 23MAR2015 (v0.7.1) Cara added weighted (by NPix) background computations
#           (especially important for vigneted filters). See Ticket #1180.
# 31MAR2015 (v0.8.0) Cara added repeated de-stripe iterations (to improve
#           corrections in the "RAW" space) and support for various
#           statistics modes. See Ticket #1183.


logging.basicConfig()
LOG = logging.getLogger(__taskname__)
LOG.setLevel(logging.INFO)


class StripeArray:
    """Class to handle data array to be destriped."""

    def __init__(self, image):
        self.hdulist = fits.open(image)
        self.ampstring = self.hdulist[0].header['CCDAMP']
        self.flatcorr = self.hdulist[0].header['FLATCORR']
        self.flshcorr = self.hdulist[0].header['FLSHCORR']
        self.darkcorr = self.hdulist[0].header['DARKCORR']
        self.configure_arrays()

    def configure_arrays(self):
        """Get the SCI and ERR data."""
        self.science = self.hdulist['sci', 1].data
        self.err = self.hdulist['err', 1].data
        self.dq = self.hdulist['dq', 1].data
        if (self.ampstring == 'ABCD'):
            self.science = np.concatenate(
                (self.science, self.hdulist['sci', 2].data[::-1, :]), axis=1)
            self.err = np.concatenate(
                (self.err, self.hdulist['err', 2].data[::-1, :]), axis=1)
            self.dq = np.concatenate(
                (self.dq, self.hdulist['dq', 2].data[::-1, :]), axis=1)
        self.ingest_dark()
        self.ingest_flash()
        self.ingest_flatfield()

    def ingest_flatfield(self):
        """Process flatfield."""

        self.invflat = extract_flatfield(
            self.hdulist[0].header, self.hdulist[1])

        # If BIAS or DARK, set flatfield to unity
        if self.invflat is None:
            self.invflat = np.ones_like(self.science)
            return

        # Apply the flatfield if necessary
        if self.flatcorr != 'COMPLETE':
            self.science = self.science * self.invflat
            self.err = self.err * self.invflat

    def ingest_flash(self):
        """Process post-flash."""

        self.flash = extract_flash(self.hdulist[0].header, self.hdulist[1])

        # Set post-flash to zeros
        if self.flash is None:
            self.flash = np.zeros_like(self.science)
            return

        # Apply the flash subtraction if necessary.
        # Not applied to ERR, to be consistent with ingest_dark()
        if self.flshcorr != 'COMPLETE':
            self.science = self.science - self.flash

    def ingest_dark(self):
        """Process dark."""

        self.dark = extract_dark(self.hdulist[0].header, self.hdulist[1])

        # If BIAS or DARK, set dark to zeros
        if self.dark is None:
            self.dark = np.zeros_like(self.science)
            return

        # Apply the dark subtraction if necessary.
        # Effect of DARK on ERR is insignificant for de-striping.
        if self.darkcorr != 'COMPLETE':
            self.science = self.science - self.dark

    def write_corrected(self, output, clobber=False):
        """Write out the destriped data."""

        # un-apply the flatfield if necessary
        if self.flatcorr != 'COMPLETE':
            self.science = self.science / self.invflat
            self.err = self.err / self.invflat

        # un-apply the post-flash if necessary
        if self.flshcorr != 'COMPLETE':
            self.science = self.science + self.flash

        # un-apply the dark if necessary
        if self.darkcorr != 'COMPLETE':
            self.science = self.science + self.dark

        # reverse the amp merge
        if (self.ampstring == 'ABCD'):
            tmp_1, tmp_2 = np.split(self.science, 2, axis=1)
            self.hdulist['sci', 1].data = tmp_1.copy()
            self.hdulist['sci', 2].data = tmp_2[::-1, :].copy()

            tmp_1, tmp_2 = np.split(self.err, 2, axis=1)
            self.hdulist['err', 1].data = tmp_1.copy()
            self.hdulist['err', 2].data = tmp_2[::-1, :].copy()
        else:
            self.hdulist['sci', 1].data = self.science.copy()
            self.hdulist['err', 1].data = self.err.copy()

        # Write the output
        self.hdulist.writeto(output, overwrite=clobber)

    def close(self):
        """Close open file(s)."""
        self.hdulist.close()


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
        mask = np.concatenate((dat1, dat2[::-1, :]), axis=1)

    # Mask must only have binary values
    if mask is not None:
        mask[mask != 0] = 1

    return mask


def clean(input, suffix, stat="pmode1", maxiter=15, sigrej=2.0,
          lower=None, upper=None, binwidth=0.3,
          mask1=None, mask2=None, dqbits=None,
          rpt_clean=0, atol=0.01, clobber=False, verbose=True):
    r"""Remove horizontal stripes from ACS WFC post-SM4 data.

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
          pixels based on an algorithm similar to IRAF's ``imagestats`` task
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

    rpt_clean : int
        An integer indicating how many *additional* times stripe cleaning
        should be performed on the input image. Default = 0.

    atol : float, None
        The threshold for maximum absolute value of bias stripe correction
        below which repeated cleanings can stop. When `atol` is `None`
        cleaning will be repeated `rpt_clean` number of times.
        Default = 0.01 [e].

    verbose : bool
        Print informational messages. Default = True.

    """
    from stsci.tools import parseinput  # Optional package dependency

    flist = parseinput.parseinput(input)[0]

    if isinstance(mask1, str):
        mlist1 = parseinput.parseinput(mask1)[0]
    elif isinstance(mask1, np.ndarray):
        mlist1 = [mask1.copy()]
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
        mlist2 = [mask2.copy()]
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
        raise ValueError("No input file(s) provided or "
                         "the file(s) do not exist")

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
        if (fits.getval(image, 'EXPSTART') <= SM4_MJD):
            LOG.warning(f'{image} is pre-SM4. Skipping...')
            continue

        # Data must be in ELECTRONS
        if (fits.getval(image, 'BUNIT', ext=1) != 'ELECTRONS'):
            LOG.warning(f'{image} is not in ELECTRONS. Skipping...')
            continue

        # Skip processing CTECORR-ed images
        if (fits.getval(image, 'PCTECORR') == 'COMPLETE'):
            LOG.warning(f'{image} already has PCTECORR applied. Skipping...')
            continue

        # generate output filename for each input based on specification
        # of the output suffix
        output = image.replace('.fits', '_' + suffix + '.fits')
        LOG.info('Processing ' + image)

        # verify masks defined (or not) simultaneously:
        if (fits.getval(image, 'CCDAMP') == 'ABCD' and
                ((mask1 is not None and mask2 is None) or
                 (mask1 is None and mask2 is not None))):
            raise ValueError("Both 'mask1' and 'mask2' must be specified "
                             "or not specified together.")

        maskdata = _read_mask(maskfile1, maskfile2)
        perform_correction(image, output, stat=stat, maxiter=maxiter,
                           sigrej=sigrej, lower=lower, upper=upper,
                           binwidth=binwidth, mask=maskdata, dqbits=dqbits,
                           rpt_clean=rpt_clean, atol=atol,
                           clobber=clobber, verbose=verbose)
        LOG.info(output + ' created')


def perform_correction(image, output, stat="pmode1", maxiter=15, sigrej=2.0,
                       lower=None, upper=None, binwidth=0.3,
                       mask=None, dqbits=None,
                       rpt_clean=0, atol=0.01, clobber=False, verbose=True):
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

    rpt_clean : int
        An integer indicating how many *additional* times stripe cleaning
        should be performed on the input image. Default = 0.

    atol : float, None
        The threshold for maximum absolute value of bias stripe correction
        below which repeated cleanings can stop. When `atol` is `None`
        cleaning will be repeated `rpt_clean` number of times.
        Default = 0.01 [e].

    verbose : bool
        Print informational messages. Default = True.

    """
    # construct the frame to be cleaned, including the
    # associated data stuctures needed for cleaning
    frame = StripeArray(image)

    # combine user mask with image's DQ array:
    mask = _mergeUserMaskAndDQ(frame.dq, mask, dqbits)

    # Do the stripe cleaning
    Success, NUpdRows, NMaxIter, Bkgrnd, STDDEVCorr, MaxCorr, Nrpt = clean_streak(
        frame, stat=stat, maxiter=maxiter, sigrej=sigrej,
        lower=lower, upper=upper, binwidth=binwidth, mask=mask,
        rpt_clean=rpt_clean, atol=atol, verbose=verbose
    )

    if Success:
        if verbose:
            LOG.info('perform_correction - =====  Overall statistics for '
                     'de-stripe corrections:  =====')

        if (STDDEVCorr > 1.5*0.9):
            LOG.warning('perform_correction - STDDEV of applied de-stripe '
                        f'corrections ({STDDEVCorr:.3g}) exceeds\n'
                        'known bias striping STDDEV of 0.9e '
                        '(see ISR ACS 2011-05) more than 1.5 times.')

        elif verbose:
            LOG.info('perform_correction - STDDEV of applied de-stripe '
                     f'corrections {STDDEVCorr:.3g}.')

        if verbose:
            LOG.info('perform_correction - Estimated background: '
                     f'{Bkgrnd:.5g}.')
            LOG.info('perform_correction - Maximum applied correction: '
                     f'{MaxCorr:.3g}.')
            LOG.info('perform_correction - Effective number of clipping '
                     f'iterations: {NMaxIter}.')
            LOG.info('perform_correction - Effective number of additional '
                     f'(repeated) cleanings: {Nrpt}.')
            LOG.info('perform_correction - Total number of corrected rows: '
                     f'{NUpdRows}.')

    frame.write_corrected(output, clobber=clobber)
    frame.close()


def _mergeUserMaskAndDQ(dq, mask, dqbits):
    from astropy.nddata.bitmask import (interpret_bit_flags,
                                        bitfield_to_boolean_mask)

    dqbits = interpret_bit_flags(dqbits)
    if dqbits is None:
        if mask is None:
            return None
        else:
            return mask.copy().astype(dtype=np.uint8)

    if dq is None:
        raise ValueError("DQ array is None while 'dqbits' is not None.")

    dqmask = bitfield_to_boolean_mask(dq, dqbits, good_mask_value=1,
                                      dtype=np.uint8)

    if mask is None:
        return dqmask

    # merge user mask with DQ mask:
    dqmask[mask == 0] = 0

    return dqmask


def clean_streak(image, stat="pmode1", maxiter=15, sigrej=2.0,
                 lower=None, upper=None, binwidth=0.3, mask=None,
                 rpt_clean=0, atol=0.01, verbose=True):
    """
    Apply destriping algorithm to input array.

    Parameters
    ----------
    image : `StripeArray` object
        Arrays are modifed in-place.

    stat : str
        Statistics for background computations
        (see :py:func:`clean` for more details)

    mask : `numpy.ndarray`
        Mask array. Pixels with zero values are masked out.

    maxiter, sigrej : see `clean`

    rpt_clean : int
        An integer indicating how many *additional* times stripe cleaning
        should be performed on the input image. Default = 0.

    atol : float, None
        The threshold for maximum absolute value of bias stripe correction
        below which repeated cleanings can stop. When `atol` is `None`
        cleaning will be repeated `rpt_clean` number of times.
        Default = 0.01 [e].

    verbose : bool
        Print informational messages. Default = True.

    Returns
    -------
    Success : bool
        Indicates successful execution.

    NUpdRows : int
        Number of updated rows in the image.

    NMaxIter : int
        Maximum number of clipping iterations performed on image rows.

    Bkgrnd, STDDEVCorr, MaxCorr : float
        Background, standard deviation of corrections and maximum correction
        applied to the non-flat-field-corrected (i.e., RAW) image rows.

    Nrpt : int
        Number of *additional* (performed *after* initial run) cleanings.

    """
    # Optional package dependency
    try:
        from stsci.imagestats import ImageStats
    except ImportError:
        ImageStats = None

    if mask is not None and image.science.shape != mask.shape:
        raise ValueError('Mask shape does not match science data shape')

    Nrpt = 0
    warn_maxiter = False
    NUpdRows = 0
    NMaxIter = 0
    STDDEVCorr = 0.0
    MaxCorr = 0.0
    wmean = 0.0

    stat = stat.lower().strip()
    if stat not in ['pmode1', 'pmode2', 'mean', 'mode', 'median', 'midpt']:
        raise ValueError("Unsupported value for 'stat'.")

    # array to hold the stripe amplitudes
    corr = np.empty(image.science.shape[0], dtype=np.float64)

    # array to hold cumulative stripe amplitudes and latest row npix:
    cumcorr = np.zeros(image.science.shape[0], dtype=np.float64)
    cnpix = np.zeros(image.science.shape[0], dtype=int)

    # other arrays
    corr_scale = np.empty(image.science.shape[0], dtype=np.float64)
    npix = np.empty(image.science.shape[0], dtype=int)
    sigcorr2 = np.zeros(image.science.shape[0], dtype=np.float64)
    updrows = np.zeros(image.science.shape[0], dtype=int)

    # for speed-up and to reduce rounding errors in ERR computations,
    # keep a copy of the squared error array:
    imerr2 = image.err**2

    # arrays for detecting oscillatory behaviour:
    nonconvi0 = np.arange(image.science.shape[0])
    corr0 = np.zeros(image.science.shape[0], dtype=np.float64)

    if stat == 'pmode1':
        # SExtractor-esque central value statistic; slightly sturdier against
        # skewness of pixel histogram due to faint source flux
        def getcorr(): return (2.5 * SMedian - 1.5 * SMean)

    elif stat == 'pmode2':
        # "Original Pearson"-ian estimate for mode:
        def getcorr(): return (3.0 * SMedian - 2.0 * SMean)

    elif stat == 'mean':
        def getcorr(): return (SMean)

    elif stat == 'median':
        def getcorr(): return (SMedian)

    elif stat == 'mode':
        if ImageStats is None:
            raise ImportError('stsci.imagestats is missing')

        def getcorr():
            imstat = ImageStats(image.science[i][BMask], 'mode',
                                lower=lower, upper=upper, nclip=0)
            if imstat.npix != NPix:
                raise ValueError(f'imstate.npix ({imstat.npix}) != '
                                 f'NPix ({NPix})')
            return (imstat.mode)

    elif stat == 'midpt':
        if ImageStats is None:
            raise ImportError('stsci.imagestats is missing')

        def getcorr():
            imstat = ImageStats(image.science[i][BMask], 'midpt',
                                lower=lower, upper=upper, nclip=0)
            if imstat.npix != NPix:
                raise ValueError(f'imstate.npix ({imstat.npix}) != '
                                 f'NPix ({NPix})')
            return (imstat.midpt)

    nmax_rpt = 1 if rpt_clean is None else max(1, rpt_clean+1)

    for rpt in range(nmax_rpt):
        Nrpt += 1

        if verbose:
            if Nrpt <= 1:
                if nmax_rpt > 1:
                    LOG.info("clean_streak - Performing initial image bias "
                             "de-stripe:")
                else:
                    LOG.info("clean_streak - Performing image bias de-stripe:")
            else:
                LOG.info("clean_streak - Performing repeated image bias "
                         f"de-stripe #{Nrpt - 1}:")

        # reset accumulators and arrays:
        corr[:] = 0.0
        corr_scale[:] = 0.0
        npix[:] = 0

        tcorr = 0.0
        tnpix = 0
        tnpix2 = 0
        NMaxIter = 0

        # loop over rows to fit the stripe amplitude
        mask_arr = None
        for i in range(image.science.shape[0]):
            if mask is not None:
                mask_arr = mask[i]

            # row-by-row iterative sigma-clipped mean;
            # sigma, iters are adjustable
            SMean, SSig, SMedian, NPix, NIter, BMask = djs_iterstat(
                image.science[i], MaxIter=maxiter, SigRej=sigrej,
                Min=lower, Max=upper, Mask=mask_arr, lineno=i+1
            )

            if NPix > 0:
                corr[i] = getcorr()
                npix[i] = NPix
                corr_scale[i] = 1.0 / np.average(image.invflat[i][BMask])
                sigcorr2[i] = corr_scale[i]**2 * \
                    np.sum((image.err[i][BMask])**2)/NPix**2
                cnpix[i] = NPix
                tnpix += NPix
                tnpix2 += NPix*NPix
                tcorr += corr[i] * NPix

            if NIter > NMaxIter:
                NMaxIter = NIter

        if tnpix <= 0:
            LOG.warning('clean_streak - No good data points; cannot de-stripe.')
            return False, 0, 0, 0.0, 0.0, 0

        if NMaxIter >= maxiter:
            warn_maxiter = True

        # require that bias stripe corrections have zero mean:
        # 1. compute weighted background of the flat-fielded image:
        wmean = tcorr / tnpix
        Bkgrnd = wmean
        # 2. estimate corrections:
        corr[npix > 0] -= wmean

        # convert corrections to the "raw" space:
        corr *= corr_scale

        # weighted mean and max value for current corrections
        # to the *RAW* image:
        trim_npix = npix[npix > 0]
        trim_corr = corr[npix > 0]
        cwmean = np.sum(trim_npix * trim_corr) / tnpix
        current_max_corr = np.amax(np.abs(trim_corr - cwmean))
        wvar = np.sum(trim_npix * (trim_corr - cwmean) ** 2) / tnpix
        uwvar = wvar / (1.0 - float(tnpix2) / float(tnpix) ** 2)
        STDDEVCorr = np.sqrt(uwvar)

        # keep track of total corrections:
        cumcorr += corr

        # apply corrections row-by-row
        for i in range(image.science.shape[0]):
            if npix[i] < 1:
                continue
            updrows[i] = 1

            ffdark = (image.dark[i] + image.flash[i]) * image.invflat[i]
            t1 = np.maximum(image.science[i] + ffdark, 0.0)

            # stripe is constant along the row, before flatfielding;
            # afterwards it has the shape of the inverse flatfield
            truecorr = corr[i] * image.invflat[i]
            #truecorr_sig2 = sigcorr2[i] * image.invflat[i]**2  # DEBUG

            # correct the SCI extension
            image.science[i] -= truecorr

            t2 = np.maximum(image.science[i] + ffdark, 0.0)

            T = (t1 - t2) * image.invflat[i]

            # correct the ERR extension
            # NOTE: np.abs() in the err array recomputation is used for safety
            #       only and, in principle, assuming no errors have been made
            #       in the derivation of the formula, np.abs() should not be
            #       necessary.
            imerr2[i] -= T
            image.err[i] = np.sqrt(np.abs(imerr2[i]))
            # NOTE: for debugging purposes, one may want to uncomment
            #       next line:
            #assert( np.all(imerr2 >= 0.0))

        if atol is not None:
            if current_max_corr < atol:
                break

            # detect oscilatory non-convergence:
            nonconvi = np.nonzero(np.abs(corr) > atol)[0]
            nonconvi_int = np.intersect1d(nonconvi, nonconvi0)
            if (nonconvi.shape[0] == nonconvi0.shape[0] and
                    nonconvi.shape[0] == nonconvi_int.shape[0] and
                    np.all(corr0[nonconvi]*corr[nonconvi] < 0.0) and Nrpt > 1):
                LOG.warning("clean_streak - Repeat bias stripe cleaning\n"
                            "process appears to be oscillatory for "
                            f"{nonconvi.shape[0]:d} image "
                            "rows.\nTry to adjust 'sigrej', 'maxiter', and/or "
                            "'dqbits' parameters.\n"
                            "In addition,  consider using masks or adjust "
                            "existing masks.")
                break

            nonconvi0 = nonconvi.copy()
            corr0 = corr.copy()

        if verbose:
            if Nrpt <= 1:
                LOG.info("clean_streak - Image bias de-stripe: Done.")
            else:
                LOG.info(f"clean_streak - Repeated (#{Nrpt - 1}) image bias "
                         "de-stripe: Done.")

    if verbose and Nrpt > 1:
        LOG.info('clean_streak - =====  Repeated de-stripe "residual" '
                 'estimates:  =====')
        LOG.info('clean_streak - STDDEV of the last applied de-stripe '
                 f'corrections {STDDEVCorr:.3g}')
        LOG.info('clean_streak - Maximum of the last applied correction: '
                 f'{current_max_corr:.3g}.')

    # add (in quadratures) an error term associated with the accuracy of
    # bias stripe correction:
    truecorr_sig2 = ((sigcorr2 * (image.invflat ** 2).T).T).astype(image.err.dtype)  # noqa

    # update the ERR extension
    image.err[:, :] = np.sqrt(np.abs(imerr2 + truecorr_sig2))

    if warn_maxiter:
        LOG.warning(
            'clean_streak - Maximum number of clipping iterations '
            f'specified by the user ({maxiter}) has been reached.')

    # weighted mean, sample variance, and max value for
    # total (cummulative) corrections to the *RAW* image:
    trim_cnpix = cnpix[cnpix > 0]
    trim_cumcorr = cumcorr[cnpix > 0]
    tcnpix = np.sum(trim_cnpix)
    tcnpix2 = np.sum(trim_cnpix ** 2)
    cwmean = np.sum(trim_cnpix * trim_cumcorr) / tcnpix
    trim_cumcorr -= cwmean
    wvar = np.sum(trim_cnpix * trim_cumcorr ** 2) / tcnpix
    uwvar = wvar / (1.0 - float(tcnpix2) / float(tcnpix) ** 2)
    STDDEVCorr = np.sqrt(uwvar)
    MaxCorr = np.amax(np.abs(trim_cumcorr))

    NUpdRows = np.sum(updrows)

    return True, NUpdRows, NMaxIter, Bkgrnd, STDDEVCorr, MaxCorr, Nrpt-1


def _write_row_number(lineno, offset=1, pad=1):
    if lineno is None:
        return ''
    return (pad * ' ' + f'(row #{lineno + offset:d})')


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
    NGood = InputArr.size
    ArrShape = InputArr.shape
    if NGood == 0:
        imrow = _write_row_number(lineno=lineno, offset=0, pad=1)
        LOG.warning(f'djs_iterstat - No data points given{imrow}')
        return 0, 0, 0, 0, 0, None
    if NGood == 1:
        imrow = _write_row_number(lineno=lineno, offset=0, pad=1)
        LOG.warning('djs_iterstat - Only one data point; '
                    f'cannot compute stats{imrow}')
        return 0, 0, 0, 0, 0, None
    if np.unique(InputArr).size == 1:
        imrow = _write_row_number(lineno=lineno, offset=0, pad=1)
        LOG.warning('djs_iterstat - Only one value in data; '
                    f'cannot compute stats{imrow}')
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
    FSig  = np.sqrt(np.sum((1.0 * InputArr - FMean) ** 2 * Mask) / (NGood - 1))

    NLast = -1
    Iter  = 0
    NGood = np.sum(Mask)
    if NGood < 2:
        imrow = _write_row_number(lineno=lineno, offset=0, pad=1)
        LOG.warning('djs_iterstat - No good data points; '
                    f'cannot compute stats{imrow}')
        return 0, 0, 0, 0, 0, None

    SaveMask = Mask.copy()
    if Iter >= MaxIter:  # to support MaxIter=0
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
            FSig = np.sqrt(np.sum(
                (1.0 * InputArr - FMean) ** 2 * Mask) / (npix - 1))
            SaveMask = Mask.copy()  # last mask used for computation of mean
            NGood = npix
            Iter += 1
        else:
            break

    logical_mask = SaveMask.astype(bool)

    if NLast > 1:
        FMedian = np.median(InputArr[logical_mask])
        NLast = NGood
    else:
        FMedian = FMean

    return FMean, FSig, FMedian, NLast, Iter, logical_mask


# --------------------------- #
# Interfaces for command line #
# --------------------------- #

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
        '--stat', type=str, default='pmode1', help='Background statistics')
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
        '--mask1', nargs=1, type=str, default=None,
        help='Mask image for [SCI,1]')
    parser.add_argument(
        '--mask2', nargs=1, type=str, default=None,
        help='Mask image for [SCI,2]')
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
        '-c', '--clobber', action="store_true", help='Clobber output')
    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='Do not print informational messages')
    parser.add_argument(
        '--version', action="version",
        version=f'{__taskname__} v{__version__} ({__vdate__})')
    args = parser.parse_args()

    if args.mask1:
        mask1 = args.mask1[0]
    else:
        mask1 = args.mask1

    if args.mask2:
        mask2 = args.mask2[0]
    else:
        mask2 = args.mask2

    clean(args.arg0, args.arg1, stat=args.stat, maxiter=args.maxiter,
          sigrej=args.sigrej, lower=args.lower, upper=args.upper,
          binwidth=args.binwidth, mask1=mask1, mask2=mask2, dqbits=args.dqbits,
          rpt_clean=args.rpt_clean, atol=args.atol,
          clobber=args.clobber, verbose=not args.quiet)


if __name__ == '__main__':
    main()
