"""General utilities for ACS calibration adapted from HSTCAL."""

# STDLIB
import os
import warnings

# THIRD-PARTY
import numpy as np
from astropy.table import Table
from astropy.utils.exceptions import AstropyUserWarning

try:
    # This supports PIXVALUE
    from stsci.tools import stpyfits as fits
except ImportError:
    # Falls back to Astropy
    from astropy.io import fits

__all__ = ['extract_dark', 'extract_flash', 'extract_flatfield',
           'from_irafpath', 'extract_ref', 'find_line', 'get_corner', 'get_lt',
           'from_lt', 'hdr_vals_for_overscan', 'check_oscntab',
           'check_overscan', 'SM4_MJD']

# The MJD date of the EVA during SM4 to restore ACS/WFC and ACS/HRC.
# This value is also defined in header file, acs.h, for use by calacs.e in hstcal
SM4_MJD = 54967.0


def extract_dark(prihdr, scihdu):
    """Extract superdark data from ``DARKFILE`` or ``DRKCFILE``.

    Parameters
    ----------
    prihdr : obj
        FITS primary header HDU.

    scihdu : obj
        Extension HDU of the science image.
        This is only used to extract subarray data.

    Returns
    -------
    dark : ndarray or `None`
        Superdark, if any. Subtract this to apply ``DARKCORR``.

    """
    if prihdr.get('PCTECORR', 'OMIT') == 'COMPLETE':
        darkfile = prihdr.get('DRKCFILE', 'N/A')
    else:
        darkfile = prihdr.get('DARKFILE', 'N/A')

    if darkfile == 'N/A':
        return None

    darkfile = from_irafpath(darkfile)
    ampstring = prihdr['CCDAMP']

    # Calculate DARKTIME
    exptime = prihdr.get('EXPTIME', 0.0)
    flashdur = prihdr.get('FLASHDUR', 0.0)
    darktime = exptime + flashdur
    if exptime > 0:  # Not BIAS
        darktime += 3.0

    with fits.open(darkfile) as hdudark:
        if ampstring == 'ABCD':
            dark = np.concatenate(
                (hdudark['sci', 1].data,
                 hdudark['sci', 2].data[::-1, :]), axis=1)
        elif ampstring in ('A', 'B', 'AB'):
            dark = extract_ref(scihdu, hdudark['sci', 2])
        else:
            dark = extract_ref(scihdu, hdudark['sci', 1])

    dark = dark * darktime

    return dark


def extract_flash(prihdr, scihdu):
    """Extract postflash data from ``FLSHFILE``.

    Parameters
    ----------
    prihdr : obj
        FITS primary header HDU.

    scihdu : obj
        Extension HDU of the science image.
        This is only used to extract subarray data.

    Returns
    -------
    flash : ndarray or `None`
        Postflash, if any. Subtract this to apply ``FLSHCORR``.

    """
    flshfile = prihdr.get('FLSHFILE', 'N/A')
    flashsta = prihdr.get('FLASHSTA', 'N/A')
    flashdur = prihdr.get('FLASHDUR', 0.0)

    if flshfile == 'N/A' or flashdur <= 0:
        return None

    if flashsta != 'SUCCESSFUL':
        warnings.warn(f'Flash status is {flashsta}', AstropyUserWarning)

    flshfile = from_irafpath(flshfile)
    ampstring = prihdr['CCDAMP']

    with fits.open(flshfile) as hduflash:
        if ampstring == 'ABCD':
            flash = np.concatenate(
                (hduflash['sci', 1].data,
                 hduflash['sci', 2].data[::-1, :]), axis=1)
        elif ampstring in ('A', 'B', 'AB'):
            flash = extract_ref(scihdu, hduflash['sci', 2])
        else:
            flash = extract_ref(scihdu, hduflash['sci', 1])

    flash = flash * flashdur

    return flash


def extract_flatfield(prihdr, scihdu):
    """Extract flatfield data from ``PFLTFILE``.

    Parameters
    ----------
    prihdr : obj
        FITS primary header HDU.

    scihdu : obj
        Extension HDU of the science image.
        This is only used to extract subarray data.

    Returns
    -------
    invflat : ndarray or `None`
        Inverse flatfield, if any. Multiply this to apply ``FLATCORR``.

    """
    for ff in ['DFLTFILE', 'LFLTFILE']:
        vv = prihdr.get(ff, 'N/A')
        if vv != 'N/A':
            warnings.warn(f'{ff}={vv} is not accounted for',
                          AstropyUserWarning)

    flatfile = prihdr.get('PFLTFILE', 'N/A')

    if flatfile == 'N/A':
        return None

    flatfile = from_irafpath(flatfile)
    ampstring = prihdr['CCDAMP']

    with fits.open(flatfile) as hduflat:
        if ampstring == 'ABCD':
            invflat = np.concatenate(
                (1 / hduflat['sci', 1].data,
                 1 / hduflat['sci', 2].data[::-1, :]), axis=1)
        elif ampstring in ('A', 'B', 'AB'):
            invflat = 1 / extract_ref(scihdu, hduflat['sci', 2])
        else:
            invflat = 1 / extract_ref(scihdu, hduflat['sci', 1])

    return invflat


def from_irafpath(irafpath):
    """Resolve IRAF path like ``jref$`` into actual file path.

    Parameters
    ----------
    irafpath : str
        Path containing IRAF syntax.

    Returns
    -------
    realpath : str
        Actual file path. If input does not follow ``path$filename``
        format, then this is the same as input.

    Raises
    ------
    ValueError
        The required environment variable is undefined.

    """
    s = irafpath.split('$')

    if len(s) != 2:
        return irafpath
    if len(s[0]) == 0:
        return irafpath

    try:
        refdir = os.environ[s[0]]
    except KeyError:
        raise ValueError(f'{s[0]} environment variable undefined')

    return os.path.join(refdir, s[1])


def extract_ref(scihdu, refhdu):
    """Extract section of the reference image that
    corresponds to the given science image.

    This only returns a view, not a copy of the
    reference image's array.

    Parameters
    ----------
    scihdu, refhdu : obj
        Extension HDU's of the science and reference image,
        respectively.

    Returns
    -------
    refdata : array-like
        Section of the relevant reference image.

    Raises
    ------
    NotImplementedError
        Either science or reference data are binned.

    ValueError
        Extracted section size mismatch.

    """
    same_size, rx, ry, x0, y0 = find_line(scihdu, refhdu)

    # Use the whole reference image
    if same_size:
        return refhdu.data

    # Binned data
    if rx != 1 or ry != 1:
        raise NotImplementedError(
            'Either science or reference data are binned')

    # Extract a view of the sub-section
    ny, nx = scihdu.data.shape
    refdata = refhdu.data[y0:y0+ny, x0:x0+nx]

    if refdata.shape != (ny, nx):
        raise ValueError(f'Extracted reference image is {refdata.shape} but '
                         f'science image is ({ny}, {nx})')

    return refdata


def find_line(scihdu, refhdu):
    """Obtain bin factors and corner location to extract
    and bin the appropriate subset of a reference image to
    match a science image.

    If the science image has zero offset and is the same size and
    binning as the reference image, ``same_size`` will be set to
    `True`. Otherwise, the values of ``rx``, ``ry``, ``x0``, and
    ``y0`` will be assigned.

    Normally the science image will be binned the same or more
    than the reference image. In that case, ``rx`` and ``ry``
    will be the bin size of the science image divided by the
    bin size of the reference image.

    If the binning of the reference image is greater than the
    binning of the science image, the ratios (``rx`` and ``ry``)
    of the bin sizes will be the reference image size divided by
    the science image bin size. This is not necessarily an error.

    .. note:: Translated from ``calacs/lib/findbin.c``.

    Parameters
    ----------
    scihdu, refhdu : obj
        Extension HDU's of the science and reference image,
        respectively.

    Returns
    -------
    same_size : bool
        `True` if zero offset and same size and binning.

    rx, ry : int
        Ratio of bin sizes.

    x0, y0 : int
        Location of start of subimage in reference image.

    Raises
    ------
    ValueError
        Science and reference data size mismatch.

    """
    sci_bin, sci_corner = get_corner(scihdu.header)
    ref_bin, ref_corner = get_corner(refhdu.header)

    # We can use the reference image directly, without binning
    # and without extracting a subset.
    if (sci_corner[0] == ref_corner[0] and sci_corner[1] == ref_corner[1] and
            sci_bin[0] == ref_bin[0] and sci_bin[1] == ref_bin[1] and
            scihdu.data.shape[1] == refhdu.data.shape[1]):
        same_size = True
        rx = 1
        ry = 1
        x0 = 0
        y0 = 0

    # Reference image is binned more than the science image.
    elif ref_bin[0] > sci_bin[0] or ref_bin[1] > sci_bin[1]:
        same_size = False
        rx = ref_bin[0] / sci_bin[0]
        ry = ref_bin[1] / sci_bin[1]
        x0 = (sci_corner[0] - ref_corner[0]) / ref_bin[0]
        y0 = (sci_corner[1] - ref_corner[1]) / ref_bin[1]

    # For subarray input images, whether they are binned or not.
    else:
        same_size = False

        # Ratio of bin sizes.
        ratiox = sci_bin[0] / ref_bin[0]
        ratioy = sci_bin[1] / ref_bin[1]

        if (ratiox * ref_bin[0] != sci_bin[0] or
                ratioy * ref_bin[1] != sci_bin[1]):
            raise ValueError('Science and reference data size mismatch')

        # cshift is the offset in units of unbinned pixels.
        # Divide by ref_bin to convert to units of pixels in the ref image.
        cshift = (sci_corner[0] - ref_corner[0], sci_corner[1] - ref_corner[1])
        xzero = cshift[0] / ref_bin[0]
        yzero = cshift[1] / ref_bin[1]

        if (xzero * ref_bin[0] != cshift[0] or
                yzero * ref_bin[1] != cshift[1]):
            warnings.warn('Subimage offset not divisible by bin size',
                          AstropyUserWarning)

        rx = ratiox
        ry = ratioy
        x0 = xzero
        y0 = yzero

    # Ensure integer index
    x0 = int(x0)
    y0 = int(y0)

    return same_size, rx, ry, x0, y0


def get_corner(hdr, rsize=1):
    """Obtain bin and corner information for a subarray.

    ``LTV1``, ``LTV2``, ``LTM1_1``, and ``LTM2_2`` keywords
    are extracted from the given extension header and converted
    to bin and corner values (0-indexed).

    ``LTV1`` for the CCD uses the beginning of the illuminated
    portion as the origin, not the beginning of the overscan region.
    Thus, the computed X-corner has the same origin as ``LTV1``,
    which is what we want, but it differs from the ``CENTERA1``
    header keyword, which has the beginning of the overscan region
    as origin.

    .. note:: Translated from ``calacs/lib/getcorner.c``.

    Parameters
    ----------
    hdr : obj
        Extension header.

    rsize : int, optional
        Size of reference pixel in units of high-res pixels.

    Returns
    -------
    bin : tuple of int
        Pixel size in X and Y.

    corner : tuple of int
        Corner of subarray in X and Y.

    """
    ltm, ltv = get_lt(hdr)
    return from_lt(rsize, ltm, ltv)


def get_lt(hdr):
    """Obtain the LTV and LTM keyword values.

    Note that this returns the values just as read from the header,
    which means in particular that the LTV values are for one-indexed
    pixel coordinates.

    LTM keywords are the diagonal elements of MWCS linear
    transformation matrix, while LTV's are MWCS linear transformation
    vector (1-indexed).

    .. note:: Translated from ``calacs/lib/getlt.c``.

    Parameters
    ----------
    hdr : obj
        Extension header.

    Returns
    -------
    ltm, ltv : tuple of float
        ``(LTM1_1, LTM2_2)`` and ``(LTV1, LTV2)``.
        Values are ``(1, 1)`` and ``(0, 0)`` if not found,
        to accomodate reference files with missing info.

    Raises
    ------
    ValueError
        Invalid LTM* values.

    """
    ltm = (hdr.get('LTM1_1', 1.0), hdr.get('LTM2_2', 1.0))

    if ltm[0] <= 0 or ltm[1] <= 0:
        raise ValueError(f'(LTM1_1, LTM2_2) = {ltm} is invalid')

    ltv = (hdr.get('LTV1', 0.0), hdr.get('LTV2', 0.0))
    return ltm, ltv


def from_lt(rsize, ltm, ltv):
    """Compute the corner location and pixel size in units
    of unbinned pixels.

    .. note:: Translated from ``calacs/lib/fromlt.c``.

    Parameters
    ----------
    rsize : int
        Reference pixel size. Usually 1.

    ltm, ltv : tuple of float
        See :func:`get_lt`.

    Returns
    -------
    bin : tuple of int
        Pixel size in X and Y.

    corner : tuple of int
        Corner of subarray in X and Y.

    """
    dbinx = rsize / ltm[0]
    dbiny = rsize / ltm[1]

    dxcorner = (dbinx - rsize) - dbinx * ltv[0]
    dycorner = (dbiny - rsize) - dbiny * ltv[1]

    # Round off to the nearest integer.
    bin = (_nint(dbinx), _nint(dbiny))
    corner = (_nint(dxcorner), _nint(dycorner))

    return bin, corner


def _nint(x):
    """Integer casting used in :func:`from_lt`."""
    if x >= 0:
        y = x + 0.5
    else:
        y = x - 0.5
    return int(y)


def hdr_vals_for_overscan(root):
    """Retrieve header keyword values from RAW and SPT
    FITS files to pass on to :func:`check_oscntab` and
    :func:`check_overscan`.

    Parameters
    ----------
    root : str
        Rootname of the observation. Can be relative path
        to the file excluding the type of FITS file and
        extension, e.g., '/my/path/jxxxxxxxq'.

    Returns
    -------
    ccdamp : str
        Amplifiers used to read out the CCDs.

    xstart : int
        Starting column of the readout in detector
        coordinates.

    ystart : int
        Starting row of the readout in detector
        coordinates.

    xsize : int
        Number of columns in the readout.

    ysize : int
        Number of rows in the readout.

    """
    with fits.open(root + '_spt.fits') as hdu:
        spthdr = hdu[0].header
    with fits.open(root + '_raw.fits') as hdu:
        prihdr = hdu[0].header
    xstart = spthdr['SS_A1CRN']
    ystart = spthdr['SS_A2CRN']
    xsize = spthdr['SS_A1SZE']
    ysize = spthdr['SS_A2SZE']
    ccdamp = prihdr['CCDAMP']

    return ccdamp, xstart, ystart, xsize, ysize


def check_oscntab(oscntab, ccdamp, xsize, ysize, leading, trailing):
    """Check if the supplied parameters are in the
    ``OSCNTAB`` reference file.

    .. note:: Even if an entry does not exist in ``OSCNTAB``,
              as long as the subarray does not have any overscan,
              it should not be a problem for CALACS.

    .. note:: This function does not check the virtual bias rows.

    Parameters
    ----------
    oscntab : str
        Path to the ``OSCNTAB`` reference file being checked against.

    ccdamp : str
        Amplifier(s) used to read out the CCDs.

    xsize : int
        Number of columns in the readout.

    ysize : int
        Number of rows in the readout.

    leading : int
        Number of columns in the bias section ("TRIMX1" to be trimmed off
        by ``BLEVCORR``) on the A/C amplifiers side of the CCDs.

    trailing : int
        Number of columns in the bias section ("TRIMX2" to be trimmed off
        by ``BLEVCORR``) on the B/D amplifiers side of the CCDs.

    Returns
    -------
    supported : bool
        Result of test if input parameters are in ``OSCNTAB``.

    """
    tab = Table.read(oscntab)
    ccdamp = ccdamp.lower().rstrip()
    for row in tab:
        if (row['CCDAMP'].lower().rstrip() in ccdamp and
                row['NX'] == xsize and row['NY'] == ysize and
                row['TRIMX1'] == leading and row['TRIMX2'] == trailing):
            return True
    return False


def check_overscan(xstart, xsize, total_prescan_pixels=24,
                   total_science_pixels=4096):
    """Check image for bias columns.

    Parameters
    ----------
    xstart : int
        Starting column of the readout in detector coordinates.

    xsize : int
        Number of columns in the readout.

    total_prescan_pixels : int
        Total prescan pixels for a single amplifier on a detector.
        Default is 24 for WFC.

    total_science_pixels : int
        Total science pixels across a detector.
        Default is 4096 for WFC (across two amplifiers).

    Returns
    -------
    hasoverscan : bool
        Indication if there are bias columns in the image.

    leading : int
        Number of bias columns on the A/C amplifiers
        side of the CCDs ("TRIMX1" in ``OSCNTAB``).

    trailing : int
        Number of bias columns on the B/D amplifiers
        side of the CCDs ("TRIMX2" in ``OSCNTAB``).

    """
    hasoverscan = False
    leading = 0
    trailing = 0

    if xstart < total_prescan_pixels:
        hasoverscan = True
        leading = abs(xstart - total_prescan_pixels)

    if (xstart + xsize) > total_science_pixels:
        hasoverscan = True
        trailing = abs(total_science_pixels -
                       (xstart + xsize - total_prescan_pixels))

    return hasoverscan, leading, trailing
