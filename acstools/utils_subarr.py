"""General utilities for ACS subarrays."""
from __future__ import absolute_import, division, print_function

# STDLIB
import warnings

# THIRD-PARTY
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['extract_ref', 'find_line', 'get_corner', 'get_lt', 'from_lt']


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
        raise ValueError('Extracted reference image is {0} but science image '
                         'is {1}'.format(refdata.shape, (ny, nx)))

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
        Values are `None` if not found.

    Raises
    ------
    ValueError
        Invalid LTM* values.

    """
    ltm = (hdr.get('LTM1_1'), hdr.get('LTM2_2'))

    if ltm[0] <= 0 or ltm[1] <= 0:
        raise ValueError('(LTM1_1, LTM2_2) = {0} is invalid'.format(ltm))

    ltv = (hdr.get('LTV1'), hdr.get('LTV2'))
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
