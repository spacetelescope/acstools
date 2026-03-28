import warnings

import numpy as np

cimport cython
cimport numpy as np

np.import_array()
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
cdef DTYPE_t d_nan = np.nan


# This is Cythonized version of radon in scikit-image but with:
#     circle=False
#     median=True
#     preserve_range=False
#     processes=1
#     print_calc_times=False
@cython.boundscheck(False)
@cython.wraparound(False)
def radon(
    np.ndarray[DTYPE_t, ndim=2] image,
    np.ndarray[DTYPE_t, ndim=1] theta,
    DTYPE_t fill_value=d_nan, bool return_length=False
):
    """
    Calculates the (median) radon transform of an image.

    This routine basically :func:`skimage.transform.radon` but
    customized for MRT.
    For further information see [1]_ and [2]_.

    Parameters
    ----------
    image : array-like
        Input image that is float 2D array.
        The rotation axis will be located in the pixel with
        indices ``(image.shape[0] // 2, image.shape[1] // 2)``.
    theta : array-like
        Projection angles (in degrees).
    fill_value : float, optional
        Value to use for regions where the transform could not be calculated.
        Default is NaN.
    return_length : bool, optional
        Option to return an array giving the length of the data array used to
        calculate the transform at every location. Default is `False`.

    Returns
    -------
    radon_image : ndarray
        Radon transform (sinogram). The tomography rotation axis will lie
        at the pixel index ``radon_image.shape[0] // 2`` along the 0th
        dimension of ``radon_image``.
    length: ndarray, optional
        Length of data array. This is only returned if ``return_length`` is
        `True`.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing
           the Discrete Radon Transform With Some Applications", Proceedings of
           the Fourth IEEE Region 10 International Conference, TENCON '89, 1989

    Notes
    -----
    Based on code of Justin K. Romberg

    """
    from skimage.transform._warps import warp

    # circle=False
    cdef int ny = image.shape[0]
    cdef int nx = image.shape[1]
    cdef DTYPE_t sqrt_2 = 1.4142135623730951
    cdef DTYPE_t diagonal
    if ny >= nx:
        diagonal = sqrt_2 * ny
    else:
        diagonal = sqrt_2 * nx
    cdef int pad_y = int(np.ceil(diagonal - ny))
    cdef int pad_x = int(np.ceil(diagonal - nx))
    cdef int nc_y = (ny + pad_y) // 2
    cdef int nc_x = (nx + pad_x) // 2
    cdef int oc_y = ny // 2
    cdef int oc_x = nx // 2
    cdef int pad_before_y = nc_y - oc_y
    cdef int pad_before_x = nc_x - oc_x
    cdef list pad_width = [
        (pad_before_y, pad_y - pad_before_y),
        (pad_before_x, pad_x - pad_before_x),
    ]
    cdef np.ndarray[DTYPE_t, ndim=2] padded_image = np.pad(
        image, pad_width, mode='constant', constant_values=fill_value)

    # padded_image is always square
    cdef int ny_padded = padded_image.shape[0]
    if ny_padded != padded_image.shape[1]:
        raise ValueError('padded_image must be a square')
    cdef int center = ny_padded // 2
    cdef int n_theta = theta.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] radon_image = np.zeros(
        (ny_padded, n_theta), dtype=DTYPE) + d_nan
    cdef np.ndarray[DTYPE_t, ndim=1] theta_rad = np.deg2rad(theta)
    cdef np.ndarray[DTYPE_t, ndim=2] lengths = np.copy(radon_image)

    # processes<=1
    cdef DTYPE_t angle
    cdef DTYPE_t cos_a
    cdef DTYPE_t sin_a
    cdef np.ndarray[DTYPE_t, ndim=2] R
    cdef np.ndarray[DTYPE_t, ndim=2] rotated
    for i in range(n_theta):
        angle = theta_rad[i]
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R = np.array([
            [cos_a, sin_a, -center * (cos_a + sin_a - 1)],
            [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
            [0, 0, 1]
        ])
        rotated = warp(padded_image, R, clip=False)

        # median = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN")
            radon_image[:, i] = np.nanmedian(rotated, axis=0)

        if return_length is True:
            lengths[:, i] = np.sum(np.isfinite(rotated), axis=0)

    if return_length is True:
        return radon_image, lengths
    else:
        return radon_image
