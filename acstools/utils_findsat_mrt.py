'''
These are various helper routines for findsat_mrt.py
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling import models, fitting
from astropy.table import Table, vstack
from skimage import transform
from astropy.stats import sigma_clip
import logging
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve
from skimage.transform._warps import warp
from skimage._shared.utils import convert_to_float
from warnings import warn
from multiprocessing import Pool
import time
from astropy.nddata import Cutout2D
from scipy import interpolate
from astropy.io import fits


__taskname__ = "utils_findsat_mrt"
__author__ = "David V. Stark"
__version__ = "1.0"
__vdate__ = "06-Dec-2022"
__all__ = ['_round_up_to_odd', 'merge_tables', 'good_indices',
           '_fit_streak_profile', '_rotate_image_trail', 'filter_sources',
           'create_mask', 'rotate', 'streak_endpoints', '_streak_persistence',
           'add_streak', '_rot_sum', '_rot_med', 'radon',
           'create_mrt_line_kernel']

# Initialize the logger
logging.basicConfig()
LOG = logging.getLogger(f'{__taskname__}')
LOG.setLevel(logging.INFO)


def _round_up_to_odd(f):
    '''
    Rounds number to nearest odd value

    Parameters
    ----------
    f : float
        Inpit number.

    Returns
    -------
    f : float
        Rounded number.

    '''
    return np.ceil(f) // 2 * 2 + 1


def merge_tables(tbls, theta_sep=10, rho_sep=10):
    '''
    Routine to merge multiple trail tables together while removing duplicates

    Parameters
    ----------
    tbls : array-like
        Array of astropy tables to compare.
    theta_sep : float, optional
        Minimum separation in angle between catalog entries. The default is 10.
    rho_sep : float, optional
        Minimum separation in rho between catalog entries. The default is 10.

    Returns
    -------
    src : astropy table object
        Merged astropy table.

    '''

    counter = 1
    for t in tbls:
        if counter == 1:
            src = t
            counter += 1
        else:
            # check that added sources are not recreations of already detected
            # ones
            keep = (np.zeros(len(t))+1).astype(bool)
            for s in src:
                dtheta = np.abs(t['xcentroid']-s['xcentroid'])
                drho = np.abs(t['ycentroid']-s['ycentroid'])
                sel = (dtheta < theta_sep) | (drho < rho_sep)
                keep[sel] = False
            src = vstack([src, t[keep]])
    return src


def good_indices(inds, shape):
    '''
    Program to make sure indices are within bounds of array. Occasionally
    useful for various tasks

    Parameters
    ----------
    inds : array of tuples
        An array of tuples of the format (ind1,ind2) for each dimension.
    sz : TYPE
        Shape of the array under consideration

    Returns
    -------
    fixed_inds: array of tuples
        Same format as input inds, but truncated to fall inside array
        dimensions

    '''
    if type(inds) is not list:
        inds = [inds]
    if type(shape) == int:
        shape = [shape]
    shape = list(shape)  # covers other sizes of shape

    good_inds = []
    for ind, sz in zip(inds, shape):
        ind1, ind2 = ind
        new_inds = (np.max([0, ind1]), np.min([sz-1, ind2]))
        good_inds.append(new_inds)

    return good_inds


def _fit_streak_profile(yarr, p0, fit_background=True, plot=False,
                        max_width=None, ax=None, bounds=None):
    '''
    Fits a Gaussian function to a 1D cross-section of a trail identified in an
    image.

    Parameters
    ----------
    yarr : float
        Values of trail cross-section at each position.
    p0 : array
        Initial guesses for amplitude, mean, and sigma parameters.
    fit_background : bool, optional
        Set to fit a polynomial to the . The default is True.
    plot : bool, optional
        Set to plot the resulting fitteed profile. The default is False.
    max_width : int, optional
        Maximum allowed width of robust satellite trail. Used to define
        background regions The default is None.
    ax : AxesSubplot, optional
        Matplotlib subplot axis of plot. The default is None.
    bounds : dict, optional
        Dictionary containing the bounds of fit parameters. Format is
        {'parameter':(lower,upper)}. The default is None.

    Returns
    -------
    g : Fittable1DModel
        Resulting astropy model with best-fit parameters.
    snr : float
        Derived signal-to-noise ratio of feature.
    width : float
        Derived width of feature, defined as 2*3*sigma, where sigma is from the
        Gaussian fit.
    mean_flux : float
        The mean flux measured within +/-sigma from the Gaussian fit.

    '''

    amp0, mean0, stdev0 = p0  # initial guesses for amplitude, mean, stddev.
    # Can be "None"

    # update mean0 if necessary; other will be updated below
    if mean0 is None:
        mean0 = np.len(yarr)/2-0.5

    xarr = np.arange(len(yarr))  # x pixel array

    # subtract background
    if fit_background is True:

        # try gitting a polynomial to background. Keep to low order to avoid
        # weirdness occuring. E.g., line
        fit = fitting.LinearLSQFitter()
        or_fit = fitting.FittingWithOutlierRemoval(fit, sigma_clip, niter=3,
                                                   sigma=3)  # rejects outliers

        # make sure there are regions to fit on either side of the initial
        # position. If not, lower order
        sel_low = np.where(np.isfinite(yarr) &
                           (xarr < (mean0-max_width/2)))[0]
        sel_high = np.where(np.isfinite(yarr) &
                            (xarr > (mean0+max_width/2)))[0]
        if (len(sel_high) == 0) or (len(sel_low) == 0):
            order = 1
        else:
            order = 3
        sel = np.concatenate([sel_low, sel_high])

        # now run fitting
        line_init = models.Polynomial1D(degree=order)
        fitted_line, data_mask = or_fit(line_init, xarr[sel], yarr[sel])

        # subtract the fitted background
        yarr = yarr - fitted_line(xarr)
    else:
        # if not fitting, subtract median using outer 50% of data
        ind = int(len(yarr)/4)  # use to select outer quarter of data
        mean = np.nanmedian(list(yarr)[:ind] + list(yarr)[-ind:])  # estimate#
        # of baseline level using outer 50% of channels
        yarr = yarr - mean

    # fit cross-sectional profile

    # update initial guesses if necessary
    if amp0 is None:
        amp0 = np.nanmax(yarr)
    if stdev0 is None:
        stdev0 = 5.

    if bounds is None:
        bounds = {'amplitude': (0, None)}
    g_init = models.Gaussian1D(amplitude=amp0, mean=mean0, stddev=stdev0,
                               bounds=bounds)
    fit_g = fitting.DogBoxLSQFitter()
    sel = np.isfinite(yarr)
    g = fit_g(g_init, xarr[sel], yarr[sel])
    if (plot is True) and (ax is not None):
        ax.plot(xarr, yarr)
        ax.plot(xarr[sel], g(xarr[sel]), color='red', label='Fit', lw=3,
                alpha=0.3)

    # characterize the fit

    # measure peak flux directly -- sometimes gaussian not the best fit and
    # can poorly represent the true amplitude
    sel = (xarr > (g.mean - 3*g.stddev)) & (xarr < (g.mean + 3*g.stddev))
    if np.sum(sel) == 0:
        peak = 0
    else:
        peak = np.nanmax(yarr[sel])

    # measure mean flux within the standard deviation
    sel = (xarr > (g.mean - g.stddev)) & (xarr < (g.mean + g.stddev))
    if np.sum(sel) == 0:
        mean_flux = 0
    else:
        mean_flux = np.nanmean(yarr[sel])

    # measure noise outside the profile region
    sel = (xarr < (g.mean - 3*g.stddev)) | (xarr > (g.mean + 3*g.stddev))
    if np.sum(sel) == 0:
        noise = 0
    else:
        noise = np.nanmedian(np.abs(yarr[sel] -
                                    np.nanmedian(yarr[sel])))/0.67449

    # use amplitude and noise to estimate snr
    snr = (peak-noise)/noise

    LOG.info('amplitude of feature: {}'.format(peak))
    LOG.info('baseline noise: {}'.format(noise))
    LOG.info('snr of feature = {}'.format(snr))

    # use fit paramters to determine the 3-sigma width
    upper, lower = (g.mean + 3*g.stddev, g.mean-3*g.stddev)
    width = upper-lower
    LOG.info('width of feature = {}'.format(width))

    # return all this
    return g, snr, width, mean_flux


def _rotate_image_trail(image, endpoints):
    '''
    Rotates an image so a given trail runs in the horizontal direction

    Parameters
    ----------
    image : ndarray
        Image to be rotated.
    endpoints : list of tuples
        List containing the endpoints of the trail. Format is
        [(x0,y0),(x1,y1)].

    Returns
    -------
    rotated : ndarray
        The rotated image.
    newendpoints : list of tuples
        The endpoints of the rotated trail.
    theta : float
        The angle by which the original image was rotated.

    '''

    # extract endpoints
    x1, y1 = endpoints[0]
    x2, y2 = endpoints[1]

    # calculate trail angle
    dy = y2-y1
    dx = x2-x1

    theta = np.arctan2(dy, dx)

    # rotate image so trail is horizontal
    rotated = transform.rotate(image, np.degrees(theta), cval=np.nan,
                               resize=True, order=3)

    # now recalculate the new streak endpoints. Need to account for change in
    # image size after rotating

    # get new image center
    xshift = (rotated.shape[1] - image.shape[1])/2
    yshift = (rotated.shape[0] - image.shape[0])/2

    # rotate streak endpoints
    rx1, ry1 = rotate(((image.shape[1]-1)/2, (image.shape[0]-1)/2), (x1, y1),
                      -theta)
    rx1, ry1 = (rx1 + xshift, ry1 + yshift)
    rx2, ry2 = rotate(((image.shape[1]-1)/2, (image.shape[0]-1)/2), (x2, y2),
                      -theta)
    rx2, ry2 = (rx2 + xshift, ry2 + yshift)

    newendpoints = [[rx1, ry1], [rx2, ry2]]

    return rotated, newendpoints, theta


def filter_sources(image, streak_positions, plot=False, buffer=100, minsnr=5,
                   max_width=75, fit_background=True,
                   min_length=50, check_persistence=True, min_persistence=0.5,
                   persistence_chunk=100, min_persistence_snr=3):
    '''
    Routine to filter a trail catalog of likely spurious sources based on S/N,
    trail width, and trail_persistence (over what fraction of the trail path
    can the trail actually be well-detected).

    Parameters
    ----------
    image : ndarray
        The original image containing the trails.
    streak_positions : array
        And array containing the endpoints of each trail being analyzed. The
        format is [[(x0,y0),(x1,y1)],[(x0_b,y0_b),(x1_b,y1_b)]...]
    plot : bool, optional
        Set to turn on plotting (WARNING: THIS CAN GENERATE A LOT OF PLOTS).
        The default is False.
    buffer : int, optional
        Size of cutout region on either side of a trail. The default is 100.
    minsnr : float, optional
        Minimum signal-to-noise ratio for a trail to be considered robust.
        The default is 5.
    max_width : int, optional
        Maximum trail width for it to be considerered robust. The default is
        75.
    fit_background : bool, optional
        Set to fit a polynomial to the background on either side of an
        extracted trail. The default is True.
    min_length : int, optional
        The minimum allowed length of trail to be analyzed. Taking the median
        along each row of the rotated image to create a 1d profile will also
        ignore any rows with fewer than this number of pixels. The default is
        50.
    check_persistence : nool, optional
        Set to turn on persistence checking. The default is True.
    min_persistence : float, optional
        Minimum allowed persistence in order for a trail to be considered
        robust. The default is 0.5.
    persistence_chunk : int, optional
        Minimum size of trail sections when breaking it up to measure
        persistence. The default is 100.
    min_persistence_snr : float, optional
        Minimum required signal-to-noise ratio of a chunk of trail in order to
        have a persistence score of 1. The default is 3.

    Returns
    -------
    properties : Table
        Same as the original table but with new columns: (1) mean trail flux,
        (2) trail width, (3) trail S/N ratio (4) trail persistence, (5) trail
        status (1 = failed snr or width requirements, 2 = passed snr and width
                requirements but failed persistence test, 3 = passed snr,
                width, and persistence requirements.

    '''

    widths = np.zeros(len(streak_positions))
    snrs = np.zeros(len(streak_positions))
    status = np.zeros(len(streak_positions)).astype(int)
    persistence = np.zeros(len(streak_positions))
    mean_fluxes = np.zeros(len(streak_positions))

    # cycle through lines and add them to the mask
    for ii, p in enumerate(streak_positions):

        rotated, [[rx1, ry1], [rx2, ry2]], theta = _rotate_image_trail(image,
                                                                       p)

        # update ry1/2 to include buffer region
        ry1_new = np.min([ry1, ry2])-buffer
        ry2_new = np.max([ry1, ry2])+buffer  # making sure ry1 lower than ry2
        streak_y_rot = (ry1 + ry2)/2  # streak position, possible slight
        # difference in ry1/ry2 due to finite angle sampling

        # buffer region could extend off edge of chip. Truncate ry1/ry2 if so
        fixed_inds = good_indices([(ry1_new, ry2_new), (rx1, rx2)],
                                  rotated.shape)
        ry1_trim, ry2_trim = fixed_inds[0]
        rx1_trim, rx2_trim = fixed_inds[1]

        # find distance of streak from current bottom of the cutout
        dy_streak = streak_y_rot - ry1_trim

        # extract final cutout
        subregion = rotated[int(ry1_trim):int(ry2_trim),
                            int(rx1_trim):int(rx2_trim)]

        # make 1D profile of trail (looking down its axis) by taking a median
        # of all pixels in each row
        medarr = np.nanmedian(subregion, axis=1)

        # get number of pixels being considered at each point; remove those
        # that are too small such that median unreliable
        narr = np.sum(np.isfinite(subregion), axis=1)
        medarr[narr < min_length] = np.nan

        # set up plot
        use_ax = None

        if plot is True:
            fig, [ax1, ax2] = plt.subplots(1, 2)
            mad = np.nanmedian(np.abs(subregion))
            ax2.imshow(subregion, vmin=-mad, vmax=5*mad, origin='lower')
            ax2.plot([0, subregion.shape[1]-1], [dy_streak, dy_streak], '--',
                     lw=2, color='magenta', alpha=0.4)
            use_ax = ax1

        g, snr, width, mean_flux = _fit_streak_profile(medarr,
                                                       (None, dy_streak, 5),
                                                       ax=use_ax,
                                                       max_width=max_width,
                                                       plot=plot)
        snrs[ii] = snr
        widths[ii] = width
        mean_fluxes[ii] = mean_flux

        if plot is True:
            ax1.set_title('snr={}, width={}'.format(snr, width))
            ax1.set_xlabel('position')
            ax1.set_ylabel('brightness')
            fig.tight_layout()

        # update status as needed
        if (snr > minsnr) & ((width) < max_width):
            status[ii] = 1

            # new step; track persistence (if toggled)
            persistence[ii] = -1
            if check_persistence is True:
                # approximate maximum number of sections and still have sn>3
                # per section (assuming uniform streak)
                maxchunk = np.floor(snr*snr/min_persistence_snr**2)

                dx = persistence_chunk  # starting dx value
                nchunk = np.floor(subregion.shape[1]/dx)

                # make sure estimated snr per section is >3
                if nchunk > maxchunk:
                    nchunk = maxchunk
                    dx = np.floor(subregion.shape[1]/nchunk)

                if nchunk < 2:
                    # want at least two chunks
                    dx = np.floor(subregion.shape[1]/2.)
                    nchunk = 2

                LOG.info('breaking into {} sections for persistence check'.
                         format(nchunk))
                LOG.info('Section size for persistence check: {}'.format(dx))

                persistence[ii] = _streak_persistence(subregion, int(dx),
                                                      g.mean.value,
                                                      g.stddev.value,
                                                      max_width=max_width)
                if persistence[ii] > min_persistence:
                    status[ii] = 2

    properties = Table()
    properties['mean flux'] = mean_fluxes
    properties['width'] = widths
    properties['snr'] = snrs
    properties['persistence'] = persistence
    properties['status'] = status
    return properties


def create_mask(image, trail_id, endpoints, widths):
    '''
    Creates an image mask given a set of trail endpoints and widths.

    Parameters
    ----------
    image : ndarray
        Input image to be masked.
    trail_id : int
        ID numbers for each trail. These should be unique.
    endpoints : array
        An array containing the endpoints of each trail to contribute to the
        mask. See filter_sources for format.
    widths : array
        Widths for each trail.

    Returns
    -------
    segment : ndarray
        A segmentation image where each pixel that is part of a masked
        satellite trail is given a value of the corresponding trail ID.
    mask : ndarray
        Boolean mask (1=masked, 0=not masked).

    '''

    segment = np.zeros_like(image).astype(int)

    # cycle through trail endpoints/widths
    for t, e, w in zip(trail_id, endpoints, widths):

        rotated, [[rx1, ry1], [rx2, ry2]], theta = _rotate_image_trail(image,
                                                                       e)

        # create submask using known width and
        ry = (ry1 + ry2)/2.  # take average, although they should be about the
        # same
        # max functionused to ensure it stays within bounds
        mask_y1 = np.maximum(0, np.floor(ry - w/2)).astype(int)
        # min function used to ensure it stays within bounds
        mask_y2 = np.minimum(rotated.shape[0]-1, np.ceil(ry+w/2)).astype(int)

        mask_x1 = np.maximum(0, np.floor(rx1)).astype(int)
        mask_x2 = np.minimum(rotated.shape[1]-1, np.ceil(rx2)).astype(int)

        # use these indices to make submask
        submask_rot = np.zeros_like(rotated)
        submask_rot[mask_y1:mask_y2, mask_x1:mask_x2] = t

        # unrotate
        submask_unrot = transform.rotate(submask_rot, -np.degrees(theta),
                                         resize=True, order=1)
        submask_unrot[submask_unrot > 0] = t
        ix0 = (submask_unrot.shape[1] - image.shape[1]) / 2
        ix1 = image.shape[1]+ix0
        iy0 = (submask_unrot.shape[0] - image.shape[0]) / 2
        iy1 = image.shape[0]+iy0
        ix0, ix1, iy0, iy1 = list(map(int, [ix0, ix1, iy0, iy1]))
        subsegment = submask_unrot[iy0:iy1, ix0:ix1]

        # add this to the existing segentation map
        segment = np.maximum(segment, subsegment)

    mask = segment > 0

    return segment, mask


def rotate(origin, point, angle):
    '''
    Rotate a point counterclockwise by a given angle around a given origin.


    Parameters
    ----------
    origin : tuple, float
        (x,y) coordinate of the origin about which rotation is centered.
    point : tuple, float
        (x,y) coordinate (in the unrotated reference frame) of the point whose
        new coordinate (in the rotated reference frame) will be calcualted.
    angle : float
        Angle of rotation. Should be given in radians.

    Returns
    -------
    qx : float
        Resulting x coordinate in the rotated reference frame.
    qy : float
        Resulting y coordinate in the rotated reference frame.

    '''
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def streak_endpoints(rho, theta, sz, plot=False):
    '''
    Calculates the endpoints of a streak parameterized by rho, theta assuming
    an image of size sz

    Parameters
    ----------
    rho : float
        offset coordinate.
    theta : float
        angle coordinate.
    sz : tuple/int
        Size of image. Streak endpoints will be truncated at the images edges

    Returns
    -------
    pos : array
        Array with form [p0,p1] where p0/p1 = (x,y) and (x,y) are the x and y
        coordinates of each end

    '''

    x0 = sz[1]/2-0.5
    y0 = sz[0]/2-0.5

    dy = rho*np.sin(np.radians(theta))
    dx = rho*np.cos(np.radians(theta))

    slope = np.sin(np.radians(theta))/np.cos(np.radians(theta))
    intercept = y0 - slope*x0

    # get perpendicular slope/intercept (this is the line that represents what
    # the streak looks like)
    slope_int = -1./slope
    b_int = (y0+dy)-slope_int*(x0+dx)

    if plot is True:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.plot([0, sz[0]-1], [sz/2-0.5,
                                        sz/2-0.5], '--',
                color='gray')
        ax.plot([sz/2-0.5, sz/2-0.5],
                [0, sz[0]-1], '--', color='gray')

        ax.plot([x0, x0+dx], [y0, y0+dy], color='red')
        ax.set_xlim(0, sz[1]-1)
        ax.set_ylim(0, sz[0]-1)

        xarr = np.arange(sz[1])
        yarr = slope_int*xarr+b_int
        ax.plot(xarr, yarr, color='blue')

    # get coordinates of streak. Start with x = 0 and max value
    xi, xf = (0, sz[1]-1)
    yi, yf = slope_int*np.array([xi, xf])+b_int

    # update coordinates if y goes out of bounds
    if yi < 0:
        yi = 0
    elif yi >= sz[0]:
        yi = sz[0]-1
    xi = (yi-b_int)/slope_int

    # same things for yf
    if yf < 0:
        yf = 0
    elif yf >= sz[0]:
        yf = sz[0]-1
    xf = (yf-b_int)/slope_int

    # update this is the slope is infinity
    if np.isfinite(slope_int) is False:
        yi = 0
        yf = sz[1]-1
        xi = x0+rho
        xf = x0+rho

    if plot is True:
        ax.scatter([xi, xf], [yi, yf], s=100, color='magenta')

    p0 = (xi, yi)
    p1 = (xf, yf)

    return [p0, p1]


def _streak_persistence(cutout, dx, streak_y0, streak_stdev, max_width=None,
                        plot=False):
    '''
    Routine to measure a sreak's persistence score. It does this by breaking
    the trail into even chunks and trying to fit a Gaussian to a 1D
    cross-section from each chunk. Successful fits contribute to the
    persistence score.

    Parameters
    ----------
    cutout : ndarray
        Cutout of the trail. This should be the same cutout generated in
        "filter_sources" so that the trail is aligned horizontally across the
        cutout.
    dx : int
        Size of each chunk to be analyzed.
    streak_y0 : float
        y coordinate of the trail derived from the analysis of its global
        cross-section.
    streak_stdev : float
        Gaussian sigma value determined from the analysis of its global
        cross-section.
    max_width : int, optional
        Maximum allowed width for a trail to be considered robust. This
        should be the same as whatever was used in "filter_sources". The
        default is None.
    plot : bool, optional
        Set to turn on plotting. The default is False.

    Returns
    -------
    pscore: float
        Persistence score of the input trail.

    '''

    guess = (None, streak_y0, streak_stdev)  # starting guess
    bounds = {'amplitude': (0, None), 'mean': (streak_y0, streak_y0 + 25)}

    snr_arr = []
    width_arr = []
    mean_arr = []
    persist = []

    # step through and fit as a function of x
    nsteps = int(np.floor(cutout.shape[1]/dx))
    for ii in range(nsteps):

        LOG.info('Checking persistence, step {} of {}'.format(ii+1, nsteps))

        ind0 = ii*dx
        ind1 = (ii+1)*dx
        chunk = np.nanmedian(cutout[:, ind0:ind1], axis=1)
        if plot is True:
            fig, ax = plt.subplots()
        else:
            ax = None

        g, snr, width, mean_flux = _fit_streak_profile(chunk, guess, ax=ax,
                                                       max_width=max_width,
                                                       plot=plot,
                                                       bounds=bounds)
        if plot is True:
            ax.set_title('{}-snr={},width={},mean={}'.format(ii, snr, width,
                                                             g.mean.value))

        LOG.info('Chunk snr, width ,mean: {}, {}, {}'.format(snr, width,
                                                             g.mean.value))
        snr_arr.append(snr)
        width_arr.append(width)
        mean_arr.append(g.mean.value)

        success = (snr > 3) & (np.isfinite(snr))
        if ii > 0:
            success = success & (np.abs(mean_arr[ii] -
                                        mean_arr[ii-1]) < width_arr[ii-1])
            if max_width is not None:
                success = success & (width <= max_width)
        if success:
            persist.append(success)
        else:
            persist.append(False)

        if success:
            # update guess
            guess = (g.amplitude.value, g.mean.value, g.stddev.value)
            bounds = {'amplitude': (0, None), 'mean': (g.mean.value-width,
                                                       g.mean.value + width)}
        else:
            LOG.info('fit failed, will not update guesses')
    pscore = np.sum(persist)/len(persist)

    LOG.info('persistance: {}'.format(pscore))

    return pscore


def add_streak(image, width, value, rho=None, theta=None, endpoints=None,
               psf_sigma=None):
    '''
    Generates a model streak and adds it to an input image. The code optionally
    convolves the trail image with a psf.

    Parameters
    ----------
    image : ndarray
        The input image onto which the trail should be added.
    width : float
        Width of the trail. Note that as of now this function does not super-
        sample the image in order to precisely model the trail width.
    value : float
        Unfiform intensity within the trail prior to convolving with a psf.
    rho : float, optional
        Trail offset from center. The default is None.
    theta : float, optional
        Trail angle with respect to x-axis. The default is None.
    endpoints : array of tuples, optional
        Endpoints of the trail. Format is [(x0,y0),(x1,y1)]. The default is
        None.
    psf_sigma : float, optional
        Sigma of Gaussian model PSF to be convolved with trail image. The
        default is None, in which case no convolution is performed.

    Returns
    -------
    image : TYPE
        DESCRIPTION.

    '''

    # make sure either rho,theta or endpoints are set
    if ((rho is not None) | (theta is not None)) & (endpoints is not None):
        LOG.warning('rho/theta and endpoints set, defaulting to using\
                    endpoints')
        use_rhotheta = False
        use_endpoints = True

    elif (rho is not None) & (theta is None) & (endpoints is None):
        LOG.error('rho set but not theta')
        return image

    elif (rho is None) & (theta is not None) & (endpoints is None):
        LOG.error('theta set but not rho')
        return image
    elif (rho is not None) & (theta is not None) & (endpoints is None):
        use_rhotheta = True
        use_endpoints = False
    elif (rho is None) & (theta is None) & (endpoints is not None):
        use_endpoints = True
        use_rhotheta = False

    # calculate endpoints from rho,theta if necessary
    if use_rhotheta:
        endpoints = streak_endpoints(rho, theta, image.shape)
        print('calculated endpoints: {}'.format(endpoints))

    # rotate image so trail horizontal
    rotated, newendpoints, rot_theta = _rotate_image_trail(image, endpoints)

    # add trail
    x1, y1 = newendpoints[0]
    x2, y2 = newendpoints[1]

    trail_y = (y1 + y2)/2
    trail_image_rot = np.zeros_like(rotated)
    # find edge of trail, avoid going beyond image boundaries
    max_ind = int(np.minimum(trail_y + width/2+1, rotated.shape[0]))
    min_ind = int(np.maximum(trail_y - width/2, 0))
    trail_image_rot[min_ind:max_ind, :] = value

    # convolve with psf if set
    if psf_sigma is not None:
        kernel = Gaussian2DKernel(x_stddev=psf_sigma, y_stddev=psf_sigma)
        trail_image_rot = convolve(trail_image_rot, kernel, boundary='extend')

    # now rotate back
    trail_image = transform.rotate(trail_image_rot, -np.degrees(rot_theta),
                                   resize=True, order=1)
    ix0 = (trail_image.shape[1] - image.shape[1]) / 2
    ix1 = image.shape[1]+ix0
    iy0 = (trail_image.shape[0] - image.shape[0]) / 2
    iy1 = image.shape[0]+iy0
    ix0, ix1, iy0, iy1 = list(map(int, [ix0, ix1, iy0, iy1]))

    trail_image = trail_image[iy0:iy1, ix0:ix1]

    image += trail_image

    return image


def _rot_sum(image, angle, return_length):
    '''
    Rotates and image by a designated angle and sums the values in each column

    Parameters
    ----------
    image : ndarray
        Image to be rotated and summed.
    angle : float
        Angle by which to rotate the input image (radians).
    return_length : bool
        Set to the return the number of valid pixels in each column.

    Returns
    -------
    medarr : ndarray
        The resulting sum in each column.
    length : ndarray, optional
        The number of valid pixels in each column.


    '''
    center = image.shape[0] // 2
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                  [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                  [0, 0, 1]])
    rotated = warp(image, R, clip=False, cval=np.nan)
    medarr = np.nansum(rotated, axis=0)
    if return_length is True:
        length = np.sum(np.isfinite(rotated), axis=0)
        return [medarr, length]
    return medarr


def _rot_med(image, angle, return_length):
    '''
    Rotates and image by a designated angle and take a median of the values
    in each column

    Parameters
    ----------
    image : ndarray
        Image to be rotated and medianed.
    angle : float
        Angle by which to rotate the input image (radians).
    return_length : bool
        Set to the return the number of valid pixels in each column.

    Returns
    -------
    medarr : ndarray
        The resulting median in each column.
    length : ndarray, optional
        The number of valid pixels in each column.


    '''
    center = image.shape[0] // 2
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                  [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                  [0, 0, 1]])
    rotated = warp(image, R, clip=False, cval=np.nan)
    medarr = np.nanmedian(rotated, axis=0)
    if return_length is True:
        length = np.sum(np.isfinite(rotated), axis=0)
        return [medarr, length]
    else:
        return medarr


def radon(image, theta=None, circle=False, *, preserve_range=False,
          fill_value=np.nan, median=True, threads=1, return_length=False,
          print_calc_times=False):
    """
    Calculates the (median) radon transform of an image given specified
    projection angles.

    Parameters
    ----------
    image : array_like
        Input image. The rotation axis will be located in the pixel with
        indices ``(image.shape[0] // 2, image.shape[1] // 2)``.
    theta : array_like, optional
        Projection angles (in degrees). If `None`, the value is set to
        np.arange(180).
    circle : boolean, optional
        Assume image is zero outside the inscribed circle, making the
        width of each projection (the first dimension of the sinogram)
        equal to ``min(image.shape)``.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    fill_value : float, optional
        Value to use for regions where the transform could not be calculated.
        Default is 0.
    median: bool, optional
        Flag to turn on Median Radon Transform instead of standard Radon
        Transform.nDefault is True.
    threads: int, optional
        Number of threads to use when calculating the transform. Default is 1
        (no multi-threading)
    return_length: bool, optional
        Option to return an array giving the length of the data array used to
        calculate the transform at every location. Default is False.

    Returns
    -------
    radon_image : ndarray
        Radon transform (sinogram).  The tomography rotation axis will lie
        at the pixel index ``radon_image.shape[0] // 2`` along the 0th
        dimension of ``radon_image``.
    length: ndarray, optional
        Length of data array

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
    (https://www.clear.rice.edu/elec431/projects96/DSP/bpanalysis.html)

    """

    total_median_time = 0.
    total_warp_time = 0.

    if median is True:
        LOG.info('Calculating median Radon Transform with {} threads'
                 .format(threads))
    else:
        LOG.info('Calculating standard Radon Transform with {} threads'
                 .format(threads))
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    if theta is None:
        theta = np.arange(180)

    image = convert_to_float(image, preserve_range)

    if circle:
        shape_min = min(image.shape)
        radius = shape_min // 2
        img_shape = np.array(image.shape)
        coords = np.array(np.ogrid[:image.shape[0], :image.shape[1]],
                          dtype=object)
        dist = ((coords - img_shape // 2) ** 2).sum(0)
        outside_reconstruction_circle = dist > radius ** 2
        if np.any(image[outside_reconstruction_circle]):
            warn('Radon transform: image must be zero outside the '
                 'reconstruction circle')
        # Crop image to make it square
        slices = tuple(slice(int(np.ceil(excess / 2)),
                             int(np.ceil(excess / 2) + shape_min))
                       if excess > 0 else slice(None)
                       for excess in (img_shape - shape_min))
        padded_image = image[slices]
    else:
        diagonal = np.sqrt(2) * max(image.shape)
        pad = [int(np.ceil(diagonal - s)) for s in image.shape]
        new_center = [(s + p) // 2 for s, p in zip(image.shape, pad)]
        old_center = [s // 2 for s in image.shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        padded_image = np.pad(image, pad_width, mode='constant',
                              constant_values=fill_value)

    # padded_image is always square
    if padded_image.shape[0] != padded_image.shape[1]:
        raise ValueError('padded_image must be a square')
    center = padded_image.shape[0] // 2
    radon_image = np.zeros((padded_image.shape[0], len(theta)),
                           dtype=image.dtype)+np.nan

    if threads <= 1:
        for i, angle in enumerate(np.deg2rad(theta)):
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                          [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                          [0, 0, 1]])
            warp_time_0 = time.time()
            rotated = warp(padded_image, R, clip=False)
            warp_time_1 = time.time()
            total_warp_time += (warp_time_1 - warp_time_0)
            if median is False:
                radon_image[:, i] = np.nansum(rotated, axis=0)
            else:
                median_time_0 = time.time()
                radon_image[:, i] = np.nanmedian(rotated, axis=0)
                median_time_1 = time.time()
                total_median_time += (median_time_1 - median_time_0)

    else:
        p = Pool(threads)
        angles = np.deg2rad(theta)
        images = [padded_image for i in range(len(angles))]
        return_lengths = [True for i in range(len(angles))]
        pairs = list(zip(images, angles, return_lengths))
        if median is False:
            result = p.starmap(_rot_sum, pairs)
            result = np.array(result)
            radon_image = result[:, 0, :]
            lengths = result[:, 1, :]
        else:
            result = p.starmap(_rot_med, pairs)
            result = np.array(result)
            print(np.shape(result))
            radon_image = result[:, 0, :]
            lengths = result[:, 1, :]
        radon_image = radon_image.T
        lengths = lengths.T

    if print_calc_times:
        LOG.info('Total time to warp images = {} seconds'.
                 format(total_warp_time))
        LOG.info('Total time to calculate medians = {} seconds'.
                 format(total_median_time))

    if return_length is True:
        return radon_image, lengths
    else:
        return radon_image

def create_mrt_line_kernel(width, sigma, outfile=None, shape=(1024, 2048),
                           plot=False, theta=np.arange(0, 180, 0.5),
                           threads=1):
    '''
    Creates a model signal MRT signal of a line of specified width and blurred
    by a psf. Used for detection of real linear signals in imaging data.

    Parameters
    ----------
    width : int
        Width of the line. Intensity is constant over this width.
    sigma : float
        Gaussian sigma of the PSF. This is NOT FWHM.
    outfile : string, optional
        Location to save an output fits file of the kernel. The default is
        None.
    sz : tuple/int, optional
        Size of the image on which to place the line. The default is
        (1024,2048).
    plot : bool, optional
        Flag to plot the original image, MRT, and kernel cutout
    theta : array, optional
        Set of angles at which to calculate the MRT, default is
        np.arange(0,180,0,5)
    threads: int, optional
        Number of threads to use when calculating MRT. Default is 1.
    Returns
    -------
    kernel : ndarray
        The resulting kernel

    '''

    # set up empty image and coordinates
    image = np.zeros(shape)
    y0 = image.shape[0]/2-0.5
    x0 = image.shape[1]/2-0.5
    xarr = np.arange(image.shape[1])-x0
    yarr = np.arange(image.shape[0])-y0
    x, y = np.meshgrid(xarr, yarr)

    # add a simple streak across the image.
    image = add_streak(image, width, 1, rho=0, theta=90, psf_sigma=sigma)

    # plot the image
    if plot is True:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.imshow(image, origin='lower')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('model image')

    # calculate the RT for this model
    rt = radon(image, circle=False, median=True, fill_value=np.nan,
                 threads=threads, return_length=False)

    # plot the RT
    if plot is True:
        fig2, ax2 = plt.subplots()
        ax2.imshow(rt, aspect='auto', origin='lower')
        ax2.set_xlabel('angle pixel')
        ax2.set_ylabel('offset pixel')

    # find the center of the signal by summing along each direction and finding
    # the max.
    rt_rho = np.nansum(rt, axis=1)
    rt_theta = np.nansum(rt, axis=0)
    fig, [ax1, ax2] = plt.subplots(1, 2)
    ax1.plot(rt_theta, '.')
    ax2.plot(rt_rho, '.')

    rho0 = np.nanargmax(rt_rho)
    theta0 = np.nanargmax(rt_theta)
    ax2.plot([rho0, rho0], [0, 1])
    ax1.plot([theta0, theta0], [0, 8])
    ax1.set_xlim(theta0-5, theta0+5)
    ax2.set_xlim(rho0-10, rho0+10)

    # may need to refine center coords. Run a Gaussian fit to see if necessary
    g_init = models.Gaussian1D(mean=rho0)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, np.arange(len(rt_rho)), rt_rho)
    rho0_gfit = g.mean.value

    g_init = models.Gaussian1D(mean=theta0)
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, np.arange(len(rt_theta)), rt_theta)
    theta0_gfit = g.mean.value

    # see if any difference between simple location of max pixel vs. gauss fit
    theta_shift = theta0_gfit - theta0
    rho_shift = rho0_gfit - rho0

    # get initial cutout
    position = (theta0, rho0)
    dtheta = 3
    drho = np.ceil(width/2+3*sigma)

    size = (_round_up_to_odd(2*drho), _round_up_to_odd(2*dtheta))
    cutout = Cutout2D(rt, position, size)

    # inteprolate onto new grid if necessary. Need to generate cutout first.
    # The rt can be too big otherwise
    do_interp = (np.abs(rho_shift) > 0.1) | (np.abs(theta_shift) > 0.1)
    if do_interp is True:
        LOG.info('Inteprolating onto new grid to center kernel')
        theta_arr = np.arange(cutout.shape[1])
        rho_arr = np.arange(cutout.shape[0])
        theta_grid, rho_grid = np.meshgrid(theta_arr, rho_arr)

        new_theta_arr = theta_arr + theta_shift
        new_rho_arr = rho_arr + rho_shift
        new_theta_grid, new_rho_grid = np.meshgrid(new_theta_arr, new_rho_arr)

        # inteprolate onto new grid
        f = interpolate.interp2d(theta_grid, rho_grid, cutout.data,
                                 kind='cubic')
        cutout = f(new_theta_arr, new_rho_arr)  # overwrite old cutout

    if plot is True:
        fig3, ax3 = plt.subplots()
        ax3.imshow(cutout.data, origin='lower', aspect='auto')

    if outfile is not None:
        fits.writeto(outfile, cutout.data, overwrite=True)
    return cutout.data