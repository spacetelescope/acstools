'''
About:
    This python module contains the function needed for satellite detection
    within an ACS/WFC image as published in ACS ISR 2015-??.

Depends:
    numpy, matplotlib, astropy, skimage, scipy, math

Author:
    Dave Borncamp, Space Telescope Science Institute

History:
    December 12, 2014 - DMB - Created for COSC 602 "Image Processing and Pattern
        Recocnition" at Towson University. Mostly detection algorithm
        development and VERY crude mask.

    February 1, 2015 - DMB - Masking algorithm refined to be useable by HFF.

    March 28, 2015 - DMB - Small bug fixes and tweaking to try not to include
        diffraction spikes.


License:
    Copyright (c) 2015, AURA
    All rights reserved.

    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

    3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from skimage import filter as filt
from skimage import transform
from skimage import morphology as morph
from skimage import draw
from skimage import exposure
from scipy import stats
import math

__version__ = '0.2.1'
__author__ = 'Dave Borncamp'


def detsat(filename, ext, sigma=2, low_thresh=.1, h_thresh=.5,
           small_edge=60, line_len=200, line_gap=75, plot=False,
           percentile=(4.5, 93), buf=200, verbose=False):
    '''

    Will detect if a satellite is present in an image and return the
    results of a probabalistic hough transform if it is.

    Returns:
        List of endpoints of line segments. These are the segments that have
        been identified as making up part of a satellite trail. These
        points are crude at best and the make_mask function should be run
        for a better approximation of the trail.
        Will return an empty list if nothing is found.

    Input:
        Filename - an ACS fits file, should be either flt or flc format

        ext - Extension to work on. For ACS, either 1 or 4 for science arrays

    keyword input:
        sigma - the size of a gaussian filter to use before edge detection.
            The default is 2 which is good for almost all images.

        low_thresh - the lower threshold for histersis linking of edge pieces.
            should be between 0 and 1, less than h_thresh.

        h_thresh - the upper threshold for histersis linkinf of edge peices.
            should be between 0 and 1, greater than low_thresh.

        small_edge - size of perimiter of small objects to remove in edge
            image. This significantly reduces noise before doing hough
            transform. but too high and you will remove the edge of the
            satellite you are trying to find.

        line_len - minimum line length for probabilistic hough transform to
            fit.

        line_gap - largest gap in points allowed for the probabillistic
            hough transform.

        percentile - the precent boundires to scale the image to before
            creating edge image.

        buf - how close to the edge of the images the satellite trail has to
            be to be considered a trail.

        plot - will make plots of edge image, hough space transformation and
            rescalesd image.

        verbose - will print a lot to the terminal, mostly for debugging.
    '''

    # minor error checking
    # for now, only allow ACS images, but in theory it should work for any
    # instrument.
    if ((ext != 1) == (ext != 4)):
        print 'select valid extension'
        print 'Only ACS science extensions allowed'
        return

    # get the data
    image = fits.getdata(filename, ext)
    # image = im.astype('float64')

    # rescale the image
    p1, p2 = np.percentile(image, percentile)
    if verbose:
        print 'p1, p2'
        print p1, p2

    # there should always be some counts in the image, anything lower should
    # be set to one. Makes things nicer for finding edges.
    if p1 < 0:
        p1 = 0.0

    image = exposure.rescale_intensity(image, in_range=(p1, p2))

    # get the edges
    edge = filt.canny(image, sigma=sigma,
                      low_threshold=np.max(image) * low_thresh,
                      high_threshold=np.max(image) * h_thresh)

    # clean up the small objects, Will make less noise
    morph.remove_small_objects(edge, min_size=small_edge, connectivity=8,
                               in_place=True)

    # create an array of angles from 0 to 180, exatly 0 will get bad columns
    # but it is unlikely that a satellite will be exatly @ 0 degrees, so
    # don't bother checking
    angle = np.arange(2, 178, .5, dtype=float)

    # convert to radians
    for i in range(len(angle)):
        angle[i] = math.radians(angle[i])

    # preform Hough Transform to detect straight lines
    # only do if plotting to visualize the image in hough space
    # otherwise just preform a probabilistic hough transform.
    if plot:
        h, theta, d = transform.hough_line(edge, theta=angle)
        plt.ion()

    # preform Probabilistic Hough Transformation to get line segments.
    result = transform.probabilistic_hough_line(edge, threshold=210,
                                                line_length=line_len,
                                                line_gap=line_gap,
                                                theta=angle)

    # initially assume there is no satellite
    satellite = False

    # only continue if there was more than one point (at least a line)
    # returned from the PHT
    if len(result) > 1:
        if verbose:
            print 'length of result: ' + str(len(result))
        # create lists for X and Y positions of lines and build points
        x0 = []
        y0 = []
        x1 = []
        y1 = []

        # populate the lists of points
        for point in result:
            x0.append(float(point[0][0]))
            y0.append(float(point[0][1]))
            x1.append(float(point[1][0]))
            y1.append(float(point[1][1]))

        # set some boundries
        ymax, xmax = image.shape
        topx = xmax - buf
        topy = ymax - buf

        if verbose:
            print 'max(x0), max(x1), max(y0), max(y1)'
            print max(x0), max(x1), max(y0), max(y1)
            print 'topx, topy'
            print topx, topy
            print 'min(x0), min(x1), min(y0), min(y1)'
            print min(x0), min(x1), min(y0), min(y1)
            print 'buf'
            print buf

        # set up trail angle "tracking" arrays
        trail_angle = []
        round_angle = []

        # find the angle of each segment and filter things out
        for i in range(len(result)):
            #  this may be wrong. Try using atan2
            angled = math.degrees(math.atan((y1[i] - y0[i]) / (x1[i] - x0[i])))
            trail_angle.append(angled)
            # round to the nearest 5 degrees, trail should not be that curved
            rounded = int(5 * round(angled / 5))
            # take out 90 degree things
            if rounded % 90 == 0:
                rounded = None
            round_angle.append(rounded)

        ang, num = stats.mode(round_angle)

        # do the filtering
        truth = []
        for r in round_angle:
            if r == ang[0]:
                truth.append(True)
            else:
                truth.append(False)
        if verbose:
            print round_angle
            print trail_angle
            print ang

        # filter out the outliers
        counter = 0
        for t in truth:
            if not t:
                result.pop(counter)
                trail_angle.pop(counter)
                counter -= 1
            counter += 1

        if verbose:
            print trail_angle
        # remake the point lists with things taken out
        x0 = []
        y0 = []
        x1 = []
        y1 = []

        if len(result) < 1:
            return []

        # add the points
        for point in result:
            x0.append(float(point[0][0]))
            y0.append(float(point[0][1]))
            x1.append(float(point[1][0]))
            y1.append(float(point[1][1]))

        # make decisions on where the trail went and determine if a trail
        # traveresd the image
        # top to bottom
        if ((min(y0) < buf) or (min(y1) < buf)) and ((max(y0) > topy) or (max(y1) > topy)):
            satellite = True
            if verbose:
                print 'Top to Bottom'

        # right to left
        if ((min(x0) < buf) or (min(x1) < buf)) and ((max(x0) > topx) or (max(x1) > topx)):
            satellite = True
            if verbose:
                print 'Right to Left'

        # bottom to left
        if ((min(x0) < buf) or (min(x1) < buf)) and ((min(y0) < buf) or (min(y1) < buf)) and (-1 > np.mean(trail_angle) > -89):
            satellite = True
            if verbose:
                print 'Bottom to Left'

        # top to left
        if ((min(x0) < buf) or (min(x1) < buf)) and ((max(y0) > topy) or (max(y1) > topy)) and (89 > np.mean(trail_angle) > 1):
            satellite = True
            if verbose:
                print 'Top to Left'

        # top to right
        if ((max(x0) > topx) or (max(x1) > topx)) and ((max(y0) > topy) or (max(y1) > topy)) and (-1 > np.mean(trail_angle) > -89):
            satellite = True
            if verbose:
                print 'Top to Right'

        # bottom to right
        if ((max(x0) > topx) or (max(x1) > topx)) and ((min(y0) < buf) or (min(y1) < buf)) and (89 > np.mean(trail_angle) > 1):
            satellite = True
            if verbose:
                print 'Bottom to Right'

        # if there is an unreasonable amount of points, it picked up garbage
        if len(result) > 300:
            print 'Way too many segments results to be correct'
            print len(result)
            print 'Rejecting detection on ' + filename + ' , ' + str(ext)
            satellite = False

    if satellite:
        if verbose:
            print 'We have a trail'
            print 'End point list: '
            print x0
            print x1
            print y0
            print y1
            print 'Trail angle list (not returned): '
            print trail_angle
            print ''

        if plot:  # plotting could really be improved, but not a big deal
            if verbose:
                print 'making a lot of plots'
            # plot everything on one plot first
#            plt.clf()
#            fig, ax = plt.subplots(1, 3, num=1)
#
#            ax[0].imshow(image, vmin=.1, vmax=700, cmap=plt.cm.gray)
#            ax[0].set_title('Input image')
#            ax[0].axis('image')
#
#            ax[1].imshow(np.log(1 + h),
#                         extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]),
#                                 d[-1], d[0]], aspect=1)
#            ax[1].set_title('Hough Transform')
#            ax[1].set_xlabel('Angle (degrees)')
#            ax[1].set_ylabel('Distance (pixels)')
#            ax[1].axis('image')
#
#            ax[2].imshow(edge, cmap=plt.cm.gray)
#            ax[2].set_title('Edge Image')
#            fig.show()

            plt.figure(num=1)
            plt.clf()
            plt.imshow(edge, cmap=plt.cm.gray)
            plt.suptitle('Edge image for ' + filename + ' exten: ' + str(ext))
            for line in result:
                p0, p1 = line
                plt.plot((p0[0], p1[0]), (p0[1], p1[1]),
                         scalex=False, scaley=False)

            plt.figure(num=2)
            plt.clf()
            plt.imshow(np.log(1 + h), extent=(np.rad2deg(theta[-1]),
                                              np.rad2deg(theta[0]),
                                              d[-1], d[0]),
                                              aspect=.02)
#            ax.set_title('Hough Transform')
#            ax.set_xlabel('Angles (degrees)')
#            ax.set_ylabel('Distance from Origin (pixels)')

            rows, cols = edge.shape

            plt.figure(num=3)
            plt.clf()
            plt.imshow(image, vmin=.7, vmax=1, cmap=plt.cm.gray)
            plt.suptitle(filename + ' exten: ' + str(ext))
            for line in result:
                p0, p1 = line
                plt.plot((p0[0], p1[0]), (p0[1], p1[1]),
                         scalex=False, scaley=False)

        return result

    else:  # length of result was too small
        if verbose:
            print 'No trail for you!'
            print 'segments found: ', len(result)

        if plot:
            plt.figure(num=1)
            plt.clf()
            plt.imshow(edge, cmap=plt.cm.gray)
            plt.suptitle(filename + ' exten: ' + str(ext))
            for line in result:
                p0, p1 = line
                plt.plot((p0[0], p1[0]), (p0[1], p1[1]),
                         scalex=False, scaley=False)

        return []


def segmask(result, shape):
    '''
    return a very quick mask for a given results
    not used. should probably delete.
    '''
    mask = np.zeros(shape)

    for line in result:
        p0, p1 = line
        rr, cc = draw.line(p0[0], p0[1], p1[0], p1[1])
        mask[cc, rr] = 1

    return mask


def mask_dq(filename, ext, mask):
    '''
    Create the mask for the satellite and update the DQ array to 16384 to
    reflect that a satellite is present.

    Returns:
        Nothing

    Input:
        filename - name of file to be updated.

        ext - extension of file to update, should be either 3 or 6 for ACS.

        mask - mask array where 0 is background and 1 is satellite.
    '''
    if ((ext != 3) == (ext != 6)):
        print 'select valid extension'
        return

    with fits.open(filename, mode='update') as hdulist:
        dqhdu = hdulist[ext]
        dq = dqhdu.data

        z = np.where(mask == 1)

        dq[z] = 16384

        hdulist.flush()


def biweightMean(inputData):
    """
    Calculate the mean of a data set using bisquare weighting.

    Based on the biweight_mean routine from the AstroIDL User's
    Library.

    Returns:
        the biweight mean of the input data

    Input:
        inputData - single dimension array of data to find the biweight mean
        of.
    """

    y = inputData.ravel()
    if type(y).__name__ == "MaskedArray":
        y = y.compressed()

    n = len(y)
    closeEnough = 0.03 * np.sqrt(0.5 / (n - 1))

    diff = 1.0e30
    nIter = 0

    y0 = np.median(y)
    deviation = y - y0
    sigma = std(deviation)

    if sigma < __epsilon:
        diff = 0
    while diff > closeEnough:
        nIter = nIter + 1
        if nIter > __iterMax:
            break
        uu = ((y - y0) / (6.0 * sigma)) ** 2.0
        uu = np.where(uu > 1.0, 1.0, uu)
        weights = (1.0 - uu) ** 2.0
        weights /= weights.sum()
        y0 = (weights * y).sum()
        deviation = y - y0
        prevSigma = sigma
        sigma = std(deviation, Zero=True)
        if sigma > __epsilon:
            diff = np.abs(prevSigma - sigma) / prevSigma
        else:
            diff = 0.0

    return y0


__iterMax = 25
__delta = 5.0e-7
__epsilon = 1.0e-20


def robust_mean(inputData, Cut=3.0):
    """
    Robust estimator of the mean of a data set.  Based on the
    resistant_mean function from the AstroIDL User's Library.

    Returns:
        The robust mean of the input data

    Input:
        inputData - single dimension array of data to find the robust mean
            of.

        cut - the value in sigma to cull the data. default is 3 sigma. Anything
            less than 1 will be set to 1.

    """

    data = inputData.ravel()
    if type(data).__name__ == "MaskedArray":
        data = data.compressed()

    data0 = np.median(data)
    maxAbsDev = np.median(np.abs(data - data0)) / 0.6745
    if maxAbsDev < __epsilon:
        maxAbsDev = (np.abs(data - data0)).mean() / 0.8000

    cutOff = Cut * maxAbsDev
    good = np.where(np.abs(data - data0) <= cutOff)
    good = good[0]
    dataMean = data[good].mean()
    dataSigma = math.sqrt(((data[good] - dataMean) ** 2.0).sum() / len(good))

    if Cut > 1.0:
        sigmaCut = Cut
    else:
        sigmaCut = 1.0
    if sigmaCut <= 4.5:
        dataSigma = dataSigma / (-0.15405 + 0.90723 * sigmaCut - 0.23584 * sigmaCut ** 2.0 + 0.020142 * sigmaCut ** 3.0)

    cutOff = Cut * dataSigma
    good = np.where(np.abs(data - data0) <= cutOff)
    good = good[0]
    dataMean = data[good].mean()
    if len(good) > 3:
        dataSigma = math.sqrt(((data[good] - dataMean) ** 2.0).sum() / len(good))

    if Cut > 1.0:
        sigmaCut = Cut
    else:
        sigmaCut = 1.0
    if sigmaCut <= 4.5:
        dataSigma = dataSigma / (-0.15405 + 0.90723 * sigmaCut - 0.23584 * sigmaCut ** 2.0 + 0.020142 * sigmaCut ** 3.0)

    dataSigma = dataSigma / math.sqrt(len(good) - 1)

    return dataMean


def std(inputData, Zero=False):
    """
    Robust estimator of the standard deviation of a data set.

    Based on the robust_sigma function from the AstroIDL User's Library.
    """

    data = inputData.ravel()
    if type(data).__name__ == "MaskedArray":
        data = data.compressed()

    if Zero:
        data0 = 0.0
    else:
        data0 = np.median(data)
    maxAbsDev = np.median(np.abs(data - data0)) / 0.6745
    if maxAbsDev < __epsilon:
        maxAbsDev = (np.abs(data - data0)).mean() / 0.8000
    if maxAbsDev < __epsilon:
        sigma = 0.0
        return sigma

    u = (data - data0) / 6.0 / maxAbsDev
    u2 = u ** 2.0
    good = np.where(u2 <= 1.0)
    good = good[0]
    if len(good) < 3:
        print "WARNING:  Distribution is too strange to compute standard deviation"
        sigma = -1.0
        return sigma

    numerator = ((data[good] - data0) ** 2.0 * (1.0 - u2[good]) ** 2.0).sum()
    nElements = (data.ravel()).shape[0]
    denominator = ((1.0 - u2[good]) * (1.0 - 5.0 * u2[good])).sum()
    sigma = nElements * numerator / (denominator * (denominator - 1.0))
    if sigma > 0:
        sigma = math.sqrt(sigma)
    else:
        sigma = 0.0

    return sigma


def rotate_point(point, angle, ishape, rshape, reverse=False):
    '''
    Transform a point from origional image coordinates to rotated image
    coordinates and back. It assumes the rotation point is the center of an
    image.

    This works on a simple rotation transformation:

    newx = (startx) * math.cos(angle) - (starty) * math.sin(angle)
    newy = (startx) * math.sin(angle) + (starty) * math.cos(angle)

    It takes into account the differences in image size.

    Returns:
        Tuple of rotated point in (x, y) as measured from origin.

    Input:
        point - Point to be rotated. should be a tuple (x, y) measured from
        origin.

        angle - The angle in degrees to rotate the point by as measured
            counter-clockwise from the X axis.

        ishape - The shape of the original image. in the form of image.shape

        rshape - The shape of the rotated image. in the form of rotate.shape

    keyword input:
        reverse - Transform from rotated coordinates back to non-rotated image.

    '''
    #  unpack the image and rotated images shapes
    if reverse:
        angle = (angle * -1)
        temp = ishape
        ishape = rshape
        rshape = temp

    # transform into center of image coordinates
    yhalf, xhalf = ishape
    yrhalf, xrhalf = rshape

    yhalf = yhalf / 2
    xhalf = xhalf / 2
    yrhalf = yrhalf / 2
    xrhalf = xrhalf / 2

    startx = point[0] - xhalf
    starty = point[1] - yhalf

    # do the rotation
    newx = (startx) * math.cos(angle) - (starty) * math.sin(angle)
    newy = (startx) * math.sin(angle) + (starty) * math.cos(angle)

    #  add back the padding from changing the size of the image
    newx = newx + xrhalf
    newy = newy + yrhalf

    return (newx, newy)


def make_mask(filename, ext, result, sublen=75, subwidth=200, order=3,
              sigma=4, pad=10, update_dq=False, plot=False):
    '''
    Create a good mask for an image based on the detsat results. And updates
    the DQ array with value of 16384 where the satellite trail crosses the
    image.

    Still a lot of debugging lines in this function...

    Returns:
        flat - list of arrays.
        mask - mask for the image.
    (These are probably not needed anymore, mostly for debugging)


    Input:
        Filename - an ACS fits file, should be either flt or flc format

        ext - extension to work on. for ACS, either 1 or 4 for science arrays.

        result - A list of tuples that contain end points of line semgents
            of the satellite trail as generated by satdet.

    keyword input:
        sublen - length of strip to use as the fitting window for the trail.
            Default is 100.

        subwidth - width of box to fit trail on.
            Default is 150.

        order - The order of the spline interpolation for image rotation. See
            SciKitImage.transform.rotate for exact documentation.
            Default is 3

        sigma - Sigma of the satellite trail to detect trail. If points are
            a given sigma above the background in the subregion then it is
            marked as a satellite. This may need to be lowered for resolved
            trails.
            Default is 3

        pad - amount of extra padding in pixels to give the satellite mask.
            Default is 10

        update_dq - flag to update the dq array of the given image or not.

    '''
    image = fits.getdata(filename, ext)
    line = result[5]
    p0, p1 = line

    #  Find out how much to rotate the image
    rad = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    newrad = (math.pi * 2) - rad
    deg = math.degrees(rad)

    print 'Rotation: ' + str(deg)

    #  rescale and rotate the image
    image = image / image.max()
    #  make sure everything is at least 0
    z = np.where(image < 0)
    image[z] = 0

    rotate = transform.rotate(image, deg, resize=True, order=order)

    #  Will do all of this in the loop, but want to make sure there is a
    #  good point first and that there is indeed a profile to fit.
    #  get starting point
    sx, sy = rotate_point(p0, newrad, image.shape, rotate.shape)

    #  start with one subarray around p0
    subr = rotate[sy - sublen:sy + sublen, sx - subwidth / 2:sx + subwidth / 2]

    #  Flatten the array so we are looking along rows
    #  Take median of each row, should filter out most outliers
    flat = [np.median(subr, axis=1)]

    if plot:
        plt.figure(num=1)
        plt.clf()
        plt.imshow(image, vmin=.0008, vmax=.008, cmap=plt.cm.gray)

        plt.ion()
        plt.figure(num=2)
        plt.clf()
        plt.imshow(rotate, vmin=.0008, vmax=.008, cmap=plt.cm.gray)

    # get the outliers
    mean = biweightMean(flat[0])
    mean = robust_mean(flat[0], Cut=sigma)
    stddev = std(flat[0])

    # only flag things that are sigma from the mean
    z = np.where(flat[0] > mean + (sigma * stddev))
    z = z[0]
    # Make sure there is something in the first pass before trying to move on
    if len(z) < 1:
        print 'First look at finding a profile failed...'
        print 'Nothing found at ' + str(sigma) + ' from background!'
        print 'Try adjusting perameters and trying again'
        if plot:
            plt.figure(num=3)
            plt.clf()
            plt.plot(flat[0], 'b.')
            plt.plot(z, flat[0][z], 'r.')
        return

    if plot:
        plt.figure(num=3)
        plt.clf()
        plt.plot(flat[0], 'b.')
        plt.plot(z, flat[0][z], 'r.')

    # get the bounds of the flagged points
    lower = z.min()
    upper = z.max()
    diff = upper - lower

    # add in a pading value to make sure all of the wings are accounted for
    lower = lower - pad
    upper = upper + pad

    #  for plotting see how the profile was made
    if plot:
        padind = np.arange(lower, upper)
        plt.plot(padind, flat[0][padind], 'yx')

    # start to create a mask
    mask = np.zeros(rotate.shape)
    lowerx = math.floor(sx - subwidth)
    lowery = math.floor(sy - sublen + lower)
    upperx = math.ceil(sx + subwidth)
    uppery = math.ceil(sy - sublen + upper)

    mask[lowery:uppery, lowerx:upperx] = 1

    done = False
    first = True
    nextx = math.ceil(sx + subwidth)
    centery = math.ceil(sy - sublen + lower + diff)
    counter = 0
    # debugging array
    test = rotate.copy()

    while not done:
        if first:  # move to the right of the centerpoint first. do the same
                   # as above but keep moving right until the edge is hit.
            subr = rotate[centery - sublen:centery + sublen, nextx - subwidth/2:nextx + subwidth/2]

            # determines the edge, if the subr is not good, then the edge was
            # hit.
            if subr.shape[1] == 0:
                first = False
                print 'subr is 0 in first'
                #print counter
                centery = sy
                nextx = sx
                continue
            flat.append(np.median(subr, axis=1))

            mean = biweightMean(flat[-1])
            mean = robust_mean(flat[0], Cut=sigma)
            stddev = std(flat[-1])

            z = np.where(flat[-1] > mean + (sigma * stddev))
            z = z[0]
            if len(z) < 1:
                print 'z is less than 1, no good profile found.'
                print 'start moving left from starting point'
                #print counter
                centery = sy
                nextx = sx
                first = False
                continue

            lower = z.min()
            upper = z.max()
            diff = upper - lower

            lower = math.floor(lower - pad)
            upper = math.ceil(upper + pad)

            lowerx = math.floor(nextx - subwidth)
            lowery = math.floor(centery - sublen + lower)
            upperx = math.ceil(nextx + subwidth)
            uppery = math.ceil(centery - sublen + upper)

            lower_p = (lowerx, lowery)
            upper_p = (upperx, uppery)

            mask[lowery:uppery, lowerx:upperx] = 1
            test[lowery:uppery, lowerx:upperx] = 1

            lower_t = rotate_point(lower_p, newrad, image.shape, rotate.shape, reverse=True)
            upper_t = rotate_point(upper_p, newrad, image.shape, rotate.shape, reverse=True)

            lowy = math.floor(lower_t[1])
            highy = math.ceil(upper_t[1])
            lowx = math.floor(lower_t[0])
            highx = math.ceil(upper_t[0])

            # Reset the next subr to be at the center of the profile
            nextx = nextx + subwidth / 2
            centery = centery - sublen + lower + diff

            if (nextx + subwidth) > rotate.shape[1]:
#                plt.figure(num=10)
#                plt.clf()
#                plt.plot(flat[-1], 'b.')
                first = False
                centery = sy
                nextx = sx
                print 'hit rotate edge'
                print counter

            if (highy > image.shape[0]) or (highx > image.shape[1]):
#                plt.figure(num=10)
#                plt.clf()
#                plt.plot(flat[-1], 'b.')
                first = False
                centery = sy
                nextx = sx
                print 'hit image edge'
                print counter

        else:  # Not first, this is the pass the other way.
            subr = rotate[centery - sublen:centery + sublen, nextx - subwidth/2:nextx + subwidth/2]
            if subr.shape[1] == 0:
                done = True
                print '\nsubr is 0'
                continue
            flat.append(np.median(subr, axis=1))

            mean = biweightMean(flat[-1])
            mean = robust_mean(flat[0], Cut=sigma)
            stddev = std(flat[-1])

            z = np.where(flat[-1] > mean + (sigma * stddev))
            z = z[0]
            if len(z) < 1:
                print 'z is less than 1'
                print subr.shape
                done = True
                continue

            lower = z.min()
            upper = z.max()
            diff = upper - lower

            lower = math.floor(lower - pad)
            upper = math.ceil(upper + pad)

            lowerx = math.floor(nextx - subwidth)
            lowery = math.floor(centery - sublen + lower)
            upperx = math.ceil(nextx + subwidth)
            uppery = math.ceil(centery - sublen + upper)

            lower_p = (lowerx, lowery)
            upper_p = (upperx, uppery)

            mask[lowery:uppery, lowerx:upperx] = 1
            test[lowery:uppery, lowerx:upperx] = 1

            lower_t = rotate_point(lower_p, newrad, image.shape, rotate.shape, reverse=True)
            upper_t = rotate_point(upper_p, newrad, image.shape, rotate.shape, reverse=True)

            lowy = math.floor(lower_t[1])
            highy = math.ceil(upper_t[1])
            lowx = math.floor(lower_t[0])
            highx = math.ceil(upper_t[0])

            # Reset the next subr to be at the center of the profile
            nextx = nextx - subwidth / 2
            centery = centery - sublen + lower + diff

            if (nextx - subwidth) < 0:
                done = True
                print 'hit rotate edge'

            if (highy > image.shape[0]) or (highx > image.shape[1]):
                done = True
                print 'hit edge'

        counter += 1
        # make sure it does not try to go infinetly
        if counter > 500:
            print 'Too many loops, exiting'
            done = True

    if plot:
        plt.figure(num=5)
        plt.clf()
        plt.imshow(test, vmin=.0008, vmax=.008, cmap=plt.cm.gray)

    rot = transform.rotate(mask, -deg, resize=True, order=2)
    mask = rot[(rot.shape[0] - image.shape[0]) / 2:image.shape[0] + (rot.shape[0] - image.shape[0]) / 2, (rot.shape[1] - image.shape[1]) / 2:image.shape[1] + (rot.shape[1] - image.shape[1]) / 2]

    if plot:
        plt.figure(num=6)
        plt.clf()
        plt.imshow(mask, cmap=plt.cm.gray)

    if update_dq:
        mask_dq(filename, ext + 2, mask)

    return flat, mask
