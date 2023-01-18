#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module contains a class called trailfinder that is used to identify
satellite trails and/or other linear features in astronomical image data. To
accomplish this goal, the Median Radon Transform (MRT) is calculated for an
image. Point sources are then extracted from the MRT and filtered to yield a
final catalog of trails. These trails can then be used to create a mask.

Example 1: Identification of trails in an ACS/WFC image, j97006j5q_flc.fits
(4th extension)

Load data

>>> from acstools.findsat_mrt import trailfinder
>>> from astropy.io import fits
>>> import numpy as np
>>> file = 'j97006j5q_flc.fits'
>>> extension = 4
>>> with fits.open(file) as h:
>>>     image = h[extension].data
>>>     dq = h[extension+2].data

Mask bad pixels, remove median background, and rebin the data to speed up MRT
calculation

>>> from astropy.nddata import bitmask
>>> import ccdproc
>>> mask = bitmask.bitfield_to_boolean_mask(dq,
                                            ignore_flags=[4096, 8192, 16384])
>>> image[mask == True]=np.nan
>>> image = image-np.nanmedian(image)
>>> image = ccdproc.block_reduce(image, 4, func=np.nansum)

Initialize trailfinder and run steps

>>> s = trailfinder(image=image, threads=8)  # initializes
>>> s.run_mrt()                       # calculates MRT
>>> s.find_mrt_sources()              # finds point sources in MRT
>>> s.filter_sources()                # filters sources from MRT
>>> s.make_mask()                     # makes a mask from the identified trails
>>> s.save_output(root='test')        # saves the output

Example 2: Quick run to find satellite trails

After loading and preprocessing the image (see example above), run

>>> s = trailfinder(image=image, threads=8)  # initializes
>>> s.run_all()                              # runs everything else

Example 3: Plot results

After running trailfinder:

>>> s = trailfinder(image=image, threads=8)  # initializes
>>> s.run_all()                        # runs everything else
>>> s.plot_mrt(show_sources=True)      # plots MRT with sources overlaid
>>> s.plot_image(overlay_mask=True)    # plots input image with mask overlaid

"""


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.detection import StarFinder
import ccdproc
import os
from astropy.table import Table
import acstools.utils_findsat_mrt as u
from astropy.nddata import bitmask
import logging


__taskname__ = "findsat_mrt"
__author__ = "David V. Stark"
__version__ = "1.0"
__vdate__ = "16-Dec-2022"
__all__ = ['trailfinder', 'wfc_wrapper']

plt.rcParams['font.size'] = '18'
package_directory = os.path.dirname(os.path.abspath(__file__))

# Initialize the logger
logging.basicConfig()
LOG = logging.getLogger(f'{__taskname__}')
LOG.setLevel(logging.INFO)


class trailfinder(object):

    def __init__(
             self,
             image=None,
             header=None,
             image_header=None,
             save_image_header_keys=[],
             threads=2,
             min_length=25,
             max_width=75,
             buffer=250,
             threshold=5,
             theta=np.arange(0, 180, 0.5),
             kernels=[package_directory+'/data/rt_line_kernel_width{}.fits'.
                      format(k) for k in [15, 7, 3]],
             plot=False,
             output_dir='.',
             output_root='',
             check_persistence=True,
             min_persistence=0.5,
             ignore_theta_range=None,
             save_catalog=True,
             save_diagnostic=True,
             save_mrt=False,
             save_mask=False
     ):
        '''
        Class to identify satellite trails in image data using the Median
        Radon Transform.

        Parameters
        ----------
        image : ndarray, optional
            Input image. The default is None, but nothing will work until this
            is defined
        header: Header,optional
            The header for the input data (0th extension). This is not used for
            anything during the analysis, but it is saved with the output mask
            and satellite trail catalog so information about the original
            observation can be easily retrieved.
        image_header: Header, optional
            The specific header for the fits extension being used. This is
            added onto the catalog
        save_image_header_keys: list, optional
            List of header keys from image_header to save in the output trail
            catalog header. Default is None.
        threads : int, optional
            Number of threads to use when calculating MRT. The default is 1.
        min_length : int, optional
            Minimum streak length to allow. The default is 25 pixels.
        max_width : int, optional
            Maximum streak width to allow. The default is 75 pixels.
        buffer : int, optional
            Size of cutout region extending perpendicular outward from a
            streak. The default is 250 pixels on each side.
        threshold : float, optional
            Minimum S/N when extracting sources from the MRT. The default is 5.
        theta : ndarray, optional
            Angles at which to calculate the MRT. The default is
            np.arange(0,180,0.5).
        kernels : list, optional
            Paths to each kernel to be used for source finding in the MRT.
            The default is to use the 3, 7, and 15 pixel wide kernels included
            with this package.
        plot : bool, optional
            Plots all intermediate steps. The default is False. Warning:
                setting this option generates A LOT of plots. It's essentially
                just for debugging purposes'
        output_dir : string, optional
            Path in which to save output. The default is './'.
        output_root : string, optional
            A prefix for all output files. The default is ''.
        check_persistence : bool, optional
            Calculates the persistence of all identified streaks. The default
            is True.
        min_persistence : float, optional
            Minimum persistence of a "true" satellite trail. Must be between 0
            and 1. The default is 0.5. Note that this does not reject satellite
            trails from the output catalog, but highlights them in a different
            color in the output plot.
        ignore_theta_range : array-like, optional
            List if ranges in theta to ignore when identifying satellite
            trails. This parameter is most useful for avoiding false positives
            due to diffraction spikes that always create streaks around the
            same angle for a given telescope/instrument. Format should be a
            list of tuples, e.g., [(theta0_a,theta1_a),(theta0_b,theta1_b)].
            Default is None.
        save_catalog: bool, optional
            Set to save the catalog of identified trails. Default is True
        save_diagnostic: bool, optional
            Set to save a diagnotic plot showing the input image and identified
            trails. Default is True.
        save_mrt: bool, optional
            Set to save the MRT in a fits file. Default is false.
        save_mask: bool, optional
            Set to save the trail mask in a fits file. Default is false.

        Returns
        -------
        None.

        '''

        # inputs
        self.image = image
        self.header = header
        self.image_header = image_header
        self.save_image_header_keys = save_image_header_keys
        self.threshold = threshold
        self.min_length = min_length
        self.max_width = max_width
        self.kernels = kernels
        self.threads = threads
        self.theta = theta
        self.plot = plot
        self.buffer = buffer
        self.check_persistence = check_persistence
        self.min_persistence = min_persistence
        self.ignore_theta_range = ignore_theta_range

        # outputs
        self.mrt = None
        self.mrt_err = None
        self.length = None
        self.rho = None
        self.mask = None

        # some internal things
        self._madrt = None
        self._medrt = None
        self._image_mad = None
        self._image_stddev = None

        # info for saving output
        self.output_dir = output_dir
        self.root = output_root
        self.save_catalog = save_catalog
        self.save_diagnostic = save_diagnostic
        self.save_mrt = save_mrt
        self.save_mask = save_mask

        # plot image upon initialization
        if (image is not None) & (self.plot is True):
            self.plot_image()

    def run_mrt(self, theta=None, threads=None):
        '''
        Runs the median radon transform on the input image

        Parameters
        ----------
        theta : TYPE, optional
            DESCRIPTION. The default is None.
        threads : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''

        if theta is None:
            theta = self.theta
        if threads is None:
            threads = self.threads

        rt, length = u.radon(self.image, circle=False, median=True,
                             fill_value=np.nan, threads=threads,
                             return_length=True, theta=theta)

        # automatically trim rt where length too short
        rt[length < self.min_length] = np.nan

        # save various outputs
        self.mrt = rt
        self.length = length

        # calculate some properties
        # median
        self._medrt = np.nanmedian(rt)
        # median abs deviation
        self._madrt = np.nanmedian(np.abs(rt[np.abs(rt) > 0])-self._medrt)

        # calculate the approximate uncertainty of the MRT at each point
        self._image_mad = np.nanmedian(np.abs(self.image))
        # using MAD to avoid influence from outliers
        self._image_stddev = self._image_mad/0.67449
        # error on median ~ 1.25x error on mean
        self.mrt_err = 1.25*self._image_stddev/np.sqrt(self.length)

        # calculate rho array
        rho0 = rt.shape[0]/2-0.5
        self.rho = np.arange(rt.shape[0])-rho0

        if self.plot is True:
            self.plot_mrt()

    def plot_image(self, ax=None, scale=[-1, 5], overlay_mask=False):
        '''
        Plots the input image

        Parameters
        ----------
        ax : AxesSubplot, optional
            A matplotlib subplot where the image should be shown. The default
            is None.
        scale : array-like, optional
            A two element array with the minimum and maximum image values used
            to set the color scale, in units of the image median absolute
            deviation. The default is [-1,5].
        overlay_mask: bool, optional
            Set to overlay the trail mask, if already calculated. Default is
            False.

        '''

        if self.image is None:
            raise ValueError('No image to plot')
            
        if ax is None:
            fig, ax = plt.subplots()

        # recaluclate mad and stdev here in case it hasn't been done yet

        self._image_mad = np.nanmedian(np.abs(self.image))
        self._image_stddev = self._image_mad/0.67449  # using MAD to avoid
        # influence from outliers
        ax.imshow(self.image, cmap='viridis', origin='lower', aspect='auto',
                  vmin=scale[0]*self._image_stddev,
                  vmax=scale[1]*self._image_stddev)
        ax.set_xlabel('X [pix]')
        ax.set_ylabel('Y [pix]')
        ax.set_title('Input Image')

        if overlay_mask is True:
            if self.mask is None:
                LOG.error('No mask to overlay')
            else:
                ax.imshow(self.mask, alpha=0.5, cmap='Reds', origin='lower',
                          aspect='auto')

    def plot_mrt(self, scale=[-1, 5], ax=None, show_sources=False):
        '''
        Plots the MRT

        Parameters
        ----------
        scale : array-like, optional
            A two element array with the minimum and maximum image values used
            to set the color scale, in units of the MRT median absolute
            deviation. The default is [-1,5].
        ax : AxesSubplot, optional
            A matplotlib subplot where the MRT should be shown. The default is
            None.
        show_sources: bool
            Indicates the positions of the detected sources. Default is False

        Returns
        -------
        ax : AxesSubplot
            Matplotlib subplot where the MRT is plotted.

        '''

        if self.mrt is None:
            LOG.error('No MRT to plot')
            return

        if (show_sources is True) and (self.source_list is None):
            show_sources = False
            LOG.info('No sources to show')

        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(self.mrt, aspect='auto', origin='lower',
                  vmin=scale[0]*self._madrt, vmax=scale[1]*self._madrt)
        ax.set_title('MRT')
        ax.set_xlabel('angle(theta) pixel')
        ax.set_ylabel('offset(rho) pixel')

        if show_sources is True:
            x = self.source_list['xcentroid']
            y = self.source_list['ycentroid']
            status = self.source_list['status']

            for s, color in zip([0, 1, 2], ['red', 'orange', 'cyan']):
                sel = (status == s)
                if np.sum(sel) > 0:
                    ax.scatter(x[sel], y[sel], edgecolor=color,
                               facecolor='none', s=100, lw=2,
                               label='status={}'.format(s))
            ax.legend(loc='upper center')

        return ax

    def plot_mrt_snr(self, scale=[1, 25], ax=None):
        '''
        Plots a map of the MRT signal-to-noise ratio

        Parameters
        ----------
        scale : array-like, optional
            A two element array with the minimum and maximum image values used
            to set the color scale. The default is [1,25].
        ax : AxesSubplot, optional
            A matplotlib subplot where the SNR should be shown. The default is
            None.

        Returns
        -------
        snr_map: array-like
            A map of the SNR

        '''

        if self.mrt is None:
            LOG.error('No MRT to plot')
            return

        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(self.mrt/self.mrt_err, aspect='auto', origin='lower',
                  vmin=scale[0], vmax=scale[1])
        ax.set_title('MRT SNR')
        ax.set_xlabel('angle(theta) pixel')
        ax.set_ylabel('offset(rho) pixel')

        return self.mrt/self.mrt_err

    def find_mrt_sources(self, kernels=None, threshold=None):
        '''
        Findings sources in the MRT consistent with satellite trails/streaks

        Parameters
        ----------
        kernels : array-like, optional
            List of kernels to use when finding sources.
        threshold : float, optional
            Minumum S/N of detected sources in the MRT.

        Returns
        -------
        source_list : Table
            Catalog containing information about detected trails

        '''

        if kernels is None:
            kernels = self.kernels
        if threshold is None:
            threshold = self.threshold

        LOG.info('Detection threshold: {}'.format(threshold))

        # cycle through kernels
        tbls = []
        for k in kernels:
            with fits.open(k) as h:
                kernel = h[0].data
            LOG.info('Using kernel {}'.format(k))
            s = StarFinder(threshold, kernel, min_separation=20,
                           exclude_border=False, brightest=None, peakmax=None)
            try:
                snrmap = self.mrt/self.mrt_err
                tbl = s.find_stars(snrmap, mask=~np.isfinite(snrmap))
            except:
                tbl = None
            if tbl is not None:
                tbl = tbl[np.isfinite(tbl['xcentroid'])]
                LOG.info('{} sources found using kernel'.format(len(tbl)))
                if (len(tbls) > 0):
                    if len(tbls[-1]['id']) > 0:
                        tbl['id'] += np.max(tbls[-1]['id'])
                        # adding max ID number from last iteration to avoid
                        # duplicate ids
                tbls.append(tbl)
            else:
                LOG.info('no sources found using kernel')
        # combine tables from each kernel and remove any duplicates
        if len(tbls) > 0:
            LOG.info('Removing duplicate sources')
            sources = u.merge_tables(tbls)
            self.source_list = sources
        else:
            self.source_list = None

        # add the theta and rho arrays
        if self.source_list is not None:
            dtheta = self.theta[1]-self.theta[0]
            self.source_list['theta'] = \
                self.theta[0]+dtheta*self.source_list['xcentroid']
            self.source_list['rho'] = \
                self.rho[0] + self.source_list['ycentroid']

        # add the status array and endpoints array. Everything will be zero
        # because no additional checks have been done
            self.source_list['endpoints'] = \
                [u.streak_endpoints(t['rho'], -t['theta'], self.image.shape)
                 for t in self.source_list]
            self.source_list['status'] = \
                np.zeros(len(self.source_list)).astype(int)

        # run the routine to remove angles if any bad ranges are specified
        if (self.source_list is not None) and (self.ignore_theta_range is not
                                               None):
            self._remove_angles()

        #print the total number of sources found
        if self.source_list is None:
            LOG.warning('No sources found')
        else:
            LOG.info('{} final sources found'.format(len(self.source_list)))
            # plot sources if set
            if (self.plot is True):
                ax = self.plot_mrt()
                for s in self.source_list:
                    ax.scatter(s['xcentroid'], s['ycentroid'], edgecolor='red',
                               facecolor='none', s=100, lw=2)

        return self.source_list

    def filter_sources(self, threshold=None, maxwidth=None, trim_catalog=False,
                       min_length=None, buffer=None, plot=None,
                       check_persistence=None, min_persistence=None):
        '''
        Filters an input catalog of trails based on their remeasured S/N,
        width, and persistence to determine which are robust.

        Parameters
        ----------
        threshold : float, optional
            Minimum S/N of trail to be considered robust. The default is None.
        maxwidth : int, optional
            Maximum width of a trail to be considered robust. The default is
            None.
        trim_catalog : bool, optional
            Flag to remove all filtered trails from the source catalog. The
            default is False.
        min_length : int, optional
            Minimum allowed length of a satellite trail. The default is None.
        buffer : int, optional
            Size of the cutout region around each trail when analyzing its
            properties. The default is None.
        plot : bool, optional
            Set to plot the MRT with the resulting filtered sources overlaid.
            The default is None.
        check_persistence : bool, optional
            Set to turn on the persistence check. The default is None.
        min_persistence : float, optional
            Minimum persistence for a trail to be considered robust. The
            default is None.

        Returns
        -------
        source_list : table
            Catalog of identified satellite trails with additional measured
            parameters appended.

        '''

        if threshold is None:
            threshold = self.threshold
        if min_length is None:
            min_length = self.min_length
        if maxwidth is None:
            maxwidth = self.max_width
        if buffer is None:
            buffer = self.buffer
        if plot is None:
            plot = self.plot
        if check_persistence is None:
            check_persistence = self.check_persistence
        if min_persistence is None:
            min_persistence = self.min_persistence

        # turn rho/theta coordinates into endpoints
        if self.source_list is not None:

            LOG.info('Filtering sources...')
            LOG.info('Min SNR : {}'.format(threshold))
            LOG.info('Max Width: {}'.format(maxwidth))
            LOG.info('Min Length: {}'.format(min_length))
            LOG.info('Check persistence: {}'.format(check_persistence))

            if check_persistence is True:
                LOG.info('Min persistence: {}'.format(min_persistence))

            # run filtering routine
            properties = u.filter_sources(self.image,
                                          self.source_list['endpoints'],
                                          max_width=maxwidth, buffer=250,
                                          plot=plot, min_length=min_length,
                                          minsnr=threshold,
                                          check_persistence=check_persistence,
                                          min_persistence=min_persistence)

            # update the status
            self.source_list.update(properties)

        if trim_catalog is True:
            sel = (self.source_list['width'] < maxwidth) & \
                (self.source_list['snr'] > threshold)
            self.source_list = self.source_list[sel]

        if plot is True:
            fig, ax = plt.subplots()
            self.plot_mrt(show_sources=True)

        return self.source_list

    def make_mask(self, include_status=[2], plot=None):
        '''
        Makes a 1/0 satellite trail mask and a segmentation image with each
        trail numbered based on the identified trails.

        Parameters
        ----------
        include_status : list, optional
            List of status flags to include. The default is [2].
        plot : bool, optional
            Set to generate a plot images of the mask and segmentation image.
            The default is None.

        Returns
        -------
        None.

        '''

        if plot is None:
            plot = self.plot

        if self.source_list is not None:

            include = [s['status'] in include_status for s in self.source_list]
            trail_id = self.source_list['id'][include]
            endpoints = self.source_list['endpoints'][include]
            widths = self.source_list['width'][include]
            segment, mask = u.create_mask(self.image, trail_id, endpoints,
                                          widths)
        else:
            mask = np.zeros_like(self.image)
            segment = np.zeros_like(self.image)
        self.segment = segment.astype(int)
        self.mask = mask

        if plot is True:
            self.plot_mask()
            self.plot_segment()

    def plot_mask(self):
        '''
        Generates a plot of the trail mask

        Returns
        -------
        ax, AxesSubplot
            The Matplotlib subplot containing the mask image
        '''
        if self.mask is None:
            LOG.error('No mask to show')

        fig, ax = plt.subplots()
        ax.imshow(self.mask, origin='lower', aspect='auto')
        ax.set_title('Mask')

        return ax

    def plot_segment(self):
        '''
        Generates a segmentation image of the identified trails.

        Returns
        -------
        ax, AxesSubplot
            A matplotlib subplot containing the segmentation map

        '''
        if self.segment is None:
            LOG.error('No segment map to show')

        # get unique values in segment
        unique_vals = np.unique(self.segment)
        data = self.segment*0
        counter = 1
        for uv in unique_vals[1:]:
            data[self.segment == uv] = counter
            counter += 1

        fig, ax = plt.subplots()
        cmap = plt.get_cmap('tab20', np.max(data) - np.min(data) + 1)
        mat = ax.imshow(data, cmap=cmap, vmin=np.min(data) - 0.5,
                        vmax=np.max(data) + 0.5, origin='lower', aspect='auto')

        # tell the colorbar to tick at integers
        ticks = np.arange(0, len(unique_vals)+1)
        cax = plt.colorbar(mat, ticks=ticks)
        cax.ax.set_yticklabels(np.concatenate([unique_vals,
                                               [unique_vals[-1]+1]]))
        cax.ax.set_ylabel('trail ID')
        ax.set_title('Segmentation Mask')

    def save_output(self, root=None, output_dir=None, save_mrt=None,
                    save_mask=None, save_catalog=None, save_diagnostic=None):
        '''
        Saves output, including (1) MRT, (2) mask/segementation image,
        (3) catalog, and (4) trail catalog.

        Parameters
        ----------
        root : string, optional
            String to prepend to all output files. The default is None.
        output_dir : string, optional
            Directory in which to save output files. The default is None.
        save_mrt : bool, optional
            Set to save the MRT in a fits file. The default is None.
        save_mask : bool, optional
            Set to save the mask and segmentation images in a fits file. The
            default is None.
        save_catalog : bool, optional
            Set to save the trail catalog in a fits table. The default is None.
        save_diagnostic : bool, optional
            Set to save a diagnostic plot (png) showing the identified trails.
            The default is None.

        Returns
        -------
        None.

        '''

        if root is None:
            root = self.root
        if output_dir is None:
            output_dir = self.output_dir
        if save_mrt is None:
            save_mrt = self.save_mrt
        if save_mask is None:
            save_mask = self.save_mask
        if save_diagnostic is None:
            save_diagnostic = self.save_diagnostic
        if save_catalog is None:
            save_catalog = self.save_catalog

        if save_mrt is True:
            if self.mrt is not None:
                fits.writeto('{}/{}_mrt.fits'.format(output_dir, root),
                             self.mrt, overwrite=True)
            else:
                LOG.error('No MRT to save')

        if save_mask is True:
            if self.mask is not None:
                hdu0 = fits.PrimaryHDU()
                if self.header is not None:
                    hdu0.header = self.header  # copying original image header
                hdu1 = fits.ImageHDU(self.mask.astype(int))
                hdu2 = fits.ImageHDU(self.segment.astype(int))
                hdul = fits.HDUList([hdu0, hdu1, hdu2])
                hdul.writeto('{}/{}_mask.fits'.format(output_dir, root),
                             overwrite=True)
            else:
                LOG.error('No mask to save')

        if save_diagnostic is True:
            fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2,
                                                         figsize=(20, 10))
            self.plot_image(ax=ax1)
            self.plot_mrt(ax=ax2)
            self.plot_image(ax=ax3)
            ax3.imshow(self.mask, alpha=0.5, origin='lower', aspect='auto',
                       cmap='Reds')
            ax3_xlim = ax3.get_xlim()
            ax3_ylim = ax3.get_ylim()

            self.plot_mrt(ax=ax4)
            if self.source_list is not None:
                for s in self.source_list:
                    color = 'red'
                    x1, y1 = s['endpoints'][0]
                    x2, y2 = s['endpoints'][1]
                    if (s['status'] == 2):
                        color = 'turquoise'
                    elif (s['status'] == 1):
                        color = 'orange'
                    ax3.plot([x1, x2], [y1, y2], color=color, lw=5, alpha=0.5)
                    ax4.scatter(s['xcentroid'], s['ycentroid'],
                                edgecolor=color, facecolor='none', s=100, lw=2)
            # sometimes overplotting the "good" trails can cause axes to change
            ax3.set_xlim(ax3_xlim)
            ax3.set_ylim(ax3_ylim)
            plt.tight_layout()
            plt.savefig('{}/{}_diagnostic.png'.format(output_dir, root))
            if self.plot is False:
                plt.close()

        if save_catalog is True:
            if self.source_list is not None:
                self.source_list.write('{}/{}_catalog.fits'.format(output_dir,
                                                                   root),
                                       overwrite=True)
            else:
                # create an empty catalog and write that. It helps to have this
                # for future analysis purposes even if it's empty
                dummy_table = Table(names=('id', 'xcentroid', 'ycentroid',
                                           'fwhm', 'roundness', 'pa',
                                           'max_value', 'flux', 'mag',
                                           'theta', 'rho', 'endpoints',
                                           'width', 'snr', 'status',
                                           'persistence'),
                                    dtype=('int64', 'float64', 'float64',
                                           'float64', 'float64', 'float64',
                                           'float64', 'float64', 'float64',
                                           'float64', 'float64', 'float64',
                                           'float64', 'float64', 'float64',
                                           'float64'))
                dummy_table.write('{}/{}_catalog.fits'.format(output_dir,
                                                              root),
                                  overwrite=True)

            # I want to append the original data header to this too
            if (self.header is not None) | (self.image_header is not None):

                h = fits.open('{}/{}_catalog.fits'.format(output_dir, root),
                              mode='update')
                hdr = h[1].header

                if self.header is not None:
                    h[0].header = self.header

                if self.image_header is not None:
                    if type(self.save_image_header_keys == tuple):
                        self.save_image_header_keys = \
                            np.squeeze(list(self.save_image_header_keys))

                    # add individal header keywords now
                    for k in self.save_image_header_keys:
                        try:
                            hdr[k] = self.image_header[k]
                        except:
                            LOG.error('\nadding image header key {} \
                                      failed\n'.format(k))

                h.flush()

    def _remove_angles(self, ignore_theta_range=None):
        '''
        Set to remove a specific range (or set of ranges) of angles from the
        trail catalog. This is primarily for removing trails at angles known to
        be overwhelmingly dominated by features that are not of interest, e.g.,
        for removing diffraction spikes.

        Parameters
        ----------
        ignore_theta_range : list, optional
            List of angle ranges to avoid.
            Format is [(min angle1,max angle1),(min angle2, max angle2) ... ].
            The default is None.

        Returns
        -------
        source_list, Table
            The source list with the specified angles removed.

        '''

        if ignore_theta_range is None:
            ignore_theta_range == self.ignore_theta_range

        if self.ignore_theta_range is None:
            LOG.error('No angles set to ignore')
            return

        # add some checks to be sure ignore_ranges is the right type
        remove = np.zeros(len(self.source_list)).astype(bool)
        for r in self.ignore_theta_range:
            r = np.sort(r)
            LOG.info('ignoring angles between {} and {}'.format(r[0], r[1]))
            remove[(self.source_list['theta'] >= r[0]) &
                   (self.source_list['theta'] <= r[1])] = True

        self.source_list = self.source_list[~remove]

    def run_all(self):
        '''
        Simple wrapper code to run the entire pipeline to identify, filter, and
        mask trails

        Returns
        -------
        None.

        '''

        self.run_mrt()
        self.find_mrt_sources()
        self.filter_sources()
        self.make_mask()
        self.save_output()


class wfc_wrapper(trailfinder):

    def __init__(self,
                 image_file,
                 extension=None,
                 binsize=None,
                 preprocess=True,
                 execute=False,
                 **kwargs):
        '''
        Wrapper for trail_finder class designed specifically for ACS/WFC data.
        Enables quick reading and preprocessing of standard ACS/WFC images.


        Parameters
        ----------
        image_file : string
            ACS/WFC data file to read. Should be a fits file.
        extension : int, optional
            Extension of input file to read. The default is None.
        binsize : int, optional
            Amount the input data should be binned by. The default is None
            (no binning).
        preprocess : bool, optional
            Flag to run all the preprocessing steps (bad pixel flagging,
            background subtraction, rebinning. The default is True.
        execute : bool, optional
            Flag to run the entire trailfinder pipeline. The default is False.
        **kwargs : dict, optional
            Additional keyword arguments for trailfinder.

        Returns
        -------
        None.

        '''

        trailfinder.__init__(self, **kwargs)
        self.image_file = image_file
        self.binsize = binsize
        self.extension = extension

        # get image type
        h = fits.open(self.image_file)

        # get suffix to determine how to process image
        suffix = (self.image_file.split('.')[0]).split('_')[-1]
        self.image_type = suffix
        LOG.info('image type is {}'.format(self.image_type))

        if suffix in ['flc', 'flt']:
            if extension is None:
                LOG.warn('No extension specified. Defaulting to 1')
                extension = 1
            elif extension not in [1, 4]:
                LOG.error('Valid extensions are 1 and 4')
                return

            self.image = h[extension].data  # main image
            self.image_mask = h[extension+2].data  # dq array

        elif suffix in ['drc', 'drz']:
            extension = 1
            self.image = h[extension].data  # main image
            self.image_mask = h[extension+1].data  # weight array

        else:
            LOG.error('Image type not recognized')
            return

        # go ahead and run the proprocessing steps if set to True
        if preprocess is True:
            self.run_preprocess()

        if execute is True:
            LOG.info('Running the trailfinding pipeline')
            self.run_all()

    def mask_bad_pixels(self, ignore_flags=[4096, 8192, 16384]):
        '''
        Masks bad pixels by replacing them with nan. Uses the bitmask arrays
        for flc/flt images, and weight arrays for drc/drz images

        Parameters
        ----------
        ignore_flags : list, optional
            List of DQ bitmasks to ignore when masking. Only relevant for
            flc/flt files. The default is [4096,8192,16384], which ignores
            cosmic ray flags

        Returns
        -------
        None.

        '''

        LOG.info('masking bad pixels')

        if self.image_type in ['flc', 'flt']:

            # for flc/flt, use dq array
            mask = bitmask.bitfield_to_boolean_mask(self.image_mask,
                                                    ignore_flags=ignore_flags)
            self.image[mask] = np.nan

        elif self.image_type in ['drz', 'drc']:

            # for drz/drc, mask everything with weight=0
            mask = self.image_mask == 0
            self.image[mask] = np.nan

    def subtract_background(self):
        '''
        Subtracts a median background from the image, ignoring NaNs.

        Returns
        -------
        None.

        '''

        LOG.info('Subtracting median background')
        self.image = self.image - np.nanmedian(self.image)

    def rebin(self, binsize=None):
        '''
        Rebins the image array. The x/y rebinning are the same. NaNs are
        ignored.

        Parameters
        ----------
        binsize : int, optional
            Bin size. The default is None.

        Returns
        -------
        None.

        '''

        if binsize is None:
            binsize = self.binsize

        if binsize is None:
            LOG.warn('No bin size defined')
            return

        LOG.info('Rebinning the data by {}'.format(binsize))

        self.image = ccdproc.block_reduce(self.image, binsize, func=np.nansum)

    def run_preprocess(self, **kwargs):
        '''
        Runs all the preprocessing steps together: mask_bad_pixels,
        subtract_background, rebin.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments for rebin and mask_bad_pixels.

        Returns
        -------
        None.

        '''

        self.mask_bad_pixels(**kwargs)
        self.subtract_background()
        self.rebin(**kwargs)

