#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module identifies satellite trails in ACS/WFC imaging with the Median Radon Transform.

It contains a class called `~acstools.findsat_mrt.TrailFinder` that is used to identify
satellite trails and/or other linear features in astronomical image data. To
accomplish this goal, the Median Radon Transform (MRT) is calculated for an
image. Point sources are then extracted from the MRT and filtered to yield a
final catalog of trails, which can then be used to create a mask.

A second class called `~acstools.findsat_mrt.WfcWrapper` is designed explicitly
to make ACS/WFC data easy to process.

This algorithm is found to be roughly 10x more sensitive compared to the
current satellite trail finding code included with acstools, :ref:`satdet`.
However, this approach can struggle with dense fields, while the performance
of :ref:`satdet` in these fields may be more reliable (but this has not yet been
tested).

For further details on this algorithm and tests of its performance, see the
`ISR ACS 2022-8 <https://ui.adsabs.harvard.edu/abs/2022acs..rept....8S/abstract>`_.

Examples
--------

**Example 1:** Identification of trails in an ACS/WFC image, ``j97006j5q_flc.fits``
(the second science and DQ extensions).

To load the data:

>>> import numpy as np
>>> from astropy.io import fits
>>> from acstools.findsat_mrt import TrailFinder
>>> file = 'j97006j5q_flc.fits'
>>> with fits.open(file) as h:
>>>     image = h['SCI', 2].data
>>>     dq = h['DQ', 2].data

Mask some bad pixels, remove median background, and rebin the data to speed up MRT
calculation:

>>> from astropy.nddata import bitmask, block_reduce
>>> mask = bitmask.bitfield_to_boolean_mask(
...     dq, ignore_flags=[4096, 8192, 16384])
>>> image[mask] = np.nan
>>> image = image - np.nanmedian(image)
>>> image = block_reduce(image, 4, func=np.nansum)

Initialize `~acstools.findsat_mrt.TrailFinder` and run these steps:

>>> s = TrailFinder(image, processes=8)  # initialization
>>> s.run_mrt()                       # calculates MRT
>>> s.find_mrt_sources()              # finds point sources in MRT
>>> s.filter_sources()                # filters sources from MRT
>>> s.make_mask()                     # makes a mask from the identified trails
>>> s.save_output(root='test')        # saves the output

The input image, mask, and MRT (with sources overlaid) can be plotted during
this process:

>>> s.plot_mrt(show_sources=True)      # plots MRT with sources overlaid
>>> s.plot_image(overlay_mask=True)    # plots input image with mask overlaid

**Example 2:** Quick run to find satellite trails.

After loading and preprocessing the image (see example above), run
the following:

>>> s = TrailFinder(image, processes=8)  # initialization
>>> s.run_all()                              # runs everything else

**Example 3:** Run identification/masking using the WFC wrapper.

`~acstools.findsat_mrt.WfcWrapper` can automatically do the binning, background subtraction, and
bad pixel flagging:

>>> from acstools.findsat_mrt import WfcWrapper
>>> w = WfcWrapper('jc8m32j5q_flc.fits', preprocess=True, extension=4, binsize=2)

In all other respects, it behaves just like `~acstools.findsat_mrt.TrailFinder`,
so to continue the process:

>>> w.run_mrt()
>>> w.find_sources()
>>> w.filter_sources()
>>> w.make_mask()

Or the entire process can be run in a single line with

>>> w = WfcWrapper('jc8m32j5q_flc.fits', preprocess=True, extension=4, binsize=2,
...                 execute=True)

"""
import logging
import multiprocessing
import os
import warnings

import numpy as np
from astropy.io import fits
from astropy.nddata import bitmask, block_reduce
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename

from acstools.utils_findsat_mrt import (create_mask, filter_sources, merge_tables,
                                        radon, streak_endpoints, update_dq)

# test for matplotlib, turn off plotting if it does not exist
try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    warnings.warn('matplotlib not found, plotting is disabled')

__taskname__ = "findsat_mrt"
__author__ = "David V. Stark"
__version__ = "1.0"
__vdate__ = "10-Feb-2023"
__all__ = ['TrailFinder', 'WfcWrapper']

# Initialize the logger
logging.basicConfig()
LOG = logging.getLogger(f'{__taskname__}')
LOG.setLevel(logging.INFO)


class TrailFinder:
    '''Top-level class to handle trail identification and masking.

    Parameters
    ----------
    image : ndarray
        Input image.
    header : `astropy.io.fits.Header`, optional
        The primary header for the input data (usually the 0th extension). This is not used for
        anything during the analysis, but it is saved with the output mask
        and satellite trail catalog so information about the original
        observation can be easily retrieved. Default is `None`.
    image_header : `astropy.io.fits.Header`, optional
        The specific header for the FITS extension being used. This is
        added onto the catalog. Default is `None`.
    save_image_header_keys : list, optional
        See :attr:`~acstools.utils_findsat_mrt.TrailFinder.save_image_header_keys`.
        Default is an empty list (nothing saved from ``image_header``).
    processes : int, optional
        See :attr:`~acstools.utils_findsat_mrt.TrailFinder.processes`.
        The default is 2.
    min_length : int, optional
        See :attr:`~acstools.utils_findsat_mrt.TrailFinder.min_length`.
        The default is 25 pixels.
    max_width : int, optional
        See :attr:`~acstools.utils_findsat_mrt.TrailFinder.max_width`.
        The default is 75 pixels.
    buffer : int, optional
        See :attr:`~acstools.utils_findsat_mrt.TrailFinder.buffer`.
        The default is 250 pixels on each side.
    threshold : float, optional
        See :attr:`~acstools.utils_findsat_mrt.TrailFinder.threshold`.
        The default is 5.
    theta : ndarray, optional
        See :attr:`~acstools.utils_findsat_mrt.TrailFinder.theta`.
        The default is `None`, which sets to ``numpy.arange(0, 180, 0.5)``.
    kernels : list, optional
        See :attr:`~acstools.utils_findsat_mrt.TrailFinder.kernels`.
        The default is `None`, which reverts to using the 3-, 7-, and 15-pixel
        wide kernels included with this package.
    mask_include_status : list, optional
        See :attr:`~acstools.utils_findsat_mrt.TrailFinder.mask_include_status`.
        The default is ``[2]``.
    plot : bool, optional
        Plot all intermediate steps. The default is `False`. Note that setting
        this option can generate A LOT of plots. It is primarily for debugging
        purposes.
    output_dir : str, optional
        Path in which to save output. The default is ``'.'`` (current directory).
    output_root : string, optional
        See :attr:`~acstools.utils_findsat_mrt.TrailFinder.root`.
        The default is ``''`` (no prefix).
    check_persistence : bool, optional
        See :attr:`~acstools.utils_findsat_mrt.TrailFinder.check_persistence`.
        The default is `True`.
    min_persistence : float, optional
        See :attr:`~acstools.utils_findsat_mrt.TrailFinder.min_persistence`.
        Must be between 0 and 1. The default is 0.5.
    ignore_theta_range : list of tuples, optional
        List of ranges in ``theta`` to ignore when identifying satellite
        trails. This parameter is most useful for avoiding false positives
        due to diffraction spikes that always create streaks around the
        same angle for a given telescope/instrument. Format should be a
        list of tuples, e.g., ``[(theta0_a, theta1_a), (theta0_b, theta1_b)]``.
        Default is `None`.
    save_catalog : bool, optional
        Save the catalog of identified trails. Default is `True`.
    save_diagnostic : bool, optional
        Save a diagnotic plot showing the input image and identified
        trails. Default is `True`.
    save_mrt : bool, optional
        Save the MRT in a FITS file. Default is `False`.
    save_mask : bool, optional
        Save the trail mask in a FITS file. Default is `False`.

    '''
    def __init__(
            self,
            image,
            header=None,
            image_header=None,
            save_image_header_keys=None,
            processes=2,
            min_length=25,
            max_width=75,
            buffer=250,
            threshold=5,
            theta=None,
            kernels=None,
            mask_include_status=[2],
            plot=False,
            output_dir='.',
            output_root='',
            check_persistence=True,
            min_persistence=0.5,
            ignore_theta_range=None,
            save_catalog=True,
            save_diagnostic=True,
            save_mrt=False,
            save_mask=False):

        # inputs
        self.image = image
        self.header = header
        self.image_header = image_header
        self.save_image_header_keys = save_image_header_keys
        self.threshold = threshold
        self.min_length = min_length
        self.max_width = max_width
        self.kernels = kernels
        self.processes = processes
        self.theta = theta
        self.plot = plot
        self.buffer = buffer
        self.check_persistence = check_persistence
        self.min_persistence = min_persistence
        self.mask_include_status = mask_include_status
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
        if plt is not None:
            self._interactive = mpl.is_interactive()
        else:
            self._interactive = False

        # info for what output to save and where
        self.output_dir = output_dir
        self.root = output_root
        self.save_catalog = save_catalog
        self.save_diagnostic = save_diagnostic
        self.save_mrt = save_mrt
        self.save_mask = save_mask

        # plot image upon initialization
        if self.plot and (plt is not None):
            self.plot_image()

    @property
    def save_image_header_keys(self):
        """List of header keys from ``self.image_header`` to save in the output trail
        catalog header.
        """
        return self._save_image_header_keys

    @save_image_header_keys.setter
    def save_image_header_keys(self, value):
        if value is None:
            value = []
        elif not isinstance(value, (list, tuple)):
            raise ValueError(f"save_image_header_keys must be list or tuple but got: {value}")
        else:
            value = np.squeeze(value).tolist()
        self._save_image_header_keys = value

    @property
    def processes(self):
        """Number of processes to use when calculating MRT."""
        return self._processes

    @processes.setter
    def processes(self, value):
        max_cpu = multiprocessing.cpu_count()
        if value < 1:
            self._processes = 1
        elif value > max_cpu:
            self._processes = max_cpu
        else:
            self._processes = value

    @property
    def min_length(self):
        """Minimum streak length allowed.
        This is the minimum allowed length of a satellite trail.
        """
        return self._min_length

    @min_length.setter
    def min_length(self, value):
        if value < 1:
            raise ValueError(f"Invalid min_length: {value}")
        self._min_length = value

    @property
    def max_width(self):
        """Maximum streak width allowed.
        This is the maximum width of a trail to be considered robust.
        """
        return self._max_width

    @max_width.setter
    def max_width(self, value):
        if value < 1:
            raise ValueError(f"Invalid max_width: {value}")
        self._max_width = value

    @property
    def buffer(self):
        """Size of cutout region extending perpendicularly outward
        on each side from a streak/trail when analyzing its properties.
        """
        return self._buffer

    @buffer.setter
    def buffer(self, value):
        if value < 1:
            raise ValueError(f"Invalid buffer: {value}")
        self._buffer = value

    @property
    def threshold(self):
        """Minimum SNR when extracting sources from the MRT."""
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        if value <= 0:
            raise ValueError(f"SNR threshold must be positive but got: {value}")
        self._threshold = value

    @property
    def theta(self):
        """Angles at which to calculate the MRT."""
        return self._theta

    @theta.setter
    def theta(self, value):
        # NOTE: Not all assumptions used in find_mrt_sources method are enforced here.
        if value is None:
            self._theta = np.arange(0, 180, 0.5)
        elif len(value) > 1:
            self._theta = value
        else:
            raise ValueError(f"Invalid theta: {value}")

    @property
    def kernels(self):
        """Paths to each kernel to be used for source finding in the MRT."""
        return self._kernels

    @kernels.setter
    def kernels(self, value):
        if value is None:
            self._kernels = [get_pkg_data_filename(f'rt_line_kernel_width{k}.fits'
                             for k in (15, 7, 3))]
        else:
            user_files = [v for v in value if os.path.isfile(v)]
            if len(user_files) < 1:
                raise ValueError(f"No usable kernel provided: {value}")
            self._kernels = user_files

    @property
    def mask_include_status(self):
        """List indicating trails with which status should be considered
        when making the mask. Statuses are generated by
        :func:`acstools.utils_findsat_mrt.filter_sources`:

        * 1 = Failed SNR or width requirements.
        * 2 = Passed SNR and width requirements but failed persistence test.
        * 3 = Passed SNR, width, and persistence requirements.
        """
        return self._mask_include_status

    @mask_include_status.setter
    def mask_include_status(self, value):
        if (not isinstance(value, list)) or (set(value) > {1, 2, 3}):
            raise ValueError(f"Status list elements must be 1, 2, or 3 but got: {value}")
        self._mask_include_status = value

    @property
    def root(self):
        """A prefix for all output files."""
        return self._root

    @root.setter
    def root(self, value):
        if not isinstance(value, str):
            raise ValueError(f"root must be a string but got: {value}")
        self._root = value

    @property
    def check_persistence(self):
        """Calculate the persistence of all identified streaks."""
        return self._check_persistence

    @check_persistence.setter
    def check_persistence(self, value):
        if not isinstance(value, bool):
            raise ValueError(f"check_persistence must be bool but got: {value}")
        self._check_persistence = value

    @property
    def min_persistence(self):
        """Minimum persistence of a "true" satellite trail to be considered robust.
        Note that this does not reject satellite trails from the output catalog,
        but highlights them in a different color in the output plot.
        """
        return self._min_persistence

    @min_persistence.setter
    def min_persistence(self, value):
        if value < 0 or value > 1:
            raise ValueError(f"min_persistence must be between 0 and 1 but got: {value}")
        self._min_persistence = value

    def run_mrt(self, plot=False):
        '''
        Run the median radon transform on the input image.
        This uses `theta` and `processes` for calculations, so update them first, if needed.

        Parameters
        ----------
        plot : bool, optional
            Flag to plot the resulting MRT. The default is `False`.

        '''
        # run MRT
        rt, length = radon(self.image, circle=False, median=True,
                           fill_value=np.nan, processes=self.processes,
                           return_length=True, theta=self.theta)

        # trim mrt where length too short
        rt[length < self.min_length] = np.nan

        # retain various outputs
        self.mrt = rt
        self.length = length

        # setting a warning filter for the nanmedian calculations. These
        # warnings are inconsequential and already accounted for in the code
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',
                                    message='All-NaN slice encountered')
            # calculate some useful properties
            # median
            self._medrt = np.nanmedian(rt)
            # median abs deviation
            self._madrt = np.nanmedian(np.abs(rt[np.abs(rt) > 0]) - self._medrt)

            # calculate the approximate uncertainty of the MRT at each point
            self._image_mad = np.nanmedian(np.abs(self.image))

        # using MAD to avoid influence from outliers
        self._image_stddev = self._image_mad / 0.67449
        # error on median ~ 1.25x error on mean. There are regions with length
        # equals zero which keeps raising warnings. Suppressing that warning
        # here
        with np.errstate(divide='ignore', invalid='ignore'):
            self.mrt_err = 1.25 * self._image_stddev / np.sqrt(self.length)

        # create the rho array
        rho0 = rt.shape[0] / 2 - 0.5
        self.rho = np.arange(rt.shape[0]) - rho0

        # plot if set
        if plot and (plt is not None):
            self.plot_mrt()

    def plot_image(self, ax=None, scale=(-1, 5), overlay_mask=False):
        '''
        Plot the input image.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            A matplotlib subplot where the image should be shown. The default
            is `None` (one will be created for you).
        scale : tuple of floats, optional
            A two element array with the minimum and maximum image values used
            to set the color scale, in units of the image median absolute
            deviation (MAD). The default is ``(-1, 5)``.
        overlay_mask : bool, optional
            Overlay the trail mask, if already calculated. Default is `False`.

        '''
        if plt is None:
            return

        if ax is None:
            fig, ax = plt.subplots()

        # recalculate mad and stdev here in case it hasn't been done yet
        # setting a warning filter for the nanmedian calculations. These
        # warnings are inconsequential and already accounted for in the code
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',
                                    message='All-NaN slice encountered')
            self._image_mad = np.nanmedian(np.abs(self.image))

        self._image_stddev = self._image_mad / 0.67449  # using MAD to avoid
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

        # write to file if interactive is off and plottin is on
        if ~self._interactive:
            file_name = os.path.join(self.output_dir, self.root + '_image')
            if overlay_mask:
                file_name = file_name + '_mask'
            plt.savefig(file_name + '.png')

    def plot_mrt(self, ax=None, scale=(-1, 5), show_sources=False):
        '''
        Plot the MRT.

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            A matplotlib subplot where MRT should be shown. The default
            is `None` (one will be created for you).
        scale : tuple of floats, optional
            A two element array with the minimum and maximum image values used
            to set the color scale, in units of the MRT median absolute
            deviation (MAD). The default is ``(-1, 5)``.
        show_sources : bool
            Mark the positions of the detected sources. Default is `False`.

        Returns
        -------
        ax : `matplotlib.axes.Axes`
            Matplotlib subplot where the MRT is plotted.

        '''
        # exit if no MRT
        if self.mrt is None:
            LOG.error('No MRT to plot')
            return

        # Letting user know if there are no sources to overplot
        if show_sources and (self.source_list is None):
            show_sources = False
            LOG.info('No sources to show')

        if ax is None:
            fig, ax = plt.subplots()

        ax.imshow(self.mrt, aspect='auto', origin='lower',
                  vmin=scale[0]*self._madrt, vmax=scale[1]*self._madrt)
        ax.set_title('MRT')
        ax.set_xlabel('angle(theta) pixel')
        ax.set_ylabel('offset(rho) pixel')

        # overplot sources if applicable
        if show_sources is True:
            x = self.source_list['xcentroid']
            y = self.source_list['ycentroid']
            status = self.source_list['status']

            for s, color in zip([0, 1, 2], ['red', 'orange', 'cyan']):
                sel = (status == s)
                if np.sum(sel) > 0:
                    ax.scatter(x[sel], y[sel], edgecolor=color,
                               facecolor='none', s=100, lw=2,
                               label=f'status={s}')
            ax.legend(loc='upper center')

        # send plot to file if interactive is off
        if ~self._interactive:
            file_name = os.path.join(self.output_dir, self.root + '_mrt')
            if show_sources:
                file_name = file_name + '_sources'
            file_name = file_name + '.png'
            plt.savefig(file_name)

        return ax

    def plot_mrt_snr(self, ax=None, scale=(1, 25)):
        '''
        Plots a map of the MRT signal-to-noise ratio (SNR).

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`, optional
            A matplotlib subplot where the SNR should be shown. The default
            is `None` (one will be created for you).
        scale : tuple of floats, optional
            A two element array with the minimum and maximum image values used
            to set the color scale. The default is ``(1, 25)``.

        Returns
        -------
        snr_map : array-like
            A map of the SNR.

        '''
        if self.mrt is None:
            LOG.error('No MRT to plot')
            return

        if ax is None:
            fig, ax = plt.subplots()

        snr = self.mrt / self.mrt_err
        ax.imshow(snr, aspect='auto', origin='lower',
                  vmin=scale[0], vmax=scale[1])
        ax.set_title('MRT SNR')
        ax.set_xlabel('angle(theta) pixel')
        ax.set_ylabel('offset(rho) pixel')

        if ~self._interactive:
            file_name = os.path.join(self.output_dir, self.root + '_mrt_snr.png')
            plt.savefig(file_name)

        return snr

    def find_mrt_sources(self, plot=False):
        '''
        Finds sources in the MRT consistent with satellite trails/streaks.
        This uses `kernels` and `threshold` for calculations, so update them first, if needed.

        Parameters
        ----------
        plot : bool, optional
            Flag to plot the MRT and overlay the sources found. The default is `False`.

        Returns
        -------
        source_list : `~astropy.table.QTable` or `None`
            Catalog containing information about detected trails, if applicable.
            This is the same info as ``self.source_list``.

        '''
        # test for photutils
        try:
            from photutils.detection import StarFinder
        except ImportError:
            LOG.error('photutils not installed. Source detection will not work.')
            return

        LOG.info('Detection threshold: {}'.format(self.threshold))

        snrmap = self.mrt / self.mrt_err
        snrmap_mask = ~np.isfinite(snrmap)

        # cycle through kernels
        tbls = []   # we'll store detected sources here
        for k in self.kernels:
            with fits.open(k) as h:
                kernel = h[0].data
                LOG.info('Using kernel {}'.format(k))

                # detect sources
                s = StarFinder(self.threshold, kernel, min_separation=20,
                               exclude_border=False, brightest=None, peakmax=None)

            # can fail for cases where nothing found. Allow code to return
            # nothing and move on
            try:
                tbl = s.find_stars(snrmap, mask=snrmap_mask)
            except Exception as e:
                tbl = None
                LOG.info('no sources found using kernel: {}'.format(repr(e)))
            else:  # append to big table of sources
                tbl = tbl[np.isfinite(tbl['xcentroid'])]
                LOG.info('{} sources found using kernel'.format(len(tbl)))
                if (len(tbls) > 0):
                    if len(tbls[-1]['id']) > 0:
                        tbl['id'] += np.max(tbls[-1]['id'])
                        # adding max ID number from last iteration to avoid
                        # duplicate ids
                tbls.append(tbl)

        # combine tables from each kernel and remove any duplicates
        if len(tbls) > 0:
            LOG.info('Removing duplicate sources')
            sources = merge_tables(tbls)
            self.source_list = sources
        else:
            self.source_list = None

        # add the theta and rho values to the table
        if self.source_list is not None:
            dtheta = self.theta[1] - self.theta[0]
            self.source_list['theta'] = self.theta[0] + dtheta * self.source_list['xcentroid']
            self.source_list['rho'] = self.rho[0] + self.source_list['ycentroid']

            # Add the status array and endpoints array. Status will be zero
            # because no additional checks have been done yet.
            self.source_list['endpoints'] = [
                streak_endpoints(t['rho'], -t['theta'], self.image.shape)
                for t in self.source_list]
            self.source_list['status'] = np.zeros(len(self.source_list), dtype=int)

            # run the routine to remove angles if any bad ranges are specified
            if self.ignore_theta_range is not None:
                self._remove_angles()

            # print the total number of sources found
            LOG.info('{} final sources found'.format(len(self.source_list)))
            # plot sources if set
            if plot and (plt is not None):
                self.plot_mrt(show_sources=True)
        else:
            LOG.warning('No sources found')

        return self.source_list

    def filter_sources(self, trim_catalog=False, plot=False, plot_streak=False):
        '''
        Filters catalog of trails based on SNR, width, and persistence.
        This uses `threshold`, `max_width`, `min_length`, `buffer`,
        `check_persistence`, and `min_persistence` for calculations,
        so update them first, if needed.

        Parameters
        ----------
        trim_catalog : bool, optional
            Flag to remove all filtered trails from the source catalog. The
            default is `False`.
        plot : bool, optional
            Set to plot the MRT with the resulting filtered sources overlaid.
            The default is `False`.
        plot_streak : bool, optional
            Set to plot diagnostics for each trail. Only works in interactive
            mode. Warning: this can generate a lot of plots depending on how
            many trails are found. Default is `False`.

        Returns
        -------
        source_list : `~astropy.table.QTable` or `None`
            Catalog of identified satellite trails with additional measured
            parameters appended. This is the same info as ``self.source_list``.

        '''
        if self.source_list is None:  # Nothing to do.
            return

        # check inputs, update class attributes as needed
        if ~self._interactive:
            plot_streak = False

        LOG.info('Filtering sources...\n'
                 'Min SNR : {}\n'
                 'Max Width: {}\n'
                 'Min Length: {}\n'
                 'Check persistence: {}'.format(
                     self.threshold, self.maxwidth, self.min_length, self.check_persistence))

        if self.check_persistence is True:
            LOG.info('Min persistence: {}'.format(self.min_persistence))

        # run filtering routine
        properties = filter_sources(self.image,
                                    self.source_list['endpoints'],
                                    max_width=self.maxwidth,
                                    buffer=self.buffer,
                                    plot_streak=plot_streak,
                                    min_length=self.min_length,
                                    minsnr=self.threshold,
                                    check_persistence=self.check_persistence,
                                    min_persistence=self.min_persistence)

        # update the status
        self.source_list.update(properties)

        # optionally trim the catalog of all rejected sources
        if trim_catalog:
            sel = ((self.source_list['width'] < self.maxwidth) &
                   (self.source_list['snr'] > self.threshold))
            self.source_list = self.source_list[sel]

        # plot if triggered
        if plot and (plt is not None):
            self.plot_mrt(show_sources=True)

        return self.source_list

    def make_mask(self, plot=False):
        '''
        Makes a satellite trail mask (bool) and a segmentation map.
        This uses `mask_include_status`, so update it first, if needed.

        The segmentation map is an image where pixels belonging to a given
        trail have values equal to the trail ID number.

        This updates ``self.segment`` and ``self.mask``.

        Parameters
        ----------
        plot : bool, optional
            Set to generate a plot images of the mask and segmentation image.
            Default is `False`.

        '''
        # crate the mask/segmentation
        if self.source_list is not None:
            include = [s['status'] in self.mask_include_status for s in self.source_list]
            trail_id = self.source_list['id'][include]
            endpoints = self.source_list['endpoints'][include]
            widths = self.source_list['width'][include]
            segment, mask = create_mask(self.image, trail_id, endpoints, widths)
        else:
            mask = np.zeros(self.image.shape, dtype=bool)
            segment = np.zeros(self.image.shape, dtype=int)

        self.segment = segment
        self.mask = mask

        # plot if triggered
        if plot and (plt is not None):
            self.plot_mask()
            self.plot_segment()

    def plot_mask(self):
        '''
        Generates a plot of the trail mask (``self.mask``).

        Returns
        -------
        ax : `matplotlib.axes.Axes`
            The matplotlib subplot containing the mask image.

        '''
        # exit if there's not mask
        if self.mask is None:
            LOG.error('No mask to show')
            return

        fig, ax = plt.subplots()
        ax.imshow(self.mask, origin='lower', aspect='auto')
        ax.set_title('Mask')

        # send mask to file if interactive off
        if ~self._interactive:
            file_name = os.path.join(self.output_dir, self.root + '_mask.png')
            plt.savefig(file_name)

        return ax

    def plot_segment(self):
        '''
        Generates a segmentation image of the identified trails (``self.segment``).

        Returns
        -------
        ax : `matplotlib.axes.Axes`
            A matplotlib subplot containing the segmentation map.

        '''
        # exit if no segmentation image
        if self.segment is None:
            LOG.error('No segment map to show')
            return

        # get unique values in segment
        unique_vals = np.unique(self.segment)
        data = np.zeros_like(self.segment)
        counter = 1
        for uv in unique_vals[1:]:
            data[self.segment == uv] = counter
            counter += 1

        data_min = np.min(data)
        data_max = np.max(data)
        fig, ax = plt.subplots()

        # update the colormap to match the segmentation IDs
        cmap = plt.get_cmap('tab20', data_max - data_min + 1)
        mat = ax.imshow(data, cmap=cmap, vmin=data_min - 0.5,
                        vmax=data_max + 0.5, origin='lower', aspect='auto')

        # tell the colorbar to tick at integers
        ticks = np.arange(len(unique_vals) + 1)
        cax = plt.colorbar(mat, ticks=ticks)
        cax.ax.set_yticklabels(np.concatenate([unique_vals, [unique_vals[-1] + 1]]))
        cax.ax.set_ylabel('trail ID')
        ax.set_title('Segmentation Mask')

        # send to file if not interactive
        if ~self._interactive:
            file_name = os.path.join(self.output_dir, self.root + '_segment.png')
            plt.savefig(file_name)

        return ax
# UNTIL HERE
    def save_output(self, output_dir=None, save_mrt=None,
                    save_mask=None, save_catalog=None, save_diagnostic=None,
                    close_plot=True):
        '''
        Save output. Any existing file will be overwritten.
        This uses `root`, so update them first, if needed.
        Output includes optionally:

        1. MRT
        2. Mask/segementation image
        3. Catalog
        4. Trail catalog

        Parameters
        ----------
        output_dir : string, optional
            Directory in which to save output files. Default is None, which
            defers to self attribute of same name.
        save_mrt : bool, optional
            Set to save the MRT in a fits file. Default is None, which
            defers to self attribute of same name.
        save_mask : bool, optional
            Set to save the mask and segmentation images in a fits file.
            Default is None, which defers to self attribute of same name.
        save_catalog : bool, optional
            Set to save the trail catalog in a fits table. Default is None,
            which defers to self attribute of same name.
        save_diagnostic : bool, optional
            Set to save a diagnostic plot (png) showing the identified trails.
            Default is None, which defers to self attribute of same name.
        close_plot : bool
            Close the plot instance after ``save_diagnostic`` is done.

        '''
        # check inputs, update class attributes as needed
        if output_dir is None:
            output_dir = self.output_dir
        else:
            self.output_dir = output_dir
        if save_mrt is None:
            save_mrt = self.save_mrt
        else:
            self.save_mrt = save_mrt
        if save_mask is None:
            save_mask = self.save_mask
        else:
            self.save_mask = save_mask
        if save_diagnostic is None:
            save_diagnostic = self.save_diagnostic
        else:
            self.save_diagnostic = save_diagnostic
        if save_catalog is None:
            save_catalog = self.save_catalog
        else:
            self.save_catalog = save_catalog

        # save the MRT image
        if save_mrt:
            if self.mrt is not None:
                mrt_filename = os.path.join(output_dir, f'{self.root}_mrt.fits')
                LOG.info('writing MRT to {}'.format(mrt_filename))
                fits.writeto(mrt_filename, self.mrt, overwrite=True)
            else:
                LOG.error('No MRT to save')

        # save the bool mask image
        if save_mask:
            if self.mask is not None and self.segment is not None:
                hdu0 = fits.PrimaryHDU()
                if self.header is not None:
                    hdu0.header = self.header  # copying original image header
                hdu1 = fits.ImageHDU(self.mask.astype(int))
                hdu2 = fits.ImageHDU(self.segment.astype(int))
                hdul = fits.HDUList([hdu0, hdu1, hdu2])
                mask_filename = os.path.join(output_dir, f'{self.root}_mask.fits')
                hdul.writeto(mask_filename, overwrite=True)
                LOG.info('writing mask to {}'.format(mask_filename))
            else:
                LOG.error('No mask to save')

        # save the diagnostic plot
        if save_diagnostic and (plt is not None):
            LOG.info('writing diagnostic plot')

            fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(20, 10))
            self.plot_image(ax=ax1)
            self.plot_mrt(ax=ax2)
            self.plot_image(ax=ax3)
            ax3.imshow(self.mask, alpha=0.5, origin='lower', aspect='auto', cmap='Reds')
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
            plt.savefig(os.path.join(output_dir, f'{self.root}_diagnostic.png'))
            if close_plot:
                plt.close()

        # save the catalog of trails
        if save_catalog:
            cat_filename = os.path.join(output_dir, f'{self.root}_catalog.fits')
            LOG.info(f'writing catalog {cat_filename}')
            if self.source_list is not None:
                # there's some meta data called "version" that cannot be added
                # to the fits header (and it is not useful). It always throws
                # an inconsequential warning so we suppress it here.
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Attribute `version` of')
                    self.source_list.write(cat_filename, overwrite=True)
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
                dummy_table.write(cat_filename, overwrite=True)
            LOG.info(f'wrote catalog {cat_filename}')

            # Append the original data header to this catalog too.
            has_pri_hdr = isinstance(self.header, fits.Header)
            has_dat_hdr = isinstance(self.image_header, fits.Header) and (len(self.save_image_header_keys) > 0)
            if has_pri_hdr or has_dat_hdr:
                with fits.open(cat_filename, mode='update') as h:
                    if has_pri_hdr:
                        h[0].header.update(self.header)

                    if has_dat_hdr:
                        # add individal header keywords now
                        for k in self.save_image_header_keys:
                            if k in self.image_header:
                                h[1].header[k] = self.image_header[k]

    def _remove_angles(self, ignore_theta_range=None):
        '''
        Remove a range (or set of ranges) of angles from the trail catalog

        This routine is primarily for removing trails at angles known to
        be overwhelmingly dominated by features that are not of interest, e.g.,
        for removing diffraction spikes.

        Parameters
        ----------
        ignore_theta_range : list of tuples, optional
            List of angle ranges to avoid.
            Format is [(min angle1,max angle1),(min angle2, max angle2) ... ].
            Default is None, which defers to self attribute of same name.

        Returns
        -------
        source_list, Table
            The source list with the specified angles removed.

        '''

        # update keyword values/class attributes as needed
        if ignore_theta_range is None:
            ignore_theta_range == self.ignore_theta_range
        else:
            self.ignore_theta_range = ignore_theta_range

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

    def run_all(self, **kwargs):
        '''
        Run the entire pipeline to identify, filter, and mask trails.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments run_mrt, find_mrt_sources,
            filter_sources, mask_mask, and save_output

        Returns
        -------
        None.

        '''

        self.run_mrt(**kwargs)
        self.find_mrt_sources(**kwargs)
        self.filter_sources(**kwargs)
        self.make_mask(**kwargs)
        self.save_output(**kwargs)


class WfcWrapper(TrailFinder):

    '''
    Wrapper for trail_finder designed specifically for ACS/WFC data.

    This class enables quick reading and preprocessing of standard
    full-frame ACS/WFC images.

    .. note::

        * This class is not designed for use with subarray images.

    Parameters
    ----------
    image_file : string
        ACS/WFC data file to read. Should be a fits file.
    extension : int, optional
        Extension of input file to read. The default is None.
    binsize : int, optional
        Amount the input data should be binned by. The default is None
        (no binning).
    ignore_flags : list
        DQ flags that lead to a pixel being ignored. Default is [4096,
        8192, 16384]. Only relevant for flt/flc files.
    preprocess : bool, optional
        Flag to run all the preprocessing steps (bad pixel flagging,
        background subtraction, rebinning. The default is True.
    execute : bool, optional
        Flag to run the entire TrailFinder pipeline. The default is False.
    **kwargs : dict, optional
        Additional keyword arguments for TrailFinder.

    Raises
    ------
    ValueError
        Image is subarray, image extension not recognized/specified, or
        image type not recognized.

        '''

    def __init__(self,
                 image_file,
                 extension=None,
                 binsize=None,
                 ignore_flags=[4096, 8192, 16384],
                 preprocess=True,
                 execute=False,
                 **kwargs):

        TrailFinder.__init__(self, **kwargs)
        self.image_file = image_file
        self.extension = extension
        self.binsize = binsize
        self.ignore_flags = ignore_flags
        self.preprocess = preprocess
        self.execute = execute

        # open image
        h = fits.open(self.image_file)

        # check that image id not subarray
        if h[0].header['SUBARRAY'] is True:
            raise ValueError('This program does not yet work on subarrays')

        # get suffix to determine how to process image
        suffix = (self.image_file.split('.')[0]).split('_')[-1]
        self.image_type = suffix
        LOG.info('image type is {}'.format(self.image_type))

        if suffix in ['flc', 'flt']:
            if extension is None:
                raise ValueError('flc/flt files must have extension specified')
            elif extension not in [1, 4]:
                raise ValueError('Valid Extensions are 1 or 4 for flc/flt \
                                 images')

            self.image = h[extension].data  # main image
            self.image_mask = h[extension+2].data  # dq array

        elif suffix in ['drc', 'drz']:
            extension = 1
            self.image = h[extension].data  # main image
            self.image_mask = h[extension+1].data  # weight array

        else:
            raise ValueError('Image type not recognized')

        # go ahead and run the proprocessing steps if set to True
        if preprocess is True:
            self.run_preprocess()

        # run the whole pipeline if set to true
        if execute is True:
            LOG.info('Running the trailfinding pipeline')
            self.run_all()

    def mask_bad_pixels(self, ignore_flags=None):
        '''
        Masks bad pixels

        Bad pixels are replacing with nan. This code the bitmask arrays
        for flc/flt images, and weight arrays for drc/drz images

        Parameters
        ----------
        ignore_flags : list, optional
            List of DQ bitmasks to ignore when masking. Only relevant for
            flc/flt files. The default is None, which defers to self attribute
            of the same name.

        Returns
        -------
        None.

        '''

        # check inputs, update class attributes as needed
        if ignore_flags is None:
            ignore_flags = self.ignore_flags
        else:
            self.ignore_flags = ignore_flags

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

        # setting a warning filter for the nanmedian calculations. These
        # warnings are inconsequential and already accounted for in the code
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',
                                    message='All-NaN slice encountered')
            self.image = self.image - np.nanmedian(self.image)

    def rebin(self, binsize=None):
        '''
        Rebins the image array.

        The binning in the x and y direction are identical.

        Parameters
        ----------
        binsize : int, optional
            Bin size. The default is None, which defers to the self attribute
            of the same name.

        Returns
        -------
        None.

        '''

        # check onputs, update class attributes as needed
        if binsize is None:
            binsize = self.binsize
        else:
            self.binsize = binsize

        if binsize is None:
            LOG.warn('No bin size defined. Will not perform binning')
            return

        LOG.info('Rebinning the data by {}'.format(binsize))

        # setting a warning filter for the nansum calculations. These
        # warnings are inconsequential and already accounted for in the code
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore',
                                    message='All-NaN slice encountered')
            self.image = block_reduce(self.image, binsize, func=np.nansum)

    def run_preprocess(self, **kwargs):
        '''
        Runs all the image preprocessing steps.

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

    def update_dq(self, **kwargs):
        '''
        Update DQ array with the satellite trail mask.

        .. note::

            * This routine only works on flc/flt images.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments for
            acstools.utils_findsat_mrt.update_dq.

        Returns
        -------
        None.

        '''

        if self.image_type not in ['flc', 'flt']:
            raise ValueError('DQ array can only be updated for flc/flt images')

        update_dq(self.image_file, self.extension, self.mask,
                  dqval=16384, verbose=True)
