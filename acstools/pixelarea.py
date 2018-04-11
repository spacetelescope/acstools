from astropy.io import fits
from drizzlepac import astrodrizzle
from drizzlepac import ablot
import numpy as np

import os
import glob

__author__ = 'Tyler D. Desjardins'
__date__ = '2018-04-11'
__version__ = '1.0'


class PixelArea(object):
    """
    Purpose
    -------
    Create a pixel area map (PAM) for the ACS instrument. For full-frame ACS/WFC
    images, two PAMs are created (one for each CCD).

    The method used to create the PAMs is based off of the IRAF code written
    by Richard Hook in February 2004.

    Inputs
    ------
    flt_image (str):
        Name of the FLT/FLC file to be used as input. The PAM is constructed
        using WCS and distortion information from this file.

    pam_root (str; optional; default=None):
        Root name of the output PAM file(s). The output will have the name
        pam_root_detector(chip)_pam.fits, where detector is one of WFC, HRC,
        or SBC. If the detector is the WFC and both CCDs were read out, then
        the CCD chip number (WFC1 or WFC2) is also used in the file name.
        If None, then the root name is either the root name of the input
        file (see use_flt_name parameter) or it is constructed using information
        in the input file header (DETECTOR, DATE-OBS, TIME-OBS, FILTER1, FILTER2).

    use_flt_name (bool; optional; default=True):
        Use the the FLT/FLC input file root name as the root name for the PAM(s)?
        If True, the output PAM(s) will use the IPSOOT of the input file as
        the pam_root.


    Outputs
    -------
    A PAM for each CCD in the input FLT/FLC file is created. These PAMs are unitless
    and express the size of the pixels on the sky relative to the native pixel scale
    of the detector. Thus, ACS/WFC PAMs are normalized to 0.05 arcsec/pixel, while
    ACS/HRC and ACS/SBC PAMs are normalized to 0.025 arcsec/pixel. Multiplying
    the input FLT/FLC file by the PAM allows for photometry on the distorted image
    using the ACS photometric calibration, which was determined for undistorted
    images. The PAM files have two extensions: a primary header and a single
    science extension containing the PAM data. A comment is added to the header
    indicating the FLT/FLC image it was based off of and the detector name.

    For more information, see the ACS website at http://www.stsci.edu/hst/acs.

    Usage
    -----
    For the FLT image j6me13qhq_flt.fits, we can generate PAM files using:


        from acstools import pixelarea
        pam = pixelarea.PixelArea('j6me13qhq_flt.fits')
        pam.make_pam()

    As this is a full-frame ACS/WFC image, it will produce two files:

        j6me13qhq_wfc1_pam.fits
        j6me13qhq_wfc2_pam.fits
    """

    def __init__(self, flt_image, pam_root=None, use_flt_name=True):

        self.flt_image = flt_image
        self.pam_root = pam_root

        if self.pam_root is None:
            if not use_flt_name:
                self.pam_root = self._pam_name()
            else:
                self.pam_root = self.flt_image.split('_')[0]

    def make_pam(self):
        """
        Purpose
        -------
        Generate the PAM(s) for a given FLT/FLC input file. This copies
        the input image to a new file to preserve the WCS and distortion
        information in the header, but changes the science array to be all
        1s and the DQ array to be all 0s (good values). This new image is
        then drizzled with the kernel set to "point." The resulting weight
        map is then blotted back to the original input FLT/FLC file. Finally,
        we take the inverse of the blotted image, which gives us the PAM.

        Inputs
        ------
        None.

        Outputs
        -------
        PAM file(s).
        """

        # Set up the input files for AstroDrizzle. Add some padding to the copy of the FLT
        # image to mitigate edge effects from the blot interpolation.

        with fits.open(self.flt_image) as hdu:
            hdr = hdu[0].header
            hdr['EXPTIME'] = 1.

            scihdr = hdu[1].header

            hdu[1].data = np.ones((scihdr['NAXIS2'] + 8, scihdr['NAXIS1'] + 8))
            hdu[2].data = np.ones((scihdr['NAXIS2'] + 8, scihdr['NAXIS1'] + 8))
            hdu[3].data = np.zeros((scihdr['NAXIS2'] + 8, scihdr['NAXIS1'] + 8)).astype('uint16')

            hdu[1].header['CRPIX1'] = hdu[1].header['CRPIX1'] + 4
            hdu[1].header['CRPIX2'] = hdu[1].header['CRPIX2'] + 4
            hdu[2].header['CRPIX1'] = hdu[2].header['CRPIX1'] + 4
            hdu[2].header['CRPIX2'] = hdu[2].header['CRPIX2'] + 4
            hdu[3].header['CRPIX1'] = hdu[3].header['CRPIX1'] + 4
            hdu[3].header['CRPIX2'] = hdu[3].header['CRPIX2'] + 4

            if '_flt.fits' in self.flt_image:
                driz_name = self.flt_image.replace('_flt.fits', '_pam_tmp.fits')
            elif '_flc.fits' in self.flt_image:
                driz_name = self.flt_image.replace('_flc.fits', '_pam_tmp.fits')

            detector = hdr['DETECTOR']

            if (detector == 'WFC') and (not hdr['SUBARRAY']):

                scihdr = hdu[4].header

                hdu[4].data = np.ones((scihdr['NAXIS2'] + 8, scihdr['NAXIS1'] + 8))
                hdu[5].data = np.ones((scihdr['NAXIS2'] + 8, scihdr['NAXIS1'] + 8))
                hdu[6].data = np.zeros((scihdr['NAXIS2'] + 8, scihdr['NAXIS1'] + 8)).astype('uint16')

                hdu[4].header['CRPIX1'] = hdu[4].header['CRPIX1'] + 4
                hdu[4].header['CRPIX2'] = hdu[4].header['CRPIX2'] + 4
                hdu[5].header['CRPIX1'] = hdu[5].header['CRPIX1'] + 4
                hdu[5].header['CRPIX2'] = hdu[5].header['CRPIX2'] + 4
                hdu[6].header['CRPIX1'] = hdu[6].header['CRPIX1'] + 4
                hdu[6].header['CRPIX2'] = hdu[6].header['CRPIX2'] + 4

                sci_ext = [1, 2]
                chip = [2, 1]
                chipname = [detector + str(ccd) for ccd in chip]
                blot_output = [self.pam_root + '_wfc{}_pam.fits'.format(ccd) for ccd in chip]

            else:
                sci_ext = [1]
                chipname = [detector]
                blot_output = [self.pam_root + '_{}_pam.fits'.format(detector.lower())]

            hdu.writeto(driz_name, overwrite=True)

        # First drizzle the input PAM image(s) with everything but the drizSeparate step
        # turned off. This will create the weight maps.

        if detector == 'WFC':
            (driz_outny, driz_outnx) = (4500, 4500)

        elif detector == 'HRC':
            (driz_outny, driz_outnx) = (1200, 1200)

        elif detector == 'SBC':
            (driz_outny, driz_outnx) = (1600, 1600)

        astrodrizzle.AstroDrizzle(driz_name, preserve=False, static=False, skysub=False,
                                  driz_sep_outny=driz_outny, driz_sep_outnx=driz_outnx, median=False,
                                  blot=False, driz_cr=False, driz_combine=False, driz_sep_kernel='square',
                                  runfile='pam_driz.log')

        # Blot the weight map back, and then take the inverse of the result. This is the pixel
        # area map.

        for idx, _ in enumerate(blot_output):
            ablot.blot(driz_name.replace('.fits', '_single_wht.fits'),
                       self.flt_image + '[sci, {}]'.format(sci_ext[idx]),
                       blot_output[idx], addsky=False, expout=1.0)

            with fits.open(blot_output[idx], mode='update') as hdu:
                hdu[1].data = 1. / hdu[1].data

                del hdu[0].header['COMMENT']
                hdu[0].header['COMMENT'] = 'Pixel area map for {}, {}'.format(self.flt_image, chipname[idx])

        # Clean up temporary files.

        junk = ['pam_driz.log'] + glob.glob(driz_name.split('.')[0] + '*')
        for foo in junk:
            os.remove(foo)

    def _pam_name(self):
        """
        Purpose
        -------
        Using an FLT/FLC file, use header keywords to generate a name for the
        pixel area map.

        Inputs
        ------
        None.

        Outputs
        -------
        pam_name (str):
            The root name of the pixel area map output file. This is defined as
            "DETECTOR_DATEOBS_TIMEOBS_FILTER1_FILTER2". The TIMEOBS value read
            from the primary header keyword TIME-OBS has been altered to be a
            string of the form HHMMSS. If any filter is clear, it is not used
            in the output string.
        """

        with fits.open(self.flt_image) as hdu:
            hdr = hdu[0].header
            detector = hdr['DETECTOR']
            dateobs = hdr['DATE-OBS']
            timeobs = hdr['TIME-OBS'].split('.')[0].replace(':', '')
            filter1 = hdr['FILTER1']
            filter2 = hdr['FILTER2']

            if 'clear' in filter1.lower():
                filter1 = ''
            if 'clear' in filter2.lower():
                filter2 = ''

        pam_name = '_'.join([detector, dateobs, timeobs, filter1, filter2]).replace('__', '_').lower()

        return pam_name
