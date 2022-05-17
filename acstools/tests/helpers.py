"""ACSTOOLS regression test helpers."""

import os
import shutil

import pytest
from ci_watson.hst_helpers import ref_from_image, download_crds

from astropy.io import fits
from astropy.io.fits import FITSDiff
from astropy.utils.data import get_pkg_data_filename

__all__ = ['calref_from_image', 'BaseACSTOOLS']


def calref_from_image(input_image):
    """
    Return a list of reference filenames, as defined in the primary
    header of the given input image, necessary for calibration.
    This is mostly needed for destriping tools.
    """

    # NOTE: Add additional mapping as needed.
    # Map *CORR to associated CRDS reference file.
    corr_lookup = {
        'DQICORR': ['BPIXTAB', 'SNKCFILE'],
        'ATODCORR': ['ATODTAB'],
        'BLEVCORR': ['OSCNTAB'],
        'SINKCORR': ['SNKCFILE'],
        'BIASCORR': ['BIASFILE'],
        'PCTECORR': ['PCTETAB', 'DRKCFILE', 'BIACFILE'],
        'FLSHCORR': ['FLSHFILE'],
        'CRCORR': ['CRREJTAB'],
        'SHADCORR': ['SHADFILE'],
        'DARKCORR': ['DARKFILE', 'TDCTAB'],
        'FLATCORR': ['PFLTFILE', 'DFLTFILE', 'LFLTFILE'],
        'PHOTCORR': ['IMPHTTAB'],
        'LFLGCORR': ['MLINTAB'],
        'GLINCORR': ['MLINTAB'],
        'NLINCORR': ['NLINFILE'],
        'ZSIGCORR': ['DARKFILE', 'NLINFILE'],
        'WAVECORR': ['LAMPTAB', 'WCPTAB', 'SDCTAB'],
        'SGEOCORR': ['SDSTFILE'],
        'X1DCORR': ['XTRACTAB', 'SDCTAB'],
        'SC2DCORR': ['CDSTAB', 'ECHSCTAB', 'EXSTAB', 'RIPTAB', 'HALOTAB',
                     'TELTAB', 'SRWTAB'],
        'BACKCORR': ['XTRACTAB'],
        'FLUXCORR': ['APERTAB', 'PHOTTAB', 'PCTAB', 'TDSTAB']}

    hdr = fits.getheader(input_image, ext=0)

    # Mandatory CRDS reference file.
    # Destriping tries to ingest some *FILE regardless of *CORR.
    ref_files = ref_from_image(input_image, ['CCDTAB', 'DARKFILE', 'PFLTFILE'])

    for step in corr_lookup:
        # Not all images have the CORR step and it is not always on.
        # Destriping also does reverse-calib.
        if ((step not in hdr) or
                (hdr[step].strip().upper() not in ('PERFORM', 'COMPLETE'))):
            continue

        ref_files += ref_from_image(input_image, corr_lookup[step])

    return list(set(ref_files))  # Remove duplicates


# Base class for actual tests.
# NOTE: Named in a way so pytest will not pick them up here.
# NOTE: remote_data because reference files need internet connection
# NOTE: _jail fixture ensures each test runs in a clean tmpdir.
@pytest.mark.remote_data
@pytest.mark.usefixtures('_jail')
class BaseACSTOOLS:
    # Timeout in seconds for file downloads.
    timeout = 30

    instrument = 'acs'
    ignore_keywords = ['filename', 'date', 'iraf-tlm', 'fitsdate',
                       'opus_ver', 'cal_ver', 'proctime', 'history']

    # To be defined by test class in actual test modules.
    detector = ''

    def get_input_file(self, filename, skip_ref=False):
        """Copy input file into the working directory.
        The associated CRDS reference files are also copied or
        downloaded, if desired.

        Input file is from ``git lfs``, while the reference files from CDBS.

        Parameters
        ----------
        filename : str
            Filename to copy over, along with its reference files, if desired.

        skip_ref : bool
            Skip downloading reference files for the given input.

        """
        # Copy over main input file: The way calibration code was written,
        # it usually assumes input is in the working directory.
        src = get_pkg_data_filename(
            os.path.join('data', 'input', filename), package='acstools.tests',
            show_progress=False, remote_timeout=self.timeout)
        dst = os.path.join(os.curdir, filename)
        shutil.copyfile(src, dst)

        if skip_ref:
            return

        ref_files = calref_from_image(src)
        for ref_file in ref_files:
            # Special reference files that live with inputs.
            if ('$' not in ref_file and
                    os.path.basename(ref_file) == ref_file):
                refsrc = get_pkg_data_filename(
                    os.path.join('data', 'input', ref_file),
                    package='acstools.tests',
                    show_progress=False, remote_timeout=self.timeout)
                refdst = os.path.join(os.curdir, ref_file)
                shutil.copyfile(refsrc, refdst)
                continue

            # Download reference files, if needed only.
            download_crds(ref_file)

    def compare_outputs(self, outputs, atol=0, rtol=1e-7, raise_error=True,
                        ignore_keywords_overwrite=None):
        """Compare ACSTOOLS output with "truth" using ``fitsdiff``.

        Parameters
        ----------
        outputs : list of tuple
            A list of tuples, each containing filename (without path)
            of CALXXX output and truth, in that order. Example::

                [('output1.fits', 'truth1.fits'),
                 ('output2.fits', 'truth2.fits'),
                 ...]

        atol, rtol : float
            Absolute and relative tolerance for data comparison.

        raise_error : bool
            Raise ``AssertionError`` if difference is found.

        ignore_keywords_overwrite : list of str or `None`
            If not `None`, these will overwrite
            ``self.ignore_keywords`` for the calling test.

        Returns
        -------
        report : str
            Report from ``fitsdiff``.
            This is part of error message if ``raise_error=True``.

        """
        all_okay = True
        creature_report = ''

        if ignore_keywords_overwrite is None:
            ignore_keywords = self.ignore_keywords
        else:
            ignore_keywords = ignore_keywords_overwrite

        for actual, desired in outputs:
            desiredpath = get_pkg_data_filename(
                os.path.join('data', 'truth', desired),
                package='acstools.tests',
                show_progress=False, remote_timeout=self.timeout)
            fdiff = FITSDiff(actual, desiredpath, rtol=rtol, atol=atol,
                             ignore_keywords=ignore_keywords)
            creature_report += fdiff.report()

            if not fdiff.identical and all_okay:
                all_okay = False

        if not all_okay and raise_error:
            raise AssertionError(os.linesep + creature_report)

        return creature_report
