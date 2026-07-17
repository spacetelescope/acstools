"""ACSTOOLS regression test helpers."""

import os
from functools import partial

import pytest
from astropy.io import fits
from astropy.io.fits import FITSDiff
from ci_watson.artifactory_helpers import generate_upload_params, generate_upload_schema
from ci_watson.artifactory_helpers import get_bigdata as _get_bigdata
from ci_watson.hst_helpers import download_crds, ref_from_image

__all__ = ["calref_from_image", "BaseACSTOOLS"]

# Overload generic get_bigdata to include repo root dir.
# This is to accomodate developers who have to run big data tests across
# several repositories using the same TEST_BIGDATA env var.
get_bigdata = partial(_get_bigdata, "scsb-hstcal")


def calref_from_image(input_image):
    """
    Return a list of reference filenames, as defined in the primary
    header of the given input image, necessary for calibration.
    This is mostly needed for destriping tools.
    """

    # NOTE: Add additional mapping as needed.
    # Map *CORR to associated CRDS reference file.
    corr_lookup = {
        "DQICORR": ["BPIXTAB", "SNKCFILE"],
        "ATODCORR": ["ATODTAB"],
        "BLEVCORR": ["OSCNTAB"],
        "SINKCORR": ["SNKCFILE"],
        "BIASCORR": ["BIASFILE"],
        "PCTECORR": ["PCTETAB", "DRKCFILE", "BIACFILE"],
        "FLSHCORR": ["FLSHFILE"],
        "CRCORR": ["CRREJTAB"],
        "SHADCORR": ["SHADFILE"],
        "DARKCORR": ["DARKFILE", "TDCTAB"],
        "FLATCORR": ["PFLTFILE", "DFLTFILE", "LFLTFILE"],
        "PHOTCORR": ["IMPHTTAB"],
        "LFLGCORR": ["MLINTAB"],
        "GLINCORR": ["MLINTAB"],
        "NLINCORR": ["NLINFILE"],
        "ZSIGCORR": ["DARKFILE", "NLINFILE"],
        "WAVECORR": ["LAMPTAB", "WCPTAB", "SDCTAB"],
        "SGEOCORR": ["SDSTFILE"],
        "X1DCORR": ["XTRACTAB", "SDCTAB"],
        "SC2DCORR": ["CDSTAB", "ECHSCTAB", "EXSTAB", "RIPTAB", "HALOTAB", "TELTAB", "SRWTAB"],
        "BACKCORR": ["XTRACTAB"],
        "FLUXCORR": ["APERTAB", "PHOTTAB", "PCTAB", "TDSTAB"],
    }

    hdr = fits.getheader(input_image, ext=0)

    # Mandatory CRDS reference file.
    # Destriping tries to ingest some *FILE regardless of *CORR.
    ref_files = ref_from_image(input_image, ["CCDTAB", "DARKFILE", "PFLTFILE"])

    for step in corr_lookup:
        # Not all images have the CORR step and it is not always on.
        # Destriping also does reverse-calib.
        if (step not in hdr) or (hdr[step].strip().upper() not in ("PERFORM", "COMPLETE")):
            continue

        ref_files += ref_from_image(input_image, corr_lookup[step])

    return list(set(ref_files))  # Remove duplicates


# Base class for actual tests.
# NOTE: Named in a way so pytest will not pick them up here.
# NOTE: bigdata marker requires TEST_BIGDATA environment variable to
#       point to a valid big data directory, whether locally or on Artifactory.
# NOTE: envopt would point tests to "dev" or "stable".
# NOTE: _jail fixture ensures each test runs in a clean tmpdir.
@pytest.mark.bigdata
@pytest.mark.usefixtures("_jail", "envopt")
class BaseACSTOOLS:
    # Timeout in seconds for file downloads.
    timeout = 30

    instrument = "acs"
    ignore_keywords = ["filename", "date", "iraf-tlm", "fitsdate", "opus_ver", "cal_ver", "proctime", "history"]

    # To be defined by test class in actual test modules.
    detector = ""

    @pytest.fixture(autouse=True)
    def setup_class(self, envopt):
        """
        Class-level setup that is done at the beginning of the test.

        Parameters
        ----------
        envopt : {'dev', 'stable'}
            This is a ``pytest`` fixture that defines the test
            environment in which input and truth files reside.

        """
        self.env = envopt

    def get_input_file(self, filename, skip_ref=False):
        """Copy input file into the working directory.
        The associated CRDS reference files are also copied or
        downloaded, if desired.

        Input file is from Artifactory, while the reference files from CDBS.

        Data directory layout same as HSTCAL::

            instrument/
                detector/
                    input/
                    truth/

        Parameters
        ----------
        filename : str
            Filename to copy over, along with its reference files, if desired.

        skip_ref : bool
            Skip downloading reference files for the given input.

        """
        # Copy over main input file: The way calibration code was written,
        # it usually assumes input is in the working directory.
        get_bigdata(self.env, self.instrument, self.detector, "input", filename)

        if skip_ref:
            return

        ref_files = calref_from_image(filename)
        for ref_file in ref_files:
            # Special reference files that live with inputs.
            if "$" not in ref_file and os.path.basename(ref_file) == ref_file:
                get_bigdata(self.env, self.instrument, self.detector, "input", ref_file)
                continue

            # Download reference files, if needed only.
            download_crds(ref_file)

    def compare_outputs(self, outputs, atol=0, rtol=1e-7, raise_error=True, ignore_keywords_overwrite=None, verbose=True):
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

        verbose : bool
            Print extra info to screen.

        Returns
        -------
        report : str
            Report from ``fitsdiff``.
            This is part of error message if ``raise_error=True``.

        """
        all_okay = True
        creature_report = ""
        updated_outputs = []  # To track outputs for Artifactory JSON schema

        if ignore_keywords_overwrite is None:
            ignore_keywords = self.ignore_keywords
        else:
            ignore_keywords = ignore_keywords_overwrite

        for actual, desired in outputs:
            desired = get_bigdata(self.env, self.instrument, self.detector, "truth", desired)
            fdiff = FITSDiff(actual, desired, rtol=rtol, atol=atol, ignore_keywords=ignore_keywords)
            creature_report += fdiff.report()

            if not fdiff.identical:
                all_okay = False
                # Only keep track of failed results which need to
                # be used to replace the truth files (if OK).
                updated_outputs.append((actual, desired))

        if not all_okay:
            if self.results_root is not None:  # pragma: no cover
                schema_pattern, tree, testname = generate_upload_params(self.results_root, updated_outputs, verbose=verbose)
                generate_upload_schema(schema_pattern, tree, testname)

            if raise_error:
                raise AssertionError(os.linesep + creature_report)

        return creature_report
