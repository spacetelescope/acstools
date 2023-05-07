'''Test satellite trail detection and masking using findsat_mrt.'''
import os

from astropy.io.fits import FITSDiff
from astropy.utils.data import get_pkg_data_filename

from acstools.findsat_mrt import WfcWrapper


def test_WfcWrapper(tmp_path):
    """Identify and mask trails in WFC extension 4."""
    timeout = 30
    rootname = 'jc8m32j5q'
    inputfile = get_pkg_data_filename(
        os.path.join('data', 'input', f'{rootname}_flc.fits'),
        package='acstools.tests', show_progress=False, remote_timeout=timeout)
    desired = get_pkg_data_filename(
        os.path.join('data', 'truth', f'{rootname}_flc_mask_ref.fits'),
        package='acstools.tests', show_progress=False, remote_timeout=timeout)
    actual = tmp_path / f'{rootname}_flc_mask.fits'

    # If the machine does not have 8 cores, it will only use up to what is
    # available. We save the catalog to make sure that code runs but we do not
    # compare catalog.
    WfcWrapper(inputfile, binsize=4, extension=4,
               output_root=f'{rootname}_flc',
               output_dir=tmp_path, max_width=37.5,
               processes=8, execute=True, save_mask=True,
               save_diagnostic=False, save_catalog=True)

    # Compare mask with truth
    fdiff = FITSDiff(actual, desired, rtol=1e-7, atol=0)
    if not fdiff.identical:
        raise AssertionError(fdiff.report())
