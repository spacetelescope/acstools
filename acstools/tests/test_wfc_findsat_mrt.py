'''
Test satellite trail detection and masking using findsat_mrt
'''

from astropy.utils.data import get_pkg_data_filename
from acstools.findsat_mrt import WfcWrapper
from acstools.tests.helpers import BaseACSTOOLS
import os
from astropy.io.fits import FITSDiff
import logging

logging.basicConfig()
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)


class TestFindsatMRT(BaseACSTOOLS):

    detector = 'wfc'

    def test_WfcWrapper(self, tmp_path):
        """Identify and mask trails in WFC extension 4."""

        rootname = 'jc8m32j5q'
        inputfile = rootname + '_flc.fits'

        # Prepare input file.
        self.get_input_file(inputfile, skip_ref=True)

        WfcWrapper(inputfile, binsize=4, extension=4,
                    output_root='jc8m32j5q_flc',
                    output_dir=tmp_path,
                    threads=8, execute=True, save_mask=True,
                    save_diagnostic=False, save_catalog=True)

        #Compare mask with truth
        creature_report = ''
        all_okay = True
        actual = '{}/{}_flc_mask.fits'.format(tmp_path, rootname)
        desired = rootname + '_flc_mask_ref.fits'
        LOG.info('actual: {}'.format(actual))
        desiredpath = get_pkg_data_filename(
            os.path.join('data', 'truth', desired),
            package='acstools.tests',
            show_progress=False, remote_timeout=self.timeout)
        LOG.info('desiredpath: {}'.format(desiredpath))
        fdiff = FITSDiff(actual, desiredpath, rtol=1e-7, atol=0)
        LOG.info('fdiff = {}'.format(fdiff))
        creature_report += fdiff.report()

        LOG.info(creature_report)

        if not all_okay:
            raise AssertionError(os.linesep + creature_report)
