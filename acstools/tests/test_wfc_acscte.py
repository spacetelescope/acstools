import pytest
from ci_watson.artifactory_helpers import get_bigdata

from .helpers import BaseACSTOOLS
from .. import acscte

class TestCTE(BaseACSTOOLS):
    """ 
    Run ACSCTE on post-SM4 full-frame data with default calibration 
    flags.
    """
    detector = 'wfc'

    def _acscte_single(self, input_file, output_file, truth_file):

        # Acquire the input file for processing
        self.get_input_file(input_file)

        # ACSCTE
        acscte.acscte(input_file, time_stamps=True, verbose=True)

        # Compare results
        self.compare_outputs([(output_file, truth_file)])

    # NOTE: Test was wfc_acscte.py
    # jbny01syq = post-SM4, full-frame data
    def test_fullframe(self):

        rootname = 'jbny01syq'
        input_file  = rootname + '_blv_tmp.fits'
        output_file = rootname + '_blc_tmp.fits'
        tra_file    = rootname + '.tra'
        truth_file  = rootname + '_blc_tmp_ref.fits'

        self._acscte_single(input_file, output_file, truth_file)

