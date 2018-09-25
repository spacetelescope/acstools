import pytest
from ci_watson.artifactory_helpers import get_bigdata

from .helpers import BaseACSTOOLS
from .. import acsccd

class TestCCD(BaseACSTOOLS):
    """ 
    Run ACSCCD on post-SM4 full-frame and subarray data with default calibration 
    flags.
    """
    detector = 'wfc'

    def _acsccd_single(self, input_file, output_file, truth_file):

        # Acquire the input file for processing
        self.get_input_file(input_file)

        # ACSCCD
        acsccd.acsccd(input_file, time_stamps=True, verbose=True)

        # Compare results
        self.compare_outputs([(output_file, truth_file)])

    # NOTE: Test was wfc_acsccd.py
    # jbdf08uf2 = post-SM4, full-frame data
    def test_fullframe(self):

        rootname = 'jbdf08uf2'
        raw_file    = rootname + '_raw.fits'
        output_file = rootname + '_blv_tmp.fits'
        tra_file    = rootname + '.tra'
        truth_file  = rootname + '_blv_tmp_ref.fits'

        self._acsccd_single(raw_file, output_file, truth_file)

    # NOTE: Test was wfc_acsccd.py
    # jb2t11se2 = post-SM4, subarray data
    def test_subarray(self):

        rootname = 'jb2t11se2'
        raw_file    = rootname + '_raw.fits'
        output_file = rootname + '_blv_tmp.fits'
        tra_file    = rootname + '.tra'
        truth_file  = rootname + '_blv_tmp_ref.fits'

        self._acsccd_single(raw_file, output_file, truth_file)
