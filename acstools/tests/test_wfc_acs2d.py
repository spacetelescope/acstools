import pytest
from ci_watson.artifactory_helpers import get_bigdata

from .helpers import BaseACSTOOLS
from .. import acs2d

class Test2D(BaseACSTOOLS):
    """ 
    Run ACS2D on post-SM4 full-frame and subarray data with default calibration 
    flags.
    """
    detector = 'wfc'

    def _acs2d_single(self, input_file, output_file, truth_file):

        # Acquire the input file for processing
        self.get_input_file(input_file)

        # ACS2D
        acs2d.acs2d(input_file, time_stamps=True, verbose=True)

        # Compare results
        self.compare_outputs([(output_file, truth_file)])

    # NOTE: Test was wfc_acs2d.py
    # jbdf08ufq = post-SM4, full-frame data
    def test_fullframe(self):

        rootname = 'jbdf08ufq'
        input_file  = rootname + '_blc_tmp.fits'
        output_file = rootname + '_flc.fits'
        tra_file    = rootname + '.tra'
        truth_file  = rootname + '_flc_ref.fits'

        self._acs2d_single(input_file, output_file, truth_file)

    # NOTE: Test was wfc_acs2d.py
    # jb2t11seq = post-SM4, subarray data
    def test_subarray(self):

        rootname = 'jb2t11seq'
        input_file  = rootname + '_blv_tmp.fits'
        output_file = rootname + '_flt.fits'
        tra_file    = rootname + '.tra'
        truth_file  = rootname + '_flt_ref.fits'

        self._acs2d_single(input_file, output_file, truth_file)
