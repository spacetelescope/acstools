import pytest
from ci_watson.artifactory_helpers import get_bigdata

from .helpers import BaseACSTOOLS
from .. import acssum

class TestSUM(BaseACSTOOLS):
    """ 
    Run ACSSUM on two post-SM4 full-frame datasets with default calibration 
    flags.
    """
    detector = 'wfc'

    def _acssum(self, input_list, output_file, truth_file):

        # Acquire the input files for processing
        with open(input_list[1:], 'r') as flist:
            for line in flist:
                infile_flt = line.split()[0]
                self.get_input_file(infile_flt)

        # ACSSUM
        acssum.acssum(input_list, output_file, time_stamps=True, verbose=True)

        # Compare results
        self.compare_outputs([(output_file, truth_file)])

    # NOTE: Test was wfc_acssum.py
    # jc4823bp2 = post-SM4, full-frame data
    # jc4823br2 = post-SM4, full-frame data
    def test_fullframe(self):

        input_list  = ['jc4823bp2_flt.fits', 'jc4823br2_flt.fits']
        output_root = 'comb2_acssum_1'
        output_file = output_root + '.fits'
        tra_file    = output_root + '.tra'
        truth_file  = output_root + '_ref.fits'

        self._acssum(input_list, output_file, truth_file)
