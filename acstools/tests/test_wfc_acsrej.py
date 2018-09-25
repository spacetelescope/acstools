import pytest
from ci_watson.artifactory_helpers import get_bigdata

from .helpers import BaseACSTOOLS
from .. import acsrej

class TestREJ(BaseACSTOOLS):
    """ 
    Run ACSREJ on two post-SM4 full-frame datasets with default calibration 
    flags.
    """
    detector = 'wfc'

    def _acsrej(self, input_list, output_file, truth_file):

        # Acquire the input files for processing
        with open(input_list[0:], 'r') as flist:
        for line in flist:
            infile_flt = line.split()[0]
            self.get_input_file(infile_flt)

        # ACSREJ
        acsrej.acsrej(input_list, output_file, crrejtab='jref$n4e12511j_crr.fits',
                      skysub='mode', scalense=30.0, time_stamps=True, verbose=True)

        outputs = [('jc4823bpq_flt.fits', 'jc4823bpq_flt_ref.fits'),
                   ('jc4823brq_flt.fits', 'jc4823brq_flt_ref.fits'),
                   (output_file, truth_file)]

        # Compare results
        self.compare_outputs([(output_file, truth_file)])

    # NOTE: Test was wfc_acsrej.py
    # jc4823bpq = post-SM4, full-frame data
    # jc4823brq = post-SM4, full-frame data
    def test_fullframe(self):

        input_list  = ['jc4823bpq_flt.fits', 'jc4823brq_flt.fits']
        #input_list  = 'wfc_acsrej_list1'
        output_root = 'crsplit2_acsrej_1'
        output_file = output_root + '.fits'
        tra_file    = output_root + '.tra'
        truth_file  = output_root + '_ref.fits'

        self._acsrej(input_list, output_file, truth_file)
