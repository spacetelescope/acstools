"""
Test satellite trail detection.

.. note:: Cannot test ``detsat()`` because PHT results change from run to run!

"""
from ci_watson.artifactory_helpers import get_bigdata

from .helpers import BaseACSTOOLS
from .. import satdet


class TestSatDet(BaseACSTOOLS):
    detector = 'wfc'

    def test_trail_mask(self):
        """Mask satellite trail on WFC EXT 6."""

        rootname = 'jc8m10syq'
        inputfile = rootname + '_flc.fits'  # This is modified in-place
        truthfile = rootname + '_flc_ref.fits'

        # Prepare input file.
        get_bigdata('scsb-acstools', self.env, self.detector, 'input',
                    inputfile)

        # Satellite trail masking.
        sciext = 4
        dqext = 6
        trail = ((1199, 1357), (2841, 1023))
        mask = satdet.make_mask(inputfile, sciext, trail, plot=False,
                                verbose=False)
        satdet.update_dq(inputfile, dqext, mask, verbose=False)

        # Compare results.
        self.compare_outputs([(inputfile, truthfile)])
