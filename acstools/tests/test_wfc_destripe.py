"""Test standalone ACS destriping for post-SM4 WFC exposures."""

import pytest

from acstools import acs_destripe
from acstools.tests.helpers import BaseACSTOOLS


class TestDestripe(BaseACSTOOLS):
    detector = 'wfc'

    def _destripe_one(self, inputfile, outsuffix, outputfile, truthfile,
                      masks=None):
        # Prepare input file.
        # acs_destripe needs access to some reference files.
        self.get_input_file(inputfile)

        # De-stripe
        if masks is None:
            acs_destripe.clean(inputfile, outsuffix)
        else:
            # Get extra input masks
            for mfile in masks:
                self.get_input_file(mfile, skip_ref=True)
            acs_destripe.clean(inputfile, outsuffix,
                               mask1=masks[0], mask2=masks[1])

        # Compare results
        self.compare_outputs([(outputfile, truthfile)], rtol=1e-6)

    # jb5g05ubq = calibrated post-SM4 WFC full-frame exposures without masks
    # ja0x03ojq = calibrated polarizer WFC subarray exposures
    # jc5001soq = partially calibrated WFC subarray exposures
    @pytest.mark.parametrize(
        'rootname', ['jb5g05ubq', 'ja0x03ojq', 'jc5001soq'])
    def test_generic(self, rootname):
        """
        Run de-striping tests on calibrated post-SM4 WFC full-frame
        exposures without masks.
        """
        out_sfx = 'destripe'
        flt_file = rootname + '_flt.fits'
        out_file = rootname + '_flt_' + out_sfx + '.fits'
        ref_file = rootname + '_flt_ref.fits'

        self._destripe_one(flt_file, out_sfx, out_file, ref_file)

    def test_fullframe_flashed(self):
        """
        Run de-striping test on semi-calibrated post-SM4 WFC full-frame
        with post-flash.
        """
        rootname = 'jc2z03cvq'
        out_sfx  = 'destripe'
        flt_file = rootname + '_blv_tmp.fits'
        out_file = rootname + '_blv_tmp_' + out_sfx + '.fits'
        ref_file = rootname + '_blv_tmp_ref.fits'

        self._destripe_one(flt_file, out_sfx, out_file, ref_file)

    def test_fullframe_masked(self):
        """
        Run de-striping tests on calibrated post-SM4 WFC full-frame
        exposures with masks.
        """
        rootname = 'jc8o04ghq'
        out_sfx  = 'destripe'
        flt_file = rootname + '_flt.fits'
        out_file = rootname + '_flt_' + out_sfx + '.fits'
        ref_file = rootname + '_flt_ref.fits'
        mask_files = [f'{rootname}_mask{i + 1}.fits' for i in range(2)]

        self._destripe_one(flt_file, out_sfx, out_file, ref_file,
                           masks=mask_files)
