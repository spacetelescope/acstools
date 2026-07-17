"""Test satellite trail detection.

.. note:: Cannot test ``detsat()`` because PHT results change from run to run!

"""

import pytest

from acstools import satdet
from acstools.tests.helpers import BaseACSTOOLS
from acstools.utils_findsat_mrt import update_dq

pytest.importorskip("skimage")


class TestSatDet(BaseACSTOOLS):
    detector = "wfc"

    def test_detsat_runs(self):
        """Test that satellite trail detection runs without error."""
        rootname = "jc8m10syq"
        inputfile = rootname + "_flc.fits"  # This is modified in-place

        # Prepare input file.
        self.get_input_file(inputfile, skip_ref=True)

        # Run detsat without multiprocessing but
        # output is non-deterministic, so we just check roughly.
        res, err = satdet.detsat(inputfile, chips=[1, 4], n_processes=1, verbose=False)
        assert list(err) == [], err[("jc8m10syq_flc.fits", 1)]
        assert sorted(res) == [("jc8m10syq_flc.fits", 1), ("jc8m10syq_flc.fits", 4)]

    def test_trail_mask(self):
        """Mask satellite trail on WFC EXT 6."""

        rootname = "jc8m10syq"
        inputfile = rootname + "_flc.fits"  # This is modified in-place
        truthfile = rootname + "_flc_ref.fits"

        # Prepare input file.
        self.get_input_file(inputfile, skip_ref=True)

        # Satellite trail masking.
        sciext = 4
        dqext = 6
        trail = ((1199, 1357), (2841, 1023))
        mask = satdet.make_mask(inputfile, sciext, trail, plot=False, verbose=False)
        update_dq(inputfile, dqext, mask, verbose=False)

        # Compare results.
        self.compare_outputs([(inputfile, truthfile)])
