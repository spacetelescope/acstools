"""Test satellite trail detection and masking using findsat_mrt."""

import pytest

pytest.importorskip("skimage")

import os  # noqa: E402

from acstools.findsat_mrt import WfcWrapper  # noqa: E402
from acstools.tests.helpers import BaseACSTOOLS  # noqa: E402


class TestWfcWrapper(BaseACSTOOLS):
    detector = "wfc"

    def test_WfcWrapper(self):
        """Identify and mask trails in WFC extension 4."""
        rootname = "jc8m32j5q"
        inputfile = f"{rootname}_flc.fits"
        desired = f"{rootname}_flc_mask_ref.fits"
        actual = f"{rootname}_flc_mask.fits"

        # Prepare input file.
        self.get_input_file(inputfile, skip_ref=True)

        # If the machine does not have 8 cores, it will only use up to what is
        # available. We save the catalog to make sure that code runs but we do not
        # compare catalog.
        WfcWrapper(
            inputfile,
            binsize=4,
            extension=4,
            output_root=f"{rootname}_flc",
            output_dir=os.curdir,
            max_width=37.5,
            processes=8,
            execute=True,
            save_mask=True,
            save_diagnostic=False,
            save_catalog=True,
        )

        # Compare mask with truth
        self.compare_outputs([(actual, desired)])
