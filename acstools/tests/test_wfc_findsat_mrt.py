'''
Test satellite trail detection and masking using findsat_mrt
'''

from acstools.findsat_mrt import wfc_wrapper
from acstools.tests.helpers import BaseACSTOOLS
import acstools as a
package_path = a.__path__[0]
import os

#update all this with my code

class test_findsat_mrt(BaseACSTOOLS):
    detector = 'wfc'
    package_directory = os.path.dirname(os.path.abspath(__file__))

    def test_wfc_wrapper(self):
        """Identify and mask trails in WFC extension 4."""

        rootname = 'jc8m32j5q' 
        inputfile = rootname + '_flc.fits'  
        truthmaskfile = rootname + '_flc_mask_ref.fits'
        truthcatalogfile = rootname + '_flc_catalog_ref.fits'

        # Prepare input file.
        self.get_input_file(inputfile, skip_ref=True)

        wfc_wrapper(inputfile, binsize=4, extension=4,
                    output_root='jc8m32j5q',
                    output_dir = package_path + '/tests/data/input/',
                    threads=8, execute=True, save_mask=True,
                    save_diagnostic=False, save_catalog=True)

        maskfile = rootname + 'flc_mask.fits'
        catalogfile = rootname + 'flc_catalog.fits'

        # Compare results.
        self.compare_outputs([(maskfile, truthmaskfile),
                              (catalogfile, truthcatalogfile)])
