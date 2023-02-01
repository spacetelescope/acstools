'''
Test satellite trail detection and masking using findsat_mrt
'''

from astropy.utils.data import get_pkg_data_filename
from acstools.findsat_mrt import wfc_wrapper
from acstools.tests.helpers import BaseACSTOOLS
import os
from astropy.io.fits import FITSDiff
import logging


logging.basicConfig()
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

#update all this with my code

class TestFindsatMRT(BaseACSTOOLS):
    detector = 'wfc'

    def test_wfc_wrapper(self,tmpdir):
        """Identify and mask trails in WFC extension 4."""

        rootname = 'jc8m32j5q' 
        inputfile = rootname + '_flc.fits'  
        truthmaskfile = rootname + '_flc_mask_ref.fits'
        truthcatalogfile = rootname + '_flc_catalog_ref.fits'

        # Prepare input file.
        self.get_input_file(inputfile, skip_ref=True)

        wfc_wrapper(inputfile, binsize=4, extension=4,
                    output_root='jc8m32j5q',
                    output_dir = tmpdir,
                    threads=8, execute=True, save_mask=True,
                    save_diagnostic=False, save_catalog=True)


        #Compare mask with truth
        creature_report = ''
        all_okay = True
        desired = tmpdir + rootname + '_flc_mask_ref.fits'
        truthpath = get_pkg_data_filename(
                    os.path.join('data', 'truth', desired),
                    package='acstools.tests',
                    show_progress=False, remote_timeout=self.timeout)
        fdiff = FITSDiff(inputfile, desiredpath, rtol=rtol, atol=atol,
                         ignore_keywords=ignore_keywords)
        creature_report += fdiff.report()
        
        if not all_okay and raise_error:
            raise AssertionError(os.linesep + creature_report)
            
        #self.compare_outputs([(maskfile, truthmaskfile),
        #                      (catalogfile, truthcatalogfile)])

