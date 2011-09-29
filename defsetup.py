from __future__ import division # confidence high

import sys
import distutils
import distutils.core
import distutils.sysconfig

try:
    import numpy
except:
    raise ImportError('NUMPY was not found. It may not be installed or it may not be on your PYTHONPATH')

pythoninc = distutils.sysconfig.get_python_inc()
numpyinc = numpy.get_include()

pkg =  "acstools"

ext = [ distutils.core.Extension(pkg + '.PixCte_FixY',
        ['src/PixCte_FixY.c',
         'src/PixCteCorr_funcs.c',
         'src/FixYCte.c'],
        include_dirs = [pythoninc,numpyinc]) ]

setupargs = {
    'version' : 		"2.0.0dev",
    'description' :	    "Python Tools for ACS Data",
    'author' : 		    "Warren Hack, Norman Grogin, Pey Lian Lim, Jay Anderson, Matt Davis",
    'author_email' : 	"help@stsci.edu",
    'license' : 		"http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'data_files' :      [(pkg,['LICENSE.txt'])],
    'scripts' :         ['lib/acstools/fitsdiff'] ,
    'platforms' : 	    ["Linux","Solaris","Mac OS X","Win"],
    'ext_modules' :     ext,
    'package_dir' :     { 'acstools':'lib/acstools', },

}

