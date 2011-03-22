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

ext = [ distutils.core.Extension('acstools2.PixCte_FixY',
        ['src/py_PixCteCorr.c',
         'src/PixCteCorr_funcs.c',
         'src/FixYCte.c'],
        include_dirs = ['include',pythoninc,numpyinc]) ]

pkg =  "acstools2"

setupargs = {
    'version' : 		"1.2.0",
    'description' :	    "Python Tools for ACS Data",
    'author' : 		    "Warren Hack, Norman Grogin, Pey Lian Lim, Jay Anderson, Matt Davis",
    'author_email' : 	"help@stsci.edu",
    'license' : 		"http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'data_files' :      [( pkg+"/pars", ['lib/pars/*']),( pkg, ['lib/*.help']),(pkg,['lib/LICENSE.txt'])],
    'scripts' :         [ 'lib/acs_destripe'] ,
    'platforms' : 	    ["Linux","Solaris","Mac OS X","Win"],
    'ext_modules' :     ext,

}

