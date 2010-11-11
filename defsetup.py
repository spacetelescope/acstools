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

ext = [ distutils.core.Extension('acstools.PixCte_FixY',['src/PixCte_FixY.c'],
        include_dirs = [pythoninc,numpyinc]) ]

pkg =  "acstools"

setupargs = {
    'version' : 		"1.1.0",
    'description' :	    "Python Tools for ACS Data",
    'author' : 		    "Warren Hack, Norman Grogin, Pey Lian Lim, Jay Anderson",
    'author_email' : 	"help@stsci.edu",
    'license' : 		"http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'data_files' :      [( pkg+"/pars", ['lib/pars/*']),( pkg, ['lib/*.help']),(pkg,['lib/LICENSE.txt'])],
    'scripts' :         [ 'lib/acs_destripe', 'lib/updatenpol'] ,
    'platforms' : 	    ["Linux","Solaris","Mac OS X","Win"],
    'ext_modules' :     ext,

}

