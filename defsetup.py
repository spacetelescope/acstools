from __future__ import division # confidence high

import sys
import distutils
import distutils.core
import distutils.sysconfig

try:
    import numpy
except:
    raise ImportError('NUMPY was not found. It may not be installed or '
                      'it may not be on your PYTHONPATH')


pythoninc = distutils.sysconfig.get_python_inc()
numpyinc = numpy.get_include()

pkg = 'acstools'

ext = [ distutils.core.Extension(
    pkg + '.PixCte_FixY',
    ['src/PixCte_FixY.c', 'src/PixCteCorr_funcs.c', 'src/FixYCte.c'],
    include_dirs = [pythoninc,numpyinc] ) ]

setupargs = {
    'version': '1.7.1',
    'description': 'Python Tools for ACS Data',
    'author': 'Warren Hack, Matt Davis, Pey Lian Lim, '
              'Jay Anderson, Norman Grogin',
    'author_email': 'help@stsci.edu',
    'license': 'http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE',
    'data_files': [(pkg + '/pars', ['lib/acstools/pars/*']),
                   (pkg, ['LICENSE.txt'])],
    'scripts': ['lib/acstools/acs_destripe', 'lib/acstools/runastrodriz'],
    'platforms': ['Linux', 'Solaris', 'Mac OS X', 'Win'],
    'ext_modules': ext,
    'package_dir': {'acstools': 'lib/acstools'}
}
