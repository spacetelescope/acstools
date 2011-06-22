"""The acstools package holds Python tasks useful for analyzing ACS data.

These tasks include:

Utility and library functions used by these tasks are also included in this
module.


"""

try:
    import svn_version
    __svn_version__ = svn_version.__svn_version__
except:
    __svn_version__ = 'Unable to determine SVN revision'

import acs_destripe
import PixCteCorr

# These lines allow TEAL to print out the names of TEAL-enabled tasks 
# upon importing this package.
import os
from stsci.tools import teal
teal.print_tasknames(__name__, os.path.dirname(__file__))
