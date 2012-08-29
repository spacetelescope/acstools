"""The acstools package holds Python tasks useful for analyzing ACS data.

These tasks include:

Utility and library functions used by these tasks are also included in this
module.


"""


try:
    from acstools.version import (__version__, __svn_revision__,
                                  __svn_full_info__, __setup_datetime__)
except ImportError:
    __version__ = ''
    __svn_revision__ = ''
    __svn_full_info__ = ''
    __setup_datetime__ = None

import acs_destripe
import PixCteCorr
import runastrodriz
import calacs

# These lines allow TEAL to print out the names of TEAL-enabled tasks
# upon importing this package.
import os
from stsci.tools import teal
teal.print_tasknames(__name__, os.path.dirname(__file__))
