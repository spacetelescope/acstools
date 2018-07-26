"""The acstools package holds Python tasks useful for analyzing ACS data.

These tasks include:

Utility and library functions used by these tasks are also included in this
module.

"""
from __future__ import absolute_import, print_function

try:
    from .version import *
except ImportError:
    __version__ = 'unknown'

from . import acs_destripe
from . import acs_destripe_plus
from . import calacs
from . import acsccd
from . import acscte
from . import acscteforwardmodel
from . import acs2d
from . import acsrej
from . import acssum
from . import satdet
from . import utils_calib

# These lines allow TEAL to print out the names of TEAL-enabled tasks
# upon importing this package.
import os
try:
    from stsci.tools import teal
except ImportError:
    pass
else:
    teal.print_tasknames(__name__, os.path.dirname(__file__))


# This is like teal.print_tasknames() above but for local warnings.
def print_deprecation_warnings():
    import sys

    # See if we can bail out early.
    # We can't use the sys.ps1 check if in PyRAF since it changes sys.
    if 'pyraf' not in sys.modules:
        # sys.ps1 is only defined in interactive mode.
        if not hasattr(sys, 'ps1'):
            return  # leave here, we're in someone's script.

    # We can remove this in the future when people no longer care about
    # PixCteCorr.
    print('PixCteCorr is no longer supported. Please use acscte.')


# Comment this out if there are no warnings anymore.
print_deprecation_warnings()
