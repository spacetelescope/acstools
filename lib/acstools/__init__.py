"""The acstools package holds Python tasks useful for analyzing ACS data.

These tasks include:

Utility and library functions used by these tasks are also included in this
module.


"""
from __future__ import absolute_import, print_function
from .version import *

from . import acs_destripe
from . import acs_destripe_plus
from . import runastrodriz
from . import calacs
from . import acsccd
from . import acscte
from . import acs2d
from . import acsrej
from . import acssum

# We can remove this in the future when people no longer care about PixCteCorr
print("\nPixCteCorr is no longer supported. Please use acscte.\n")

# These lines allow TEAL to print out the names of TEAL-enabled tasks
# upon importing this package.
import os
from stsci.tools import teal
teal.print_tasknames(__name__, os.path.dirname(__file__))
