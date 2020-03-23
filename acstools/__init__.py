"""The acstools package holds Python tasks useful for analyzing ACS data.

These tasks include:

Utility and library functions used by these tasks are also included in this
module.

"""
from pkg_resources import get_distribution, DistributionNotFound


try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
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
from . import acszpt
from . import satdet
from . import utils_calib
