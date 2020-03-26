"""Custom ``pytest`` configurations."""
try:
    from pytest_astropy_header.display import (PYTEST_HEADER_MODULES,
                                               TESTED_VERSIONS)
except ImportError:
    PYTEST_HEADER_MODULES = {}
    TESTED_VERSIONS = {}

try:
    from acstools.version import version
except ImportError:
    version = 'unknown'

# Uncomment the following line to treat all DeprecationWarnings as
# exceptions
# NOTE: socks warning fixed by https://github.com/Anorov/PySocks/pull/106
#       but not released yet.
from astropy.tests.helper import enable_deprecations_as_exceptions
enable_deprecations_as_exceptions(warnings_to_ignore_entire_module=['socks'])

# Uncomment and customize the following lines to add/remove entries
# from the list of packages for which version numbers are displayed
# when running the tests.
PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
PYTEST_HEADER_MODULES['beautifulsoup4'] = 'bs4'
PYTEST_HEADER_MODULES['requests'] = 'requests'
PYTEST_HEADER_MODULES['stsci.tools'] = 'stsci.tools'
PYTEST_HEADER_MODULES.pop('Pandas')
PYTEST_HEADER_MODULES.pop('h5py')

TESTED_VERSIONS['acstools'] = version
