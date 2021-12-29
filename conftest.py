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

# Uncomment and customize the following lines to add/remove entries
# from the list of packages for which version numbers are displayed
# when running the tests.
PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
PYTEST_HEADER_MODULES['beautifulsoup4'] = 'bs4'
PYTEST_HEADER_MODULES['requests'] = 'requests'
PYTEST_HEADER_MODULES['stsci.tools'] = 'stsci.tools'
PYTEST_HEADER_MODULES.pop('Pandas', None)
PYTEST_HEADER_MODULES.pop('h5py', None)

TESTED_VERSIONS['acstools'] = version
