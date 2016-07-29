import subprocess
import pytest

# Check for external executables

CALACS = 'calacs.e'

try:
    subprocess.check_call(['which', CALACS])
except:
    HAS_CALACS = False
else:
    HAS_CALACS = True

# pytest marker to mark tests which get data from the web
remote_data = pytest.mark.remote_data

# pytest marker to mark tests that use CALACS
use_calacs = pytest.mark.skipif(not HAS_CALACS, reason='no CALACS')
