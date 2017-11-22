""" Helper functions for ACS wrappers."""

import subprocess

def _callAndRaiseOnSignal(call_list):
    try:
        subprocess.check_call(call_list)
    except subprocess.CalledProcessError as err:
        if err.returncode < 0:
            raise err
        else:
            return err.returncode
    except:
        raise

    return 0
