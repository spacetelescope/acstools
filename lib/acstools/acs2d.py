"""
The acs2d module contains a function `acs2d` that calls the ACS2D executable.
Use this function to facilitate batch runs of ACS2D, or for the TEAL interface.

Examples
--------

In Python without TEAL:

>>> from acstools import acs2d
>>> acs2d.acs2d('*blv_tmp.fits')

In Python with TEAL:

>>> from stsci.tools import teal
>>> from acstools import acs2d
>>> teal.teal('acs2d')

In Pyraf::

    --> import acstools
    --> epar acs2d

"""
# STDLIB
import os.path
import subprocess

# STSCI
from stsci.tools import parseinput
try:
    from stsci.tools import teal
except:
    teal = None


__taskname__ = "acs2d"
__version__ = "1.0"
__vdate__ = "18-Dec-2012"


def acs2d(input, exec_path='', time_stamps=False, verbose=False, quiet=False,
          dqicorr=False, glincorr=False, lflgcorr=False, darkcorr=False,
          flshcorr=False, flatcorr=False, shadcorr=False, photcorr=False):
    """
    Run the acs2d.e executable as from the shell.

    Output is automatically named based on input suffix:

        +--------------------+----------------+------------------------------+
        | INPUT              | OUTPUT         | EXPECTED DATA                |
        +====================+================+==============================+
        | ``*_raw.fits``     | ``*_flt.fits`` | SBC image.                   |
        +--------------------+----------------+------------------------------+
        | ``*_blv_tmp.fits`` | ``*_flt.fits`` | ACSCCD output.               |
        +--------------------+----------------+------------------------------+
        | ``*_blc_tmp.fits`` | ``*_flc.fits`` | ACSCCD output with PCTECORR. |
        +--------------------+----------------+------------------------------+
        | ``*_crj_tmp.fits`` | ``*_crj.fits`` | ACSREJ output.               |
        +--------------------+----------------+------------------------------+
        | ``*_crc_tmp.fits`` | ``*_crc.fits`` | ACSREJ output with PCTECORR. |
        +--------------------+----------------+------------------------------+

    Parameters
    ----------
    input : str or list of str
        Input filenames in one of these formats:

            * a single filename ('j1234567q_blv_tmp.fits')
            * a Python list of filenames
            * a partial filename with wildcards ('\*blv_tmp.fits')
            * filename of an ASN table ('j12345670_asn.fits')
            * an at-file (``@input``)

    exec_path : str, optional
        The complete path to ACS2D executable.
        If not given, run ACS2D given by 'acs2d.e'.

    time_stamps : bool, optional
        Set to True to turn on the printing of time stamps.

    verbose : bool, optional
        Set to True for verbose output.

    quiet : bool, optional
        Set to True for quiet output.

    dqicorr, glincorr, lflgcorr, darkcorr, flshcorr, flatcorr, shadcorr, photcorr : bool, optional
        Enable XXXXCORR.
        If all False, will set all but FLSHCORR and SHADCORR to PERFORM.
        If any is True, will set that to PERFORM and the rest to OMIT.
        GLINCORR and LFLGCORR are used for SBC MAMA only.
        FLSHCORR and SHADCORR are used for CCD only.

    """
    if exec_path:
        if not os.path.exists(exec_path):
            raise OSError('Executable not found: ' + exec_path)
        call_list = [exec_path]
    else:
        call_list = ['acs2d.e']

    # Parse input to get list of filenames to process.
    # acs2d.e only takes 'file1,file2,...'
    infiles, dummy_out = parseinput.parseinput(input)
    call_list.append(','.join(infiles))

    if time_stamps:
        call_list.append('-t')

    if verbose:
        call_list.append('-v')

    if quiet:
        call_list.append('-q')

    if dqicorr:
        call_list.append('-dqi')

    if glincorr:
        call_list.append('-glin')

    if lflgcorr:
        call_list.append('-lflg')

    if darkcorr:
        call_list.append('-dark')

    if flshcorr:
        call_list.append('-flash')

    if flatcorr:
        call_list.append('-flat')

    if shadcorr:
        call_list.append('-shad')

    if photcorr:
        call_list.append('-phot')

    subprocess.call(call_list)


def getHelpAsString():
    """
    Returns documentation on the `acs2d` function. Required by TEAL.

    """
    return acs2d.__doc__


def run(configobj=None):
    """
    TEAL interface for the `acs2d` function.

    """
    acs2d(configobj['input'],
          exec_path=configobj['exec_path'],
          time_stamps=configobj['time_stamps'],
          verbose=configobj['verbose'],
          quiet=configobj['quiet'],
          dqicorr=configobj['dqicorr'],
          glincorr=configobj['glincorr'],
          lflgcorr=configobj['lflgcorr'],
          darkcorr=configobj['darkcorr'],
          flshcorr=configobj['flshcorr'],
          flatcorr=configobj['flatcorr'],
          shadcorr=configobj['shadcorr'],
          photcorr=configobj['photcorr'])
