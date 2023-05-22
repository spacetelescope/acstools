"""
The acscte module contains a function `acscte` that calls the ACSCTE executable.
Use this function to facilitate batch runs of ACSCTE.

Only WFC full-frame and some 2K subarrays are currently supported. See
`Table 7.6 in the ACS Instrument Handbook <https://hst-docs.stsci.edu/acsihb/chapter-7-observing-techniques/7-3-operating-modes#id-7.3OperatingModes-table7.6>`_
for more details.

.. note:: Calibration flags are controlled by primary header.

Examples
--------

>>> from acstools import acscte
>>> acscte.acscte('*blv_tmp.fits')

For help usage use ``exe_args=['--help']``

"""
# STDLIB
import os
import subprocess  # nosec

__taskname__ = "acscte"
__version__ = "1.0"
__vdate__ = "13-Aug-2013"
__all__ = ['acscte']


def acscte(input, exec_path='', time_stamps=False, verbose=False, quiet=False,
           single_core=False, exe_args=None):
    r"""
    Run the acscte.e executable as from the shell.

    Expect input to be ``*_blv_tmp.fits``.
    Output is automatically named ``*_blc_tmp.fits``.

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
        The complete path to ACSCTE executable.
        If not given, run ACSCTE given by 'acscte.e'.

    time_stamps : bool, optional
        Set to True to turn on the printing of time stamps.

    verbose : bool, optional
        Set to True for verbose output.

    quiet : bool, optional
        Set to True for quiet output.

    single_core : bool, optional
        CTE correction in ACSCTE will by default try to use all available
        CPUs on your computer. Set this to True to force the use of just
        one CPU.

    exe_args : list, optional
        Arbitrary arguments passed to underlying executable call.
        Note: Implementation uses subprocess.call and whitespace is not
        permitted. E.g. use exe_args=['--nThreads', '1']

    """
    from stsci.tools import parseinput  # Optional package dependency

    if exec_path:
        if not os.path.exists(exec_path):
            raise OSError('Executable not found: ' + exec_path)
        call_list = [exec_path]
    else:
        call_list = ['acscte.e']

    # Parse input to get list of filenames to process.
    # acscte.e only takes 'file1,file2,...'
    infiles, dummy_out = parseinput.parseinput(input)
    call_list.append(','.join(infiles))

    if time_stamps:
        call_list.append('-t')

    if verbose:
        call_list.append('-v')

    if quiet:
        call_list.append('-q')

    if single_core:
        call_list.append('-1')

    if exe_args:
        call_list.extend(exe_args)

    subprocess.check_call(call_list)  # nosec
