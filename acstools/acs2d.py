"""
The acs2d module contains a function `acs2d` that calls the ACS2D executable.
Use this function to facilitate batch runs of ACS2D.

Examples
--------

>>> from acstools import acs2d
>>> acs2d.acs2d('*blv_tmp.fits')

For help usage use ``exe_args=['--help']``.

"""
# STDLIB
import os
import subprocess  # nosec

__taskname__ = "acs2d"
__version__ = "2.0"
__vdate__ = "10-Oct-2014"
__all__ = ['acs2d']


def acs2d(input, exec_path='', time_stamps=False, verbose=False, quiet=False,
          exe_args=None):
    r"""
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

    if exe_args:
        call_list.extend(exe_args)

    subprocess.check_call(call_list)  # nosec
