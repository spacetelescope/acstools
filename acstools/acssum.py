"""
The acssum module contains a function `acssum` that calls the ACSSUM executable.
Use this function to facilitate batch runs of ACSSUM.

Examples
--------

>>> from acstools import acssum
>>> acssum.acssum('*flt.fits', 'combined_image.fits')

For help usage use ``exe_args=['--help']``

"""
# STDLIB
import os
import subprocess  # nosec

__taskname__ = "acssum"
__version__ = "1.0"
__vdate__ = "18-Dec-2012"
__all__ = ['acssum']


def acssum(input, output, exec_path='', time_stamps=False, verbose=False,
           quiet=False, exe_args=None):
    r"""
    Run the acssum.e executable as from the shell.

    Parameters
    ----------
    input : str or list of str
        Input filenames in one of these formats:

            * a Python list of filenames
            * a partial filename with wildcards ('\*flt.fits')
            * filename of an ASN table ('j12345670_asn.fits')
            * an at-file (``@input``)

    output : str
        Output filename.
        If `output` is '' and `input` is '\*_asn.fits',
        `output` will be automatically set to '\*_sfl.fits'.
        Otherwise, it is an error not to provide a specific `output`.

    exec_path : str, optional
        The complete path to ACSSUM executable.
        If not given, run ACSSUM given by 'acssum.e'.

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
        call_list = ['acssum.e']

    # Parse input to get list of filenames to process.
    # acssum.e only takes 'file1,file2,...'
    infiles, dummy_out = parseinput.parseinput(input)
    call_list.append(','.join(infiles))

    call_list.append(output)

    if time_stamps:
        call_list.append('-t')

    if verbose:
        call_list.append('-v')

    if quiet:
        call_list.append('-q')

    if exe_args:
        call_list.extend(exe_args)

    subprocess.check_call(call_list)  # nosec
