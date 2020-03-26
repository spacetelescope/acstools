"""
The acsccd module contains a function `acsccd` that calls the ACSCCD executable.
Use this function to facilitate batch runs of ACSCCD.

.. note:: Calibration flags are controlled by primary header.

.. warning:: Do not use with SBC MAMA images.

Examples
--------

>>> from acstools import acsccd
>>> acsccd.acsccd('*raw.fits')

For help usage use ``exe_args=['--help']``

"""
# STDLIB
import os
import subprocess  # nosec

__taskname__ = "acsccd"
__version__ = "2.0"
__vdate__ = "13-Aug-2013"
__all__ = ['acsccd']


#
# These keywords do not work because they get overwritten by header in
# acsccd/getacsflags.c
#
# dqicorr=False, atodcorr=False, blevcorr=False, biascorr=False
#
# dqicorr, atodcorr, blevcorr, biascorr : bool, optional
#     Enable XXXXCORR.
#     If all False, will set all but ATODCORR to PERFORM.
#     If any is True, will set that to PERFORM and the rest to OMIT.
#
def acsccd(input, exec_path='', time_stamps=False, verbose=False, quiet=False,
           exe_args=None):
    r"""
    Run the acsccd.e executable as from the shell.

    Expect input to be ``*_raw.fits``.
    Output is automatically named ``*_blv_tmp.fits``.

    Parameters
    ----------
    input : str or list of str
        Input filenames in one of these formats:

            * a single filename ('j1234567q_raw.fits')
            * a Python list of filenames
            * a partial filename with wildcards ('\*raw.fits')
            * filename of an ASN table ('j12345670_asn.fits')
            * an at-file (``@input``)

    exec_path : str, optional
        The complete path to ACSCCD executable.
        If not given, run ACSCCD given by 'acsccd.e'.

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
        call_list = ['acsccd.e']

    # Parse input to get list of filenames to process.
    # acsccd.e only takes 'file1,file2,...'
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

    #if dqicorr:
    #    call_list.append('-dqi')

    #if atodcor:
    #    call_list.append('-atod')

    #if blevcorr:
    #    call_list.append('-blev')

    #if biascorr:
    #    call_list.append('-bias')

    subprocess.check_call(call_list)  # nosec
