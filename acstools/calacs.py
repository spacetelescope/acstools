"""
The calacs module contains a function `calacs` that calls the CALACS executable.
Use this function to facilitate batch runs of CALACS.

Examples
--------

>>> from acstools import calacs
>>> calacs.calacs(filename)

For help usage use ``exe_args=['--help']``

"""
import os
import subprocess  # nosec

__all__ = ['calacs']


def calacs(input_file, exec_path=None, time_stamps=False, temp_files=False,
           verbose=False, debug=False, quiet=False, single_core=False,
           exe_args=None):
    """
    Run the calacs.e executable as from the shell.

    By default this will run the calacs given by 'calacs.e'.

    Parameters
    ----------
    input_file : str
        Name of input file.

    exec_path : str, optional
        The complete path to a calacs executable.

    time_stamps : bool, optional
        Set to True to turn on the printing of time stamps.

    temp_files : bool, optional
        Set to True to have CALACS save temporary files.

    verbose : bool, optional
        Set to True for verbose output.

    debug : bool, optional
        Set to True to turn on debugging output.

    quiet : bool, optional
        Set to True for quiet output.

    single_core : bool, optional
        CTE correction in CALACS will by default try to use all available
        CPUs on your computer. Set this to True to force the use of just
        one CPU.

    exe_args : list, optional
        Arbitrary arguments passed to underlying executable call.
        Note: Implementation uses subprocess.call and whitespace is not
        permitted. E.g. use exe_args=['--nThreads', '1']

    """
    if exec_path:
        if not os.path.exists(exec_path):
            raise OSError('Executable not found: ' + exec_path)

        call_list = [exec_path]
    else:
        call_list = ['calacs.e']

    if time_stamps:
        call_list.append('-t')

    if temp_files:
        call_list.append('-s')

    if verbose:
        call_list.append('-v')

    if debug:
        call_list.append('-d')

    if quiet:
        call_list.append('-q')

    if single_core:
        call_list.append('-1')

    if not os.path.exists(input_file):
        raise IOError('Input file not found: ' + input_file)

    call_list.append(input_file)

    if exe_args:
        call_list.extend(exe_args)

    subprocess.check_call(call_list)  # nosec
