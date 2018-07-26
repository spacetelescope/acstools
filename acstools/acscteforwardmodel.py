"""
The acscte module contains a function `acscte` that calls the ACSCTE forward
model executable.
Use this function to facilitate batch runs of the forward model, or for the
TEAL interface.

Only WFC full-frame and some 2K subarrays are currently supported. See
`ACS Data Handbook <http://www.stsci.edu/hst/acs/documents/handbooks/currentDHB/>`_
for more details.

.. note:: Calibration flags are controlled by primary header.

Examples
--------

In Python without TEAL:

>>> from acstools import acscteforwardmodel
>>> acscteforwardmodel.acscteforwardmodel('*blc_tmp.fits')

In Python with TEAL:

>>> from stsci.tools import teal
>>> from acstools import acscteforwardmodel
>>> teal.teal('acscteforwardmodel')

In Pyraf::

    --> import acstools
    --> epar acscteforwardmodel

For help usage use ``exe_args=['--help']``

"""
# STDLIB
import os
import subprocess

__taskname__ = "acscteforwardmodel"
__version__ = "1.0"
__vdate__ = "19-Jul-2018"
__all__ = ['acscteforwardmodel']


def acscteforwardmodel(input, exec_path='', time_stamps=False, verbose=False, 
               quiet=False, single_core=False, exe_args=None):
    """
    Run the acscteforwardmodel.e executable as from the shell.

    Expect input to be ``*_blc_tmp.fits`` or ``*_flc.fits``.
    Output is automatically named ``*_ctefmod.fits``.

    Parameters
    ----------
    input : str or list of str
        Input filenames in one of these formats:

            * a single filename ('j1234567q_blc_tmp.fits')
            * a Python list of filenames
            * a partial filename with wildcards ('\*blc_tmp.fits')
            * filename of an ASN table ('j12345670_asn.fits')
            * an at-file (``@input``)

    exec_path : str, optional
        The complete path to ACSCTE forward model executable.
        If not given, run ACSCTE given by 'acscteforwardmodel.e'.

    time_stamps : bool, optional
        Set to True to turn on the printing of time stamps.

    verbose : bool, optional
        Set to True for verbose output.

    quiet : bool, optional
        Set to True for quiet output.

    single_core : bool, optional
        CTE correction in the ACSCTE forward model will by default try to use
        all available CPUs on your computer. Set this to True to force the use
        of just one CPU.

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
        call_list = ['acscteforwardmodel.e']

    # Parse input to get list of filenames to process.
    # acscte.e only takes 'file1,file2,...'
    infiles, dummy_out = parseinput.parseinput(input)

    # Make temporary files for input into forward model
    tmp_files = make_tmp_files(infiles)

    call_list.append(','.join(tmp_files))

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

    subprocess.check_call(call_list)

    # Restore extra FLC extensions to forward model
    restore_files(infiles)


def make_tmp_files(infiles):
    """
    If more than 6 extensions, take SCI, ERR, and DQ extensions and make a
    temporary fits image to run the forward model on.

    Parameters
    ----------
    infiles : list of strings
        List of input files parsed by parseinput

    Returns
    -------
    tmp_files : list of strings
        List of temporary files containing 6 or fewer extensions

    """

    from astropy.io import fits

    tmp_files = []

    for file in infiles:

        hdu = fits.open(file)
        hdr = hdu[0].header

        outfile = '{}_tmp.fits'.format(file.split('.fits')[0])

        # Update NEXTEND keyword and select extensions for temporary file
        if hdr['NEXTEND'] > 6:

            if hdr['SUBARRAY']:

                new_hdu = hdu[:4]
                hdr['NEXTEND'] = 3

            else:

                new_hdu = hdu[:7]
                hdr['NEXTEND'] = 6

            # Write temporary file
            new_hdu.writeto(outfile)

            tmp_files.append(outfile)

        # If file has typical number of FLCs, output original input
        else:

            tmp_files.append(file)

    return tmp_files


def restore_files(infiles):
    """
    Copy SCI, ERR, and DQ extensions from forward-modeled temporary file to
    a copy of the original FLC.

    Parameters
    ----------
    infiles : list of strings
        List of input files parsed by parseinput

    """

    from astropy.io import fits
    from shutil import copyfile
    import glob

    for file in infiles:

        base = file.split('.fits')[0]
        tmp_file = '{}_tmp_ctefmod.fits'.format(base)
        
        if os.path.exists(tmp_file):
        
            # Make a copy of input file in which to dump forward modeled
            # extensions
            copy = '{}_ctefmod.fits'.format(base)

            copyfile(file, copy)
            
            hdu = fits.open(copy, mode='update')
            tmp_hdu = fits.open(tmp_file)

            hdr = hdu[0].header

            if hdr['SUBARRAY']:

                hdu[1].header = tmp_hdu[1].header
                hdu[1].data = tmp_hdu[1].data

            else:

                hdu[1].header = tmp_hdu[1].header
                hdu[1].data = tmp_hdu[1].data
                hdu[4].header = tmp_hdu[4].header
                hdu[4].data = tmp_hdu[4].data

            hdu.close()

            # Remove temporary files
            for tmp in glob.glob('{}_tmp*'.format(base)):
                os.remove(tmp)


def getHelpAsString():
    """
    Returns documentation on the `acscteforwardmodel` function. Required by TEAL.

    """
    return acscteforwardmodel.__doc__


def run(configobj=None):
    """
    TEAL interface for the `acscteforwardmodel` function.

    """
    acscteforwardmodel(configobj['input'],
                       exec_path=configobj['exec_path'],
                       time_stamps=configobj['time_stamps'],
                       verbose=configobj['verbose'],
                       quiet=configobj['quiet'],
                       single_core=configobj['single_core'],
                       exe_args=configobj['exe_args']
                       )
