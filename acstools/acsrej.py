"""
The acsrej module contains a function `acsrej` that calls the ACSREJ executable.
Use this function to facilitate batch runs of ACSREJ, or for the TEAL interface.

Examples
--------

In Python without TEAL:

>>> from acstools import acsrej
>>> acsrej.acsrej('*flt.fits', 'combined_image.fits')

In Python with TEAL:

>>> from stsci.tools import teal
>>> from acstools import acsrej
>>> teal.teal('acsrej')

In Pyraf::

    --> import acstools
    --> epar acsrej

"""
# STDLIB
import os.path
import subprocess

# STSCI
try:
    from stsci.tools import parseinput
except ImportError:  # So RTD would build
    pass
try:
    from stsci.tools import teal
except:
    teal = None


__taskname__ = "acsrej"
__version__ = "1.0"
__vdate__ = "18-Dec-2012"


def acsrej(input, output, exec_path='', time_stamps=False, verbose=False,
          shadcorr=False, crrejtab='', crmask=False, scalense=None, initgues='',
          skysub='', crsigmas='', crradius=None, crthresh=None, badinpdq=None,
          newbias=False):
    """
    Run the acsrej.e executable as from the shell.

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

    exec_path : str, optional
        The complete path to ACSREJ executable.
        If not given, run ACSREJ given by 'acsrej.e'.

    time_stamps : bool, optional
        Set to True to turn on the printing of time stamps.

    verbose : bool, optional
        Set to True for verbose output.

    shadcorr : bool, optional
        Perform shutter shading correction.
        If this is False but SHADCORR is set to PERFORM in
        the header of the first image, the correction will
        be applied anyway.
        Only use this with CCD image, not SBC MAMA.

    crrejtab : str, optional
        CRREJTAB to use. If not given, will use CRREJTAB
        given in the primary header of the first input image.

    crmask : bool, optional
        Flag CR-rejected pixels in input files.
        If False, will use CRMASK value in CRREJTAB.

    scalense : float, optional
        Multiplicative scale factor (in percents) applied to noise.
        Acceptable values are 0 to 100, inclusive.
        If None, will use SCALENSE from CRREJTAB.

    initgues : {'med', 'min'}, optional
        Scheme for computing initial-guess image.
        If not given, will use INITGUES from CRREJTAB.

    skysub : {'none', 'mode'}, optional
        Scheme for computing sky levels to be subtracted.
        If not given, will use SKYSUB from CRREJTAB.

    crsigmas : str, optional
        Cosmic ray rejection thresholds given in the format of 'sig1,sig2,...'.
        Number of sigmas given will be the number of rejection
        iterations done. At least 1 and at most 20 sigmas accepted.
        If not given, will use CRSIGMAS from CRREJTAB.

    crradius : float, optional
        Radius (in pixels) to propagate the cosmic ray.
        If None, will use CRRADIUS from CRREJTAB.

    crthresh : float, optional
        Cosmic ray rejection propagation threshold.
        If None, will use CRTHRESH from CRREJTAB.

    badinpdq : int, optional
        Data quality flag used for cosmic ray rejection.
        If None, will use BADINPDQ from CRREJTAB.

    newbias : bool, optional
        ERR is just read noise, not Poisson noise.
        This is used for BIAS images.

    """
    if exec_path:
        if not os.path.exists(exec_path):
            raise OSError('Executable not found: ' + exec_path)
        call_list = [exec_path]
    else:
        call_list = ['acsrej.e']

    # Parse input to get list of filenames to process.
    # acsrej.e only takes 'file1,file2,...'
    infiles, dummy_out = parseinput.parseinput(input)
    call_list.append(','.join(infiles))

    call_list.append(output)

    if time_stamps:
        call_list.append('-t')

    if verbose:
        call_list.append('-v')

    if shadcorr:
        call_list.append('-shadcorr')

    if crrejtab:
        call_list += ['-table', crrejtab]

    if crmask:
        call_list.append('-crmask')

    if scalense is not None:
        if scalense < 0 or scalense > 100:
            raise ValueError('SCALENSE must be 0 to 100')
        call_list += ['-scale', str(scalense)]

    if initgues:
        if initgues not in ('med', 'min'):
            raise ValueError('INITGUES must be "med" or "min"')
        call_list += ['-init', initgues]

    if skysub:
        if skysub not in ('none', 'mode'):
            raise ValueError('SKYSUB must be "none" or "mode"')
        call_list += ['-sky', skysub]

    if crsigmas:
        call_list += ['-sigmas', crsigmas]

    if crradius is not None:
        call_list += ['-radius', str(crradius)]

    if crthresh is not None:
        call_list += ['-thresh ', str(crthresh)]

    if badinpdq is not None:
        call_list += ['-pdq', str(badinpdq)]

    if newbias:
        call_list.append('-newbias')

    subprocess.call(call_list)


def getHelpAsString():
    """
    Returns documentation on the `acsrej` function. Required by TEAL.

    """
    return acsrej.__doc__


def run(configobj=None):
    """
    TEAL interface for the `acsrej` function.

    """
    acsrej(configobj['input'],
           configobj['output'],
           exec_path=configobj['exec_path'],
           time_stamps=configobj['time_stamps'],
           verbose=configobj['verbose'],
           shadcorr=configobj['shadcorr'],
           crrejtab=configobj['crrejtab'],
           crmask=configobj['crmask'],
           scalense=configobj['scalense'],
           initgues=configobj['initgues'],
           skysub=configobj['skysub'],
           crsigmas=configobj['crsigmas'],
           crradius=configobj['crradius'],
           crthresh=configobj['crthresh'],
           badinpdq=configobj['badinpdq'],
           newbias=configobj['newbias'])
