.. _acsdestripe:

*****************
ACS_DESTRIPE Task
*****************
This task has been written to remove the bias stripe pattern imposed on
post-SM4 full frame ACS/WFC images.  

It is assumed that the data is an ACS/WFC FLT image - with two SCI extensions.
The program needs access to the flatfield specified in the image header
PFLTFILE.

.. note::
    If PFLTFILE has the value "N/A", as is the case with biases and darks,
    then the program assumes a unity flatfield.

    This program also expects an `_flt.fits` file as input, **NOT** a
    `_raw.fits` file.

Parameters
----------
input : str or list of str 
    The name of a single FLT image, or list of FLT images using 
    either wild-cards (`\*flt.fits`) or an IRAF-style 
    at-list (`@filename`).

output : str
    The string to use to add to each input file name to
    indicate an output product. This string will be appended
    to the _flt suffix in each input file's name to create the
    new output filename.  For example, setting 'output=csck' will
    result in output images with suffixes of '_flt_csck.fits'.

clobber : bool 
    Specify whether or not to 'clobber' (delete then replace)
    previously generated products with the same names.  

maxiter : int
    This parameter controls the maximum number of iterations
    to perform when computing the statistics used to compute the
    row-by-row corrections.

sigrej : float
    This parameters sets the sigma level for the rejection applied
    during each iteration of statistics computations for the
    row-by-row corrections. 
    
Examples
--------
To run this task from within Python, make sure the `acstools` package is on
your Python path:

>>> from acstools import acs_destripe
>>> acs_destripe.clean('uncorrected_flt.fits','csck', clobber=False, maxiter=15, sigrej=2.0)

To run this task using the TEAL GUI to set the parameters under PyRAF:

>>> import acstools
>>> from stsci.tools import teal
>>> teal.teal('acs_destripe')

To run this task from the operating system command line, make sure the file
`acs_destripe.py` is on your executable path:

    % ./acs_destripe [-h][-c] uncorrected_flt.fits uncorrected_flt_csck.fits [15 [2.0]]

Version
-------

.. autodata:: acstools.acs_destripe.__version__
.. autodata:: acstools.acs_destripe.__vdate__

Author
------

.. autodata:: acstools.acs_destripe.__author__
