*****************
calacs.e (HSTCAL)
*****************

A detailed description of this new and improved CALACS is available in
`ACS Data Handbook v7.0 or later <http://www.stsci.edu/hst/acs/documents/handbooks/currentDHB/>`_.
If you have questions not answered in the documentation, please contact
`STScI Help Desk <https://hsthelp.stsci.edu>`_.


Running CALACS
==============


Where to Find CALACS
--------------------

CALACS is now part of
`HSTCAL package <https://github.com/spacetelescope/hstcal>`_.


Usage
-----

From the command line::

   calacs.e jb1f89eaq_raw.fits [command line options]


Command Line Options
--------------------

CALACS supports several command line options:

* -t

  * Print verbose time stamps.

* -s

  * Save temporary files.

* -v

  * Turn on verbose output.

* -d

  * Turn on debug output.

* -q

  * Turn on quiet output.

* -1

  * Turn off parallel processing for PCTECORR. Try this option if you encounter
    problems running CALACS with PCTECORR=PERFORM.

* --nthreads <N>

  * Specify the number of threads used for PCTECORR.
    The default is 'greedy' and will use that specified by the system environment variable OMP_NUM_THREADS.
    If N > ``OMP_NUM_THREADS``, ``OMP_NUM_THREADS`` will be used instead. The option -1 takes precedence.

* --ctegen <1|2>

  * Specify which generation CTE correction to use, the default is 2. Gen 1 (officially deprecated) refers to
    the correction algorithm used in calacs version pre 9.2.0 in relation to the following ISRs
    ACS ISR 2010-03 and 2012-03. Gen 2 refers to the new CTE correction algorithm implemented in calacs
    version 9.2.0 (HSTCAL 1.3.0) in relation to the ISR (to be updated).

* --pctetab <filename>

  * Override the PCTETAB reference file specified in the image header.

Parallel Processing with OpenMP
-------------------------------

By default, CALACS will attempt to perform PCTECORR using all available CPUs on
your machine. You can set the maximum number of CPUs available for CALACS by
either using the command line option --nthreads <N> or by
setting the ``OMP_NUM_THREADS`` environmental variable.

In tcsh::

  setenv OMP_NUM_THREADS 2

In bash::

  export OMP_NUM_THREADS=2


Batch CALACS
------------

The recommended method for running CALACS in batch mode is to use Python and
the `acstools` package.

For example::

    from acstools import calacs
    import glob

    for fits in glob.iglob('j*_raw.fits'):
        calacs.calacs(fits)


BIASCORR
========

BIASCORR is now performed before BLEVCORR. This should not significantly affect
science results. This change was necessary to accomodate BIASFILE subtraction in
DN with the rest of the calculations done in ELECTRONS.


Unit Conversion to Electrons
============================

The image is multiplied by gain right after BIASCORR, converting it to
ELECTRONS. This step is no longer embedded within FLATCORR.


BLEVCORR
========

BLEVCORR is now performed after BIASCORR. Calculations are done in ELECTRONS.

For post-SM4 full-frame WFC exposures, it also includes:

    * de-striping to remove stripes introduced by new hardware installed during
      SM-4 (J. Anderson; ACS ISR 2011-05); and
    * if JWROTYPE=DS_int and CCDGAIN=2, also correct for bias shift
      (ACS ISR 2012-02) and cross-talk (N. Grogin; ACS ISR 2010-02).


SINKCORR
========

SINKCORR flags sink pixels and the adjacent affected pixels with the value
1024 in the DQ array of WFC images using the SNKCFILE. Only performed on images
taken after January 2015.


Pixel-Based CTE Correction (PCTECORR)
=====================================

For all full-frame WFC exposures, pixel-based CTE correction (ACS ISR to be updated)
is applied in ACSCTE that occurs after the ACSCCD series;
i.e., after BLEVCORR.

Because the CTE correction is applied before DARKCORR and FLSHCORR, it is
necessary to use a CTE-corrected dark (DRKCFILE) if
the PCTECORR step is enabled.

Parameters characterizing the CTE correction are stored in a reference table,
PCTETAB.

.. note::

    CALACS 8.2 and later uses a slightly different PCTETAB format, where
    the COL_SCALE extension does not include overscan columns.

Required Keywords
-----------------

Running CALACS with pixel-based CTE correction requires the following header
keywords:

* PCTECORR

  * By default, set to PERFORM for all full-frame WFC exposures, except BIAS.

* PCTETAB

  * Reference table containing CTE correction parameters. By default, it should
    be in the ``jref`` directory and have the suffix ``_cte.fits``.

* DRKCFILE

  * Similar to DARKFILE but with CTE correction performed. By default, it should
    be in the ``jref`` directory and have the suffix ``_dkc.fits``. This is
    necessary because PCTECORR is done before DARKCORR.

Optional Keywords
-----------------

You may adjust some CTE correction algorithm parameters by changing the
following keywords in the RAW image header. The default values are picked for
optimal results in a typical WFC full-frame exposure. Changing these values is
not recommended unless you know what you are doing.

* FIXROCR

  * Account for and correct readout cosmic ray over-subtraction.

    * 0 - Off: do not correct
    * 1 - On: correct

* PCTENPAR

  * Number of parallel transfer iterations.

* PCTENSMD

  * Read noise mitigation mode:

    * 0 - No mitigation
    * 1 - Perform noise smoothing
    * 2 - No noise smoothing

* PCTERNOI

  * Read noise amplitude in ELECTRONS.

* PCTETLEN

  * Maximum length of CTE trail.

* PCTENFOR

  * Number of iterations used for forward CTE model.


Dark Current Subtraction (DARKCORR)
===================================

It uses DARKFILE if PCTECORR=OMIT, otherwise it uses DRKCFILE (CTE-corrected
dark reference file).

Dark image is now scaled by EXPTIME and FLASHDUR. For post-SM4 non-BIAS
WFC images, extra 3 seconds are also added to account for idle time before
readout. Any image with non-zero EXPTIME is considered not a BIAS.


Post-Flash Correction (FLSHCORR)
================================

Post-flash correction is now performed after DARKCORR in the ACS2D step.
When FLSHCORR=PERFORM, it uses FLSHFILE.


FLATCORR
========

Conversion from DN to ELECTRONS no longer depends on FLATCORR=PERFORM. Unit
conversion is done for all exposures after BIASCORR.


Photometry Keywords (PHOTCORR)
==============================

The PHOTCORR step is now performed using tables of precomputed values instead
of calls  to SYNPHOT. The correct table for a given image must be specified
in the IMPHTTAB header keyword in order for CALACS to perform the PHOTCORR step.
By default, it should be in the ``jref`` directory and have the suffix
``_imp.fits``. Each DETECTOR uses a different table.

If you do not wish to use this feature, set PHOTCORR to OMIT.


CALACS Output
=============

Using RAW as input:

    * flt.fits: Same as existing FLT.
    * flc.fits: Similar to FLT, except with pixel-based CTE correction applied.

Using ASN as input with ACSREJ:

    * crj.fits: Same as existing CRJ.
    * crc.fits: Similar to CRJ, except with pixel-based CTE correction applied.

CALACS uses HSTIO that utilizes ``PIXVALUE`` keyword to represent a data
extension with constant value. However, this is not a standard FITS behavior
and is not recognized by PyFITS. Therefore, one should use
``stsci.tools.stpyfits``, which is distributed as part of ``stsci_python``,
instead of ``pyfits`` or `astropy.io.fits` when working with CALACS products.
To use ``stpyfits`` in Python::

    from stsci.tools import stpyfits as pyfits
