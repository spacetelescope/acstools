*****************
calacs.e (HSTCAL)
*****************

A detailed description of this new and improved CALACS is available in
`ACS Data Handbook v7.0 or later <http://www.stsci.edu/hst/acs/documents/handbooks/currentDHB/acs_cover.html>`_.
If you have questions not answered in the documentation, please contact
STScI Help Desk (``help[at]stsci.edu``).


Running CALACS
==============


Where to Find CALACS
--------------------

CALACS is now part of HSTCAL package, which can be downloaded from
`STSDAS download page <http://www.stsci.edu/institute/software_hardware/stsdas/download-stsdas>`_.


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


Parallel Processing with OpenMP
-------------------------------

By default, CALACS will attempt to perform PCTECORR using all available CPUs on
your machine. You can set the maximum number of CPUs available for CALACS by
setting the ``OMP_NUM_THREADS`` environmental variable.

In tcsh::

  setenv OMP_NUM_THREADS 2

In bash::

  export OMP_NUM_THREADS=2


Batch CALACS
------------

The recommended method for running CALACS in batch mode is to use Python and
the `acstools` package in `STSDAS distribution
<http://www.stsci.edu/institute/software_hardware/stsdas/download-stsdas>`_.

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


Pixel-Based CTE Correction (PCTECORR)
=====================================

For all full-frame WFC exposures, pixel-based CTE correction (ACS ISR 2010-03
and 2012-03) is applied in ACSCTE that occurs after the ACSCCD series;
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
following keywords in RAW image header. The default values are picked for
optimum results in a typical WFC full-frame exposure. Changing these values is
not recommended unless you know what you are doing.

* PCTENSMD

  * Read noise mitigation mode:

    * 0 - No mitigation
    * 1 - Perform noise smoothing
    * 2 - No noise smoothing

  * Overwrites NSEMODEL in PCTETAB.

* PCTERNCL

  * Read noise level of image in ELECTRONS. This is not used if you specified
    no mitigation in read noise mitigation mode.
  * Overwrites RN_CLIP in PCTETAB.

* PCTETRSH

  * Over-subtraction correction threshold. Pixel below this value in ELECTRONS
    after CTE correction is considered over-corrected and will re-corrected with
    smaller correction.
  * Overwrites SUBTHRSH in PCTETAB.

* PCTESMIT

  * Number of iterations of readout simulation per column.
  * Overwrites SIM_NIT in PCTETAB.

* PCTESHFT

  * Number of shifts each readout simulation is broken up into.
  * Overwrites SHFT_NIT in PCTETAB.


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
`stsci.tools.stpyfits`, which is distributed as part of ``stsci_python``,
instead of `pyfits` or `astropy.io.fits` when working with CALACS products.
To use ``stpyfits`` in Python::

    from stsci.tools import stpyfits as pyfits
