.. _calacs-landing:

******
CALACS
******

CALACS Python Wrapper
=====================

.. automodapi:: acstools.calacs
    :no-heading:
    :headings: =-
    :no-inheritance-diagram:

ACSCCD Python Wrapper
=====================

.. automodapi:: acstools.acsccd
    :no-heading:
    :headings: =-
    :no-inheritance-diagram:

ACSCTE Python Wrapper
=====================

.. automodapi:: acstools.acscte
    :no-heading:
    :headings: =-
    :no-inheritance-diagram:

ACSREJ Python Wrapper
=====================

.. automodapi:: acstools.acsrej
    :no-heading:
    :headings: =-
    :no-inheritance-diagram:

ACS2D Python Wrapper
====================

.. automodapi:: acstools.acs2d
    :no-heading:
    :headings: =-
    :no-inheritance-diagram:

ACSSUM Python Wrapper
=====================

.. automodapi:: acstools.acssum
    :no-heading:
    :headings: =-
    :no-inheritance-diagram:

CTE Forward Model
=================

This functionality is provided to simulate the CTE effects associated with
readout of the WFC detectors. It is not involved in standard data calibration with the CALACS pipeline.

.. automodapi:: acstools.acscteforwardmodel
    :no-heading:
    :headings: =-
    :no-inheritance-diagram:

PIXVALUE in FITS File
=====================

CALACS uses HSTIO, which utilizes the ``PIXVALUE`` keyword to represent a data
extension with a constant value. However, this is not a standard FITS behavior
and is not recognized by ``astropy.io.fits``. While you should not encounter errors or warnings, 
constant value data extensions may exhibit unexpected behavior when reading, writing, or 
manipulating them with ``astropy.io.fits``. Therefore, if issues such as these arise, we recommend use of
``stsci.tools.stpyfits``, which is distributed as part of ``stsci_python``,
instead of `astropy.io.fits` when working with CALACS products.
To use ``stpyfits`` in Python::

    from stsci.tools import stpyfits as fits

calacs.e (C Program)
====================

A detailed description of CALACS is available in the
`ACS Data Handbook <https://hst-docs.stsci.edu/acsdhb>`_.
If you have questions not answered in the documentation, please contact the
`STScI Help Desk <https://hsthelp.stsci.edu>`_.

Where to Find CALACS
--------------------

CALACS is part of the
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

    * Turn off parallel processing for PCTECORR. Try this option if you
      encounter problems running CALACS with PCTECORR=PERFORM.

* --nthreads <N>

    * Specify the number of threads used for PCTECORR.
      The default is 'greedy' and will use that specified by the system environment variable OMP_NUM_THREADS.
      If N > ``OMP_NUM_THREADS``, ``OMP_NUM_THREADS`` will be used instead. The option -1 takes precedence.

* --ctegen <1|2>

    * Specify which generation CTE correction to use, the default is 2. Gen 1 (officially deprecated) refers to
      the correction algorithm used in calacs version pre 9.2.0, described in
      `ACS ISR 2010-03 <https://ui.adsabs.harvard.edu/abs/2010PASP..122.1035A/abstract>`_. Gen 2 refers to the 
      new CTE correction algorithm implemented in calacs version 9.2.0 (HSTCAL 1.3.0) described in `ACS ISR 2018-04 <https://ui.adsabs.harvard.edu/abs/2018acs..rept....4A/abstract>`_.

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

CALACS Processing Steps
=======================

Please see
`Chapter 3 of the ACS Data Handbook <https://hst-docs.stsci.edu/acsdhb/chapter-3-acs-calibration-pipeline/3-4-calacs-processing-steps>`_
for a more thorough description of the steps within CALACS.

Bias Image Subtraction (BIASCORR)
---------------------------------

Bias correction is done by subtracting the BIASFILE in units of DN, removing the
low-order quasi-static structure of the bias, including the bias gradient (post-SM4).

Unit Conversion to Electrons
----------------------------

The image is multiplied by the gain, converting it to ELECTRONS.

Bias Level Correction (BLEVCORR)
--------------------------------

BLEVCORR is performed after BIASCORR. Calculations are done in ELECTRONS.

For post-SM4 full-frame WFC exposures, this also includes:

* de-striping to remove stripes introduced by new hardware installed during
  SM-4 (`ACS ISR 2011-05 <https://ui.adsabs.harvard.edu/abs/2011acs..rept....5G/abstract>`_)
* if JWROTYPE=DS_int and CCDGAIN=2, also correct for bias shift
  (`ACS ISR 2012-02 <https://ui.adsabs.harvard.edu/abs/2012acs..rept....2G/abstract>`_) and cross-talk (`ACS ISR 2010-02 <https://ui.adsabs.harvard.edu/abs/2010acs..rept....2S/abstract>`_).

Sink Pixel Flagging (SINKCORR)
------------------------------

SINKCORR flags sink pixels and adjacent affected pixels with the value
1024 in the DQ array of WFC images using the SNKCFILE. It is only performed on images
taken after January 2015.

Pixel-Based CTE Correction (PCTECORR)
-------------------------------------

For all full-frame WFC exposures, pixel-based CTE correction 
(`ACS ISR 2018-04 <https://ui.adsabs.harvard.edu/abs/2018acs..rept....4A/abstract>`_)
is applied during ACSCTE, which runs after BLEVCORR in ACSCCD.

Because the CTE correction is applied before DARKCORR and FLSHCORR, it is
necessary to use a CTE-corrected dark reference file (DRKCFILE) if the PCTECORR step is enabled.

Parameters characterizing the CTE correction are stored in a reference table,
PCTETAB.

.. note::

    CALACS 8.2 and later uses a slightly different PCTETAB format, where
    the COL_SCALE extension does not include overscan columns.

Required Keywords
^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^

You may adjust some CTE correction algorithm parameters by changing the
following keywords in the RAW image header. The default values are picked for
optimal results in a typical WFC full-frame exposure. **Changing these values is
not recommended unless you know what you are doing.**

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

    * Number of iterations used for the CTE forward model.

Dark Current Subtraction (DARKCORR)
-----------------------------------

Dark correction uses DARKFILE if PCTECORR=OMIT, otherwise it uses DRKCFILE (CTE-corrected
dark reference file).

The dark image is scaled by DARKTIME, which is the sum of EXPTIME, FLASHDUR,
and for WFC, a commanding overhead based on the observing mode, which is listed
in the CCDTAB. Any image with non-zero EXPTIME is assumed to not be a BIAS.

Post-Flash Correction (FLSHCORR)
--------------------------------

Post-flash correction is performed after DARKCORR in the ACS2D step.
When FLSHCORR=PERFORM, it scales FLSHFILE by FLASHDUR for correction.

Flat-Field Correction (FLATCORR)
--------------------------------

PFLTFILE is used for flat-field correction, which is a combination of the
pixel-to-pixel flats and low-order flats. This corrects for pixel-to-pixel and
large-scale sensitivity gradients across the detector by dividing the data by
the flat-field image.

Photometry Keywords (PHOTCORR)
------------------------------

The PHOTCORR step is performed using tables of precomputed values (IMPHTTAB).
The correct table for a given image must be specified
in the IMPHTTAB header keyword in order for CALACS to perform the PHOTCORR step.
By default, it should be in the ``jref`` directory and have the suffix
``_imp.fits``. Each DETECTOR uses a different table.

CALACS Output
=============

Using RAW as input:

* flt.fits: Also called FLT.
* flc.fits: Similar to FLT, except with pixel-based CTE correction applied.
* Temporary files: blv_tmp.fits (BLV_TMP), blc_tmp.fits (BLC_TMP)

Using ASN as input with ACSREJ:

* crj.fits: Also called CRJ.
* crc.fits: Similar to CRJ, except with pixel-based CTE correction applied.
* Temporary files: crj_tmp.fits (CRJ_TMP), crc_tmp.fits (CRC_TMP)
