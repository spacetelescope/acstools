"""
Functions to apply pixel-based CTE correction to ACS images.

The algorithm implemented in this code was
described in detail by [Anderson]_ as available
online at:

http://adsabs.harvard.edu/abs/2010PASP..122.1035A

:Authors: Pey Lian Lim and W.J. Hack (Python), J. Anderson (Fortran)

:Organization: Space Telescope Science Institute

:History:
    * 2010/09/01 PLL created this module.
    * 2010/10/13 WH added/modified documentations.
    * 2010/10/15 PLL fixed PCTEFILE lookup logic.
    * 2010/10/26 WH added support for multiple file processing
    * 2010/11/09 PLL modified `YCte`, `_PixCteParams` and `_DecomposeRN` to reflect noise improvement by JA. Also updated documentations.
    * 2011/04/26 MRD Began updates for new CTE algorithm.

References
----------
.. [Anderson] Anderson J. & Bedin, L.R., 2010, PASP, 122, 1035

Notes
------
* This code only works for ACS/WFC but can be modified to work on other detectors.
* It was developed for use with full-frame GAIN=2 FLT images as input.
* It has not been fully tested with any other formats.
* Noise is slightly enhanced in the output (see [Anderson]_).
* This code assumes a linear time dependence for a given set of coefficients.
* This algorithm does not account for traps with very long release timescale 
  but it is not an issue for ACS/WFC.
* This code also does not account for second-exposure effect.
* Multi-threading support was not implemented in this version as it would 
  interfere with eventual pipeline operation.

"""

# External modules
import os, shutil, time, numpy, pyfits

try:
    from pytools import teal
except:
    teal = None

from pytools import parseinput

# Local modules
import ImageOpByAmp
import PixCte_FixY as pcfy # C extension

__taskname__ = "PixCteCorr"
__version__ = "0.3.0"
__vdate__ = "21-Mar-2011"

# general error for things related to his module
class PixCteError(Exception):
    pass

#--------------------------
def CteCorr(input, outFits='', noise=1, nits=0, intermediateFiles=False):
    """
    Run all the CTE corrections on all the input files.
    
    This function simply calls `YCte()` on each input image
    parsed from the `input` parameter, and passes all remaining
    parameter values through unchanged.
    
    Examples
    --------
    1.  This task can be used to correct a set of ACS images simply with:

            >>> import PixCteCorr
            >>> PixCteCorr.CteCorr('j*q_flt.fits')

        This task will generate a new CTE-corrected image for each of the FLT images.

    2.  The TEAL GUI can be used to run this task using:

            >>> epar PixCteCorr  # under PyRAF only

        or from a general Python command line:

            >>> from pytools import teal
            >>> teal.teal('PixCteCorr')

    Parameters
    ----------
    input: string or list of strings
        name of FLT image(s) to be corrected. The name(s) can be specified
        either as:
         
          * a single filename ('j1234567q_flt.fits')
          * a Python list of filenames
          * a partial filename with wildcards ('\*flt.fits')
          * filename of an ASN table ('j12345670_asn.fits')
          * an at-file ('@input')
        
    outFits: string
        *USE DEFAULT IF `input` HAS MULTIPLE FILES.*
        CTE corrected image in the same
        directory as input. If not given, will use
        ROOTNAME_cte.fits instead. Existing file will
        be overwritten.

    noise: int
        Noise mitigation algorithm. As CTE
        loss occurs before noise is added at readout,
        not removing noise prior to CTE correction
        will enhance the noise in output image.  
         
            - 0: None.
            - 1: Vertical linear, +/- 1 pixel.

    intermediateFiles: bool 
        Generate intermediate files in the same directory as input? 
        Useful for debugging. These are:
            
            1. ROOTNAME_cte_rn_tmp.fits - Noise image.
            2. ROOTNAME_cte_wo_tmp.fits - Noiseless
               image.
            3. ROOTNAME_cte_log.txt - Log file.

    nits: int 
        Not used. *Future work.*

    """
    # Parse input to get list of filenames to process
    infiles, output = parseinput.parseinput(input)
    
    # Process each file
    for file in infiles:
        YCte(file, outFits=outFits, noise=noise, nits=nits, intermediateFiles=intermediateFiles)
        
#--------------------------
def XCte():
    """
    *FUTURE WORK.*
    Not Implemented yet.
    
    Apply correction to serial CTE loss. This is to
    be done before parallel CTE loss correction.
    
    Probably easier to call as routine from `YCte()`.
    """

    print 'Not yet available'

#--------------------------
def YCte(inFits, outFits='', noise=1, nits=0, intermediateFiles=False):
    """
    Apply correction for parallel CTE loss.

    Input image that is already de-striped is desired
    but not compulsory. Using image with striping
    will enhance the stripes in output. Calibrations
    that have been applied to FLT should not
    significantly affect the result.

    Notes
    -----
    * EXT 0 header will be updated. ERR arrays will be
      added in quadrature with 10% of the correction.
      DQ not changed.

    * Does not work on RAW but can be modified
      to do so.

    Examples
    --------
    1.  This task can be used to correct a single FLT image with:

            >>> import PixCteCorr
            >>> PixCteCorr.YCte('j12345678_flt.fits')

        This task will generate a new CTE-corrected image.

    Parameters
    ----------
    inFits: string 
        FLT image to be corrected.

    outFits: string 
        CTE corrected image in the same
        directory as input. If not given, will use
        ROOTNAME_cte.fits instead. Existing file will
        be overwritten.

    noise: int
        Noise mitigation algorithm. As CTE
        loss occurs before noise is added at readout,
        not removing noise prior to CTE correction
        will enhance the noise in output image.  
         
            - 0: None.
            - 1: Vertical linear, +/- 1 pixel.

    intermediateFiles: bool 
        Generate intermediate
        files in the same directory as input? Useful
        for debugging. These are:
            
            1. ROOTNAME_cte_rn_tmp.fits - Noise image.
            2. ROOTNAME_cte_wo_tmp.fits - Noiseless
               image.
            3. ROOTNAME_cte_log.txt - Log file.

    nits: int 
        Not used. *Future work.*

    """

    # Start timer
    timeBeg = time.time()

    # For output files naming.
    # Store in same path as input.
    outPath = os.path.dirname( os.path.abspath(inFits) ) + os.sep
    rootname = pyfits.getval(inFits, 'ROOTNAME')
    print os.linesep, 'Performing pixel-based CTE correction on', rootname
    rootname = outPath + rootname
    
    # Construct output filename
    if not outFits: 
        outFits = rootname + '_cte.fits'

    # Copy input to output
    shutil.copyfile(inFits, outFits)

    # Open output for correction
    pf_out = pyfits.open(outFits, mode='update')

    # For detector-specific operations
    detector = pf_out['PRIMARY'].header['DETECTOR']

    # For epoch-specific operations
    expstart = pf_out['PRIMARY'].header['EXPSTART']

    # Calculate CTE_FRAC
    if detector == 'WFC':
      cte_frac = pcfy.CalcCteFrac(expstart, 1)
    else:
      raise PixCteError('Invalid detector: PixCteCorr only supports ACS WFC.')

    # Read CTE params from file
    pctefile = pf_out['PRIMARY'].header['PCTEFILE']
    sim_nit, shft_nit, rn2_nit, q_dtde, dtde_l, psi_node, chg_leak, levels = \
      _PixCteParams(pctefile)

    # N in charge tail
    chg_leak_kt, chg_open_kt = pcfy.InterpolatePsi(chg_leak, psi_node)

    # dtde_q: Marginal PHI at a given chg level.
    # q_pix_array: Maps Q (cumulative charge) to P (dependent var).
    # pix_q_array: Maps P to Q.
    dtde_q = pcfy.InterpolatePhi(dtde_l, q_dtde, shft_nit)

    # Extract data for amp quadrants.
    # For each amp, view of image is created with amp on bottom left.
    quadObj = ImageOpByAmp.ImageOpByAmp(pf_out)
    ampList = quadObj.GetAmps()
    gain = quadObj.GetHdrValByAmp('gain')
    # DQ needs to be read if new flags are to be added.
    sciQuadData = quadObj.DataByAmp()
    errQuadData = quadObj.DataByAmp(extName='ERR')

    # Optional readnoise from header.
    # Only needed when NOISE=100, which is hidden from user.
    if noise != 100:
        rdns = {}
        for amp in ampList: rdns[amp] = 0.0 # Dummy
    else:
        rdns = quadObj.GetHdrValByAmp('noise')
    # End if

    # Intermediate files
    outLog = ''
    if intermediateFiles:
        # Images
        mosWo = quadObj.MosaicTemplate()
        mosRn = mosWo.copy()

        # Log file name
        outLog = rootname + '_cte_log.txt'
    # End if

    # Compute open spaces. Overwrite log file.
    chg_leak_tq, chg_open_tq = _TrackChargeTrap(pix_q_array, chg_leak_kt, 
                                                ycte_qmax, pFile=outLog, psiNode=psi_node)

    # Choose one amp to log detailed results
    ampPriorityOrder = ['C','D','A','B'] # Might be instrument dependent
    amp2log = ''
    for amp in ampPriorityOrder:
        if amp in ampList:
            amp2log = amp
            break
    # End for

    # Process each amp readout
    for amp in ampList:
        print os.linesep, 'AMP', amp, ', GAIN', gain[amp]
        
        # Keep a copy of original SCI for error calculations.
        # Assume unit of electrons.
        sciAmpOrig = sciQuadData[amp].copy().astype('float')
        
        # Separate noise and signal.
        # Must be in unit of electrons.
        sciAmpSig, sciAmpNse = pcfy.DecomposeRN(sciAmpOrig, noise, rn2_nit, rdns[amp])
        
        if intermediateFiles:
            mosX1, mosX2, mosY1, mosY2, tCode = quadObj.MosaicPars(amp)
            mosWo[mosY1:mosY2,mosX1:mosX2] = quadObj.FlipAmp(sciAmpSig, tCode, trueCopy=True)
            mosRn[mosY1:mosY2,mosX1:mosX2] = quadObj.FlipAmp(sciAmpNse, tCode, trueCopy=True)
        # End if

        # Convert noiseless image from electrons to DN.
        sciAmpSig /= gain[amp]

        # Only log pre-selected amp.
        if amp == amp2log:
            outLog2 = outLog
        else:
            outLog2 = ''
        # End if

        # CTE correction in DN.
        sciAmpCor = pcfy.FixYCte(sciAmpSig, ycte_qmax, q_pix_array, 
                              chg_leak_tq, chg_open_tq, amp, outLog2)

        # Convert corrected noiseless data back to electrons.
        # Add noise in electrons back to corrected image.
        sciAmpFin = sciAmpCor * gain[amp] + sciAmpNse
        sciQuadData[amp][:,:] = sciAmpFin.astype(sciQuadData[amp].dtype)

        # Apply 10% correction to ERR in quadrature.
        # Assume unit of electrons.
        dcte = 0.1 * numpy.abs(sciAmpFin - sciAmpOrig)
        errAmpSig = errQuadData[amp].copy().astype('float')
        errAmpFin = numpy.sqrt(errAmpSig**2 + dcte**2)
        errQuadData[amp][:,:] = errAmpFin.astype(errQuadData[amp].dtype)
    # End of amp loop

    # Update header
    pf_out['PRIMARY'].header.update('PCTECORR', 'COMPLETE')
    pf_out['PRIMARY'].header.update('PCTEFRAC', cte_frac)
    pf_out['PRIMARY'].header.add_history('PCTE noise model is %i' % noise)
    pf_out['PRIMARY'].header.add_history('PCTE NITS is %i' % nits)
    pf_out['PRIMARY'].header.add_history('PCTECORR complete ...')

    # Close output file
    pf_out.close()
    print os.linesep, outFits, 'written'

    # Write intermediate files
    if intermediateFiles:
        outWo = rootname + '_cte_wo_tmp.fits'
        hdu = pyfits.PrimaryHDU(mosWo)
        hdu.writeto(outWo, clobber=True) # Overwrite
        
        outRn = rootname + '_cte_rn_tmp.fits'
        hdu = pyfits.PrimaryHDU(mosRn)
        hdu.writeto(outRn, clobber=True) # Overwrite

        print os.linesep, 'Intermediate files:'
        print outWo
        print outRn
        print outLog
    # End if

    # Stop timer
    timeEnd = time.time()
    print os.linesep, 'Run time:', timeEnd - timeBeg, 'secs'

#--------------------------
def _PixCteParams(fitsTable):
    """
    Read params from PCTEFILE.

    .. note: Environment variable pointing to
             reference file directory must exist.

    Parameters
    ----------
    fitsTable: string 
        PCTEFILE from header.

    Returns
    -------
    sim_nit: integer
        Number of readout simulations to do for each column of data
        
    shft_nit: integer
        Number of shifts to break each readout simulation into
        
    rn2_nit: int
        Number of iterations for `noise`=1 in
        `_DecomposeRN`.
        
    dtde_q: array
        Charge levels at which dtde_l is parameterized
    
    dtde_l: array
        PHI(Q).

    psi_node: array
        N values for PSI(Q,N).

    chg_leak: array
        PSI(Q,N).
        
    levels: array
        Charge levels at which to do CTE evaluation

    """

    # Resolve path to PCTEFILE
    refFile = _ResolveRefFile(fitsTable)
    if not os.path.isfile(refFile): 
        raise IOError, 'PCTEFILE not found: %s' % refFile

    # Open FITS table
    pf_ref = pyfits.open(refFile)

    # Read RN2_NIT value from header
    rn2_nit = pf_ref['PRIMARY'].header['RN2_NIT']
    
    # read SIM_NIT value from header
    sim_nit = pf_ref['PRIMARY'].header['SIM_NIT']
    
    # read SHFT_NIT value from header
    shft_nit = pf_ref['PRIMARY'].header['SHFT_NIT']

    # read dtde data from DTDE extension
    dtde_l = pf_ref['DTDE'].data['DTDE']
    q_dtde = pf_ref['DTDE'].data['Q']
    
    # read chg_leak data from CHG_LEAK extension
    psi_node = pf_ref['CHG_LEAK'].data['NODE']
    chg_leak = numpy.array(pf_ref['CHG_LEAK'].data.tolist(), dtype=numpy.float32)[:,1:]
    
    # read levels data from LEVELS extension
    levels = pf_ref['LEVELS'].data['LEVEL']

    # Close FITS table
    pf_ref.close()

    return sim_nit, shft_nit, rn2_nit, q_dtde, dtde_l, psi_node, chg_leak, levels

#--------------------------
def _ResolveRefFile(refText, sep='$'):
    """
    Resolve the full path to reference file.
    This could be replaced with existing STSDAS
    library function, if necessary.

    Assume standard syntax: dir$file.fits

    Parameters
    ----------
    refText: string
        The text to process.

    sep: char 
        Separator between directory and file name.

    Returns
    -------
    f: string
        Full path to reference file.
    """

    s = refText.split(sep)
    n = len(s)
    if n > 1:
        p = os.getenv(s[0])
        if p:
            p += os.sep
        else:
            p = ''
        # End if
        f = p + s[1]
    else:
        f = os.path.abspath(refText)
    # End if
    return f

#--------------------------
def _CalcCteFrac(mjd, detector):
    """
    Calculate CTE_FRAC used for linear time dependency.
    
    .. math::
        CTE_FRAC = (mjd - C1) / (C2 - C1)

    Formula is defined such that `CTE_FRAC` is 0 for
    `mjd=C1` and 1 for `mjd=C2`.

    WFC: `C1` and `C2` are MJD equivalents for ``2002-03-02``
    (ACS installation) and ``2009-10-01`` (Anderson's test
    data), respectively.

    .. note: Only works on ACS/WFC but can be modified
             to work on other detectors.

    Parameters
    ----------
    mjd: float
        EXPSTART from header.

    detector: string
        DETECTOR from header.

    Returns
    -------
    CTE_FRAC: float
        Time scaling factor.
    """
    
    # Calculate CTE_FRAC
    if detector == 'WFC':
      cte_frac = pcfy.CalcCteFrac(expstart, 1)
    else:
      raise StandardError('Invalid detector: PixCteCorr only supports ACS WFC.')
      
    return cte_frac
    
#--------------------------
def _InterpolatePsi(chg_leak, psi_node):
    """
    Interpolates the `PSI(Q,N)` curve at all N from
    1 to 100.

    `PSI(Q,N)` models the CTE tail profile across N
    pixels from the original pixel for a given
    charge, Q. Up to 100 pixels are tracked. For
    post-SM4 ACS/WFC, CTE loss is within 60 pixels.
    Might be worse for WFPC2 since it is older and
    has faster readout time.

    .. note: As this model is refined, future release
             might only have PSI(N) independent of Q.

    Parameters
    ----------
    chg_leak: array_like
        PSI table data from PCTEFILE.

    psi_node: array_like
        PSI node data from PCTEFILE.

    Returns
    -------
    chg_leak_kt: array_like
        Interpolated PSI.

    """
    
    chg_leak_kt = pcfy.InterpolatePsi(chg_leak, psi_node.astype(numpy.int32))
    
    return chg_leak_kt
    
#--------------------------
def _InterpolatePhi(dtde_l, cte_frac):
    """
    Interpolates the `PHI(Q)` at all Q from 1 to
    49999 (log scale).

    `PHI(Q)` models the amount of charge in CTE
    tail, i.e., probability of an electron being
    grabbed by a charge trap.
    
    Parameters
    ----------
    dtde_l: array_like
        PHI data from PCTEFILE.

    cte_frac: float
        Time dependency factor.

    Returns
    -------
    dtde_q: array_like

    q_pix_array: array_like

    pix_q_array: array_like
    
    ycte_qmax: integer
    
    """
    
    dtde_q, q_pix_array, pix_q_array, ycte_qmax = pcfy.InterpolatePhi(dtde_l, cte_frac)
    
    return dtde_q, q_pix_array, pix_q_array, ycte_qmax

#--------------------------
def _TrackChargeTrap(pix_q_array, chg_leak_kt, ycte_qmax, pFile=None, psiNode=None):
    """
    Calculate the trails (N pix downstream) for each
    block of charge that amounts to a single electron
    worth of traps. Determine what the trails look
    like for each of the traps bring tracked.

    Parameters
    ----------
    pix_q_array: array_like
        Maps P to cumulative charge.

    chg_leak_kt: array_like
        Interpolated PSI(Q,N).
        
    ycte_qmax: integer

    pFile: string, optional 
        Optional log file name.

    psiNode: array_like
        PSI nodes from PCTEFILE. Only used with `pFile`.

    Returns
    -------
    chg_leak_tq: array_like

    chg_open_tq: array_like
    
    """

    chg_leak_tq, chg_open_tq = pcfy.TrackChargeTrap(pix_q_array, chg_leak_kt, ycte_qmax)

    # Write results to log file
    if pFile:
        i_open = 100
        i2 = i_open - 1
        psinode2 = psiNode - 1
        fLog = open(pFile,'w') # Overwrite

        fLog.write('%-1s%4s %5s ' % ('#', 'Q', 'P'))
        for t in psiNode: fLog.write('NODE_%-3i ' % t)
        fLog.write('OPEN_%-3i%s' % (i_open, os.linesep))

        for q in q_range:
            fLog.write('%5i %5.0f ' % (q+1, pix_q_array[q]))
            for t in psinode2: fLog.write('%8.4f ' % chg_leak_tq[t,q])
            fLog.write('%8.4f%s' % (chg_open_tq[i2,q], os.linesep))
        # End of q loop

        fLog.close()
    # End if

    return chg_leak_tq, chg_open_tq
    
#--------------------------
def _DecomposeRN(data_e, model=1, nitrn=7, readNoise=5.0):
    """
    Separate noise and signal.
    
        REAL DATA = SIGNAL + NOISE

    .. note: Assume data only has 1 amp readout with
             amp on lower left when displayed with default
             plot settings.

    Parameters
    ----------
    data_e: array_like
        SCI data in electrons.

    model: int, optional
        Noise mitigation algorithm.
        Calculations done in Y only.

            - 0: None.
            - 1: Vertical linear, +/- 1 pixel.
            - 100: Simpler version of `model`=1.
              Not used anymore. Kept for testing.

    nitrn: int, optional
        Only used if `model`=1. Number of iterations
        for noise mitigation, each one removing one
        extra electron.

    readNoise: float, optional
        Only used if `model`=100. Read noise in
        electrons.

    Returns
    -------
    sigArr: array_like
        Noiseless signal component in electrons.

    nseArr: array_like
        Noise component in electrons.

    """
    
    sigArr, nseArr = pcfy.DecomposeRN(data_e, model, nitrn, readNoise)

    return sigArr, nseArr

#--------------------------
# TEAL Interface functions
#--------------------------
def run(configObj):
    
    CteCorr(configObj['inFits'],outFits=configObj['outFits'],noise=configObj['noise'],
        intermediateFiles=configObj['debug'],nits=configObj['nits'])
    
def getHelpAsString():
    helpString = ''
    if teal:
        helpString += teal.getHelpFileAsString(__taskname__,__file__)

    if helpString.strip() == '':
        helpString += __doc__+'\n'+YCte.__doc__

    return helpString
