"""
Functions to apply pixel-based CTE correction to ACS images.

The algorithm implemented in this code was
described in detail by [Anderson]_ as available
online at:

http://adsabs.harvard.edu/abs/2010PASP..122.1035A

:Authors: P.L. Lim and W. Hack (Python), J. Anderson (Fortran)

:Organization: Space Telescope Science Institute

:History: 2010/09/01 PLL created this module.

References
----------
.. [Anderson] Anderson J. & Bedin, L.R., 2010, PASP, 122, 1035

Notes
------
* This code only works for ACS/WFC but can be modified to work on other detectors.
* It was developed for use with full-frame GAIN=2 FLT images as input.
* It has not been fully tested with any other formats.
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

# Local modules
import ImageOpByAmp
import PixCte_FixY # C extension

__taskname__ = "PixCteCorr"
__version__ = "0.1"
__vdate__ = "13-Oct-2010"

# Global variable
_YCTE_QMAX = 10000

#--------------------------
def XCte():
    """
    *FUTURE WORK*
    
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
    1.  This task can be used to correct an image simply with:

            >>> import PixCteCorr
            >>> PixCteCorr.YCte('j12345678_flt.fits')

        This task will generate a new CTE-corrected image.

    2.  The TEAL GUI can be used to run this task using:

            >>> epar PixCteCorr  # under PyRAF only

        or from a general Python command line:

            >>> from pytools import teal
            >>> teal.teal('PixCteCorr')

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
    if not outFits: outFits = rootname + '_cte.fits'

    # Copy input to output
    shutil.copyfile(inFits, outFits)

    # Open output for correction
    pf_out = pyfits.open(outFits, mode='update')

    # For detector-specific operations
    detector = pf_out['PRIMARY'].header['DETECTOR']

    # For epoch-specific operations
    expstart = pf_out['PRIMARY'].header['EXPSTART']

    # Calculate CTE_FRAC
    cte_frac = _CalcCteFrac(expstart, detector)

    # Read CTE params from file
    pctefile = pf_out['PRIMARY'].header['PCTEFILE']
    dtde_l, chg_leak, psi_node = _PixCteParams(pctefile, expstart)

    # N in charge tail
    chg_leak_kt = _InterpolatePsi(chg_leak, psi_node)

    # dtde_q: Marginal PHI at a given chg level.
    # q_pix_array: Maps Q (cumulative charge) to P (dependent var).
    # pix_q_array: Maps P to Q.
    dtde_q, q_pix_array, pix_q_array = _InterpolatePhi(dtde_l, cte_frac)

    # Extract data for amp quadrants.
    # For each amp, view of image is created with amp on bottom left.
    quadObj = ImageOpByAmp.ImageOpByAmp(pf_out)
    ampList = quadObj.GetAmps()
    gain = quadObj.GetHdrValByAmp('gain')
    rdns = quadObj.GetHdrValByAmp('noise')
    # DQ needs to be read if new flags are to be added.
    sciQuadData = quadObj.DataByAmp()
    errQuadData = quadObj.DataByAmp(extName='ERR')

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
    chg_leak_tq, chg_open_tq = _TrackChargeTrap(pix_q_array, chg_leak_kt, pFile=outLog, psiNode=psi_node)

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
        
        # Keep a copy of original SCI for error calculations
        sciAmpOrig = sciQuadData[amp].copy().astype('float')
        
        # Convert SCI from electrons to DN.
        # This is not needed for RAW.
        sciAmpDN = sciAmpOrig / gain[amp]

        # Separate noise and signal
        sciAmpSig, sciAmpNse = _DecomposeRN(sciAmpDN, rdns[amp]/gain[amp], model=noise)
        
        if intermediateFiles:
            mosX1, mosX2, mosY1, mosY2, tCode = quadObj.MosaicPars(amp)
            mosWo[mosY1:mosY2,mosX1:mosX2] = quadObj.FlipAmp(sciAmpSig, tCode, trueCopy=True)
            mosRn[mosY1:mosY2,mosX1:mosX2] = quadObj.FlipAmp(sciAmpNse, tCode, trueCopy=True)
        # End if

        # Only log pre-selected amp.
        if amp == amp2log:
            outLog2 = outLog
        else:
            outLog2 = ''
        # End if

        # CTE correction
        sciAmpCor = numpy.zeros(sciAmpOrig.shape)
        retCode = PixCte_FixY.FixYCte(sciAmpSig, sciAmpCor, _YCTE_QMAX, q_pix_array, chg_leak_tq, chg_open_tq, amp, outLog2)
        if retCode != 0:
            print 'C-extension call failed for AMP', amp
            continue

        # Add noise back to corrected image.
        # Put corrected data back to image as electrons.
        # Gain is not needed for RAW.
        sciAmpFin = (sciAmpCor + sciAmpNse) * gain[amp]
        sciQuadData[amp][:,:] = sciAmpFin.astype(sciQuadData[amp].dtype)

        # Apply 10% correction to ERR in quadrature
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
def _PixCteParams(fitsTable, mjd):
    """
    Read params from PCTEFILE.

    .. note: Environment variable pointing to
             reference file directory must exist.

    Parameters
    ----------
    fitsTable: string 
        PCTEFILE from header.

    mjd: float 
        EXPSTART from header.

    Returns
    -------
    dtde_l, chg_leak, psi_node
        2 Numpy float arrays, Numpy int array
    """

    # Resolve path to PCTEFILE
    refFile = _ResolveRefFile(fitsTable)

    # Open FITS table
    pf_ref = pyfits.open(refFile)

    # Read PHI array from header
    s = pf_ref['PRIMARY'].header['PHI_*']
    dtde_l = numpy.array( s.values() )

    # Find matching extension with MJD
    chg_leak = numpy.array([])
    psiKey = ('PSIMJD1', 'PSIMJD2')
    for ext in pf_ref:
        if chg_leak.size > 0: break
        if not ext.header.has_key(psiKey[0]): continue

        # Read PSI from table, skip first column
        if mjd >= ext.header[psiKey[0]] and mjd < ext.header[psiKey[1]]:
            s = numpy.array( ext.data.tolist() )
            psi_node = s[:,0]
            chg_leak = s[:,1:]
        # End if
    # End of ext loop

    # Close FITS table
    pf_ref.close()

    return dtde_l, chg_leak, psi_node.astype('int')

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
        Separator between directory
        and file name.

    Returns
    -------
    refPath: string
        Full path to reference file.
    """

    s = refText.split(sep)
    n = len(s)

    if n > 1:
        refPath = os.getenv(s[0])
    else:
        refPath = os.getcwd()
    # End if

    return refPath + os.sep + s[1]

#--------------------------
def _CalcCteFrac(mjd, detector):
    """
    Calculate CTE_FRAC used for linear time
    dependency.
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

    """

    c1 = {'WFC':52335.0}
    c2 = {'WFC':55105.0}
    return (mjd - c1[detector]) / (c2[detector] - c1[detector])

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
    chg_leak: Numpy float array
        PSI table data from PCTEFILE.

    psi_node: Numpy int array
        PSI node data from PCTEFILE.

    Returns
    -------
    chg_leak_kt: Numpy float array
        Interpolated PSI.

    """

    max_node = 100
    chg_leak_kt = numpy.zeros((max_node, chg_leak.shape[1]))
    kRange = range(chg_leak.shape[1])

    for n1 in range(chg_leak.shape[0] - 1):
        n2 = n1 + 1
        psi1 = psi_node[n1]
        psi2 = psi_node[n2]

        rangeInterp = numpy.array( range(psi1, psi2), dtype='float' )
        ftt = (rangeInterp - psi1) / (psi2 - psi1)

        for k in kRange: chg_leak_kt[psi1-1:psi2-1,k] = chg_leak[n1,k] + ftt * (chg_leak[n2,k] - chg_leak[n1,k])
    # End of n1 loop

    chg_leak_kt[-1,:] = chg_leak[-1,:]
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
    dtde_l: Numpy float array
        PHI data from PCTEFILE.

    cte_frac: float
        Time dependency factor.

    Returns
    -------
    dtde_q, q_pix_array, pix_q_array: numpy arrays
    
    """

    global _YCTE_QMAX
    p_max  = 49999 # Jay: No need to change.
    p0_range = numpy.arange(p_max)
    p1_range = p0_range + 1

    # Distance along the nodes.
    # rl = 1 is at node 1.
    # rl = 1.5 is halfway between nodes 1 and 2.
    rl = 1.0 + 2.0 * numpy.log10(p1_range)
    ll = rl.astype('int')
    fl = rl - ll  # Distance between nlow and nhigh
    kl = ll - 1   # Lower node ; nlow = nhigh-1

    # Interpolated PHI at charge P (in DN_2).
    # This is then rescaled by CTE_FRAC to set the
    # level of CTE losses appropriate for this exposure.
    dtde_q = (dtde_l[kl] + fl*(dtde_l[ll]-dtde_l[kl])) * cte_frac

    # Running sum
    ctde_q = numpy.zeros(dtde_q.shape)
    sumt = 0.0
    for p in p0_range:
        sumt += dtde_q[p]
        ctde_q[p] = sumt
    # End of p loop

    q_pix_array = ctde_q.astype('int').clip(1)
    pix_q_array = numpy.zeros(p_max)
    pix_q_array[q_pix_array - 1] = p1_range

    # Total amount of charge under the curve.
    # Max loss of electrons. Max traps to track.
    _YCTE_QMAX = int( dtde_q.sum() )

    return dtde_q, q_pix_array, pix_q_array

#--------------------------
def _TrackChargeTrap(pix_q_array, chg_leak_kt, pFile=None, psiNode=None):
    """
    Calculate the trails (N pix downstream) for each
    block of charge that amounts to a single electron
    worth of traps. Determine what the trails look
    like for each of the traps bring tracked.

    Parameters
    ----------
    pix_q_array: Numpy array 
        Maps P to cumulative charge.

    chg_leak_kt: Numpy array
        Interpolated PSI(Q,N).

    pFile: string, optional 
        Optional log file name.

    psiNode: Numpy int array
        PSI nodes from PCTEFILE. Only used with `pFile`.

    Returns
    -------
    chg_leak_tq, chg_open_tq: 2 Numpy arrays
    
    """

    max_node = chg_leak_kt.shape[0]
    t_range = range(max_node)
    q_range = range(_YCTE_QMAX)
    chg_leak_tq = numpy.zeros((max_node, _YCTE_QMAX))
    chg_open_tq = chg_leak_tq.copy()

    # LOG_Q for PSI(Q,N)
    lp_min, lp_max = 1.0, 4.0 # LOG_Q for PSI(Q,N)
    lp = numpy.log10(pix_q_array[:_YCTE_QMAX]).clip(lp_min, lp_max)

    # Log_Q index
    k = numpy.ones(lp.shape, dtype='int')
    k[ numpy.where(lp < 2) ] = 0
    k[ numpy.where(lp >= 3) ] = 2
    k2 = k + 1

    flp = lp - k2
    for t in t_range: chg_leak_tq[t,:] = chg_leak_kt[t,k] + flp*(chg_leak_kt[t,k2]-chg_leak_kt[t,k])

    for q in q_range:
        # Normalize so sum is 1.
        # When sum is 1, every grabbed electron will be released.
        chg_leak_tq[:,q] /= chg_leak_tq[:,q].sum()

        # Cumulative sum
        sumt = 0.0
        for t in t_range:
            sumt += chg_leak_tq[t,q]
            chg_open_tq[t,q] = sumt
        # End of t loop
    # End of q loop

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
def _DecomposeRN(dataDN, readNoise, model=1, sigCut=2.0):
    """
    Separate noise and signal.
    
        REAL DATA = SIGNAL + NOISE

    .. note: Assume data only has 1 amp readout with
             amp on lower left when displayed with default
             plot settings.

    Parameters
    ----------
    dataDN: Numpy float array
        SCI data in DN.

    readNoise: float
        Read noise in DN.

    model: int, optional
        Noise mitigation algorithm.
        Calculations done in Y only.

            - 0: None.
            - 1: Vertical linear, +/- 1 pixel.

    sigCut: float, optional
        Read noise sigma cuts.

    Returns
    --------
    sigArr, nseArr: 2 Numpy float arrays
        Signal and noise components.

    """

    sigArr = dataDN.copy()
    nseArr = numpy.zeros(dataDN.shape)

    # Assume no noise
    if model == 0: return sigArr, nseArr

    # ----- model=1 from here onwards

    nseSigLo = sigCut * readNoise # Is noise below this
    nseSigHi = sigCut * nseSigLo  # Not noise above this

    # Exclude 1 pix from side near amp, 4 pix from side far from amp
    y1 = 1
    y2 = dataDN.shape[0] - 4

    # Views of regions to use for calculations
    sigCen = dataDN[y1:y2,:]
    sigLow = dataDN[y1-1:y2-1,:]
    sigUpp = dataDN[y1+1:y2+1,:]
    
    # Initial model of signal
    sigArr[y1:y2,:] = 0.333*sigLow + 0.334*sigCen + 0.333*sigUpp

    # Compute model of noise
    # -----
    # 1. If the readnoise image has an amplitude < +/- 5 DN
    #    (2 sig of GAIN=2 ACS/WFC noise), then it is consistent with
    #    being pure readnoise, so just leave as is.
    # 2. If the readnoise image has an amplitude > +/- 10 DN, then
    #    it is not at all consistent with being readnoise; so assume
    #    NO readnoise in this pixel.
    # 3. If the readnoise image has an amplitude between 5 and 10
    #    (or -5 and -10), then taper from 5 to 0 (or -5 to 0).
    # 4. End result is that the "readnoise only" image has values
    #    between -5 and 5, and makes the image smoother.
    # -----
    nseArr = dataDN - sigArr
    nseArrAbs = numpy.abs(nseArr)
    idx = numpy.where(nseArrAbs > nseSigHi)
    nseArr[idx] = 0.0
    idx = numpy.where(nseArrAbs > nseSigLo)
    nseArr[idx] = (nseSigHi - nseArrAbs[idx]) * numpy.sign(nseArr[idx])

    # Final model of signal
    sigArr = dataDN - nseArr

    return sigArr, nseArr

#--------------------------
def _InterpolatePsi_NOT_USED(chg_leak, psi_node):
    """
    Interpolates the PSI(Q,N) curve at all N from
    1 to 100.

    Same as `_InterpolatePsi()` but a slower
    implementation. Indexing starts from 1 like
    Fortran. Kept for testing only.

    Parameters
    ----------
    chg_leak: Numpy float array
        PSI table data from PCTEFILE.

    psi_node: Numpy INt array
        PSI node data from PCTEFILE.

    Returns
    -------
    chg_leak_kt: Numpy float array
        Interpolated PSI.

    """

    max_node = 100
    k_range = range(1, chg_leak.shape[1] + 1) # 1 to 4
    t_range = range(1, max_node + 1)          # 1 to 100
    p_range = range(1, psi_node.size)
    chg_leak_kt = numpy.zeros((max_node+1, chg_leak.shape[1]+1))

    tlist = numpy.zeros(psi_node.size + 1)
    tlist[1:] = psi_node[:]

    chg_leak2 = numpy.zeros((chg_leak.shape[0]+1, chg_leak.shape[1]+1))
    chg_leak2[1:,1:] = chg_leak[:,:]

    for k in k_range:
        for t in t_range:
            ttu = 1
            for tt in p_range:
                if tlist[tt] <= t: ttu = tt
            # End of tt loop
            ftt = (t - tlist[ttu])*1.0 / (tlist[ttu+1]-tlist[ttu])
            chg_leak_kt[t,k] = chg_leak2[ttu,k] + ftt*(chg_leak2[ttu+1,k]-chg_leak2[ttu,k])
        # End of k loop
    # End of t loop

    return chg_leak_kt[1:,1:]

#--------------------------
def _InterpolatePhi_NOT_USED(dtde_l2, cte_frac):
    """
    Interpolates the PHI(Q) at all Q from 1 to
    49999 (log scale).

    Same as `_InterpolatePhi()` but a slower
    implementation. Indexing starts from 1 like
    Fortran. Kept for testing only.

    Parameters
    ----------
    dtde_l2: Numpy float array
        PHI data from PCTEFILE.

    cte_frac: float
        Time dependency factor.

    Returns
    -------
    dtde_q, q_pix_array, pix_q_array: 3 Numpy arrays

    """

    global _YCTE_QMAX
    p_max = 50000 # Jay: No need to change.

    # Jay's calculations started from index 1.
    dtde_l = numpy.zeros(dtde_l2.size + 1)
    dtde_l[1:] = dtde_l2[:]

    # Initialization
    dtde_q = numpy.zeros(p_max)
    ctde_q = dtde_q.copy()
    q_pix_array = dtde_q.copy()
    pix_q_array = dtde_q.copy()
    sumt = 0.0

    for p in range(1,p_max):
        # Distance along the nodes.
        # rl = 1 is at node 1.
        # rl = 1.5 is halfway between nodes 1 and 2.
        rl = 1 + 2*numpy.log10(p)
        ll = int(rl) # Lower node ; nhigh = nlow+1
        fl = rl-ll   # Distance between nlow and nhigh

        # Interpolated PHI at charge P (in DN_2).
        # This is then rescaled by CTE_FRAC to set the
        # level of CTE losses appropriate for this exposure.
        dtde_q[p] = ( dtde_l[ll] + fl*(dtde_l[ll+1]-dtde_l[ll]) ) * cte_frac

        sumt +=  dtde_q[p] # Total sum
        ctde_q[p] = sumt   # Running sum
        q = int(sumt)      # q after last iter is QMAX

        # Cumulative amount of charge in traps as a function of Q
        if q < 1: q = 1
        q_pix_array[p] = q
        pix_q_array[q] = p
    # End of p loop

    # Total amount of charge under the curve.
    # Max loss of electrons. Max traps to track.
    _YCTE_QMAX = q

    return dtde_q[1:], q_pix_array[1:], pix_q_array[1:]

#--------------------------
def _TrackChargeTrap_NOT_USED(pix_q_array, chg_leak_kt, pFile=None, psiNode=None):
    """
    Calculate the trails (N pix downstream) for each
    block of charge that amounts to a single electron
    worth of traps. Determine what the trails look
    like for each of the traps bring tracked.

    Same as `_TrackChargeTrap()` but a slower
    implementation. Indexing starts from 1 like
    Fortran. Kept for testing only.

    Parameters
    ----------
    pix_q_array: Numpy float array
        Maps P to cumulative charge.

    chg_leak_kt: Numpy float array
        Interpolated PSI(Q,N).

    pFile: string, optional
        Optional log file name.

    psiNode: Numpy int array
        PSI nodes from PCTEFILE. Only used with `pFile`.

    Returns
    -------
    chg_leak_tq, chg_open_tq: 2 Numpy arrays

    """

    max_node = chg_leak_kt.shape[0]
    t_range = range(1, max_node+1)
    q_range = range(1,_YCTE_QMAX+1)
    lp_min, lp_max = 1.0, 4.0 # LOG_Q for PSI(Q,N)

    pix_q_array2 = numpy.zeros(pix_q_array.size + 1)
    pix_q_array2[1:] = pix_q_array[:]
    
    chg_leak_kt2 = numpy.zeros((chg_leak_kt.shape[0]+1, chg_leak_kt.shape[1]+1))
    chg_leak_kt2[1:,1:] = chg_leak_kt[:,:]

    chg_leak_tq = numpy.zeros((max_node+1, _YCTE_QMAX+1))
    chg_open_tq = chg_leak_tq.copy()

    # Q=1 corresponds to the charge level of 1 electron
    # worth of charge.
    for q in q_range:
        p = pix_q_array2[q]

        # LOG_Q with min/max clipping
        lp = numpy.log10(p)
        if lp < lp_min:
            lp = lp_min
        elif lp > lp_max:
            lp = lp_max
        # End if

        # Log_Q index
        k = 3
        if lp < 3: k = 2
        if lp < 2: k = 1

        flp = lp - k
        sumt = 0.0
        for t in t_range:
            chg_leak_tq[t,q] = chg_leak_kt2[t,k] + flp*(chg_leak_kt2[t,k+1]-chg_leak_kt2[t,k])
            sumt += chg_leak_tq[t,q]
        # End of t loop

        # Normalize so the sum is 1
        for t in t_range: chg_leak_tq[t,q] /= sumt

        # Calculate the cumulative of the curve
        sumt = 0.0
        for t in t_range:
            sumt += chg_leak_tq[t,q]
            chg_open_tq[t,q] = sumt
        # End of t loop
    # End of q loop

    # Write results to log file
    if pFile:
        i_open = 100
        fLog = open(pFile,'w') # Overwrite

        fLog.write('%-1s%4s %5s ' % ('#', 'Q', 'P'))
        for t in psiNode: fLog.write('NODE_%-3i ' % t)
        fLog.write('OPEN_%-3i%s' % (i_open, os.linesep))

        for q in q_range:
            fLog.write('%5i %5.0f ' % (q, pix_q_array2[q]))
            for t in psiNode: fLog.write('%8.4f ' % chg_leak_tq[t,q])
            fLog.write('%8.4f%s' % (chg_open_tq[i_open,q], os.linesep))
        # End of q loop

        fLog.close()
    # End if

    return chg_leak_tq[1:,1:], chg_open_tq[1:,1:]

#--------------------------
# TEAL Interface functions
#--------------------------
def run(configObj):
    
    YCte(configObj['inFits'],outFits=configObj['outFits'],noise=configObj['noise'],
        intermediateFiles=configObj['debug'],nits=configObj['nits'])
    
def getHelpAsString():
    helpString = ''
    if teal:
        helpString += teal.getHelpFileAsString(__taskname__,__file__)

    if helpString.strip() == '':
        helpString += __doc__+'\n'+YCte.__doc__

    return helpString

#--------------------------
if __name__ == '__main__':
    print 'Command line call not supported'
