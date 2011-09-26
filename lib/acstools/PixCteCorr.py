"""
Functions to apply pixel-based CTE correction to ACS images.

The algorithm implemented in this code was
described in detail by [Anderson]_ as available
online at:

http://adsabs.harvard.edu/abs/2010PASP..122.1035A

:Authors: Pey Lian Lim and W.J. Hack (Python), J. Anderson (Fortran), Matt Davis

:Organization: Space Telescope Science Institute

:History:
    * 2010/09/01 PLL created this module.
    * 2010/10/13 WH added/modified documentations.
    * 2010/10/15 PLL fixed PCTEFILE lookup logic.
    * 2010/10/26 WH added support for multiple file processing
    * 2010/11/09 PLL modified `YCte`, `_PixCteParams` and `_DecomposeRN` to reflect noise improvement by JA. Also updated documentations.
    * 2011/04/26 MRD Began updates for new CTE algorithm.
    * 2011/07/20 MRD Updated to handle new PCTETAB containing time dependent
      CTE characterization.
    * 2011/09/22 MRD Trimmed down module to contain only CTE removal, and prepare
      it for best parallelization.

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
import os, shutil, time
import numpy
import pyfits

# Local modules
import PixCte_FixY as pcfy # C extension

__taskname__ = "PixCteCorr"
__version__ = "0.5.0"
__vdate__ = "22-Sep-2011"

# general error for things related to his module
class PixCteError(Exception):
    pass

#--------------------------
def YCte(inFits, outFits=''):
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

    Parameters
    ----------
    inFits : str
        FLT image to be corrected.

    outFits : str, optional
        CTE corrected image in the same
        directory as input. If not given, will use
        ROOTNAME_cte.fits instead. Existing file will
        be overwritten.
            
    Examples
    --------
    1.  This task can be used to correct a single FLT image with:

            >>> import PixCteCorr
            >>> PixCteCorr.YCte('j12345678_flt.fits')

        This task will generate a new CTE-corrected image.

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

    # This is just for WFC for now.
    if detector != 'WFC':
        raise PixCteError('Invalid detector: PixCteCorr only supports ACS WFC.')

    # Read CTE params from file
    pctefile = pf_out['PRIMARY'].header['PCTETAB']
    cte_frac, sim_nit, shft_nit, rn_clip, q_dtde, dtde_l, psi_node, chg_leak, levels = \
      _PixCteParams(pctefile, expstart)

    # N in charge tail
    chg_leak_kt, chg_open_kt = pcfy.InterpolatePsi(chg_leak, psi_node)
    del chg_leak, psi_node

    # dtde_q: Marginal PHI at a given chg level.
    # q_pix_array: Maps Q (cumulative charge) to P (dependent var).
    # pix_q_array: Maps P to Q.
    dtde_q = pcfy.InterpolatePhi(dtde_l, q_dtde, shft_nit)
    del dtde_l, q_dtde
 
    # finish interpolation along the Q dimension and reduce arrays to contain
    # only info at the levels specified in the levels array
    chg_leak_lt, chg_open_lt, dpde_l, tail_len = \
      pcfy.FillLevelArrays(chg_leak_kt, chg_open_kt, dtde_q, levels)
    del chg_leak_kt, chg_open_kt, dtde_q
    
    # make some slices to map each chip to the large array
    chip2_slice = slice(0,4096)
    chip1_slice = slice(4096,4096*2)
    
    # combine oringal data to one big array with the two chips side by side
    # with the readout at the bottom of the array
    original = numpy.empty((pf_out[1].data.shape[0],2*pf_out[1].data.shape[1]),
                            dtype=numpy.float)                      
    original[:,chip2_slice] = pf_out[1].data.copy().astype(numpy.float)
    original[:,chip1_slice] = pf_out[4].data.copy().astype(numpy.float)[::-1,:]
    
    # make some slice objects to map each amp to the large array
    ampSlices = {}
    ampSlices['ampC'] = slice(2048*0,2048*1)
    ampSlices['ampD'] = slice(2048*1,2048*2)
    ampSlices['ampA'] = slice(2048*2,2048*3)
    ampSlices['ampB'] = slice(2048*3,2048*4)
    
    # large arrays to hold the separated signal and noise
#    signal = original.copy()
#    noise = numpy.zeros_like(original)
    signal = numpy.empty_like(original)
    noise = numpy.empty_like(original)
    
    # separate signal and noise
    print 'Smoothing read noise.'
    for amp in ampSlices:
        # these amps have the amp in the right corner so flip them so it's on
        # the left for this step. not sure why this makes a difference.
        if amp in ('ampD','ampB'):
            sig,nse = pcfy.DecomposeRN(original[:,ampSlices[amp]][:,::-1], rn_clip)
        
            signal[:,ampSlices[amp]] = sig[:,::-1]
            noise[:,ampSlices[amp]] = nse[:,::-1]
        else:
            sig,nse = pcfy.DecomposeRN(original[:,ampSlices[amp]], rn_clip)
        
            signal[:,ampSlices[amp]] = sig[:,:]
            noise[:,ampSlices[amp]] = nse[:,:]
    
    # CTE correction
    print 'Applying CTE correction.'
    corrected = pcfy.FixYCte(signal, cte_frac, sim_nit, shft_nit,
                              levels, dpde_l, tail_len,
                              chg_leak_lt, chg_open_lt)
                              
    # recombine noise and signal
    final = corrected + noise
    del corrected, noise
    
    # apply 10% of correction to error in quadrature
    error = numpy.empty_like(original)
    error[:,chip2_slice] = pf_out[2].data.copy().astype(numpy.float)
    error[:,chip1_slice] = pf_out[5].data.copy().astype(numpy.float)[::-1,:]
    
    error_correction = 0.1 * numpy.abs(final - original)
    error_final = numpy.sqrt(error**2 + error_correction**2)
    del error,error_correction,original
    
    # copy everything back to fits file
    pf_out[1].data[:,:] = final.astype(pf_out[1].data.dtype)[:,chip2_slice]
    pf_out[4].data[:,:] = final.astype(pf_out[4].data.dtype)[:,chip1_slice][::-1,:]
    pf_out[2].data[:,:] = error_final.astype(pf_out[2].data.dtype)[:,chip2_slice]
    pf_out[5].data[:,:] = error_final.astype(pf_out[5].data.dtype)[:,chip1_slice][::-1,:]
    del final, error_final

    # Update header
    pf_out['PRIMARY'].header.update('PCTECORR', 'COMPLETE')
    pf_out['PRIMARY'].header.update('PCTEFRAC', cte_frac)
    pf_out['PRIMARY'].header.update('PCTERNCL', rn_clip)
    pf_out['PRIMARY'].header.update('PCTESMIT', sim_nit)
    pf_out['PRIMARY'].header.update('PCTESHFT', shft_nit)
    pf_out['PRIMARY'].header.add_history('PCTECORR complete ...')

    # Close output file
    pf_out.close()
    print os.linesep, outFits, 'written'

    # Stop timer
    timeEnd = time.time()
    print os.linesep, 'Run time:', timeEnd - timeBeg, 'secs'

#--------------------------
def _PixCteParams(fitsTable, expstart):
    """
    Read params from PCTEFILE.

    .. note: Environment variable pointing to
             reference file directory must exist.

    Parameters
    ----------
    fitsTable : str
        PCTEFILE from header.
        
    expstart : float
        MJD of exposure start time, EXPSTART in image header

    Returns
    -------
    sim_nit : int
        Number of readout simulations to do for each column of data
        
    shft_nit : int
        Number of shifts to break each readout simulation into
        
    rn_clip : float
        Maximum amplitude of read noise removed by DecomposeRN.
        
    dtde_q : ndarray
        Charge levels at which dtde_l is parameterized
    
    dtde_l : ndarray
        PHI(Q).

    psi_node : ndarray
        N values for PSI(Q,N).

    chg_leak : ndarray
        PSI(Q,N).
        
    levels : ndarray
        Charge levels at which to do CTE evaluation

    """

    # Resolve path to PCTEFILE
    refFile = _ResolveRefFile(fitsTable)
    if not os.path.isfile(refFile): 
        raise IOError, 'PCTEFILE not found: %s' % refFile

    # Open FITS table
    pf_ref = pyfits.open(refFile)

    # Read RN_CLIP value from header
    rn_clip = pf_ref['PRIMARY'].header['RN_CLIP']
    
    # read SIM_NIT value from header
    sim_nit = pf_ref['PRIMARY'].header['SIM_NIT']
    
    # read SHFT_NIT value from header
    shft_nit = pf_ref['PRIMARY'].header['SHFT_NIT']
    
    # read number of CHG_LEAK# extensions from the header
    nchg_leak = pf_ref['PRIMARY'].header['NCHGLEAK']

    # read dtde data from DTDE extension
    dtde_l = pf_ref['DTDE'].data['DTDE']
    q_dtde = pf_ref['DTDE'].data['Q']
    
    # read levels data from LEVELS extension
    levels = pf_ref['LEVELS'].data['LEVEL']
    
    # read scale data from CTE_SCALE extension
    scalemjd = pf_ref['CTE_SCALE'].data['MJD']
    scaleval = pf_ref['CTE_SCALE'].data['SCALE']
    
    cte_frac = _CalcCteFrac(expstart, scalemjd, scaleval)
    
    # there are nchg_leak CHG_LEAK# extensions. we need to find out which one
    # is the right one for our data.
    chg_leak_names = ['CHG_LEAK{}'.format(i) for i in range(1,nchg_leak+1)]
    
    for n in chg_leak_names:
        mjd1 = pf_ref[n].header['MJD1']
        mjd2 = pf_ref[n].header['MJD2']
        
        if (expstart >= mjd1) and (expstart < mjd2):
            # read chg_leak data from CHG_LEAK extension
            psi_node = pf_ref[n].data['NODE']
            chg_leak = numpy.array(pf_ref[n].data.tolist(), dtype=numpy.float32)[:,1:]
            break

    # Close FITS table
    pf_ref.close()

    return cte_frac, sim_nit, shft_nit, rn_clip, q_dtde, dtde_l, psi_node, chg_leak, levels

#--------------------------
def _ResolveRefFile(refText, sep='$'):
    """
    Resolve the full path to reference file.
    This could be replaced with existing STSDAS
    library function, if necessary.

    Assume standard syntax: dir$file.fits

    Parameters
    ----------
    refText : str
        The text to process.

    sep : char 
        Separator between directory and file name.

    Returns
    -------
    f : str
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
def _CalcCteFrac(expstart, scalemjd, scaleval):
    """
    Calculate CTE_FRAC used to scale CTE according to time dependence.

    Parameters
    ----------
    expstart : float
        EXPSTART from header.

    scalemjd : ndarray
        MJD points for corresponding CTE scale values in scaleval
        
    scaleval : ndarray
        CTE scale values corresponding to MJDs in scalemjd

    Returns
    -------
    cte_frac : float
        Time scaling factor.
    """
    
    # Calculate CTE_FRAC
    cte_frac = pcfy.CalcCteFrac(expstart, scalemjd, scaleval)
          
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
    chg_leak : ndarray
        PSI table data from PCTEFILE.

    psi_node : ndarray
        PSI node data from PCTEFILE.

    Returns
    -------
    chg_leak : ndarray
        Interpolated PSI.
        
    chg_open : ndarray
        Interpolated tail profile data.

    """
    
    chg_leak, chg_open = pcfy.InterpolatePsi(chg_leak, psi_node.astype(numpy.int32))
    
    return chg_leak, chg_open
    
#--------------------------
def _InterpolatePhi(dtde_l, q_dtde, shft_nit):
    """
    Interpolates the `PHI(Q)` at all Q from 1 to
    99999 (log scale).

    `PHI(Q)` models the amount of charge in CTE
    tail, i.e., probability of an electron being
    grabbed by a charge trap.
    
    Parameters
    ----------
    dtde_l : ndarray
        PHI data from PCTEFILE.

    q_dtde : ndarray
        Q levels at which dtde_l is defined, read from PCTEFILE
        
    shft_int : int
        Number of shifts performed reading out CCD

    Returns
    -------
    dtde_q : ndarray
        dtde_l interpolated at all PHI levels
    
    """
    
    dtde_q = pcfy.InterpolatePhi(dtde_l, q_dtde, shft_nit)
    
    return dtde_q
    
def _FillLevelArrays(chg_leak, chg_open, dtde_q, levels):
    """
    Interpolates CTE parameters to the charge levels specified in levels.
    
    Parameters
    ----------
    chg_leak : ndarray
        Interpolated chg_leak tail profile data returned by _InterpolatePsi.
        
    chg_open : ndarray
        Interpolated chg_open tail profile data returned by _InterpolatePsi.
        
    dtde_q : ndarray
        PHI data interpolated at all PHI levels as returned by
        _InterpolatePhi.
        
    levels : ndarray
        Charge levels at which output arrays will be interpolated.
        Read from PCTEFILE.
        
    Returns
    -------
    chg_leak_lt : ndarray
        chg_leak tail profile data interpolated at the specified charge levels.
        
    chg_open_lt : ndarray
        chg_open tail profile data interpolated at the specified charge levels.
        
    dpde_l : ndarray
        dtde_q interpolated and summed for the specified charge levels.
        
    tail_len : ndarray
        Array of maximum tail lengths for the specified charge levels.
    
    """
    
    chg_leak_lt, chg_open_lt, dpde_l, tail_len = \
      pcfy.FillLevelArrays(chg_leak, chg_open, dtde_q, levels)
      
    return chg_leak_lt, chg_open_lt, dpde_l, tail_len
    
#--------------------------
def _DecomposeRN(data_e, rn_clip=10.0):
    """
    Separate noise and signal.
    
        REAL DATA = SIGNAL + NOISE

    .. note: Assume data only has 1 amp readout with
             amp on lower left when displayed with default
             plot settings.

    Parameters
    ----------
    data_e : ndarray
        SCI data in electrons.

    rn_clip : float
        Maximum amplitude of read noise removed.
        Defaults to 10.0.

    Returns
    -------
    sigArr : ndarray
        Noiseless signal component in electrons.

    nseArr : ndarray
        Noise component in electrons.

    """
    
    sigArr, nseArr = pcfy.DecomposeRN(data_e, rn_clip)

    return sigArr, nseArr
    
def _FixYCte(detector, cte_data, cte_frac, sim_nit, shft_nit, levels, dpde_l, 
              tail_len, chg_leak_lt, chg_open_lt, amp='', outLog2=''):
    """
    Perform CTE correction on input data. It is best to perform some kind
    of readnoise smoothing on the data, otherwise the CTE algorithm will
    amplify the read noise. (In the read out process readnoise is added to
    the data after CTE blurring.)
    
    Parameters
    ----------
    detector : str
        DETECTOR from header.
        Currently only 'WFC' is supported.
        
    cte_data : ndarray
        Data in need of CTE correction. For proper results cte_data[0,x] should
        be next to the readout register and cte_data[-1,x] should be furthest.
        Data are processed a column at a time, e.g. cte_data[:,x] is corrected,
        then cte_data[:,x+1] and so on.
        
    cte_frac : float
        Time dependent CTE scaling parameter.
        
    sim_nit : int
        Number of readout simulation iterations to perform.
        
    shft_nit : int
        Number of readout shifts to do.
        
    levels : ndarray
        Levels at which CTE is evaluated as read from PCTEFILE.
        
    dpde_l : ndarray
        Parameterized amount of charge in CTE trails as a function of
        specific charge levels, as returned by _FillLevelArrays.
        
    tail_len : ndarray
        Maximum tail lengths for CTE tails at the specific charge levels
        specified by levels, as returned by _FillLevelArrays.
        
    chg_leak_lt : ndarray
        Tail profile data at charge levels specified by levels, as returned
        by _FillLevelArrays.
        
    chg_open_lt : ndarray
        Tail profile data at charge levels specified by levels, as returned
        by _FillLevelArrays.
        
    amp : char
        Amp name for this data, used in log file.
        Optional, but must be specified if outLog2 is specified.
        
    outLog2 : str
        Name of optional log file.
        
    Returns
    -------
    corrected : ndarray
        Data CTE correction algorithm applied. Same size and shape as input
        cte_data.
    
    """
    
    if outLog2 != '' and amp == '':
        raise PixCteError('amp argument must be specified if log file is specified.')

    if detector == 'WFC':
        corrected = pcfy.FixYCte(cte_data, cte_frac, sim_nit, shft_nit,
                                levels, dpde_l, tail_len,
                                chg_leak_lt, chg_open_lt, amp, outLog2)
    else:
        raise PixCteError('Invalid detector: PixCteCorr only supports ACS WFC.')
                              
    return corrected
