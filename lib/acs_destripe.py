#!/usr/bin/env python
""" acs_destripe - try to clean out horizontal stripes and crosstalk
                   from ACS WFC post-SM4 data.

"""
__author__ = "Norman Grogin, STScI, June 2010."
__usage__ = """

1. To run this task from within Python::

    >>> from acstools import acs_destripe
    >>> acs_destripe.clean('uncorrected_flt.fits','csck', clobber=False, maxiter=15, sigrej=2.0)

.. note:: make sure the `acstools` package is on your Python path

2.  To run this task using the TEAL GUI to set the parameters under PyRAF::

    >>> import acstools
    >>> epar acs_destripe  # or `teal acs_destripe`
        
3. To run this task from the operating system command line::

    % ./acs_destripe [-h][-c] uncorrected_flt.fits uncorrected_flt_csck.fits [15 [2.0]]

.. note:: make sure the file `acs_destripe.py` is on your executable path

"""

__taskname__ = 'acs_destripe'
__version__ = '0.2.2'
__vdate__ = '10-Feb-2011'

import os, pyfits, sys
import numpy as np
from numpy import sqrt, empty, unique, ones, zeros, byte, median, average, sum, concatenate

from pytools import parseinput,teal
        
def clean(input,suffix,clobber=False,maxiter=15,sigrej=2.0):

    flist,alist = parseinput.parseinput(input)
        
    for image in flist:
        # Implement check to skip processing sub-array images
        if (pyfits.getval(image,'subarray') == 'T') or \
            ((pyfits.getval(image,'ltv1',ext=1) != 0.0) or (pyfits.getval(image,'ltv2',ext=1) != 0.0)):
            sys.stdout.write('Error: Not processing %s...subarray images not supported!'%image)
            continue
        # generate output filename for each input based on specification
        # of the output suffix
        output = image.replace('_flt','_flt_'+suffix)
        if os.path.lexists(output):
            if (clobber):
                sys.stdout.write('Warning: Deleting previous output: %s \n'%output)
                os.remove(output)
            else:
                sys.stderr.write('Error: Skipping processing of ',image,'...Output file \''+output+'\' already exists and needs to be removed.\n') 
                continue
        sys.stdout.write('Processing %s ...'%image)
        sys.stdout.flush()
        perform_correction(image,output,maxiter=maxiter,sigrej=sigrej)
        sys.stdout.write("Created corrected image: %s\n"%output)
        
def perform_correction(image,output,maxiter=15,sigrej=2.0):
    # Open the input file and find out about it
    hdulist = pyfits.open(image)

    # Get the science data
    science1 = hdulist['sci',1].data
    science2 = hdulist['sci',2].data

    # Get the err data
    err1 = hdulist['err',1].data
    err2 = hdulist['err',2].data


    # Assuming here that both CCDs have same units
    units = hdulist['sci',1].header['BUNIT']
    
    ### read the flatfield filename from the header
    flatfile = hdulist[0].header['PFLTFILE']

    ### if BIAS or DARK, set flatfield to unity
    if flatfile == 'N/A':
         invflat1 = invflat2 = np.ones(science1.shape)
    else:     
        ### resolve the filename into an absolute pathname (if necessary)
        flatparts = flatfile.partition('$')
        if flatparts[1] == '$':
            
            flatdir = os.getenv(flatparts[0])
            if flatdir is None:
                errstr = 'Environment variable ',flatparts[0],' not defined!'
                raise ValueError,errstr
            
            flatfile = flatdir+flatparts[2]
            if not os.path.exists(flatfile):
                errstr = 'Flat-field file ',flatfile,' could not be found!'
                raise IOError, errstr

        ### open the flatfield
        hduflat = pyfits.open(flatfile)

        invflat1 = 1/hduflat['sci',1].data
        invflat2 = 1/hduflat['sci',2].data

    ### apply GAIN and flatfield corrections if necessary
    if units == 'COUNTS':
        ### *** NOT YET FLATFIELDED! ***

        ### read the gain settings from the header
        gainA = hdulist[0].header['ATODGNA']
        gainB = hdulist[0].header['ATODGNB']
        gainC = hdulist[0].header['ATODGNC']
        gainD = hdulist[0].header['ATODGND']

        ### apply the gains
        science1[:,:2048] *= gainC
        science1[:,2048:] *= gainD
        science2[:,:2048] *= gainA
        science2[:,2048:] *= gainB
        err1[:,:2048] *= gainC
        err1[:,2048:] *= gainD
        err2[:,:2048] *= gainA
        err2[:,2048:] *= gainB
        
        # Kill the crosstalk (empirical amplitudes) in the SCI and ERR extensions

        # the following two lines are purposefully odd-looking because a 
        # direct assignment will break the linkage to the hdulist structure
        err1 += np.sqrt(err1**2 + science1[:,::-1] * 7.0e-5) - err1
        err2 += np.sqrt(err2**2 + science2[:,::-1] * 7.0e-5) - err2

        science1 += science1[:,::-1] * 7.0e-5
        science2 += science2[:,::-1] * 7.0e-5

        ### apply the flatfield
        science1 *= invflat1
        science2 *= invflat2
        err1 *= invflat1
        err2 *= invflat2
    else:
        ### already flatfielded (and gain-corrected): remove crosstalk pre-flatfield

        ### un-apply the flatfield
        science1 /= invflat1
        science2 /= invflat2
        err1 /= invflat1
        err2 /= invflat2
   
        # Kill the crosstalk (empirical amplitudes) in the SCI and ERR extensions
        science1 += science1[:,::-1] * 1.0e-4
        science2 += science2[:,::-1] * 7.0e-5
        # the following two lines are purposefully odd-looking because a 
        # direct assignment will break the linkage to the hdulist structure
        err1 += np.sqrt(err1**2 + science1[:,::-1] * 1.0e-4) - err1
        err2 += np.sqrt(err2**2 + science2[:,::-1] * 7.0e-5) - err2

        ### re-apply the flatfield
        science1 *= invflat1
        science2 *= invflat2
        err1 *= invflat1
        err2 *= invflat2

    # Do the stripe cleaning
    clean_streak(science1,invflat1,err1,science2,invflat2,err2,maxiter=maxiter,sigrej=sigrej)

    # Undo the GAIN and flatfield corrections if applied above
    if units == 'COUNTS':
        ### un-apply the gains
        science1[:,:2048] /= gainC
        science1[:,2048:] /= gainD
        science2[:,:2048] /= gainA
        science2[:,2048:] /= gainB
        err1[:,:2048] /= gainC
        err1[:,2048:] /= gainD
        err2[:,:2048] /= gainA
        err2[:,2048:] /= gainB
        ### un-apply the flatfield
        science1 /= invflat1
        science2 /= invflat2
        err1 /= invflat1
        err2 /= invflat2

    # Write the output
    hdulist.writeto(output)
   
def clean_streak(image1,invflat1,err1,image2,invflat2,err2,maxiter=15,sigrej=2.0):

    ### create the array to hold the stripe amplitudes
    corr = np.empty(2048)
   
    ### loop over rows to fit the stripe amplitude
    for i in range(2048):
        ### row-by-row iterative sigma-clipped mean; sigma, iters are adjustable
        SMean, SSig, SMedian, SMask = djs_iterstat(np.concatenate((image1[i],image2[2047-i])),SigRej=sigrej,MaxIter=maxiter)
        
        ### SExtractor-esque central value statistic; slightly sturdier against
        ### skewness of pixel histogram due to faint source flux
        corr[i]=2.5*SMedian-1.5*SMean
    
    ### preserve the original mean level of the image
    corr -= np.average(corr)

    ### apply the correction row-by-row
    for i in range(2048):
        ### stripe is constant along the row, before flatfielding; 
        ### afterwards it has the shape of the inverse flatfield
        truecorr1 = corr[i] * invflat1[i] / np.average(invflat1[i])
        truecorr2 = corr[2047-i] * invflat2[i] / np.average(invflat2[i])

        ### correct the SCI extension
        image1[i] -= truecorr1
        image2[i] -= truecorr2

        ### correct the ERR extension
        err1[i] = np.sqrt(err1[i]**2 - truecorr1)
        err2[i] = np.sqrt(err2[i]**2 - truecorr2)

def djs_iterstat(InputArr, SigRej=3.0, MaxIter=10, Mask=0,\
                 Max='', Min='', RejVal=''):
### routine for iterative sigma-clipping
  NGood    = InputArr.size
  ArrShape = InputArr.shape
  if NGood == 0: 
    print 'No data points given'
    return 0, 0, 0, 0
  if NGood == 1:
    print 'Only one data point; cannot compute stats'
    return 0, 0, 0, 0

  #Determine Max and Min
  if Max == '':
    Max = InputArr.max()
  if Min == '':
    Min = InputArr.min()
 
  if np.unique(InputArr).size == 1:
    return 0, 0, 0, 0
 

  Mask  = np.zeros(ArrShape, dtype=np.byte)+1
  #Reject those above Max and those below Min
  Mask[InputArr > Max] = 0
  Mask[InputArr < Min] = 0
  if RejVal != '' :  Mask[InputArr == RejVal]=0
  FMean = np.sum(1.*InputArr*Mask) / NGood
  FSig  = np.sqrt(np.sum((1.*InputArr-FMean)**2*Mask) / (NGood-1))

  NLast = -1
  Iter  =  0
  NGood = np.sum(Mask)
  if NGood < 2:
    return -1, -1, -1, -1

  while (Iter < MaxIter) and (NLast != NGood) and (NGood >= 2) :

    LoVal = FMean - SigRej*FSig
    HiVal = FMean + SigRej*FSig
    NLast = NGood

    Mask[InputArr < LoVal] = 0
    Mask[InputArr > HiVal] = 0
    NGood = np.sum(Mask)

    if NGood >= 2:
      FMean = np.sum(1.*InputArr*Mask) / NGood
      FSig  = np.sqrt(np.sum((1.*InputArr-FMean)**2*Mask) / (NGood-1))

    SaveMask = Mask.copy()

    Iter = Iter+1
  if np.sum(SaveMask) > 2:
    FMedian = np.median(InputArr[SaveMask == 1])
  else:
    FMedian = FMean

  return FMean, FSig, FMedian, SaveMask 

#
#### Interfaces used by TEAL
#
def run(configobj=None):
    ''' TEAL interface for running this code. '''
### version Tue Jun 01 2010
### removes row striping (and row-averaged src flux if not amenable to sigma-clipping)
###
### !!!expects an flt-format file, not a raw-format file!!!

    clean(configobj['input'],configobj['output'],
        clobber = configobj['clobber'],
        maxiter= configobj['maxiter'], sigrej = configobj['sigrej'])

def getHelpAsString(fulldoc=True):

    if fulldoc:
        basedoc = __doc__
    else:
        basedoc = ""
    helpString = basedoc+'\n'
    helpString += 'Version '+__version__+'\n'

    """ 
    return useful help from a file in the script directory called module.help
    """
    helpString += teal.getHelpFileAsString(__taskname__,__file__)

    helpString += __usage__

    return helpString

# Set up doc string without the module level docstring included for
# use with Sphinx, since Sphinx will already include module level docstring
clean.__doc__ = getHelpAsString(fulldoc=False)

def help():
    print getHelpAsString()
    
def main():
    import getopt

    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'hc')
    except getopt.error, e:
        print str(e)
        print __doc__
        print "\t", __version__

    # initialize default values
    help = 0
    clobber = False
    maxiter = 15
    sigrej = 2.0
    
    # read options
    for opt, value in optlist:
        if opt == "-h":
            help = 1
        if opt == "-c":
            clobber = True

    if len(args) < 2:
        sys.stderr.write('Usage: acs_destripe <input FLT-structured file> <output FLT-structured file>\n')

    if len(args) > 2:
        # User provided parameters for maxiter, and possibly sigrej
        maxiter = int(args[2])
        if len(args) == 4:
            sigrej = float(args[3])
           
    if (help):
        print __doc__
        print __usage__
        print "\t", __version__+'('+__vdate__+')'
    else:    
        clean(args[0],args[1],clobber=clobber,maxiter=maxiter,sigrej=sigrej)

if __name__ == '__main__':

    main()
    
