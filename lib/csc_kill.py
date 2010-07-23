#!/usr/bin/env python
""" csc_kill - try to clean out horizontal stripes and crosstalk
               from ACS WFC post-SM4 data.

It is assumed that the data is an ACS/WFC FLT image - with two SCI extensions.
The program needs access to the flatfield specified in the image header PFLTFILE.
Norman Grogin, STScI, June 2010. """

__usage__ = """
Usage: 

- From within Python:
--- make sure this file is on your Python path

>>> import csc_kill
>>> csc_kill.run('uncorrected_flt.fits','uncorrected_flt_csck.fits',clobber=False,maxiter=15,sigrej=2.0)

- From the command line:
--- make sure this file is on your executable path

% ./csc_kill.py [-h][-c] uncorrected_flt.fits uncorrected_flt_csck.fits [15 [2.0]]
"""

__taskname__ = 'csc_kill'
__version__ = '0.2.1'
__vdate__ = '23-Jul-2010'

import os, pyfits, sys
from numpy import sqrt, empty, unique, zeros, byte, median, average, sum

from pytools import parseinput
        
def clean(input,suffix,clobber=False,maxiter=15,sigrej=2.0):
    ''' Primary user interface for running this task on either single or multiple
        input images. 
    '''
    flist,alist = parseinput.parseinput(input)
        
    for image in flist:
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

    ### resolve the filename into an absolute pathname (if necessary)
    flatparts = flatfile.partition('$')
    if flatparts[1] == '$':
    	flatfile = os.getenv(flatparts[0])+flatparts[2]

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
        science1 += science1[:,::-1] * 1.0e-4
        science2 += science2[:,::-1] * 7.0e-5
        # the following two lines are purposefully odd-looking because a 
        # direct assignment will break the linkage to the hdulist structure
        err1 += sqrt(err1**2 + science1[:,::-1] * 1.0e-4) - err1
        err2 += sqrt(err2**2 + science2[:,::-1] * 7.0e-5) - err2

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
        err1 += sqrt(err1**2 + science1[:,::-1] * 1.0e-4) - err1
        err2 += sqrt(err2**2 + science2[:,::-1] * 7.0e-5) - err2

        ### re-apply the flatfield
        science1 *= invflat1
        science2 *= invflat2
        err1 *= invflat1
        err2 *= invflat2

    # Do the stripe cleaning
    clean_streak(science1,invflat1,err1,maxiter=maxiter,sigrej=sigrej)
    clean_streak(science2,invflat2,err2,maxiter=maxiter,sigrej=sigrej)

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
   
def clean_streak(image,invflat,err,maxiter=15,sigrej=2.0):

    ### create the array to hold the stripe amplitudes
    corr = empty(2048)
   
    ### loop over rows to fit the stripe amplitude
    for i in range(2048):
        ### row-by-row iterative sigma-clipped mean; sigma, iters are adjustable
        SMean, SSig, SMedian, SMask = djs_iterstat(image[i],SigRej=sigrej,MaxIter=maxiter)
        
        ### SExtractor-esque central value statistic; slightly sturdier against
        ### skewness of pixel histogram due to faint source flux
        corr[i]=2.5*SMedian-1.5*SMean
    
    ### preserve the original mean level of the image
    corr -= average(corr)

    ### apply the correction row-by-row
    for i in range(2048):
        ### stripe is constant along the row, before flatfielding; 
        ### afterwards it has the shape of the inverse flatfield
        truecorr = corr[i] * invflat[i] / average(invflat[i])

        ### correct the SCI extension
        image[i] -= truecorr

        ### correct the ERR extension
        err[i] = sqrt(err[i]**2 - truecorr)

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
 
  if unique(InputArr).size == 1:
    return 0, 0, 0, 0
 

  Mask  = zeros(ArrShape, dtype=byte)+1
  #Reject those above Max and those below Min
  Mask[InputArr > Max] = 0
  Mask[InputArr < Min] = 0
  if RejVal != '' :  Mask[InputArr == RejVal]=0
  FMean = sum(1.*InputArr*Mask) / NGood
  FSig  = sqrt(sum((1.*InputArr-FMean)**2*Mask) / (NGood-1))

  NLast = -1
  Iter  =  0
  NGood = sum(Mask)
  if NGood < 2:
    return -1, -1, -1, -1

  while (Iter < MaxIter) and (NLast != NGood) and (NGood >= 2) :

    LoVal = FMean - SigRej*FSig
    HiVal = FMean + SigRej*FSig
    NLast = NGood

    Mask[InputArr < LoVal] = 0
    Mask[InputArr > HiVal] = 0
    NGood = sum(Mask)

    if NGood >= 2:
      FMean = sum(1.*InputArr*Mask) / NGood
      FSig  = sqrt(sum((1.*InputArr-FMean)**2*Mask) / (NGood-1))

    SaveMask = Mask.copy()

    Iter = Iter+1
  if sum(SaveMask) > 2:
    FMedian = median(InputArr[SaveMask == 1])
  else:
    FMedian = FMean

  return FMean, FSig, FMedian, SaveMask 

#
#### Interfaces used by TEAL
#
def run(configobj=None):
    ''' Teal interface for running this code. '''
### version Tue Jun 01 2010
### removes row striping (and row-averaged src flux if not amenable to sigma-clipping)
###
### example usage: csc_kill.run('striped_flt.fits','striped_flt_csck.fits')
###
### !!!expects an flt-format file, not a raw-format file!!!

    clean(configobj['input'],configobj['output'],
        clobber = configobj['clobber'],
        maxiter= configobj['maxiter'], sigrej = configobj['sigrej'])

def getHelpAsString():
    # Does NOT work with TEAL/teal.teal()
    helpString = __doc__+'\n'
    helpString += 'Version '+__version__+'\n'

    """ 
    return useful help from a file in the script directory called module.help
    """
    #get the local library directory where the code is stored
    localDir=os.path.split(__file__)
    helpfile=__taskname__.split(".")
    
    helpfile=localDir[0]+"/"+helpfile[0]+".help"
    
    if os.access(helpfile,os.R_OK):
        fh=open(helpfile,'r')
        ss=fh.readlines()
        fh.close()
        #helpString=""
        for line in ss:
            helpString+=line
    else:    
        helpString=__doc__

    return helpString


if __name__ == '__main__':

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
        sys.stderr.write('Usage: csc_kill.py <input FLT-structured file> <output FLT-structured file>\n')

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
