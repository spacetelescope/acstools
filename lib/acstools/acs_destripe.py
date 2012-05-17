#!/usr/bin/env python
""" acs_destripe - try to clean out horizontal stripes from ACS WFC post-SM4 data.
  
  """
__author__ = "Norman Grogin, STScI, March 2012."
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
__version__ = '0.2.4'
__vdate__ = '17-May-2012'

import os, pyfits, sys
import numpy as np

from stsci.tools import parseinput,teal

### here are defined the fractional crosstalk values : 
### flux-independent; intra-chip only; reflexive; 
### (values had been 7.0e-5 before tweak to fit Saturn obsvs.)
CDcrosstalk = 9.1e-5
ABcrosstalk = 9.1e-5

class stripearray(object):
  
  def __init__(self,image):
    self.hdulist = pyfits.open(image)
    self.ampstring = self.hdulist[0].header['CCDAMP']
    self.flatcorr = self.hdulist[0].header['FLATCORR']
    self.configure_arrays()
  
  def configure_arrays(self):
    # Get the science and err data
    self.science = self.hdulist['sci',1].data
    self.err = self.hdulist['err',1].data
    if (self.ampstring == 'ABCD'): 
      self.science = np.concatenate((self.science,self.hdulist['sci',2].data[::-1,:]),axis=1)
      self.err = np.concatenate((self.err,self.hdulist['err',2].data[::-1,:]),axis=1)
    self.ingest_flatfield()

  def ingest_flatfield(self):
    flatfile = self.hdulist[0].header['PFLTFILE']
    
    ### if BIAS or DARK, set flatfield to unity
    if flatfile == 'N/A':
      self.invflat = np.ones(self.science.shape)
      return
    else: hduflat = self.resolve_flatname(flatfile)
    
    if (self.ampstring == 'ABCD'): 
      self.invflat = np.concatenate((1/hduflat['sci',1].data,1/hduflat['sci',2].data[::-1,:]),axis=1)
    else:
      ### complex algorithm to determine proper subarray of flatfield to use
      
      ### which amp?
      if (self.ampstring == 'A' or self.ampstring == 'B' or self.ampstring == 'AB'):
        self.invflat = 1/hduflat['sci',2].data
      else:
        self.invflat = 1/hduflat['sci',1].data
      
      ### now, which section?
      sizaxis1 = self.hdulist[1].header['SIZAXIS1']
      sizaxis2 = self.hdulist[1].header['SIZAXIS2']
      centera1 = self.hdulist[1].header['CENTERA1']
      centera2 = self.hdulist[1].header['CENTERA2']
      
      ### configure the offset appropriate to left- or right-side of CCD
      if (self.ampstring[0] == 'A' or self.ampstring[0] == 'C'):
        xdelta = 13
      else:
        xdelta = 35
      
      xlo = centera1 - xdelta - sizaxis1/2 - 1
      xhi = centera1 - xdelta + sizaxis1/2 - 1
      ylo = centera2 - sizaxis2/2 - 1
      yhi = centera2 + sizaxis2/2 - 1
      
      self.invflat = self.invflat[ylo:yhi,xlo:xhi]
    
    ### apply the flatfield if necessary
    if self.flatcorr != 'COMPLETE':
      self.science = self.science / self.invflat
      self.err = self.err / self.invflat
  
  def resolve_flatname(self,flatfile):
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
    return(pyfits.open(flatfile))
  
  def write_corrected(self,output):
    
    ### un-apply the flatfield if necessary
    if self.flatcorr != 'COMPLETE':
      self.science = self.science * self.invflat
      self.err = self.err * self.invflat
    
    ### reverse the amp merge
    if (self.ampstring == 'ABCD'):
      tmp_1, tmp_2 = np.split(self.science, 2, axis=1)
      self.hdulist['sci',1].data = tmp_1.copy()
      self.hdulist['sci',2].data = tmp_2[::-1,:].copy()

      tmp_1, tmp_2 = np.split(self.err, 2, axis=1)
      self.hdulist['err',1].data = tmp_1.copy()
      self.hdulist['err',2].data = tmp_2[::-1,:].copy()
    
    # Write the output
    self.hdulist.writeto(output)

def clean(input,suffix,clobber=False,maxiter=15,sigrej=2.0):
  
  flist,alist = parseinput.parseinput(input)
  
  for image in flist:
    ### Skip processing pre-SM4 images   
    if (pyfits.getval(image,'EXPSTART') <= 54967):
      sys.stdout.write('Error: Not processing %s ...pre-SM4 images not supported!'%image)
      continue
    
    ### Skip processing non-DARKCORR'd non-BIAS images
    if (pyfits.getval(image,'IMAGETYP') != 'BIAS' and pyfits.getval(image,'DARKCORR') != 'COMPLETE'):
      sys.stdout.write('Error: Not processing %s ...non-BIAS images must have DARKCORR=COMPLETE!'%image)
      continue
    
    ### newly requiring images be in electrons (revised CALACS now does this at beginning rather than end)
    if (pyfits.getval(image,'BUNIT',ext=1) != 'ELECTRONS'):
      sys.stdout.write('Error: Not processing %s ...SCI and ERR extensions must be in units of electrons!'%image)
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

  ### construct the frame to be cleaned, including the
  ### associated data stuctures needed for cleaning
  frame = stripearray(image)

  # Do the stripe cleaning
  clean_streak(frame,maxiter=maxiter,sigrej=sigrej)
  
  frame.write_corrected(output)

def clean_streak(image,maxiter=15,sigrej=2.0):
  
  ### create the array to hold the stripe amplitudes
  corr = np.empty(image.science.shape[0])
  
  ### loop over rows to fit the stripe amplitude
  for i in range(image.science.shape[0]):
    ### row-by-row iterative sigma-clipped mean; sigma, iters are adjustable
    SMean, SSig, SMedian, SMask = djs_iterstat(image.science[i],SigRej=sigrej,MaxIter=maxiter)
    
    ### SExtractor-esque central value statistic; slightly sturdier against
    ### skewness of pixel histogram due to faint source flux
    corr[i]=2.5*SMedian-1.5*SMean
  
  ### preserve the original mean level of the image
  corr -= np.average(corr)
  
  ### apply the correction row-by-row
  for i in range(image.science.shape[0]):
    ### stripe is constant along the row, before flatfielding; 
    ### afterwards it has the shape of the inverse flatfield
    truecorr = corr[i] * image.invflat[i] / np.average(image.invflat[i])
    
    ### correct the SCI extension
    image.science[i] -= truecorr
    
    ### correct the ERR extension
    image.err[i] = np.sqrt(image.err[i]**2 - truecorr)

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
#clean.__doc__ = getHelpAsString(fulldoc=False)

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
    ### added by NAG to run on command-line
    return
  
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
