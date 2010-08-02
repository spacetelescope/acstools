#!/usr/bin/env python

# $Id: readgeis.py 8609 2010-01-19 16:22:48Z dencheva $

"""
    updatenpol:  Update the header of ACS file(s) with the names of new
                NPOLFILE and D2IMFILE reference files for use with the 
                C version of MultiDrizzle.
    
        License: http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE
        
        Usage:
            updatenpol [options] input [refdir]
            
                'input' is the specification of the files to be updated, either
                as a single filename, an ASN table name, or wild-card specification
                of a list of files
                
                'refdir' is the name of the directory containing all the new
                reference files (*_npl.fits and *_d2i.fits files). 
                If no directory is given, it will look in 'jref$' by default. 
                
                Option:

                -h
                    print the help (this text)
                -l
                    if specified, copy NPOLFILEs and D2IMFILEs to
                    local directory for use with the input files
                -i
                    if specified, the program will interactively request
                    the exact names of the NPOLFILE and D2IMFILE reference files
                    to be used for updating the header of each file. The value of
                    'refdir' will be ignored in interactive mode.
                    ===> WARNING: 
                         It will ask for this information for EACH separate INPUT file. 

        Examples:
            updatenpol *flt.fits myjref$
            
            This command will update all the FLT files in the current directory
            with the new NPOLFILE and D2IMFILE reference files found in the 'myjref' 
            directory as defined in the environment.  
    
        Python Syntax:
            >>> import updatenpol
            >>> updatenpol.update('*flt.fits','myjref$')
            
            Another use under Python would be to feed it a specific list of files
            to be updated using:
            >>> updatenpol.update(['file1_flt.fits','file2_flt.fits'],'myjref$')
            
            Files in another directory can also be processed using:
            >>> updatenpol.update('data$*flt.fits','../new/ref/')
            
"""

from __future__ import division
__docformat__ = 'restructuredtext'

__version__ = '1.0.2'
__vdate__ = '25-May-2010'
import os,sys,shutil

import pyfits
from pytools import fileutil as fu
from pytools import parseinput

def update(input,refdir="jref$",local=None,interactive=False):
    """ 
    Purpose
    =======    
    Updates files given in input (as a single filename, a list or any other
    input recognized by parseinput) to use the new reference files required with
    the new C version of MultiDrizzle. 
    
    Example
    =======
    >>>import updatenpol
    >>>updatenpol.update('j8bt06010_asn.fits', 'myref$')
    
    :Parameters:

    `input`: string or list
                Name of input file or files
                Acceptable forms: 
                  - single filename with or without directory
                  - @-file
                  - association table
                  - python list of filenames
                  - wildcard specification of filenames
    `refdir`: string
                path to directory containing new reference files, either 
                environment variable or full path
    `local`: boolean
                Specifies whether or not to copy new reference files to local
                directory for use with the input files
    `interactive': boolean 
                Specifies whether or not to interactively ask the user for the
                exact names of the new reference files instead of automatically
                searching a directory for them.
    """ 
    print 'UPDATENPOL Version',__version__+'('+__vdate__+')'
    # expand (as needed) the list of input files
    files,output = parseinput.parseinput(input)

    if not interactive:
        # expand reference directory name (if necessary) to 
        # interpret IRAF or environment variable names
        rdir = fu.osfn(refdir)
        ngeofiles,ngout = parseinput.parseinput(os.path.join(rdir,'*npl.fits'))
        # Find D2IMFILE in refdir for updating input file header as well
        d2ifiles,d2iout = parseinput.parseinput(os.path.join(rdir,"*d2i.fits"))

    # Now, build a matched list of input files and DGEOFILE reference files
    # to use for selecting the appropriate new reference file from the 
    # refdir directory. 
    for f in files: 
        print 'Updating: ',f
        fdir = os.path.split(f)[0]
        # Open each file...
        fimg = pyfits.open(f,mode='update')
        phdr = fimg['PRIMARY'].header
        fdet = phdr['detector']
        # get header of DGEOFILE 
        dhdr = pyfits.getheader(fu.osfn(phdr['DGEOFILE']))
        if not interactive:
            # search all new NPOLFILEs for one that matches current DGEOFILE config
            npol = find_npolfile(ngeofiles,fdet,[phdr['filter1'],phdr['filter2']])
        else:
            npol = raw_input("Enter name of NPOLFILE for %s:"%f)
            if npol == "": npol = None
            
        if npol is None:
            errstr =  "No valid NPOLFILE found in "+rdir+" for detector="+fdet+"\n"
            errstr += " filters = "+phdr['filter1']+","+phdr['filter2']
            raise ValueError,errstr
            
        npolname = os.path.split(npol)[1]
        if local:
            npolname = os.path.join(fdir,npolname)
            # clobber any previous copies of this reference file
            if os.path.exists(npolname): os.remove(npolname)
            shutil.copy(npol,npolname)
        else:
            if '$' in refdir:
                npolname = refdir+npolname
            else:
                npolname = os.path.join(refdir,npolname)
        phdr.update('NPOLFILE',npolname,comment="Non-polynomial corrections in Paper IV LUT",after='DGEOFILE')

        # Now find correct D2IFILE
        if not interactive:
            d2i = find_d2ifile(d2ifiles,fdet)
        else:
            d2i = raw_input("Enter name of D2IMFILE for %s:"%f)
            if d2i == "": d2i = None
            
        if d2i is None:
            print '=============\nWARNING:'
            print "    No valid D2IMFILE found in "+rdir+" for detector ="+fdet
            print "    D2IMFILE correction will not be applied."
            print '=============\n'
            d2iname = ""
        else:
            d2iname = os.path.split(d2i)[1]
            if local:
                # Copy D2IMFILE to local data directory alongside input file as well
                d2iname = os.path.join(fdir,d2iname)
                # clobber any previous copies of this reference file
                if os.path.exists(d2iname): os.remove(d2iname)
                shutil.copy(d2i,d2iname)
            else:
                if '$' in refdir:
                    d2iname = refdir+d2iname
                else:
                    d2iname = os.path.join(refdir,d2iname)
            
        phdr.update('D2IMFILE',d2iname,comment="Column correction table",after='DGEOFILE')

        # Close this input file header and go on to the next
        fimg.close()
        
        
def find_d2ifile(flist,detector):
    """ Search a list of files for one that matches the detector specified
    """
    d2ifile = None
    d2idate = None
    for f in flist:
        fdet = pyfits.getval(f,'detector') 
        if fdet == detector:
            fdate = fu.convertDate(pyfits.getval(f,'DATE'))
            if d2idate is None or fdate > d2idate:
                d2idate = fdate
                d2ifile = f
    return d2ifile
        
def find_npolfile(flist,detector,filters):
    """ Search a list of files for one that matches the configuration 
        of detector and filters used.
    """      
    npolfile = None
    npoldate = None
    for f in flist:
        fdet = pyfits.getval(f,'detector')
        if fdet == detector:
            filt1 = pyfits.getval(f,'filter1')
            filt2 = pyfits.getval(f,'filter2')
            fdate = pyfits.getval(f,'date')
            if filt1 == 'ANY' or \
             (filt1 == filters[0] and filt2 == filters[1]):
                ddate = fu.convertDate(fdate)
                if npoldate is None or ddate > npoldate:
                    npoldate = ddate
                    npolfile = f
    return npolfile

def help():
    print 'updatenpol Version ',__version__+__vdate__
    print update.__doc__

def main():

    import getopt

    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'hli')
    except getopt.error, e:
        print str(e)
        print __doc__
        print "\t", __version__

    # initialize default values
    help = 0
    local = False
    interactive = False
    
    # read options
    for opt, value in optlist:
        if opt == "-h":
            help = 1
        if opt == "-l":
            local = True
        if opt == "-i":
            interactive = True
    if len(args) < 2:
        args.append('jref$')
    if (help):
        print __doc__
        print "\t", __version__+'('+__vdate__+')'
    else:
        update(args[:-1],args[-1],local=local,interactive=interactive)

if __name__ == "__main__":
    main()
"""

Copyright (C) 2003 Association of Universities for Research in Astronomy (AURA)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    3. The name of AURA and its representatives may not be used to
      endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY AURA ``AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL AURA BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""
