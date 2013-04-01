"""
Class for image operations by amp readout.

Written for ACS/WFC pixel-based CTE correction
but could be used for other programs as needed.
Could also be modified to work on other
detectors.

"""
# External modules
import numpy as np


class ImageOpByAmp(object):
    """
    Image op by amp readout class.

    .. note:: Can be modified to exclude overscans in RAW.

    Parameters
    ----------
    fitsPointer : `astropy.io.fits` image pointer

    """

    def __init__(self, fitsPointer):
        self._pf = fitsPointer
        hasMode = False

        self._ccdamp = self._pf['PRIMARY'].header['CCDAMP']
        self._ampList = list(self._ccdamp)
        n_amp = len(self._ccdamp)

        instrume = self._pf['PRIMARY'].header['INSTRUME']
        detector = self._pf['PRIMARY'].header['DETECTOR']
        naxis1 = self._pf['SCI',1].header['NAXIS1']
        naxis2 = self._pf['SCI',1].header['NAXIS2']
        xHalf = naxis1 / 2
        yMos  = naxis2 * 2

        # ----- ACS
        if instrume == 'ACS':

            # ----- WFC
            # Amps: A,B,C,D,AC,AD,BC,BD,ABCD
            if detector == 'WFC':

                hasMode = True

                # ----- Extension
                self._extNum = {'SCI':{'A':4, 'B':4, 'C':1, 'D':1},
                                'ERR':{'A':5, 'B':5, 'C':2, 'D':2},
                                'DQ': {'A':6, 'B':6, 'C':3, 'D':3}}
                # Special for subarray
                if n_amp == 1:
                    self._extNum = {'SCI':{'A':1, 'B':1, 'C':1, 'D':1},
                                    'ERR':{'A':2, 'B':2, 'C':2, 'D':2},
                                    'DQ': {'A':3, 'B':3, 'C':3, 'D':3}}
                # End if

                # ----- Quadrants
                # Overscans are not excluded.
                self._y1, self._y2 = 0, naxis2
                self._x1 = {'A':0, 'B':0, 'C':0, 'D':0}
                self._x2 = {'A':naxis1, 'B':naxis1, 'C':naxis1, 'D':naxis1}
                # Special for full frame
                if n_amp == 4:
                    self._x1 = {'A':0, 'B':xHalf, 'C':0, 'D':xHalf}
                    self._x2 = {'A':xHalf, 'B':naxis1, 'C':xHalf, 'D':naxis1}
                # End if

                # ----- Amp transformation code
                self._transCode = {'A':1, 'B':3, 'C':0, 'D':2}

                # ----- Gain and noise keywords
                # Only works on FLT, not populated in RAW.
                self._hdrKeys =  {'gain':{'A':'ATODGNA', 'B':'ATODGNB',
                                          'C':'ATODGNC', 'D':'ATODGND'},
                                  'noise':{'A':'READNSEA', 'B':'READNSEB',
                                           'C':'READNSEC', 'D':'READNSED'}}

                # ----- Mosaic size and coordinates
                self._mosaicXsize, self._mosaicYsize = naxis1, yMos
                self._mosaicX1 = self._x1
                self._mosaicX2 = self._x2
                self._mosaicY1 = {'A':naxis2, 'B':naxis2, 'C':0, 'D':0}
                self._mosaicY2 = {'A':yMos, 'B':yMos, 'C':naxis2, 'D':naxis2}
                # Special for subarray
                if n_amp == 1:
                    self._mosaicYsize = naxis2
                    self._mosaicY1 = {'A':0, 'B':0, 'C':0, 'D':0}
                    self._mosaicY2 = {'A':naxis2, 'B':naxis2,
                                      'C':naxis2, 'D':naxis2}
                # End if

            # End of DETECTOR check
        # End of INSTRUME check

        if not hasMode:
            raise ValueError('Unsupported mode')

    def GetAmps(self):
        """
        Get list of amps used.

        Returns
        -------
        self._ampList : list of str

        """
        return self._ampList

    def GetHdrValByAmp(self, key):
        """
        Get gain or noise for each amp.

        Parameters
        ----------
        key : {'gain', 'noise'}

        Returns
        -------
        dataOut : dict
            Values for each amp.

        """
        dataOut = {}
        for amp in self._ampList:
            dataOut[amp] = self._pf['PRIMARY'].header[ self._hdrKeys[key][amp] ]
        return dataOut

    def DataByAmp(self, extName='SCI'):
        """
        Separate data by amp readout such that amp will
        always be on the lower left of the data (when
        displayed in DS9 or Matplotlib in default settings).

        Parameters
        ----------
        extName : {'SCI', 'ERR', 'DQ'}
            Extension name of data to extract.

        Returns
        -------
        dataOut : `numpy.ndarray`
            View of data with adjusted amp position.

        """
        dataOut = {}

        for amp in self._ampList:
            arr1 = self._pf[ self._extNum[extName][amp] ].data[
                self._y1:self._y2, self._x1[amp]:self._x2[amp] ]
            dataOut[amp] = self.FlipAmp(arr1, self._transCode[amp])
        # End of amp loop

        return dataOut

    def FlipAmp(self, dataArray, transCode, trueCopy=False):
        """
        Flip array with given transformation.

        Parameters
        ----------
        dataArray : `numpy.ndarray`
            Array to flip.

        transCode : int
            Transformation code:
                * 0 - None
                * 1 - Flip vertical
                * 2 - Flip horizontal
                * 3 - Flip both

        trueCopy : bool
            Return copy instead of view?

        Returns
        -------
        arr : `numpy.ndarray`
            Flipped array.

        """
        if transCode == 1: # Flip vertical
            arr = dataArray[::-1,:]
        elif transCode == 2: # Flip horizontal
            arr = dataArray[:,::-1]
        elif transCode == 3: # Flip both
            arr = dataArray[::-1,::-1]
        else: # None
            arr = dataArray
        # End if

        if trueCopy:
            return arr.copy()
        else:
            return arr

    def MosaicTemplate(self):
        """
        Blank array template for mosaic.

        Returns
        -------
        `numpy.ndarray`

        """
        return np.zeros((self._mosaicYsize, self._mosaicXsize))

    def MosaicPars(self, amp):
        """
        Mosaic parameters for a given amp.

        Parameters
        ----------
        amp : str
            Amplifier to use.

        Returns
        -------
        X1, X2, Y1, Y2, transformation code : tuple of int

        """
        return self._mosaicX1[amp], self._mosaicX2[amp], \
               self._mosaicY1[amp], self._mosaicY2[amp], self._transCode[amp]
