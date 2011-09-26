"""The acstools package holds Python tasks useful for analyzing ACS data.

These tasks include:

Utility and library functions used by these tasks are also included in this
module.


"""

if False :
    __version__ = ''

    __svn_version__ = 'Unable to determine SVN revision'
    __full_svn_info__ = ''
    __setup_datetime__ = None

    try:
        __version__ = __import__('pkg_resources').\
                            get_distribution('acstools').version
    except:
        pass

else :
    __version__ = '2.0.0dev'

try:
    from acstools.svninfo import (__svn_version__, __full_svn_info__,
                                  __setup_datetime__)
except ImportError:
    pass

import PixCteCorr
