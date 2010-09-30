from __future__ import division # confidence high

pkg =  "acstools"

setupargs = {
    'version' : 		"1.0.2",
    'description' :	    "Python Tools for ACS Data",
    'author' : 		    "Warren Hack, Norman Grogin",
    'author_email' : 	"help@stsci.edu",
    'license' : 		"http://www.stsci.edu/resources/software_hardware/pyraf/LICENSE",
    'data_files' :      [( pkg+"/pars", ['lib/pars/*']),( pkg, ['lib/*.help'])],
    'scripts' :         [ 'lib/acs_destripe', 'lib/updatenpol'] ,
    'platforms' : 	    ["Linux","Solaris","Mac OS X","Win"],
}

