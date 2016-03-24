#!/usr/bin/env python
import os
import subprocess
import sys
from setuptools import setup, find_packages


if os.path.exists('relic'):
    sys.path.insert(1, 'relic')
    import relic.release
else:
    try:
        import relic.release
    except ImportError:
        try:
            subprocess.check_call(['git', 'clone', 
                'https://github.com/jhunkeler/relic.git'])
            sys.path.insert(1, 'relic')
            import relic.release
        except subprocess.CalledProcessError as e:
            print(e)
            exit(1)


version = relic.release.get_info()
relic.release.write_template(version, 'lib/acstools')

setup(
    name = 'acstools',
    version = version.pep386,
    author = 'Matt Davis, Warren Hack, Norman Grogin, Pey Lian Lim, Sara Ogaz, Leornado Ubeda, Mihai Cara, David Borncamp',
    author_email = 'help@stsci.edu',
    description = 'Python Tools for ACS (Advanced Camera for Surveys) Data',
    url = 'https://github.com/spacetelescope/acstools',
    classifiers = [
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires = [
        'astropy',
        'nose',
        'numpy',
        'scikit-image',
        'scipy',
        'sphinx',
        'stsci.imagestats',
        'stsci.sphinxext',
        'stsci.tools',
    ],

    package_dir = {
        '': 'lib'
    },
    packages = find_packages('lib'),
    package_data = {
        'acstools': [
            'pars/*',
            'LICENSE.txt'
        ]
    },
    entry_points = {
        'console_scripts': [
            'acs_destripe=acstools.acs_destripe:main',
            'acs_destripe_plus=acstools.acs_destripe_plus:main'
        ],
    },
)
