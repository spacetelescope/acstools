#!/usr/bin/env python
import sys
from setuptools import setup

# RELIC submodule
sys.path.insert(1, 'relic')
import relic.release

# Get some values from the setup.cfg
try:
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser
conf = ConfigParser()
conf.read(['setup.cfg'])
metadata = dict(conf.items('metadata'))

PACKAGENAME = metadata.get('package_name', 'packagename')
DESCRIPTION = metadata.get('description', 'package')
AUTHOR = metadata.get('author', '')
AUTHOR_EMAIL = metadata.get('author_email', '')
LICENSE = metadata.get('license', 'unknown')
URL = metadata.get('url', 'http://www.stsci.edu')
CLASSIFIERS = metadata.get('classifier', [''])

# Version from git tag
version = relic.release.get_info()
relic.release.write_template(version, PACKAGENAME)

# Define entry points for command-line scripts
entry_points = {'console_scripts': []}
entry_point_list = conf.items('entry_points')
for entry_point in entry_point_list:
    entry_points['console_scripts'].append('{0} = {1}'.format(entry_point[0],
                                                              entry_point[1]))
setup(
    name=PACKAGENAME,
    version=version.pep386,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    url=URL,
    classifiers=CLASSIFIERS,
    packages=[PACKAGENAME, PACKAGENAME + '.tests'],
    package_data={PACKAGENAME: ['pars/*']},
    entry_points=entry_points,
    install_requires = [
        'astropy>=1.1',
        'numpy'
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    use_2to3=False,
    zip_safe=False
)
