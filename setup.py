#!/usr/bin/env python
import os
import pkgutil
import sys
from setuptools import setup
from subprocess import check_call, CalledProcessError


if not pkgutil.find_loader('relic'):
    relic_local = os.path.exists('relic')
    relic_submodule = (relic_local and
                       os.path.exists('.gitmodules') and
                       not os.listdir('relic'))
    try:
        if relic_submodule:
            check_call(['git', 'submodule', 'update', '--init', '--recursive'])
        elif not relic_local:
            check_call(['git', 'clone', 'https://github.com/spacetelescope/relic.git'])

        sys.path.insert(1, 'relic')
    except CalledProcessError as e:
        print(e)
        exit(1)

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
CLASSIFIERS = [c for c in metadata.get('classifier', ['']).splitlines() if c]

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
    packages=[PACKAGENAME],
    package_dir={PACKAGENAME: PACKAGENAME},
    package_data={PACKAGENAME: ['pars/*']},
    entry_points=entry_points,
    install_requires=[
        'astropy>=1.1',
        'numpy'
    ],
    use_2to3=False,
    zip_safe=False
)
