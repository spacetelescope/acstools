#!/usr/bin/env python
from setuptools import setup

# Get some values from the setup.cfg
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

# Define entry points for command-line scripts
entry_points = {'console_scripts': []}
entry_point_list = conf.items('entry_points')
for entry_point in entry_point_list:
    entry_points['console_scripts'].append('{0} = {1}'.format(entry_point[0],
                                                              entry_point[1]))
setup(
    name=PACKAGENAME,
    use_scm_version=True,
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
    python_requires='>=3.5',
    setup_requires=['setuptools_scm'],
    install_requires=[
        'astropy>=2',
        'numpy',
        'beautifulsoup4'
    ],
    tests_require=['pytest'],
    zip_safe=False
)
