.. _acstools_doc:

########
ACSTOOLS
########

Software tools for
`Advanced Camera for Surveys (ACS) <http://www.stsci.edu/hst/acs/>`_.

Different ways to install the latest release of this package::

    pip install acstools

    conda install acstools -c conda-forge

To install the development version of this package::

    pip install git+https://github.com/spacetelescope/acstools

.. note::

    The information here reflects the *latest* software and might not
    be in-sync with the
    `ACS Data Handbook <https://hst-docs.stsci.edu/acsdhb>`_.

.. note::

    Standalone CTE correction (``PixCteCorr``) is no longer supported.
    Please use `acstools.acscte`.

.. note::

    Python 2 is no longer supported. Please use Python 3.8 or later.

.. toctree::
   :maxdepth: 2

   calacs
   acs_destripe
   acszpt
   acsphotcte
   satdet
   findsat_mrt
   polarization_tools
   utils_calib

******************
Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
