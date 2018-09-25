// Obtain files from source control system.
if (utils.scm_checkout()) return

// Configuration data needed to build HSTCAL.
CFLAGS = ''
LDFLAGS = ''
DEFAULT_FLAGS = "${CFLAGS} ${LDFLAGS}"
// Some waf flags cause a prompt for input during configuration, hence the 'yes'.
configure_cmd = "yes '' | hstcal/waf configure --prefix=./_install ${DEFAULT_FLAGS}"

// Define each build configuration, copying and overriding values as necessary.
bc0 = new BuildConfig()
bc0.nodetype = "linux-stable"
bc0.name = "egg"
bc0.build_cmds = ["python setup.py egg_info"]

bc1 = utils.copy(bc0)
bc1.name = "release"
// Would be nice if Jenkins can access /grp/hst/cdbs/xxxx directly.
bc1.env_vars = ['TEST_BIGDATA=https://bytesalad.stsci.edu/artifactory']
bc1.conda_channels = ['http://ssb.stsci.edu/astroconda']
bc1.conda_packages = ['python=3.6',
                      'requests',
                      'numpy',
                      'matplotlib',
                      'scipy',
                      'scikit-image',
                      'stsci.tools']
bc1.build_cmds = ["pip install ci-watson",
                  "python setup.py install"]
bc1.test_cmds = ["pytest --basetemp=tests_output --junitxml results.xml --bigdata -v"]
bc1.failedUnstableThresh = 1
bc1.failedFailureThresh = 6

// Build HSTCAL, and run with astropy dev and Python 3.7
bc2 = utils.copy(bc1)
bc2.name = "dev"
bc2.env_vars += ['PATH=./_install/bin:$PATH',
                 'OMP_NUM_THREADS=8']
bc2.conda_packages[0] = "python=3.7"
bc2.conda_packages += ['cfitsio', 'pkg-config']
bc2.build_cmds = ["git clone https://github.com/spacetelescope/hstcal.git",
                  "${configure_cmd} --release-with-symbols",
                  "hst/waf build",
                  "hst/waf install",
                  "calacs.e --version",
                  "pip install ci-watson",
                  "pip install git+https://github.com/astropy/astropy.git#egg=astropy --upgrade --no-deps",
                  "python setup.py install"]

// Run PEP 8 check
bc3 = utils.copy(bc0)
bc3.name = "pep8"
bc3.conda_packages = ['python=3.6', 'flake8']
bc3.test_cmds = ["flake8 acstools --count"]

// Run doc build
bc4 = utils.copy(bc0)
bc4.name = "doc"
bc4.conda_channels = ['http://ssb.stsci.edu/astroconda', 'astropy']
bc4.conda_packages = ['python=3.6', 'numpydoc', 'matplotlib','sphinx-automodapi']
bc4.build_cmds = ["pip install sphinx_rtd_theme",
                  "python setup.py install"]
bc4.test_cmds = ["cd doc; make html"]

// Iterate over configurations that define the (distibuted) build matrix.
// Spawn a host of the given nodetype for each combination and run in parallel.
utils.run([bc0, bc1, bc2, bc3, bc4])
