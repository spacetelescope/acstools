// Obtain files from source control system.
if (utils.scm_checkout()) return

// Define each build configuration, copying and overriding values as necessary.
bc1 = new BuildConfig()
bc1.nodetype = "linux"
bc1.name = "release"
// Would be nice if Jenkins can access /grp/hst/cdbs/xxxx directly.
bc1.env_vars = ['TEST_BIGDATA=https://bytesalad.stsci.edu/artifactory']
bc1.conda_channels = ['http://ssb.stsci.edu/astroconda']
bc1.conda_packages = ['python=3.6']
bc1.build_cmds = ["pip install -e .[test,all]"]
bc1.test_cmds = ["pytest --basetemp=tests_output --junitxml results.xml --bigdata -v"]
bc1.failedUnstableThresh = 1
bc1.failedFailureThresh = 6

// Run with astropy dev and Python 3.8
bc2 = utils.copy(bc1)
bc2.name = "dev"
bc2.conda_packages[0] = "python=3.8"
bc2.build_cmds = ["pip install git+https://github.com/astropy/astropy.git#egg=astropy --upgrade --no-deps",
                  "pip install -e .[test,all]"]

// Iterate over configurations that define the (distibuted) build matrix.
// Spawn a host of the given nodetype for each combination and run in parallel.
utils.run([bc1, bc2])
