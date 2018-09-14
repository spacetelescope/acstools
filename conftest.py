"""Custom ``pytest`` configurations."""

from astropy.tests.helper import enable_deprecations_as_exceptions

# Turn deprecation warnings into exceptions.
# TODO: What we really want to do.
#enable_deprecations_as_exceptions()
# TODO: Remove this workaround eventually.
#       https://github.com/astropy/astropy/issues/7822
enable_deprecations_as_exceptions(include_astropy_deprecations=False,
                                  warnings_to_ignore_entire_module=['socks'])

# Require these pytest plugins to run.
pytest_plugins = ["pytest_ciwatson"]


# For easy inspection on what dependencies were used in test.
def pytest_report_header(config):
    import sys
    import warnings
    from astropy.utils.introspection import resolve_name

    s = "\nFull Python Version: \n{0}\n\n".format(sys.version)

    for module_name in ('numpy', 'astropy', 'scipy', 'matplotlib',
                        'stsci.tools'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                module = resolve_name(module_name)
        except ImportError:
            s += "{0}: not available\n".format(module_name)
        else:
            try:
                version = module.__version__
            except AttributeError:
                version = 'unknown (no __version__ attribute)'
            s += "{0}: {1}\n".format(module_name, version)

    return s
