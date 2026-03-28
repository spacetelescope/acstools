import os

from numpy import get_include as get_numpy_include
from setuptools import Extension

ROOT = os.path.relpath(os.path.dirname(__file__))


def get_extensions():
    _radon_ext = Extension(
        name="acstools._radon",
        sources=[os.path.join(ROOT, "_radon.pyx")],
        include_dirs=[get_numpy_include()],
    )

    return [_radon_ext]
