"""CALACS: WFC ASN Set Full-frame Pre-SM4

Process a WFC dataset using CR-SPLIT=2 with all standard calibration steps
turned on.

"""
from __future__ import absolute_import, division, print_function

from .helpers import remote_data, use_calacs


@remote_data
@use_calacs
def test_wfc_asn_fullframe_presm4():
    pass


@remote_data
def test1():
    pass


@use_calacs
def test2():
    pass
