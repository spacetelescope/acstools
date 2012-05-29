from nose import tools

from acstools import calacs

@tools.raises(OSError)
def test_raises_oserror():
    exec_path = 'IamNOTanEXECUTABLE'

    input_file = 'IamNOTaFILE'

    calacs.calacs(input_file, exec_path=exec_path)


@tools.raises(IOError)
def test_raises_ioerror():
    input_file = 'IamNOTaFILE'

    calacs.calacs(input_file)
