"""
Compute sRGB colour values of CIE 224:2017's 99 test color samples under D65.
"""

from colour.colorimetry import (SpectralShape, SDS_ILLUMINANTS, sd_to_XYZ)
from colour.models import XYZ_to_sRGB
from colour.notation import RGB_to_HEX
from colour.quality.cfi2017 import get_tcs_CFI2017

if __name__ == '__main__':
    sds = get_tcs_CFI2017(SpectralShape(380, 780, 5)).to_sds()
    D65 = SDS_ILLUMINANTS['D65']

    for sd in sds:
        XYZ = sd_to_XYZ(sd, illuminant=D65) / 100
        RGB = XYZ_to_sRGB(XYZ, cctf_encoding=True)
        HEX = RGB_to_HEX(RGB)
        print('\'%s\', ' % HEX, end='')

    print('')
