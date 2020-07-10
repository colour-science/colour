# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.recovery.jakob2019` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.characterisation import COLOURCHECKER_SDS
from colour.colorimetry import (ILLUMINANTS, ILLUMINANT_SDS,
                                STANDARD_OBSERVER_CMFS, SpectralDistribution,
                                sd_to_XYZ)
from colour.difference import delta_E_CIE1976
from colour.models import XYZ_to_Lab
from colour.recovery import XYZ_to_sd_Otsu2018

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestXYZ_to_sd_Otsu2018']

CMFS = STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer']
D65 = SpectralDistribution(ILLUMINANT_SDS['D65'])
D65_XY = ILLUMINANTS['CIE 1931 2 Degree Standard Observer']["D65"]


class TestXYZ_to_sd_Otsu2018(unittest.TestCase):
    """
    Defines :func:`colour.recovery.otsu2018.XYZ_to_sd_otsu2018`
    definition unit tests methods.
    """

    def test_roundtrip_colourchecker(self):
        """
        Tests :func:`colour.recovery.otsu2018.XYZ_to_sd_otsu2018` definition
        round-trip errors using a colour checker.
        """

        for _name, sd in COLOURCHECKER_SDS['ColorChecker N Ohta'].items():
            XYZ = sd_to_XYZ(sd, illuminant=D65) / 100
            Lab = XYZ_to_Lab(XYZ, D65_XY)

            recovered_sd = XYZ_to_sd_Otsu2018(XYZ, CMFS, D65, clip=False)
            recovered_XYZ = sd_to_XYZ(recovered_sd, illuminant=D65) / 100
            recovered_Lab = XYZ_to_Lab(recovered_XYZ, D65_XY)

            error = delta_E_CIE1976(Lab, recovered_Lab)
            self.assertLess(error, 1e-12)


if __name__ == '__main__':
    unittest.main()
