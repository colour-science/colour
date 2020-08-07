# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.recovery.jakob2019` module.
"""

from __future__ import division, unicode_literals

import unittest
import numpy as np

from colour.characterisation import COLOURCHECKER_SDS
from colour.colorimetry import (SpectralShape, STANDARD_OBSERVER_CMFS,
                                ILLUMINANT_SDS, ILLUMINANTS, sd_to_XYZ)
from colour.difference import JND_CIE1976, delta_E_CIE1976
from colour.models import sRGB_COLOURSPACE, XYZ_to_RGB, XYZ_to_Lab
from colour.recovery import (spectral_primary_decomposition_Mallett2019,
                             RGB_to_sd_Mallett2019, sRGB_to_sd_Mallett2019)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestSPD', 'TestsRGB_to_sd']

D65 = ILLUMINANT_SDS['D65']
D65_XY = ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']


def test_function(test, RGB_to_sd, *args):
    """
    Tests :func:`RGB_to_sd_Mallett2019` or the more specialised
    :func:`sRGB_to_sd_Mallett2019`.
    """

    # Make sure the white point is reconstructed as a perfectly flat spectrum
    RGB = np.full(3, 1.0)
    sd = RGB_to_sd(RGB, *args)
    test.assertLess(np.var(sd.values), 1e-5)

    # Check if the primaries or their combination exceeds the [0, 1] range
    lower = np.zeros_like(sd.values) - 1e-12
    upper = np.ones_like(sd.values) + 1e+12
    for RGB in [[1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        sd = RGB_to_sd(RGB, *args)
        np.testing.assert_array_less(sd.values, upper)
        np.testing.assert_array_less(lower, sd.values)

    # Check delta E's using a colourchecker
    for name, sd in COLOURCHECKER_SDS['ColorChecker N Ohta'].items():
        XYZ = sd_to_XYZ(sd, illuminant=D65) / 100
        Lab = XYZ_to_Lab(XYZ, D65_XY)
        RGB = XYZ_to_RGB(XYZ, sRGB_COLOURSPACE.whitepoint, D65_XY,
                         sRGB_COLOURSPACE.XYZ_to_RGB_matrix)

        recovered_sd = RGB_to_sd(RGB, *args)
        recovered_XYZ = sd_to_XYZ(recovered_sd, illuminant=D65) / 100
        recovered_Lab = XYZ_to_Lab(recovered_XYZ, D65_XY)

        error = delta_E_CIE1976(Lab, recovered_Lab)

        # This method has relatively high delta E's using datasets generated
        # quickly, so this threshold has to be increased for unit tests.
        if error > 5 * JND_CIE1976:
            test.fail('Delta E for \'{0}\' is {1}!'.format(name, error))


class TestSPD(unittest.TestCase):
    """
    Defines :func:`colour.recovery.spectral_primary_decomposition_Mallett2019`
    definition unit tests methods.
    """

    def test_generate(self):
        """
        Generates a basis and tests it.
        """

        shape = SpectralShape(380, 730, 10)
        cmfs = STANDARD_OBSERVER_CMFS['CIE 1931 2 Degree Standard Observer']
        cmfs = cmfs.copy().align(shape)
        illuminant = D65.copy().align(shape)

        basis = spectral_primary_decomposition_Mallett2019(
            sRGB_COLOURSPACE, shape, cmfs, illuminant)

        test_function(self, RGB_to_sd_Mallett2019, basis)


class TestsRGB_to_sd(unittest.TestCase):
    """
    Defines :func:`colour.recovery.sRGB_to_sd_Mallett2019` definition unit
    tests methods.
    """

    def test_sRGB_to_sd(self):
        """
        Tests the pre-computed basis.
        """

        test_function(self, sRGB_to_sd_Mallett2019)


if __name__ == '__main__':
    unittest.main()
