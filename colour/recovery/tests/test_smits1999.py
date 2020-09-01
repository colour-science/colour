# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.recovery.smits1999` module.
"""

from __future__ import division, unicode_literals

import colour.ndarray as np
import unittest

from colour.colorimetry import sd_to_XYZ_integration
from colour.recovery import RGB_to_sd_Smits1999
from colour.recovery.smits1999 import XYZ_to_RGB_Smits1999
from colour.utilities import domain_range_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestRGB_to_sd_Smits1999']


class TestRGB_to_sd_Smits1999(unittest.TestCase):
    """
    Defines :func:`colour.recovery.smits1999.RGB_to_sd_Smits1999`
    definition unit tests methods.
    """

    def test_RGB_to_sd_Smits1999(self):
        """
        Tests :func:`colour.recovery.smits1999.RGB_to_sd_Smits1999`
        definition.
        """

        np.testing.assert_array_almost_equal(
            RGB_to_sd_Smits1999(
                XYZ_to_RGB_Smits1999(
                    np.array([0.21781186, 0.12541048, 0.04697113]))).values,
            np.array([
                0.07691923, 0.05870050, 0.03943195, 0.03024978, 0.02750692,
                0.02808645, 0.34298985, 0.41185795, 0.41185795, 0.41180754
            ]),
            decimal=7)

        np.testing.assert_array_almost_equal(
            RGB_to_sd_Smits1999(
                XYZ_to_RGB_Smits1999(
                    np.array([0.15434689, 0.22960951, 0.09620221]))).values,
            np.array([
                0.06981477, 0.06981351, 0.07713379, 0.25139495, 0.30063408,
                0.28797045, 0.11990414, 0.08186170, 0.08198613, 0.08272671
            ]),
            decimal=7)

        np.testing.assert_array_almost_equal(
            RGB_to_sd_Smits1999(
                XYZ_to_RGB_Smits1999(
                    np.array([0.07683480, 0.06006092, 0.25833845]))).values,
            np.array([
                0.29091152, 0.29010285, 0.26572455, 0.13140471, 0.05160646,
                0.05162034, 0.02765638, 0.03199188, 0.03472939, 0.03504156
            ]),
            decimal=7)

        np.testing.assert_array_almost_equal(
            RGB_to_sd_Smits1999(
                XYZ_to_RGB_Smits1999(np.array([0.0, 1.0, 0.0]))).values,
            np.array([
                -0.2549796, -0.2848386, -0.1634905, 1.5254829, 1.9800433,
                1.8510762, -0.7327702, -1.2758621, -1.2758621, -1.2703551
            ]),
            decimal=7)

        np.testing.assert_array_almost_equal(
            RGB_to_sd_Smits1999(
                XYZ_to_RGB_Smits1999(np.array([1.0, 1.0, 0.0]))).values,
            np.array([
                -0.1168428, -0.1396982, -0.0414535, 0.581391, 0.9563091,
                0.9562111, 1.3366949, 1.3742666, 1.3853491, 1.4027005
            ]),
            decimal=7)

        np.testing.assert_array_almost_equal(
            RGB_to_sd_Smits1999(
                XYZ_to_RGB_Smits1999(np.array([0.5, 0.0, 1.0]))).values,
            np.array([
                1.1938776, 1.1938776, 1.1213867, -0.067889, -0.4668587,
                -0.4030985, 0.703056, 0.9407334, 0.9437298, 0.9383386
            ]),
            decimal=7)

    def test_domain_range_scale_RGB_to_sd_Smits1999(self):
        """
        Tests :func:`colour.recovery.smits1999.RGB_to_sd_Smits1999`
        definition domain and range scale support.
        """

        RGB_i = XYZ_to_RGB_Smits1999(
            np.array([0.21781186, 0.12541048, 0.04697113]))
        XYZ_o = sd_to_XYZ_integration(RGB_to_sd_Smits1999(RGB_i))

        d_r = (('reference', 1, 1), (1, 1, 0.01), (100, 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    sd_to_XYZ_integration(
                        RGB_to_sd_Smits1999(RGB_i * factor_a)),
                    XYZ_o * factor_b,
                    decimal=7)


if __name__ == '__main__':
    unittest.main()
