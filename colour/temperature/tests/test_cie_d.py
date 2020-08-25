# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.temperature.cie_d` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.temperature import xy_to_CCT_CIE_D, CCT_to_xy_CIE_D
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestXy_to_CCT_CIE_D', 'TestCCT_to_xy_CIE_D']


class TestXy_to_CCT_CIE_D(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cie_d.xy_to_CCT_CIE_D` definition units
    tests methods.
    """

    def test_xy_to_CCT_CIE_D(self):
        """
        Tests :func:`colour.temperature.cie_d.xy_to_CCT_CIE_D` definition.
        """

        np.testing.assert_allclose(
            xy_to_CCT_CIE_D(
                np.array([0.382343625000000, 0.383766261015578]),
                {'method': 'Nelder-Mead'}),
            4000,
            rtol=0.0000001,
            atol=0.0000001)

        np.testing.assert_allclose(
            xy_to_CCT_CIE_D(
                np.array([0.305357431486880, 0.321646345474552]),
                {'method': 'Nelder-Mead'}),
            7000,
            rtol=0.0000001,
            atol=0.0000001)

        np.testing.assert_allclose(
            xy_to_CCT_CIE_D(
                np.array([0.24985367, 0.254799464210944]),
                {'method': 'Nelder-Mead'}),
            25000,
            rtol=0.0000001,
            atol=0.0000001)

    def test_n_dimensional_xy_to_CCT_CIE_D(self):
        """
        Tests :func:`colour.temperature.cie_d.xy_to_CCT_CIE_D` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.382343625000000, 0.383766261015578])
        CCT = xy_to_CCT_CIE_D(xy)

        xy = np.tile(xy, (6, 1))
        CCT = np.tile(CCT, 6)
        np.testing.assert_almost_equal(xy_to_CCT_CIE_D(xy), CCT, decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        CCT = np.reshape(CCT, (2, 3))
        np.testing.assert_almost_equal(xy_to_CCT_CIE_D(xy), CCT, decimal=7)

    @ignore_numpy_errors
    def test_nan_xy_to_CCT_CIE_D(self):
        """
        Tests :func:`colour.temperature.cie_d.xy_to_CCT_CIE_D` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            xy_to_CCT_CIE_D(case)


class TestCCT_to_xy_CIE_D(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cie_d.CCT_to_xy_CIE_D` definition
    unit tests methods.
    """

    def test_CCT_to_xy_CIE_D(self):
        """
        Tests :func:`colour.temperature.cie_d.CCT_to_xy_CIE_D` definition.
        """

        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(4000),
            np.array([0.382343625000000, 0.383766261015578]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(7000),
            np.array([0.305357431486880, 0.321646345474552]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(25000),
            np.array([0.24985367, 0.254799464210944]),
            decimal=7)

    def test_n_dimensional_CCT_to_xy_CIE_D(self):
        """
        Tests :func:`colour.temperature.cie_d.CCT_to_xy_CIE_D` definition
        n-dimensional arrays support.
        """

        CCT = 4000
        xy = CCT_to_xy_CIE_D(CCT)

        CCT = np.tile(CCT, 6)
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(CCT_to_xy_CIE_D(CCT), xy, decimal=7)

        CCT = np.reshape(CCT, (2, 3))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(CCT_to_xy_CIE_D(CCT), xy, decimal=7)

    @ignore_numpy_errors
    def test_nan_CCT_to_xy_CIE_D(self):
        """
        Tests :func:`colour.temperature.cie_d.CCT_to_xy_CIE_D` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            CCT_to_xy_CIE_D(case)


if __name__ == '__main__':
    unittest.main()
