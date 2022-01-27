# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.rgb.cylindrical` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.models.rgb.cylindrical import (
    RGB_to_HSV,
    HSV_to_RGB,
    RGB_to_HSL,
    HSL_to_RGB,
    RGB_to_HCL,
    HCL_to_RGB,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestRGB_to_HSV',
    'TestHSV_to_RGB',
    'TestRGB_to_HSL',
    'TestHSL_to_RGB',
    'TestRGB_to_HCL',
    'TestHCL_to_RGB',
]


class TestRGB_to_HSV(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.cylindrical.RGB_to_HSV` definition unit
    tests methods.
    """

    def test_RGB_to_HSV(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.RGB_to_HSV` definition.
        """

        np.testing.assert_almost_equal(
            RGB_to_HSV(np.array([0.45620519, 0.03081071, 0.04091952])),
            np.array([0.99603944, 0.93246304, 0.45620519]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_HSV(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_HSV(np.array([1.00000000, 1.00000000, 1.00000000])),
            np.array([0.00000000, 0.00000000, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_HSV(np.array([0.00000000, 1.00000000, 1.00000000])),
            np.array([0.50000000, 1.00000000, 1.00000000]),
            decimal=7)

    def test_n_dimensional_RGB_to_HSV(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.RGB_to_HSV` definition
        n-dimensional arrays support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HSV = RGB_to_HSV(RGB)

        RGB = np.tile(RGB, (6, 1))
        HSV = np.tile(HSV, (6, 1))
        np.testing.assert_almost_equal(RGB_to_HSV(RGB), HSV, decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        HSV = np.reshape(HSV, (2, 3, 3))
        np.testing.assert_almost_equal(RGB_to_HSV(RGB), HSV, decimal=7)

    def test_domain_range_scale_RGB_to_HSV(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.RGB_to_HSV` definition
        domain and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HSV = RGB_to_HSV(RGB)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    RGB_to_HSV(RGB * factor), HSV * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_HSV(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.RGB_to_HSV` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_HSV(RGB)


class TestHSV_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.cylindrical.HSV_to_RGB` definition unit
    tests methods.
    """

    def test_HSV_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.HSV_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            HSV_to_RGB(np.array([0.99603944, 0.93246304, 0.45620519])),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

        np.testing.assert_almost_equal(
            HSV_to_RGB(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            HSV_to_RGB(np.array([0.00000000, 0.00000000, 1.00000000])),
            np.array([1.00000000, 1.00000000, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            HSV_to_RGB(np.array([0.50000000, 1.00000000, 1.00000000])),
            np.array([0.00000000, 1.00000000, 1.00000000]),
            decimal=7)

    def test_n_dimensional_HSV_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.HSV_to_RGB` definition
        n-dimensional arrays support.
        """

        HSV = np.array([0.99603944, 0.93246304, 0.45620519])
        RGB = HSV_to_RGB(HSV)

        HSV = np.tile(HSV, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_almost_equal(HSV_to_RGB(HSV), RGB, decimal=7)

        HSV = np.reshape(HSV, (2, 3, 3))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_almost_equal(HSV_to_RGB(HSV), RGB, decimal=7)

    def test_domain_range_scale_HSV_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.HSV_to_RGB` definition
        domain and range scale support.
        """

        HSV = np.array([0.99603944, 0.93246304, 0.45620519])
        RGB = HSV_to_RGB(HSV)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    HSV_to_RGB(HSV * factor), RGB * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_HSV_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.HSV_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            HSV = np.array(case)
            HSV_to_RGB(HSV)


class TestRGB_to_HSL(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.cylindrical.RGB_to_HSL` definition unit
    tests methods.
    """

    def test_RGB_to_HSL(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.RGB_to_HSL` definition.
        """

        np.testing.assert_almost_equal(
            RGB_to_HSL(np.array([0.45620519, 0.03081071, 0.04091952])),
            np.array([0.99603944, 0.87347144, 0.24350795]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_HSL(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_HSL(np.array([1.00000000, 1.00000000, 1.00000000])),
            np.array([0.00000000, 0.00000000, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_HSL(np.array([1.00000000, 0.00000000, 0.00000000])),
            np.array([0.00000000, 1.00000000, 0.50000000]),
            decimal=7)

    def test_n_dimensional_RGB_to_HSL(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.RGB_to_HSL` definition
        n-dimensional arrays support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HSL = RGB_to_HSL(RGB)

        RGB = np.tile(RGB, (6, 1))
        HSL = np.tile(HSL, (6, 1))
        np.testing.assert_almost_equal(RGB_to_HSL(RGB), HSL, decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        HSL = np.reshape(HSL, (2, 3, 3))
        np.testing.assert_almost_equal(RGB_to_HSL(RGB), HSL, decimal=7)

    def test_domain_range_scale_RGB_to_HSL(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.RGB_to_HSL` definition
        domain and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HSL = RGB_to_HSL(RGB)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    RGB_to_HSL(RGB * factor), HSL * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_HSL(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.RGB_to_HSL` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_HSL(RGB)


class TestHSL_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.cylindrical.HSL_to_RGB` definition unit
    tests methods.
    """

    def test_HSL_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.HSL_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            HSL_to_RGB(np.array([0.99603944, 0.87347144, 0.24350795])),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

        np.testing.assert_almost_equal(
            HSL_to_RGB(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            HSL_to_RGB(np.array([0.00000000, 0.00000000, 1.00000000])),
            np.array([1.00000000, 1.00000000, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            HSL_to_RGB(np.array([0.00000000, 1.00000000, 0.50000000])),
            np.array([1.00000000, 0.00000000, 0.00000000]),
            decimal=7)

    def test_n_dimensional_HSL_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.HSL_to_RGB` definition
        n-dimensional arrays support.
        """

        HSL = np.array([0.99603944, 0.87347144, 0.24350795])
        RGB = HSL_to_RGB(HSL)

        HSL = np.tile(HSL, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_almost_equal(HSL_to_RGB(HSL), RGB, decimal=7)

        HSL = np.reshape(HSL, (2, 3, 3))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_almost_equal(HSL_to_RGB(HSL), RGB, decimal=7)

    def test_domain_range_scale_HSL_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.HSL_to_RGB` definition
        domain and range scale support.
        """

        HSL = np.array([0.99603944, 0.87347144, 0.24350795])
        RGB = HSL_to_RGB(HSL)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    HSL_to_RGB(HSL * factor), RGB * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_HSL_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.HSL_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            HSL = np.array(case)
            HSL_to_RGB(HSL)


class TestRGB_to_HCL(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.cylindrical.RGB_to_HCL` definition unit
    tests methods.
    """

    def test_RGB_to_HCL(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.RGB_to_HCL` definition.
        """

        np.testing.assert_almost_equal(
            RGB_to_HCL(np.array([0.45620519, 0.03081071, 0.04091952])),
            np.array([-0.03167854, 0.2841715, 0.22859647]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_HCL(np.array([1.00000000, 2.00000000, 0.50000000])),
            np.array([1.83120102, 1.0075282, 1.00941024]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_HCL(np.array([2.00000000, 1.00000000, 0.50000000])),
            np.array([0.30909841, 1.0075282, 1.00941024]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_HCL(np.array([0.50000000, 1.00000000, 2.00000000])),
            np.array([-2.40349351, 1.0075282, 1.00941024]),
            decimal=7)

    def test_n_dimensional_RGB_to_HCL(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.RGB_to_HCL` definition
        n-dimensional arrays support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HCL = RGB_to_HCL(RGB)

        RGB = np.tile(RGB, (6, 1))
        HCL = np.tile(HCL, (6, 1))
        np.testing.assert_almost_equal(RGB_to_HCL(RGB), HCL, decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        HCL = np.reshape(HCL, (2, 3, 3))
        np.testing.assert_almost_equal(RGB_to_HCL(RGB), HCL, decimal=7)

    def test_domain_range_scale_RGB_to_HCL(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.RGB_to_HCL` definition
        domain and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HCL = RGB_to_HCL(RGB)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    RGB_to_HCL(RGB * factor), HCL * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_HCL(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.RGB_to_HCL` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_HCL(RGB)


class TestHCL_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.cylindrical.HCL_to_RGB` definition unit
    tests methods.
    """

    def test_HCL_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.HCL_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            HCL_to_RGB(np.array([-0.03167854, 0.28417150, 0.22859647])),
            np.array([0.45620333, 0.03081048, 0.04091925]),
            decimal=7)

        np.testing.assert_almost_equal(
            HCL_to_RGB(np.array([1.00000000, 2.00000000, 0.50000000])),
            np.array([0.92186029, 0.71091922, -2.26364935]),
            decimal=7)

        np.testing.assert_almost_equal(
            HCL_to_RGB(np.array([2.00000000, 1.00000000, 0.50000000])),
            np.array([-0.31368585, 1.00732462, -0.51534497]),
            decimal=7)

        np.testing.assert_almost_equal(
            HCL_to_RGB(np.array([0.50000000, 1.00000000, 2.00000000])),
            np.array([3.88095422, 3.11881934, 2.40881719]),
            decimal=7)

    def test_n_dimensional_HCL_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.HCL_to_RGB` definition
        n-dimensional arrays support.
        """

        HCL = np.array([0.99603944, 0.87347144, 0.24350795])
        RGB = HCL_to_RGB(HCL)

        HCL = np.tile(HCL, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_almost_equal(HCL_to_RGB(HCL), RGB, decimal=7)

        HCL = np.reshape(HCL, (2, 3, 3))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_almost_equal(HCL_to_RGB(HCL), RGB, decimal=7)

    def test_domain_range_scale_HCL_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.HCL_to_RGB` definition
        domain and range scale support.
        """

        HCL = np.array([0.99603944, 0.87347144, 0.24350795])
        RGB = HCL_to_RGB(HCL)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    HCL_to_RGB(HCL * factor), RGB * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_HCL_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.cylindrical.HCL_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            HCL = np.array(case)
            HCL_to_RGB(HCL)


if __name__ == '__main__':
    unittest.main()
