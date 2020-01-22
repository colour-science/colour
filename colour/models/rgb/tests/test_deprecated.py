# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.deprecated` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models.rgb.deprecated import (RGB_to_HSV, HSV_to_RGB, RGB_to_HSL,
                                          HSL_to_RGB, RGB_to_CMY, CMY_to_RGB,
                                          CMY_to_CMYK, CMYK_to_CMY)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestRGB_to_HSV', 'TestHSV_to_RGB', 'TestRGB_to_HSL', 'TestHSL_to_RGB',
    'TestRGB_to_CMY', 'TestCMY_to_RGB', 'TestCMY_to_CMYK', 'TestCMYK_to_CMY'
]


class TestRGB_to_HSV(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.deprecated.RGB_to_HSV` definition unit
    tests methods.
    """

    def test_RGB_to_HSV(self):
        """
        Tests :func:`colour.models.rgb.deprecated.RGB_to_HSV` definition.
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

    def test_n_dimensional_RGB_to_HSV(self):
        """
        Tests :func:`colour.models.rgb.deprecated.RGB_to_HSV` definition
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
        Tests :func:`colour.models.rgb.deprecated.RGB_to_HSV` definition domain
        and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HSV = RGB_to_HSV(RGB)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    RGB_to_HSV(RGB * factor), HSV * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_HSV(self):
        """
        Tests :func:`colour.models.rgb.deprecated.RGB_to_HSV` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_HSV(RGB)


class TestHSV_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.deprecated.HSV_to_RGB` definition unit
    tests methods.
    """

    def test_HSV_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.deprecated.HSV_to_RGB` definition.
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

    def test_n_dimensional_HSV_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.deprecated.HSV_to_RGB` definition
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
        Tests :func:`colour.models.rgb.deprecated.HSV_to_RGB` definition domain
        and range scale support.
        """

        HSV = np.array([0.99603944, 0.93246304, 0.45620519])
        RGB = HSV_to_RGB(HSV)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    HSV_to_RGB(HSV * factor), RGB * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_HSV_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.deprecated.HSV_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            HSV = np.array(case)
            HSV_to_RGB(HSV)


class TestRGB_to_HSL(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.deprecated.RGB_to_HSL` definition unit
    tests methods.
    """

    def test_RGB_to_HSL(self):
        """
        Tests :func:`colour.models.rgb.deprecated.RGB_to_HSL` definition.
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

    def test_n_dimensional_RGB_to_HSL(self):
        """
        Tests :func:`colour.models.rgb.deprecated.RGB_to_HSL` definition
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
        Tests :func:`colour.models.rgb.deprecated.RGB_to_HSL` definition domain
        and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HSL = RGB_to_HSL(RGB)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    RGB_to_HSL(RGB * factor), HSL * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_HSL(self):
        """
        Tests :func:`colour.models.rgb.deprecated.RGB_to_HSL` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_HSL(RGB)


class TestHSL_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.deprecated.HSL_to_RGB` definition unit
    tests methods.
    """

    def test_HSL_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.deprecated.HSL_to_RGB` definition.
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

    def test_n_dimensional_HSL_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.deprecated.HSL_to_RGB` definition
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
        Tests :func:`colour.models.rgb.deprecated.HSL_to_RGB` definition domain
        and range scale support.
        """

        HSL = np.array([0.99603944, 0.87347144, 0.24350795])
        RGB = HSL_to_RGB(HSL)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    HSL_to_RGB(HSL * factor), RGB * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_HSL_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.deprecated.HSL_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            HSL = np.array(case)
            HSL_to_RGB(HSL)


class TestRGB_to_CMY(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.deprecated.RGB_to_CMY` definition unit
    tests methods.
    """

    def test_RGB_to_CMY(self):
        """
        Tests :func:`colour.models.rgb.deprecated.RGB_to_CMY` definition.
        """

        np.testing.assert_almost_equal(
            RGB_to_CMY(np.array([0.45620519, 0.03081071, 0.04091952])),
            np.array([0.54379481, 0.96918929, 0.95908048]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_CMY(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([1.00000000, 1.00000000, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_CMY(np.array([1.00000000, 1.00000000, 1.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7)

    def test_n_dimensional_RGB_to_CMY(self):
        """
        Tests :func:`colour.models.rgb.deprecated.RGB_to_CMY` definition
        n-dimensional arrays support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        CMY = RGB_to_CMY(RGB)

        RGB = np.tile(RGB, (6, 1))
        CMY = np.tile(CMY, (6, 1))
        np.testing.assert_almost_equal(RGB_to_CMY(RGB), CMY, decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        CMY = np.reshape(CMY, (2, 3, 3))
        np.testing.assert_almost_equal(RGB_to_CMY(RGB), CMY, decimal=7)

    def test_domain_range_scale_RGB_to_CMY(self):
        """
        Tests :func:`colour.models.rgb.deprecated.RGB_to_CMY` definition domain
        and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        CMY = RGB_to_CMY(RGB)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    RGB_to_CMY(RGB * factor), CMY * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_CMY(self):
        """
        Tests :func:`colour.models.rgb.deprecated.RGB_to_CMY` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_CMY(RGB)


class TestCMY_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.deprecated.CMY_to_RGB` definition unit
    tests methods.
    """

    def test_CMY_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.deprecated.CMY_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            CMY_to_RGB(np.array([0.54379481, 0.96918929, 0.95908048])),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

        np.testing.assert_almost_equal(
            CMY_to_RGB(np.array([1.00000000, 1.00000000, 1.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            CMY_to_RGB(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([1.00000000, 1.00000000, 1.00000000]),
            decimal=7)

    def test_n_dimensional_CMY_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.deprecated.CMY_to_RGB` definition
        n-dimensional arrays support.
        """

        CMY = np.array([0.54379481, 0.96918929, 0.95908048])
        RGB = CMY_to_RGB(CMY)

        CMY = np.tile(CMY, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_almost_equal(CMY_to_RGB(CMY), RGB, decimal=7)

        CMY = np.reshape(CMY, (2, 3, 3))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_almost_equal(CMY_to_RGB(CMY), RGB, decimal=7)

    def test_domain_range_scale_CMY_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.deprecated.CMY_to_RGB` definition domain
        and range scale support.
        """

        CMY = np.array([0.54379481, 0.96918929, 0.95908048])
        RGB = CMY_to_RGB(CMY)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    CMY_to_RGB(CMY * factor), RGB * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_CMY_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.deprecated.CMY_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            CMY = np.array(case)
            CMY_to_RGB(CMY)


class TestCMY_to_CMYK(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.deprecated.CMY_to_CMYK` definition unit
    tests methods.
    """

    def test_CMY_to_CMYK(self):
        """
        Tests :func:`colour.models.rgb.deprecated.CMY_to_CMYK` definition.
        """

        np.testing.assert_almost_equal(
            CMY_to_CMYK(np.array([0.54379481, 0.96918929, 0.95908048])),
            np.array([0.00000000, 0.93246304, 0.91030457, 0.54379481]),
            decimal=7)

        np.testing.assert_almost_equal(
            CMY_to_CMYK(np.array([0.15000000, 1.00000000, 1.00000000])),
            np.array([0.00000000, 1.00000000, 1.00000000, 0.15000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            CMY_to_CMYK(np.array([0.15000000, 0.00000000, 0.00000000])),
            np.array([0.15000000, 0.00000000, 0.00000000, 0.00000000]),
            decimal=7)

    def test_n_dimensional_CMY_to_CMYK(self):
        """
        Tests :func:`colour.models.rgb.deprecated.CMY_to_CMYK` definition
        n-dimensional arrays support.
        """

        CMY = np.array([0.54379481, 0.96918929, 0.95908048])
        CMYK = CMY_to_CMYK(CMY)

        CMY = np.tile(CMY, (6, 1))
        CMYK = np.tile(CMYK, (6, 1))
        np.testing.assert_almost_equal(CMY_to_CMYK(CMY), CMYK, decimal=7)

        CMY = np.reshape(CMY, (2, 3, 3))
        CMYK = np.reshape(CMYK, (2, 3, 4))
        np.testing.assert_almost_equal(CMY_to_CMYK(CMY), CMYK, decimal=7)

    def test_domain_range_scale_CMY_to_CMYK(self):
        """
        Tests :func:`colour.models.rgb.deprecated.CMY_to_CMYK` definition
        domain and range scale support.
        """

        CMY = np.array([0.54379481, 0.96918929, 0.95908048])
        CMYK = CMY_to_CMYK(CMY)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    CMY_to_CMYK(CMY * factor), CMYK * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_CMY_to_CMYK(self):
        """
        Tests :func:`colour.models.rgb.deprecated.CMY_to_CMYK` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            CMY = np.array(case)
            CMY_to_CMYK(CMY)


class TestCMYK_to_CMY(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.deprecated.CMYK_to_CMY` definition unit
    tests methods.
    """

    def test_CMYK_to_CMY(self):
        """
        Tests :func:`colour.models.rgb.deprecated.CMYK_to_CMY` definition.
        """

        np.testing.assert_almost_equal(
            CMYK_to_CMY(
                np.array([0.00000000, 0.93246304, 0.91030457, 0.54379481])),
            np.array([0.54379481, 0.96918929, 0.95908048]),
            decimal=7)

        np.testing.assert_almost_equal(
            CMYK_to_CMY(
                np.array([0.00000000, 1.00000000, 1.00000000, 0.15000000])),
            np.array([0.15000000, 1.00000000, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            CMYK_to_CMY(
                np.array([0.15000000, 0.00000000, 0.00000000, 0.00000000])),
            np.array([0.15000000, 0.00000000, 0.00000000]),
            decimal=7)

    def test_n_dimensional_CMYK_to_CMY(self):
        """
        Tests :func:`colour.models.rgb.deprecated.CMYK_to_CMY` definition
        n-dimensional arrays support.
        """

        CMYK = np.array([0.00000000, 0.93246304, 0.91030457, 0.54379481])
        CMY = CMYK_to_CMY(CMYK)

        CMYK = np.tile(CMYK, (6, 1))
        CMY = np.tile(CMY, (6, 1))
        np.testing.assert_almost_equal(CMYK_to_CMY(CMYK), CMY, decimal=7)

        CMYK = np.reshape(CMYK, (2, 3, 4))
        CMY = np.reshape(CMY, (2, 3, 3))
        np.testing.assert_almost_equal(CMYK_to_CMY(CMYK), CMY, decimal=7)

    def test_domain_range_scale_CMYK_to_CMY(self):
        """
        Tests :func:`colour.models.rgb.deprecated.CMYK_to_CMY` definition
        domain and range scale support.
        """

        CMYK = np.array([0.00000000, 0.93246304, 0.91030457, 0.54379481])
        CMY = CMYK_to_CMY(CMYK)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    CMYK_to_CMY(CMYK * factor), CMY * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_CMYK_to_CMY(self):
        """
        Tests :func:`colour.models.rgb.deprecated.CMYK_to_CMY` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=4))
        for case in cases:
            CMYK = np.array(case)
            CMYK_to_CMY(CMYK)


if __name__ == '__main__':
    unittest.main()
