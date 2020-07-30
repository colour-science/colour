# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.hanbury2002` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations
from colour.models.rgb.hanbury2002 import (RGB_to_IHLS, IHLS_to_RGB)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestRGB_to_IHLS', 'TestIHLS_to_RGB']


class TestRGB_to_IHLS(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.hanbury2002.RGB_to_IHLS` definition unit
    tests methods.
    """

    def test_RGB_to_IHLS(self):
        """
        Tests :func:`colour.models.rgb.HLS.RGB_to_HLS` definition.
        """
        np.testing.assert_almost_equal(
            RGB_to_IHLS(np.array([0.45620519, 0.03081071, 0.04091952])),
            np.array([3.59979176e+02, 1.21979433e-01, -1.57572751e-01]),
            decimal=7)
        np.testing.assert_almost_equal(
            RGB_to_IHLS(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([np.nan, 0., 0.]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_IHLS(np.array([1.00000000, 1.00000000, 1.00000000])),
            np.array([np.nan, 1.0, 0.]),
            decimal=7)

    def test_n_dimensional_RGB_to_IHLS(self):
        """
        Tests :func:`colour.models.rgb.hanbury2002.RGB_to_IHLS` definition
        n-dimensional arrays support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        IHLS = RGB_to_IHLS(RGB)

        RGB = np.tile(RGB, (6, 1))
        IHLS = np.tile(IHLS, (6, 1))
        np.testing.assert_almost_equal(RGB_to_IHLS(RGB), IHLS, decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        IHLS = np.reshape(IHLS, (2, 3, 3))
        np.testing.assert_almost_equal(RGB_to_IHLS(RGB), IHLS, decimal=7)

    def test_domain_range_scale_RGB_to_IHLS(self):
        """
        Tests :func:`colour.models.rgb.hanbury2002.RGB_to_IHLS` definition
        domain and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        IHLS = RGB_to_IHLS(RGB)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    RGB_to_IHLS(RGB * factor), IHLS * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_IHLS(self):
        """
        Tests :func:`colour.models.rgb.hanbury2002.RGB_to_IHLS` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_IHLS(RGB)


class TestIHLS_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.hanbury2002.RGB_to_IHLS` definition unit
    tests methods.
    """

    def test_IHLS_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.hanbury2002.IHLS_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            IHLS_to_RGB(np.array([0.99603944, 0.93246304, 0.45620519])),
            np.array([0.80741911, 0.87090447, 1.91218565]),
            decimal=7)

        np.testing.assert_almost_equal(
            IHLS_to_RGB(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([0., 0., 0.]),
            decimal=7)

        np.testing.assert_almost_equal(
            IHLS_to_RGB(np.array([0.00000000, 0.00000000, 1.00000000])),
            np.array([1.17461569, -0.31695979, -0.31695979]),
            decimal=7)

    def test_n_dimensional_IHLS_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.hanbury2002.IHLS_to_RGB` definition
        n-dimensional arrays support.
        """

        IHLS = np.array([0.99603944, 0.93246304, 0.45620519])
        RGB = IHLS_to_RGB(IHLS)

        IHLS = np.tile(IHLS, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_almost_equal(IHLS_to_RGB(IHLS), RGB, decimal=7)

        IHLS = np.reshape(IHLS, (2, 3, 3))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_almost_equal(IHLS_to_RGB(IHLS), RGB, decimal=7)

    def test_domain_range_scale_IHLS_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.hanbury2002.IHLS_to_RGB` definition
        domain and range scale support.
        """

        IHLS = np.array([0.99603944, 0.93246304, 0.45620519])
        RGB = IHLS_to_RGB(IHLS)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    IHLS_to_RGB(IHLS * factor), RGB * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_IHLS_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.hanbury2002.IHLS_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            IHLS = np.array(case)
            IHLS_to_RGB(IHLS)


if __name__ == '__main__':
    unittest.main()
