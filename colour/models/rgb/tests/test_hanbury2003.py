# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.hanbury2003` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.models.rgb import RGB_to_IHLS, IHLS_to_RGB
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestRGB_to_IHLS',
    'TestIHLS_to_RGB',
]


class TestRGB_to_IHLS(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.hanbury2003.RGB_to_IHLS` definition unit
    tests methods.
    """

    def test_RGB_to_IHLS(self):
        """
        Tests :func:`colour.models.rgb.hanbury2003.RGB_to_IHLS` definition.
        """

        np.testing.assert_almost_equal(
            RGB_to_IHLS(np.array([0.45620519, 0.03081071, 0.04091952])),
            np.array([6.26236117, 0.12197943, 0.42539448]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_IHLS(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_IHLS(np.array([1.00000000, 1.00000000, 1.00000000])),
            np.array([0.00000000, 1.00000000, 0.00000000]),
            decimal=7)

    def test_n_dimensional_RGB_to_IHLS(self):
        """
        Tests :func:`colour.models.rgb.hanbury2003.RGB_to_IHLS` definition
        n-dimensional arrays support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HYS = RGB_to_IHLS(RGB)

        RGB = np.tile(RGB, (6, 1))
        HYS = np.tile(HYS, (6, 1))
        np.testing.assert_almost_equal(RGB_to_IHLS(RGB), HYS, decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        HYS = np.reshape(HYS, (2, 3, 3))
        np.testing.assert_almost_equal(RGB_to_IHLS(RGB), HYS, decimal=7)

    def test_domain_range_scale_RGB_to_IHLS(self):
        """
        Tests :func:`colour.models.rgb.hanbury2003.RGB_to_IHLS` definition
        domain and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HYS = RGB_to_IHLS(RGB)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    RGB_to_IHLS(RGB * factor), HYS * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_IHLS(self):
        """
        Tests :func:`colour.models.rgb.hanbury2003.RGB_to_IHLS` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_IHLS(RGB)


class TestIHLS_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.hanbury2003.RGB_to_IHLS` definition unit
    tests methods.
    """

    def test_IHLS_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.hanbury2003.IHLS_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            IHLS_to_RGB(np.array([6.26236117, 0.12197943, 0.42539448])),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            decimal=7)

        np.testing.assert_almost_equal(
            IHLS_to_RGB(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            IHLS_to_RGB(np.array([0.00000000, 1.00000000, 0.00000000])),
            np.array([1.00000000, 1.00000000, 1.00000000]),
            decimal=7)

    def test_n_dimensional_IHLS_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.hanbury2003.IHLS_to_RGB` definition
        n-dimensional arrays support.
        """

        HYS = np.array([6.26236117, 0.12197943, 0.42539448])
        RGB = IHLS_to_RGB(HYS)

        HYS = np.tile(HYS, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_almost_equal(IHLS_to_RGB(HYS), RGB, decimal=7)

        HYS = np.reshape(HYS, (2, 3, 3))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_almost_equal(IHLS_to_RGB(HYS), RGB, decimal=7)

    def test_domain_range_scale_IHLS_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.hanbury2003.IHLS_to_RGB` definition
        domain and range scale support.
        """

        HYS = np.array([6.26236117, 0.12197943, 0.42539448])
        RGB = IHLS_to_RGB(HYS)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    IHLS_to_RGB(HYS * factor), RGB * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_IHLS_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.hanbury2003.IHLS_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            HYS = np.array(case)
            IHLS_to_RGB(HYS)


if __name__ == '__main__':
    unittest.main()
