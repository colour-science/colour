# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.ycocg` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models.rgb import RGB_to_YCoCg, YCoCg_to_RGB
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Development'

__all__ = ['TestRGB_to_YCoCg', 'TestYCoCg_to_RGB']


class TestRGB_to_YCoCg(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ycocg.RGB_to_YCoCg` definition unit tests
    methods.
    """

    def test_RGB_to_YCoCg(self):
        """
        Tests :func:`colour.models.rgb.ycocg.RGB_to_YCoCg` definition.
        """

        np.testing.assert_array_equal(
            RGB_to_YCoCg(np.array([0.75, 0.75, 0.0])),
            np.array([0.5625, 0.375, 0.1875]))

        np.testing.assert_array_equal(
            RGB_to_YCoCg(np.array([0.25, 0.5, 0.75])),
            np.array([0.5, -0.25, 0.0]))

        np.testing.assert_array_equal(
            RGB_to_YCoCg(np.array([0.0, 0.75, 0.75])),
            np.array([0.5625, -0.375, 0.1875]))

    def test_n_dimensional_RGB_to_YCoCg(self):
        """
        Tests :func:`colour.models.rgb.ycocg.RGB_to_YCoCg` definition
        n-dimensional arrays support.
        """

        RGB = np.array([0.75, 0.75, 0.0])
        YCoCg = np.array([0.5625, 0.375, 0.1875])
        np.testing.assert_array_equal(RGB_to_YCoCg(RGB), YCoCg)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 3))
        YCoCg = np.tile(YCoCg, 4)
        YCoCg = np.reshape(YCoCg, (4, 3))
        np.testing.assert_array_equal(RGB_to_YCoCg(RGB), YCoCg)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 4, 3))
        YCoCg = np.tile(YCoCg, 4)
        YCoCg = np.reshape(YCoCg, (4, 4, 3))
        np.testing.assert_array_equal(RGB_to_YCoCg(RGB), YCoCg)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 4, 4, 3))
        YCoCg = np.tile(YCoCg, 4)
        YCoCg = np.reshape(YCoCg, (4, 4, 4, 3))
        np.testing.assert_array_equal(RGB_to_YCoCg(RGB), YCoCg)

    @ignore_numpy_errors
    def test_nan_RGB_to_YCoCg(self):
        """
        Tests :func:`colour.models.rgb.ycocg.RGB_to_YCoCg` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_YCoCg(RGB)


class TestYCoCg_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.ycocg.YCoCg_to_RGB` definition unit tests
    methods.
    """

    def test_YCoCg_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ycocg.YCoCg_to_RGB` definition.
        """

        np.testing.assert_array_equal(
            YCoCg_to_RGB(np.array([0.5625, 0.375, 0.1875])),
            np.array([0.75, 0.75, 0.0]))

        np.testing.assert_array_equal(
            YCoCg_to_RGB(np.array([0.5, -0.25, 0.0])),
            np.array([0.25, 0.5, 0.75]))

        np.testing.assert_array_equal(
            YCoCg_to_RGB(np.array([0.5625, -0.375, 0.1875])),
            np.array([0.0, 0.75, 0.75]))

    def test_n_dimensional_YCoCg_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ycocg.YCoCg_to_RGB` definition
        n-dimensional arrays support.
        """

        YCoCg = np.array([0.5625, 0.375, 0.1875])
        RGB = np.array([0.75, 0.75, 0.0])
        np.testing.assert_array_equal(YCoCg_to_RGB(YCoCg), RGB)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 3))
        YCoCg = np.tile(YCoCg, 4)
        YCoCg = np.reshape(YCoCg, (4, 3))
        np.testing.assert_array_equal(YCoCg_to_RGB(YCoCg), RGB)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 4, 3))
        YCoCg = np.tile(YCoCg, 4)
        YCoCg = np.reshape(YCoCg, (4, 4, 3))
        np.testing.assert_array_equal(YCoCg_to_RGB(YCoCg), RGB)

        RGB = np.tile(RGB, 4)
        RGB = np.reshape(RGB, (4, 4, 4, 3))
        YCoCg = np.tile(YCoCg, 4)
        YCoCg = np.reshape(YCoCg, (4, 4, 4, 3))
        np.testing.assert_array_equal(YCoCg_to_RGB(YCoCg), RGB)

    @ignore_numpy_errors
    def test_nan_YCoCg_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.ycocg.YCoCg_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            YCoCg = np.array(case)
            YCoCg_to_RGB(YCoCg)


if __name__ == '__main__':
    unittest.main()
