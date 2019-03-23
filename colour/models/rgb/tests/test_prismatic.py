# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.prismatic` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models.rgb import RGB_to_Prismatic, Prismatic_to_RGB
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRGB_to_Prismatic', 'TestPrismatic_to_RGB']


class TestRGB_to_Prismatic(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.prismatic.TestRGB_to_Prismatic` definition
    unit tests methods.
    """

    def test_RGB_to_Prismatic(self):
        """
        Tests :func:`colour.models.rgb.prismatic.RGB_to_Prismatic` definition.
        """

        np.testing.assert_almost_equal(
            RGB_to_Prismatic(np.array([0.0, 0.0, 0.0])),
            np.array([0.0, 0.0, 0.0, 0.0]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_Prismatic(np.array([0.25, 0.50, 0.75])),
            np.array([0.7500000, 0.1666667, 0.3333333, 0.5000000]),
            decimal=7)

    def test_n_dimensional_RGB_to_Prismatic(self):
        """
        Tests :func:`colour.models.rgb.prismatic.RGB_to_Prismatic` definition
        n-dimensions support.
        """

        RGB = np.array([0.25, 0.50, 0.75])
        Lrgb = np.array([0.7500000, 0.1666667, 0.3333333, 0.5000000])
        np.testing.assert_almost_equal(RGB_to_Prismatic(RGB), Lrgb, decimal=7)

        RGB = np.tile(RGB, (6, 1))
        Lrgb = np.tile(Lrgb, (6, 1))
        np.testing.assert_almost_equal(RGB_to_Prismatic(RGB), Lrgb, decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        Lrgb = np.reshape(Lrgb, (2, 3, 4))
        np.testing.assert_almost_equal(RGB_to_Prismatic(RGB), Lrgb, decimal=7)

    def test_domain_range_scale_RGB_to_Prismatic(self):
        """
        Tests :func:`colour.models.rgb.prismatic.RGB_to_Prismatic` definition
        domain and range scale support.
        """

        RGB = np.array([0.25, 0.50, 0.75])
        Lrgb = RGB_to_Prismatic(RGB)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    RGB_to_Prismatic(RGB * factor), Lrgb * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_Prismatic(self):
        """
        Tests :func:`colour.models.rgb.prismatic.RGB_to_Prismatic` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_Prismatic(RGB)


class TestPrismatic_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.prismatic.Prismatic_to_RGB` definition
    unit tests methods.
    """

    def test_Prismatic_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.prismatic.Prismatic_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            Prismatic_to_RGB(np.array([0.0, 0.0, 0.0, 0.0])),
            np.array([0.0, 0.0, 0.0]),
            decimal=7)

        np.testing.assert_almost_equal(
            Prismatic_to_RGB(
                np.array([0.7500000, 0.1666667, 0.3333333, 0.5000000])),
            np.array([0.25, 0.50, 0.75]),
            decimal=7)

    def test_n_dimensional_Prismatic_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.prismatic.Prismatic_to_RGB` definition
        n-dimensions support.
        """

        Lrgb = np.array([0.7500000, 0.1666667, 0.3333333, 0.5000000])
        RGB = np.array([0.25, 0.50, 0.75])
        np.testing.assert_almost_equal(Prismatic_to_RGB(Lrgb), RGB, decimal=7)

        Lrgb = np.tile(Lrgb, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_almost_equal(Prismatic_to_RGB(Lrgb), RGB, decimal=7)

        Lrgb = np.reshape(Lrgb, (2, 3, 4))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_almost_equal(Prismatic_to_RGB(Lrgb), RGB, decimal=7)

    def test_domain_range_scale_Prismatic_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.prismatic.Prismatic_to_RGB` definition
        domain and range scale support.
        """

        Lrgb = np.array([0.7500000, 0.1666667, 0.3333333, 0.5000000])
        RGB = Prismatic_to_RGB(Lrgb)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    Prismatic_to_RGB(Lrgb * factor), RGB * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_Prismatic_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.prismatic.Prismatic_to_RGB` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Prismatic = np.array(case)
            Prismatic_to_RGB(Prismatic)


if __name__ == '__main__':
    unittest.main()
