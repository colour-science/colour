#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.notation.triplet` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.notation.triplet import (
    RGB_to_HEX,
    HEX_to_RGB)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRGB_to_HEX',
           'TestHEX_to_RGB']


class TestRGB_to_HEX(unittest.TestCase):
    """
    Defines :func:`colour.notation.triplet.RGB_to_HEX` definition unit tests
    methods.
    """

    def test_RGB_to_HEX(self):
        """
        Tests :func:`colour.notation.triplet.RGB_to_HEX` definition.
        """

        self.assertEqual(
            RGB_to_HEX(np.array([0.25000000, 0.60000000, 0.05000000])),
            '#3f990c')

        self.assertEqual(
            RGB_to_HEX(np.array([0.00000000, 0.00000000, 0.00000000])),
            '#000000')

        self.assertEqual(
            RGB_to_HEX(np.array([1.00000000, 1.00000000, 1.00000000])),
            '#ffffff')

    def test_n_dimensional_RGB_to_HEX(self):
        """
        Tests :func:`colour.notation.triplet.RGB_to_HEX` definition
        n-dimensional arrays support.
        """

        RGB = np.array([0.25000000, 0.60000000, 0.05000000])
        HEX = '#3f990c'
        self.assertEqual(RGB_to_HEX(RGB), HEX)

        RGB = np.tile(RGB, (6, 1))
        HEX = np.tile(HEX, 6)
        self.assertListEqual(RGB_to_HEX(RGB).tolist(), HEX.tolist())

        RGB = np.reshape(RGB, (2, 3, 3))
        HEX = np.reshape(HEX, (2, 3))
        self.assertListEqual(RGB_to_HEX(RGB).tolist(), HEX.tolist())

    @ignore_numpy_errors
    def test_nan_RGB_to_HEX(self):
        """
        Tests :func:`colour.notation.triplet.RGB_to_HEX` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_HEX(RGB)


class TestHEX_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.notation.triplet.HEX_to_RGB` definition unit tests
    methods.
    """

    def test_HEX_to_RGB(self):
        """
        Tests :func:`colour.notation.triplet.HEX_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            HEX_to_RGB('#3f990c'),
            np.array([0.25000000, 0.60000000, 0.05000000]),
            decimal=2)

        np.testing.assert_almost_equal(
            HEX_to_RGB('#000000'),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=2)

        np.testing.assert_almost_equal(
            HEX_to_RGB('#ffffff'),
            np.array([1.00000000, 1.00000000, 1.00000000]),
            decimal=2)

    def test_n_dimensional_HEX_to_RGB(self):
        """
        Tests :func:`colour.notation.triplet.HEX_to_RGB` definition
        n-dimensional arrays support.
        """

        HEX = '#3f990c'
        RGB = np.array([0.25000000, 0.60000000, 0.05000000])
        np.testing.assert_almost_equal(HEX_to_RGB(HEX), RGB, decimal=2)

        HEX = np.tile(HEX, 6)
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_almost_equal(HEX_to_RGB(HEX), RGB, decimal=2)

        HEX = np.reshape(HEX, (2, 3))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_almost_equal(HEX_to_RGB(HEX), RGB, decimal=2)


if __name__ == '__main__':
    unittest.main()
