#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.cie_xyy` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import XYZ_to_xyY, xyY_to_XYZ, xy_to_XYZ, XYZ_to_xy

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_xyY',
           'TestxyY_to_XYZ',
           'Testxy_to_XYZ',
           'TestXYZ_to_xy']


class TestXYZ_to_xyY(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.XYZ_to_xyY` definition unit tests
    methods.
    """

    def test_XYZ_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xyY` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0.07049534, 0.1008, 0.09558313])),
            np.array([0.26414772, 0.37770001, 0.1008]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0.4709771, 0.3495, 0.11301649])),
            np.array([0.50453169, 0.3744, 0.3495]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0.25506814, 0.1915, 0.08849752])),
            np.array([0.47670437, 0.3579, 0.1915]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0, 0, 0])),
            np.array([0.34567, 0.3585, 0.]),
            decimal=7)


class TestxyY_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.xyY_to_XYZ` definition unit tests
    methods.
    """

    def test_xyY_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.26414772, 0.37770001, 0.1008])),
            np.array([0.07049534, 0.1008, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.50453169, 0.3744, 0.3495])),
            np.array([0.4709771, 0.3495, 0.11301649]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.47670437, 0.3579, 0.1915])),
            np.array([0.25506814, 0.1915, 0.08849752]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.34567, 0.3585, 0.])),
            np.array([0, 0, 0]),
            decimal=7)


class Testxy_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.xy_to_XYZ` definition unit tests
    methods.
    """

    def test_xy_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            xy_to_XYZ((0.26414772, 0.37770001)),
            np.array([0.69935852, 1., 0.94824533]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_XYZ((0.50453169, 0.3744)),
            np.array([1.34757396, 1., 0.32336621]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_XYZ((0.47670437, 0.3579)),
            np.array([1.33194851, 1., 0.46212805]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_XYZ((0.34567, 0.3585)),
            np.array([0.96421199, 1., 0.82518828]),
            decimal=7)


class TestXYZ_to_xy(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.XYZ_to_xy` definition unit tests
    methods.
    """

    def test_XYZ_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xy` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_xy(np.array([0.07049534, 0.1008, 0.09558313])),
            (0.26414772, 0.37770001),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xy(np.array([0.4709771, 0.3495, 0.11301649])),
            (0.50453169, 0.3744),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xy(np.array([0.25506814, 0.1915, 0.08849752])),
            (0.47670437, 0.3579),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xy(np.array([0, 0, 0])),
            (0.34567, 0.3585),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
