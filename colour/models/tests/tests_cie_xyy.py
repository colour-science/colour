#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.cie_xyy` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import (
    XYZ_to_xyY,
    xyY_to_XYZ,
    xy_to_xyY,
    xyY_to_xy,
    xy_to_XYZ,
    XYZ_to_xy)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_xyY',
           'TestxyY_to_XYZ',
           'Testxy_to_xyY',
           'TestxyY_to_xy',
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
            XYZ_to_xyY(np.array([0.07049534, 0.10080000, 0.09558313])),
            np.array([0.26414772, 0.37770001, 0.10080000]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0.47097710, 0.34950000, 0.11301649])),
            np.array([0.50453169, 0.37440000, 0.34950000]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0.25506814, 0.19150000, 0.08849752])),
            np.array([0.47670437, 0.35790000, 0.19150000]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([0.34570000, 0.35850000, 0.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0.00000000, 0.00000000, 1.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([[0.07049534, 0.10080000, 0.09558313],
                                 [0.00000000, 0.00000000, 0.00000000],
                                 [0.00000000, 0.00000000, 1.00000000]])),
            np.array([[0.26414772, 0.37770001, 0.10080000],
                      [0.34570000, 0.35850000, 0.00000000],
                      [0.00000000, 0.00000000, 0.00000000]]),
            decimal=7)

    def test_n_dimensional_XYZ_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xyY` definition n-dimensions
        support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        illuminant = np.array([0.34570, 0.35850])
        xyY = np.array([0.26414772, 0.37770001, 0.10080000])
        np.testing.assert_almost_equal(
            XYZ_to_xyY(XYZ, illuminant),
            xyY,
            decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        xyY = np.tile(xyY, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_xyY(XYZ, illuminant),
            xyY,
            decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_xyY(XYZ, illuminant),
            xyY,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        xyY = np.reshape(xyY, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_xyY(XYZ, illuminant),
            xyY,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xyY` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            illuminant = np.array(case[0:2])
            XYZ_to_xyY(XYZ, illuminant)


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
            xyY_to_XYZ(np.array([0.26414772, 0.37770001, 0.10080000])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.50453169, 0.37440000, 0.34950000])),
            np.array([0.47097710, 0.34950000, 0.11301649]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.47670437, 0.35790000, 0.19150000])),
            np.array([0.25506814, 0.19150000, 0.08849752]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.34567, 0.3585, 0.00000000])),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            decimal=7)

    def test_n_dimensional_xyY_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_XYZ` definition n-dimensions
        support.
        """

        xyY = np.array([0.26414772, 0.37770001, 0.10080000])
        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        np.testing.assert_almost_equal(
            xyY_to_XYZ(xyY),
            XYZ,
            decimal=7)

        xyY = np.tile(xyY, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            xyY_to_XYZ(xyY),
            XYZ,
            decimal=7)

        xyY = np.reshape(xyY, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            xyY_to_XYZ(xyY),
            XYZ,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_xyY_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_XYZ` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            xyY = np.array(case)
            xyY_to_XYZ(xyY)


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
            xy_to_XYZ(np.array([0.26414772, 0.37770001])),
            np.array([0.69935852, 1.00000000, 0.94824533]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_XYZ(np.array([0.50453169, 0.37440000])),
            np.array([1.34757396, 1.00000000, 0.32336621]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_XYZ(np.array([0.47670437, 0.35790000])),
            np.array([1.33194851, 1.00000000, 0.46212805]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_XYZ(np.array([0.34570000, 0.35850000])),
            np.array([0.96429568, 1.00000000, 0.82510460]),
            decimal=7)

    def test_n_dimensional_xy_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_XYZ` definition n-dimensions
        support.
        """

        xy = np.array([0.26414772, 0.37770001])
        XYZ = np.array([0.69935852, 1.00000000, 0.94824533])
        np.testing.assert_almost_equal(
            xy_to_XYZ(xy),
            XYZ,
            decimal=7)

        xy = np.tile(xy, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            xy_to_XYZ(xy),
            XYZ,
            decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            xy_to_XYZ(xy),
            XYZ,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_xy_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_XYZ` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            xy = np.array(case)
            xy_to_XYZ(xy)


class Testxy_to_xyY(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.xy_to_xyY` definition unit tests
    methods.
    """

    def test_xy_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_xyY` definition.
        """

        np.testing.assert_almost_equal(
            xy_to_xyY(np.array([0.26414772, 0.37770001])),
            np.array([0.26414772, 0.37770001, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_xyY(np.array([0.50453169, 0.37440000])),
            np.array([0.50453169, 0.37440000, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_xyY(np.array([0.47670437, 0.35790000])),
            np.array([0.47670437, 0.35790000, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_xyY(np.array([0.34570000, 0.35850000, 0.10080000])),
            np.array([0.34570000, 0.35850000, 0.10080000]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_xyY(np.array([0.34570000, 0.35850000]), 100),
            np.array([0.34570000, 0.35850000, 100.00000000]),
            decimal=7)

    def test_n_dimensional_xy_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_xyY` definition n-dimensions
        support.
        """

        xy = np.array([0.26414772, 0.37770001])
        XYZ = np.array([0.26414772, 0.37770001, 1.00000000])
        np.testing.assert_almost_equal(
            xy_to_xyY(xy),
            XYZ,
            decimal=7)

        xy = np.tile(xy, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            xy_to_xyY(xy),
            XYZ,
            decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            xy_to_xyY(xy),
            XYZ,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_xy_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_xyY` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            xy = np.array(case)
            xy_to_xyY(xy)


class TestxyY_to_xy(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.xyY_to_xy` definition unit tests
    methods.
    """

    def test_xyY_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_xy` definition.
        """

        np.testing.assert_almost_equal(
            xyY_to_xy(np.array([0.26414772, 0.37770001, 1.00000000])),
            np.array([0.26414772, 0.37770001]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_xy(np.array([0.50453169, 0.37440000, 1.00000000])),
            np.array([0.50453169, 0.37440000]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_xy(np.array([0.47670437, 0.35790000, 1.00000000])),
            np.array([0.47670437, 0.35790000]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_xy(np.array([0.34570, 0.35850])),
            np.array([0.34570000, 0.35850000]),
            decimal=7)

    def test_n_dimensional_xyY_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_xy` definition n-dimensions
        support.
        """

        xyY = np.array([0.26414772, 0.37770001, 1.00000000])
        xy = np.array([0.26414772, 0.37770001])
        np.testing.assert_almost_equal(
            xyY_to_xy(xyY),
            xy,
            decimal=7)

        xyY = np.tile(xyY, (6, 1))
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(
            xyY_to_xy(xyY),
            xy,
            decimal=7)

        xyY = np.reshape(xyY, (2, 3, 3))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(
            xyY_to_xy(xyY),
            xy,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_xyY_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_xy` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            xyY = np.array(case)
            xyY_to_xy(xyY)


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
            XYZ_to_xy(np.array([0.07049534, 0.10080000, 0.09558313])),
            np.array([0.26414772, 0.37770001]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xy(np.array([0.47097710, 0.34950000, 0.11301649])),
            np.array([0.50453169, 0.37440000]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xy(np.array([0.25506814, 0.19150000, 0.08849752])),
            np.array([0.47670437, 0.35790000]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xy(np.array([0.00000000, 0.00000000, 0.00000000])),
            np.array([0.34570000, 0.35850000]),
            decimal=7)

    def test_n_dimensional_XYZ_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xy` definition n-dimensions
        support.
        """

        XYZ = np.array([0.69935853, 1.00000000, 0.94824534])
        illuminant = np.array([0.34570, 0.35850])
        xy = np.array([0.26414772, 0.37770001])
        np.testing.assert_almost_equal(
            XYZ_to_xy(XYZ, illuminant),
            xy,
            decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_xy(XYZ, illuminant),
            xy,
            decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_xy(XYZ, illuminant),
            xy,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(xy, (2, 3, 2))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(
            XYZ_to_xy(XYZ, illuminant),
            xy,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xy` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            illuminant = np.array(case[0:2])
            XYZ_to_xy(XYZ, illuminant)


if __name__ == '__main__':
    unittest.main()
