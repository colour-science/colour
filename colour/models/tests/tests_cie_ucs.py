#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.cie_ucs` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_UCS, UCS_to_XYZ, UCS_to_uv, UCS_uv_to_xy
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_UCS',
           'TestUCS_to_XYZ',
           'TestUCS_to_uv',
           'TestUCS_uv_to_xy']


class TestXYZ_to_UCS(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_ucs.XYZ_to_UCS` definition unit tests
    methods.
    """

    def test_XYZ_to_UCS(self):
        """
        Tests :func:`colour.models.cie_ucs.XYZ_to_UCS` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_UCS(np.array([0.07049534, 0.10080000, 0.09558313])),
            np.array([0.04699689, 0.10080000, 0.1637439]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UCS(np.array([0.47097710, 0.34950000, 0.11301649])),
            np.array([0.31398473, 0.34950000, 0.34526969]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UCS(np.array([0.25506814, 0.19150000, 0.08849752])),
            np.array([0.17004543, 0.19150000, 0.20396469]),
            decimal=7)

    def test_n_dimensional_XYZ_to_UCS(self):
        """
        Tests :func:`colour.models.cie_ucs.XYZ_to_UCS` definition n-dimensions
        support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        UCS = np.array([0.04699689, 0.10080000, 0.16374390])
        np.testing.assert_almost_equal(
            XYZ_to_UCS(XYZ),
            UCS,
            decimal=7)

        UCS = np.tile(UCS, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_UCS(XYZ),
            UCS,
            decimal=7)

        UCS = np.reshape(UCS, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_UCS(XYZ),
            UCS,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_UCS(self):
        """
        Tests :func:`colour.models.cie_ucs.XYZ_to_UCS` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_to_UCS(XYZ)


class TestUCS_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_ucs.UCS_to_XYZ` definition unit tests
    methods.
    """

    def test_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            UCS_to_XYZ(np.array([0.04699689, 0.10080000, 0.16374390])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_XYZ(np.array([0.31398473, 0.34950000, 0.34526969])),
            np.array([0.47097710, 0.34950000, 0.11301649]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_XYZ(np.array([0.17004543, 0.19150000, 0.20396469])),
            np.array([0.25506814, 0.19150000, 0.08849752]),
            decimal=7)

    def test_n_dimensional_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_XYZ` definition n-dimensions
        support.
        """

        UCS = np.array([0.04699689, 0.10080000, 0.16374390])
        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        np.testing.assert_almost_equal(
            UCS_to_XYZ(UCS),
            XYZ,
            decimal=7)

        UCS = np.tile(UCS, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            UCS_to_XYZ(UCS),
            XYZ,
            decimal=7)

        UCS = np.reshape(UCS, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            UCS_to_XYZ(UCS),
            XYZ,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_XYZ` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            UCS = np.array(case)
            UCS_to_XYZ(UCS)


class TestUCS_to_uv(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_ucs.UCS_to_uv` definition unit tests
    methods.
    """

    def test_UCS_to_uv(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_uv` definition.
        """

        np.testing.assert_almost_equal(
            UCS_to_uv(np.array([0.04699689, 0.10080000, 0.16374390])),
            np.array([0.15085309, 0.32355314]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_uv(np.array([0.31398473, 0.34950000, 0.34526969])),
            np.array([0.31125983, 0.34646688]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_uv(np.array([0.17004543, 0.19150000, 0.20396469])),
            np.array([0.30069388, 0.33863231]),
            decimal=7)

    def test_n_dimensional_UCS_to_uv(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_uv` definition n-dimensions
        support.
        """

        UCS = np.array([0.04699689, 0.10080000, 0.16374390])
        uv = np.array([0.15085309, 0.32355314])
        np.testing.assert_almost_equal(
            UCS_to_uv(UCS),
            uv,
            decimal=7)

        UCS = np.tile(UCS, (6, 1))
        uv = np.tile(uv, (6, 1))
        np.testing.assert_almost_equal(
            UCS_to_uv(UCS),
            uv,
            decimal=7)

        UCS = np.reshape(UCS, (2, 3, 3))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_almost_equal(
            UCS_to_uv(UCS),
            uv,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_UCS_to_uv(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_uv` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            UCS = np.array(case)
            UCS_to_uv(UCS)


class TestUCS_uv_to_xy(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_ucs.UCS_uv_to_xy` definition unit tests
    methods.
    """

    def test_UCS_uv_to_xy(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_uv_to_xy` definition.
        """

        np.testing.assert_almost_equal(
            UCS_uv_to_xy(np.array([0.15085309, 0.32355314])),
            np.array([0.26414771, 0.37770001]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_uv_to_xy(np.array([0.31125983, 0.34646688])),
            np.array([0.50453169, 0.37440000]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_uv_to_xy(np.array([0.30069388, 0.33863231])),
            np.array([0.47670437, 0.35789998]),
            decimal=7)

    def test_n_dimensional_UCS_uv_to_xy(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_uv_to_xy` definition
        n-dimensional arrays support.
        """

        uv = np.array([0.15085309, 0.32355314])
        xy = np.array([0.26414771, 0.37770001])
        np.testing.assert_almost_equal(
            UCS_uv_to_xy(uv),
            xy,
            decimal=7)

        uv = np.tile(uv, (6, 1))
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(
            UCS_uv_to_xy(uv),
            xy,
            decimal=7)

        uv = np.reshape(uv, (2, 3, 2))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(
            UCS_uv_to_xy(uv),
            xy,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_UCS_uv_to_xy(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_uv_to_xy` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            uv = np.array(case)
            UCS_uv_to_xy(uv)


if __name__ == '__main__':
    unittest.main()
