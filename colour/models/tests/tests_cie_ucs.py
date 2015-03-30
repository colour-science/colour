#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.cie_ucs` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import XYZ_to_UCS, UCS_to_XYZ, UCS_to_uv, UCS_uv_to_xy

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
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
            XYZ_to_UCS(np.array([0.07049534, 0.1008, 0.09558313])),
            np.array([0.04699689, 0.1008, 0.1637439]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UCS(np.array([0.4709771, 0.3495, 0.11301649])),
            np.array([0.31398473, 0.3495, 0.34526969]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UCS(np.array([0.25506814, 0.1915, 0.08849752])),
            np.array([0.17004543, 0.1915, 0.20396469]),
            decimal=7)

    def test_n_dimensions_XYZ_to_UCS(self):
        """
        Tests :func:`colour.models.cie_ucs.XYZ_to_UCS` definition n-dimensions
        support.
        """

        XYZ = np.array([0.07049534, 0.1008, 0.09558313])
        UCS = np.array([0.04699689, 0.1008, 0.1637439])
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
            UCS_to_XYZ(np.array([0.04699689, 0.1008, 0.1637439])),
            np.array([0.07049534, 0.1008, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_XYZ(np.array([0.31398473, 0.3495, 0.34526969])),
            np.array([0.4709771, 0.3495, 0.11301649]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_XYZ(np.array([0.17004543, 0.1915, 0.20396469])),
            np.array([0.25506814, 0.1915, 0.08849752]),
            decimal=7)

    def test_n_dimensions_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_XYZ` definition n-dimensions
        support.
        """

        UCS = np.array([0.04699689, 0.1008, 0.1637439])
        XYZ = np.array([0.07049534, 0.1008, 0.09558313])
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
            UCS_to_uv(np.array([0.04699689, 0.1008, 0.1637439])),
            np.array([0.15085309, 0.32355314]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_uv(np.array([0.31398473, 0.3495, 0.34526969])),
            np.array([0.31125983, 0.34646688]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_to_uv(np.array([0.17004543, 0.1915, 0.20396469])),
            np.array([0.30069388, 0.33863231]),
            decimal=7)

    def test_n_dimensions_UCS_to_uv(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_to_uv` definition n-dimensions
        support.
        """

        UCS = np.array([0.04699689, 0.1008, 0.1637439])
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
            np.array([0.50453169, 0.3744]),
            decimal=7)

        np.testing.assert_almost_equal(
            UCS_uv_to_xy(np.array([0.30069388, 0.33863231])),
            np.array([0.47670437, 0.35789998]),
            decimal=7)

    def test_UCS_uv_to_xy(self):
        """
        Tests :func:`colour.models.cie_ucs.UCS_uv_to_xy` definition.
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


if __name__ == '__main__':
    unittest.main()
