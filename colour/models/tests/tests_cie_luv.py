#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.cie_luv` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import (
    XYZ_to_Luv,
    Luv_to_XYZ,
    Luv_to_uv,
    Luv_uv_to_xy,
    Luv_to_LCHuv,
    LCHuv_to_Luv)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_Luv',
           'TestLuv_to_XYZ',
           'TestLuv_to_uv',
           'TestLuv_to_LCHuv',
           'TestLCHuv_to_Luv']


class TestXYZ_to_Luv(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.XYZ_to_Luv` definition unit tests
    methods.
    """

    def test_XYZ_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.XYZ_to_Luv` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([0.07049534, 0.1008, 0.09558313])),
            np.array([37.9856291, -28.79229446, -1.3558195]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([0.4709771, 0.3495, 0.11301649])),
            np.array([65.7097188, 87.21709531, 27.01490816]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([0.25506814, 0.1915, 0.08849752])),
            np.array([50.86223896, 60.52359443, 13.14030896]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([0.07049534, 0.1008, 0.09558313]),
                       (0.44757, 0.40745)),
            np.array([37.9856291, -51.90523525, -19.24118281]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([0.07049534, 0.1008, 0.09558313]),
                       (0.31271, 0.32902)),
            np.array([37.9856291, -23.19754103, 8.3936094]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([0.07049534, 0.1008, 0.09558313]),
                       (0.37208, 0.37529)),
            np.array([37.9856291, -34.23840374, -7.09461715]),
            decimal=7)


class TestLuv_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.Luv_to_XYZ` definition unit tests
    methods.
    """

    def test_Luv_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([37.9856291, -28.79229446, -1.3558195])),
            np.array([0.07049534, 0.1008, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([65.7097188, 87.21709531, 27.01490816])),
            np.array([0.4709771, 0.3495, 0.11301649]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([50.86223896, 60.52359443, 13.14030896])),
            np.array([0.25506814, 0.1915, 0.08849752]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([37.9856291, -51.90523525, -19.24118281]),
                       (0.44757, 0.40745)),
            np.array([0.07049534, 0.1008, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([37.9856291, -23.19754103, 8.3936094]),
                       (0.31271, 0.32902)),
            np.array([0.07049534, 0.1008, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([37.9856291, -34.23840374, -7.09461715]),
                       (0.37208, 0.37529)),
            np.array([0.07049534, 0.1008, 0.09558313]),
            decimal=7)


class TestLuv_to_uv(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.Luv_to_uv` definition unit tests
    methods.
    """

    def test_Luv_to_uv(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_uv` definition.
        """

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([37.9856291, -28.79229446, -1.3558195])),
            (0.1508531, 0.48532971),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([65.7097188, 87.21709531, 27.01490816])),
            (0.31125983, 0.51970032),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([50.86223896, 60.52359443, 13.14030896])),
            (0.30069387, 0.50794847),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([37.9856291, -51.90523525, -19.24118281]),
                      (0.44757, 0.40745)),
            (0.1508531, 0.48532971),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([37.9856291, -23.19754103, 8.3936094]),
                      (0.31271, 0.32902)),
            (0.1508531, 0.48532971),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([37.9856291, -34.23840374, -7.09461715]),
                      (0.37208, 0.37529)),
            (0.1508531, 0.48532971),
            decimal=7)


class TestLuv_to_LCHuv(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.Luv_to_LCHuv` definition unit tests
    methods.
    """

    def test_Luv_to_LCHuv(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_LCHuv` definition.
        """

        np.testing.assert_almost_equal(
            Luv_to_LCHuv(np.array([37.9856291, -28.79229446, -1.3558195])),
            np.array([37.9856291, 28.82419933, 182.69604747]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_LCHuv(np.array([65.7097188, 87.21709531, 27.01490816])),
            np.array([65.7097188, 91.30513117, 17.21001524]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_LCHuv(np.array([50.86223896, 60.52359443, 13.14030896])),
            np.array([50.86223896, 61.93361932, 12.24941097]),
            decimal=7)


class TestLCHuv_to_Luv(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.LCHuv_to_Luv` definition unit tests
    methods.
    """

    def test_LCHuv_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.LCHuv_to_Luv` definition.
        """

        np.testing.assert_almost_equal(
            LCHuv_to_Luv(np.array([37.9856291, 28.82419933, 182.69604747])),
            np.array([37.9856291, -28.79229446, -1.3558195]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHuv_to_Luv(np.array([65.7097188, 91.30513117, 17.21001524])),
            np.array([65.7097188, 87.21709531, 27.01490816]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHuv_to_Luv(np.array([50.86223896, 61.93361932, 12.24941097])),
            np.array([50.86223896, 60.52359443, 13.14030896]),
            decimal=7)


class TestLuv_uv_to_xy(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.Luv_uv_to_xy` definition unit tests
    methods.
    """

    def test_Luv_uv_to_xy(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_uv_to_xy` definition.
        """

        np.testing.assert_almost_equal(
            Luv_uv_to_xy((0.1508531, 0.48532971)),
            (0.26414773, 0.37770001),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_uv_to_xy((0.31125983, 0.51970032)),
            (0.50453169, 0.3744),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_uv_to_xy((0.30069387, 0.50794847)),
            (0.47670437, 0.3579),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
