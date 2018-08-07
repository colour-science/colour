# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.cie_luv` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import (XYZ_to_Luv, Luv_to_XYZ, Luv_to_uv, Luv_uv_to_xy,
                           xy_to_Luv_uv, Luv_to_LCHuv, LCHuv_to_Luv)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_Luv', 'TestLuv_to_XYZ', 'TestLuv_to_uv', 'TestLuv_uv_to_xy',
    'TestXy_to_Luv_uv', 'TestLuv_to_LCHuv', 'TestLCHuv_to_Luv'
]


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
            XYZ_to_Luv(np.array([0.07049534, 0.10080000, 0.09558313])),
            np.array([37.98562910, -28.80219593, -1.35800706]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([0.47097710, 0.34950000, 0.11301649])),
            np.array([65.70971880, 87.19996716, 27.01112399]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([0.25506814, 0.19150000, 0.08849752])),
            np.array([50.86223896, 60.51033649, 13.13737985]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(
                np.array([0.07049534, 0.10080000, 0.09558313]),
                np.array([0.44757, 0.40745])),
            np.array([37.98562910, -51.90523525, -19.24118281]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(
                np.array([0.07049534, 0.10080000, 0.09558313]),
                np.array([0.31270, 0.32900])),
            np.array([37.98562910, -23.19781615, 8.39962073]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(
                np.array([0.07049534, 0.10080000, 0.09558313]),
                np.array([0.37208, 0.37529])),
            np.array([37.98562910, -34.23840374, -7.09461715]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(
                np.array([0.07049534, 0.10080000, 0.09558313]),
                np.array([0.37208, 0.37529, 0.10080])),
            np.array([100.00000000, -90.13514992, -18.67710847]),
            decimal=7)

    def test_n_dimensional_XYZ_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.XYZ_to_Luv` definition n-dimensions
        support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        illuminant = np.array([0.34570, 0.35850])
        Luv = np.array([37.98562910, -28.80219593, -1.35800706])
        np.testing.assert_almost_equal(
            XYZ_to_Luv(XYZ, illuminant), Luv, decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        Luv = np.tile(Luv, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Luv(XYZ, illuminant), Luv, decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Luv(XYZ, illuminant), Luv, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        Luv = np.reshape(Luv, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_Luv(XYZ, illuminant), Luv, decimal=7)

    def test_domain_range_scale_XYZ_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.XYZ_to_Luv` definition
        domain and range scale support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        illuminant = np.array([0.34570, 0.35850])
        Luv = XYZ_to_Luv(XYZ, illuminant)

        d_r = (('reference', 1, 1), (1, 1, 0.01), (100, 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_Luv(XYZ * factor_a, illuminant),
                    Luv * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.XYZ_to_Luv` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            illuminant = np.array(case[0:2])
            XYZ_to_Luv(XYZ, illuminant)


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
            Luv_to_XYZ(np.array([37.98562910, -28.80219593, -1.35800706])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([65.70971880, 87.19996716, 27.01112399])),
            np.array([0.47097710, 0.34950000, 0.11301649]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([50.86223896, 60.51033649, 13.13737985])),
            np.array([0.25506814, 0.19150000, 0.08849752]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(
                np.array([37.98562910, -51.90523525, -19.24118281]),
                np.array([0.44757, 0.40745])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(
                np.array([37.98562910, -23.19781615, 8.39962073]),
                np.array([0.31270, 0.32900])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(
                np.array([37.98562910, -34.23840374, -7.09461715]),
                np.array([0.37208, 0.37529])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(
                np.array([37.98562910, -34.23840374, -7.09461715]),
                np.array([0.37208, 0.37529, 0.10080])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

    def test_n_dimensional_Luv_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_XYZ` definition n-dimensions
        support.
        """

        Luv = np.array([37.98562910, -28.80219593, -1.35800706])
        illuminant = np.array([0.34570, 0.35850])
        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        np.testing.assert_almost_equal(
            Luv_to_XYZ(Luv, illuminant), XYZ, decimal=7)

        Luv = np.tile(Luv, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            Luv_to_XYZ(Luv, illuminant), XYZ, decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            Luv_to_XYZ(Luv, illuminant), XYZ, decimal=7)

        Luv = np.reshape(Luv, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            Luv_to_XYZ(Luv, illuminant), XYZ, decimal=7)

    def test_domain_range_scale_Luv_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_XYZ` definition
        domain and range scale support.
        """

        Luv = np.array([37.98562910, -28.80219593, -1.35800706])
        illuminant = np.array([0.34570, 0.35850])
        XYZ = Luv_to_XYZ(Luv, illuminant)

        d_r = (('reference', 1, 1), (1, 0.01, 1), (100, 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    Luv_to_XYZ(Luv * factor_a, illuminant),
                    XYZ * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_Luv_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_XYZ` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Luv = np.array(case)
            illuminant = np.array(case[0:2])
            Luv_to_XYZ(Luv, illuminant)


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
            Luv_to_uv(np.array([37.98562910, -28.80219593, -1.35800706])),
            np.array([0.15085310, 0.48532971]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([65.70971880, 87.19996716, 27.01112399])),
            np.array([0.31125983, 0.51970032]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([50.86223896, 60.51033649, 13.13737985])),
            np.array([0.30069387, 0.50794847]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(
                np.array([37.98562910, -51.90523525, -19.24118281]),
                np.array([0.44757, 0.40745])),
            np.array([0.15085310, 0.48532971]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(
                np.array([37.98562910, -23.19781615, 8.39962073]),
                np.array([0.31270, 0.32900])),
            np.array([0.15085310, 0.48532971]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(
                np.array([37.98562910, -34.23840374, -7.09461715]),
                np.array([0.37208, 0.37529])),
            np.array([0.15085310, 0.48532971]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(
                np.array([37.98562910, -34.23840374, -7.09461715]),
                np.array([0.37208, 0.37529, 0.10080])),
            np.array([0.15085310, 0.48532971]),
            decimal=7)

    def test_n_dimensional_Luv_to_uv(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_uv` definition n-dimensions
        support.
        """

        Luv = np.array([37.98562910, -28.80219593, -1.35800706])
        illuminant = np.array([0.34570, 0.35850])
        uv = np.array([0.15085310, 0.48532971])
        np.testing.assert_almost_equal(
            Luv_to_uv(Luv, illuminant), uv, decimal=7)

        Luv = np.tile(Luv, (6, 1))
        uv = np.tile(uv, (6, 1))
        np.testing.assert_almost_equal(
            Luv_to_uv(Luv, illuminant), uv, decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            Luv_to_uv(Luv, illuminant), uv, decimal=7)

        Luv = np.reshape(Luv, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_almost_equal(
            Luv_to_uv(Luv, illuminant), uv, decimal=7)

    def test_domain_range_scale_Luv_to_uv(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_uv` definition
        domain and range scale support.
        """

        Luv = np.array([37.98562910, -28.80219593, -1.35800706])
        illuminant = np.array([0.34570, 0.35850])
        uv = Luv_to_uv(Luv, illuminant)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    Luv_to_uv(Luv * factor, illuminant), uv, decimal=7)

    @ignore_numpy_errors
    def test_nan_Luv_to_uv(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_uv` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Luv = np.array(case)
            illuminant = np.array(case[0:2])
            Luv_to_uv(Luv, illuminant)


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
            Luv_uv_to_xy(np.array([0.15085310, 0.48532971])),
            np.array([0.26414773, 0.37770001]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_uv_to_xy(np.array([0.31125983, 0.51970032])),
            np.array([0.50453169, 0.37440000]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_uv_to_xy(np.array([0.30069387, 0.50794847])),
            np.array([0.47670437, 0.35790000]),
            decimal=7)

    def test_n_dimensional_Luv_uv_to_xy(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_uv_to_xy` definition
        n-dimensional arrays support.
        """

        uv = np.array([0.15085310, 0.48532971])
        xy = np.array([0.26414773, 0.37770001])
        np.testing.assert_almost_equal(Luv_uv_to_xy(uv), xy, decimal=7)

        uv = np.tile(uv, (6, 1))
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(Luv_uv_to_xy(uv), xy, decimal=7)

        uv = np.reshape(uv, (2, 3, 2))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(Luv_uv_to_xy(uv), xy, decimal=7)

    @ignore_numpy_errors
    def test_nan_Luv_uv_to_xy(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_uv_to_xy` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            uv = np.array(case)
            Luv_uv_to_xy(uv)


class TestXy_to_Luv_uv(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.xy_to_Luv_uv` definition unit tests
    methods.
    """

    def test_xy_to_Luv_uv(self):
        """
        Tests :func:`colour.models.cie_luv.xy_to_Luv_uv` definition.
        """

        np.testing.assert_almost_equal(
            xy_to_Luv_uv(np.array([0.26414773, 0.37770001])),
            np.array([0.15085310, 0.48532971]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_Luv_uv(np.array([0.50453169, 0.37440000])),
            np.array([0.31125983, 0.51970032]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_Luv_uv(np.array([0.47670437, 0.35790000])),
            np.array([0.30069387, 0.50794847]),
            decimal=7)

    def test_n_dimensional_xy_to_Luv_uv(self):
        """
        Tests :func:`colour.models.cie_luv.xy_to_Luv_uv` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.26414773, 0.37770001])
        uv = np.array([0.15085310, 0.48532971])
        np.testing.assert_almost_equal(xy_to_Luv_uv(xy), uv, decimal=7)

        xy = np.tile(xy, (6, 1))
        uv = np.tile(uv, (6, 1))
        np.testing.assert_almost_equal(xy_to_Luv_uv(xy), uv, decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_almost_equal(xy_to_Luv_uv(xy), uv, decimal=7)

    @ignore_numpy_errors
    def test_nan_xy_to_Luv_uv(self):
        """
        Tests :func:`colour.models.cie_luv.xy_to_Luv_uv` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            xy = np.array(case)
            xy_to_Luv_uv(xy)


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
            Luv_to_LCHuv(np.array([37.98562910, -28.80219593, -1.35800706])),
            np.array([37.98562910, 28.83419279, 182.69946404]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_LCHuv(np.array([65.70971880, 87.19996716, 27.01112399])),
            np.array([65.70971880, 91.28765027, 17.21092723]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_LCHuv(np.array([50.86223896, 60.51033649, 13.13737985])),
            np.array([50.86223896, 61.92004176, 12.24936515]),
            decimal=7)

    def test_n_dimensional_Luv_to_LCHuv(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_LCHuv` definition
        n-dimensional arrays support.
        """

        Luv = np.array([37.98562910, -28.80219593, -1.35800706])
        LCHuv = np.array([37.98562910, 28.83419279, 182.69946404])
        np.testing.assert_almost_equal(Luv_to_LCHuv(Luv), LCHuv, decimal=7)

        Luv = np.tile(Luv, (6, 1))
        LCHuv = np.tile(LCHuv, (6, 1))
        np.testing.assert_almost_equal(Luv_to_LCHuv(Luv), LCHuv, decimal=7)

        Luv = np.reshape(Luv, (2, 3, 3))
        LCHuv = np.reshape(LCHuv, (2, 3, 3))
        np.testing.assert_almost_equal(Luv_to_LCHuv(Luv), LCHuv, decimal=7)

    def test_domain_range_scale_Luv_to_LCHuv(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_LCHuv` definition domain and
        range scale support.
        """

        Luv = np.array([37.98562910, -28.80219593, -1.35800706])
        LCHuv = Luv_to_LCHuv(Luv)

        d_r = (('reference', 1, 1), (1, 0.01, np.array([0.01, 0.01, 1 / 360])),
               (100, 1, np.array([1, 1, 1 / 3.6])))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    Luv_to_LCHuv(Luv * factor_a), LCHuv * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_Luv_to_LCHuv(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_LCHuv` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Luv = np.array(case)
            Luv_to_LCHuv(Luv)


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
            LCHuv_to_Luv(np.array([37.98562910, 28.83419279, 182.69946404])),
            np.array([37.98562910, -28.80219593, -1.35800706]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHuv_to_Luv(np.array([65.70971880, 91.28765027, 17.21092723])),
            np.array([65.70971880, 87.19996716, 27.01112399]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHuv_to_Luv(np.array([50.86223896, 61.92004176, 12.24936515])),
            np.array([50.86223896, 60.51033649, 13.13737985]),
            decimal=7)

    def test_n_dimensional_LCHuv_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.LCHuv_to_Luv` definition
        n-dimensional arrays support.
        """

        LCHuv = np.array([37.98562910, 28.83419279, 182.69946404])
        Luv = np.array([37.98562910, -28.80219593, -1.35800706])
        np.testing.assert_almost_equal(LCHuv_to_Luv(LCHuv), Luv, decimal=7)

        Luv = np.tile(Luv, (6, 1))
        LCHuv = np.tile(LCHuv, (6, 1))
        np.testing.assert_almost_equal(LCHuv_to_Luv(LCHuv), Luv, decimal=7)

        Luv = np.reshape(Luv, (2, 3, 3))
        LCHuv = np.reshape(LCHuv, (2, 3, 3))
        np.testing.assert_almost_equal(LCHuv_to_Luv(LCHuv), Luv, decimal=7)

    def test_domain_range_scale_LCHuv_to_Lab(self):
        """
        Tests :func:`colour.models.cie_luv.LCHuv_to_Luv` definition domain and
        range scale support.
        """

        LCHuv = np.array([37.98562910, 28.83419279, 182.69946404])
        Luv = LCHuv_to_Luv(LCHuv)

        d_r = (('reference', 1, 1), (1, np.array([0.01, 0.01, 1 / 360]), 0.01),
               (100, np.array([1, 1, 1 / 3.6]), 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    LCHuv_to_Luv(LCHuv * factor_a), Luv * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_LCHuv_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.LCHuv_to_Luv` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            LCHuv = np.array(case)
            LCHuv_to_Luv(LCHuv)


if __name__ == '__main__':
    unittest.main()
