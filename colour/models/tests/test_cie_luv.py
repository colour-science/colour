# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.cie_luv` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import (XYZ_to_Luv, Luv_to_XYZ, Luv_to_uv, uv_to_Luv,
                           Luv_uv_to_xy, xy_to_Luv_uv, Luv_to_LCHuv,
                           LCHuv_to_Luv)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_Luv', 'TestLuv_to_XYZ', 'TestLuv_to_uv', 'Testuv_to_Luv',
    'TestLuv_uv_to_xy', 'TestXy_to_Luv_uv', 'TestLuv_to_LCHuv',
    'TestLCHuv_to_Luv'
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
            XYZ_to_Luv(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([41.52787529, 96.83626054, 17.75210149]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([55.11636304, -37.59308176, 44.13768458]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([29.80565520, -10.96316802, -65.06751860]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.44757, 0.40745])),
            np.array([41.52787529, 65.45180940, -12.46626977]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.34570, 0.35850])),
            np.array([41.52787529, 90.70925962, 7.08455273]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.34570, 0.35850, 1.00000])),
            np.array([41.52787529, 90.70925962, 7.08455273]),
            decimal=7)

    def test_n_dimensional_XYZ_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.XYZ_to_Luv` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
        Luv = XYZ_to_Luv(XYZ, illuminant)

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

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
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
            Luv_to_XYZ(np.array([41.52787529, 96.83626054, 17.75210149])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([55.11636304, -37.59308176, 44.13768458])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([29.80565520, -10.96316802, -65.06751860])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(
                np.array([41.52787529, 65.45180940, -12.46626977]),
                np.array([0.44757, 0.40745])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(
                np.array([41.52787529, 90.70925962, 7.08455273]),
                np.array([0.34570, 0.35850])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(
                np.array([41.52787529, 90.70925962, 7.08455273]),
                np.array([0.34570, 0.35850, 1.00000])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

    def test_n_dimensional_Luv_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_XYZ` definition n-dimensional
        support.
        """

        Luv = np.array([41.52787529, 96.83626054, 17.75210149])
        illuminant = np.array([0.31270, 0.32900])
        XYZ = Luv_to_XYZ(Luv, illuminant)

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

        Luv = np.array([41.52787529, 96.83626054, 17.75210149])
        illuminant = np.array([0.31270, 0.32900])
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
            Luv_to_uv(np.array([41.52787529, 96.83626054, 17.75210149])),
            np.array([0.37720213, 0.50120264]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([55.11636304, -37.59308176, 44.13768458])),
            np.array([0.14536327, 0.52992069]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([29.80565520, -10.96316802, -65.06751860])),
            np.array([0.16953603, 0.30039234]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(
                np.array([41.52787529, 65.45180940, -12.46626977]),
                np.array([0.44757, 0.40745])),
            np.array([0.37720213, 0.50120264]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(
                np.array([41.52787529, 90.70925962, 7.08455273]),
                np.array([0.34570, 0.35850])),
            np.array([0.37720213, 0.50120264]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(
                np.array([41.52787529, 90.70925962, 7.08455273]),
                np.array([0.34570, 0.35850, 1.00000])),
            np.array([0.37720213, 0.50120264]),
            decimal=7)

    def test_n_dimensional_Luv_to_uv(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_uv` definition n-dimensional
        support.
        """

        Luv = np.array([41.52787529, 96.83626054, 17.75210149])
        illuminant = np.array([0.31270, 0.32900])
        uv = Luv_to_uv(Luv, illuminant)

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

        Luv = np.array([41.52787529, 96.83626054, 17.75210149])
        illuminant = np.array([0.31270, 0.32900])
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


class Testuv_to_Luv(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.uv_to_Luv` definition unit tests
    methods.
    """

    def test_uv_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.uv_to_Luv` definition.
        """

        np.testing.assert_almost_equal(
            uv_to_Luv(np.array([0.37720213, 0.50120264])),
            np.array([100.00000000, 233.18376036, 42.74743858]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_Luv(np.array([0.14536327, 0.52992069])),
            np.array([100.00000000, -68.20675764, 80.08090358]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_Luv(np.array([0.16953603, 0.30039234])),
            np.array([100.00000000, -36.78216964, -218.3059514]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_Luv(
                np.array([0.37720213, 0.50120264]),
                np.array([0.44757, 0.40745])),
            np.array([100.00000000, 157.60933976, -30.01903705]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_Luv(
                np.array([0.37720213, 0.50120264]),
                np.array([0.34570, 0.35850])),
            np.array([100.00000000, 218.42981284, 17.05975609]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_Luv(
                np.array([0.37720213, 0.50120264]),
                np.array([0.34570, 0.35850, 1.00000])),
            np.array([100.00000000, 218.42981284, 17.05975609]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_Luv(np.array([0.37720213, 0.50120264]), Y=0.18),
            np.array([49.49610761, 115.41688496, -243.29048251]),
            decimal=7)

    def test_n_dimensional_uv_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.uv_to_Luv` definition n-dimensional
        support.
        """

        uv = np.array([0.37720213, 0.50120264])
        illuminant = np.array([0.31270, 0.32900])
        Luv = uv_to_Luv(uv, illuminant)

        uv = np.tile(uv, (6, 1))
        Luv = np.tile(Luv, (6, 1))
        np.testing.assert_almost_equal(
            uv_to_Luv(uv, illuminant), Luv, decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            uv_to_Luv(uv, illuminant), Luv, decimal=7)

        uv = np.reshape(uv, (2, 3, 2))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        Luv = np.reshape(Luv, (2, 3, 3))
        np.testing.assert_almost_equal(
            uv_to_Luv(uv, illuminant), Luv, decimal=7)

    def test_domain_range_scale_uv_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.uv_to_Luv` definition
        domain and range scale support.
        """

        uv = np.array([0.37720213, 0.50120264])
        illuminant = np.array([0.31270, 0.32900])
        Y = 1
        Luv = uv_to_Luv(uv, illuminant, Y)

        d_r = (('reference', 1, 1), (1, 1, 0.01), (100, 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    uv_to_Luv(uv, illuminant, Y * factor_a),
                    Luv * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_uv_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.uv_to_Luv` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            uv = np.array(case)
            illuminant = np.array(case[0:2])
            uv_to_Luv(uv, illuminant)


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
            Luv_uv_to_xy(np.array([0.37720213, 0.50120264])),
            np.array([0.54369558, 0.32107944]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_uv_to_xy(np.array([0.14536327, 0.52992069])),
            np.array([0.29777734, 0.48246445]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_uv_to_xy(np.array([0.16953603, 0.30039234])),
            np.array([0.18582824, 0.14633764]),
            decimal=7)

    def test_n_dimensional_Luv_uv_to_xy(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_uv_to_xy` definition
        n-dimensional arrays support.
        """

        uv = np.array([0.37720213, 0.50120264])
        xy = Luv_uv_to_xy(uv)

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
            xy_to_Luv_uv(np.array([0.54369558, 0.32107944])),
            np.array([0.37720213, 0.50120264]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_Luv_uv(np.array([0.29777734, 0.48246445])),
            np.array([0.14536327, 0.52992069]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_Luv_uv(np.array([0.18582824, 0.14633764])),
            np.array([0.16953603, 0.30039234]),
            decimal=7)

    def test_n_dimensional_xy_to_Luv_uv(self):
        """
        Tests :func:`colour.models.cie_luv.xy_to_Luv_uv` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.54369558, 0.32107944])
        uv = xy_to_Luv_uv(xy)

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
            Luv_to_LCHuv(np.array([41.52787529, 96.83626054, 17.75210149])),
            np.array([41.52787529, 98.44997950, 10.38816348]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_LCHuv(np.array([55.11636304, -37.59308176, 44.13768458])),
            np.array([55.11636304, 57.97736624, 130.42180076]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_LCHuv(np.array([29.80565520, -10.96316802, -65.06751860])),
            np.array([29.80565520, 65.98464238, 260.43611196]),
            decimal=7)

    def test_n_dimensional_Luv_to_LCHuv(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_LCHuv` definition
        n-dimensional arrays support.
        """

        Luv = np.array([41.52787529, 96.83626054, 17.75210149])
        LCHuv = Luv_to_LCHuv(Luv)

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

        Luv = np.array([41.52787529, 96.83626054, 17.75210149])
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
            LCHuv_to_Luv(np.array([41.52787529, 98.44997950, 10.38816348])),
            np.array([41.52787529, 96.83626054, 17.75210149]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHuv_to_Luv(np.array([55.11636304, 57.97736624, 130.42180076])),
            np.array([55.11636304, -37.59308176, 44.13768458]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHuv_to_Luv(np.array([29.80565520, 65.98464238, 260.43611196])),
            np.array([29.80565520, -10.96316802, -65.06751860]),
            decimal=7)

    def test_n_dimensional_LCHuv_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.LCHuv_to_Luv` definition
        n-dimensional arrays support.
        """

        LCHuv = np.array([41.52787529, 98.44997950, 10.38816348])
        Luv = LCHuv_to_Luv(LCHuv)

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

        LCHuv = np.array([41.52787529, 98.44997950, 10.38816348])
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
