# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.appearance.ciecam02` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.appearance import (
    VIEWING_CONDITIONS_CIECAM02,
    InductionFactors_CIECAM02,
    CAM_Specification_CIECAM02,
    XYZ_to_CIECAM02,
    CIECAM02_to_XYZ,
)
from colour.utilities import (
    as_float_array,
    domain_range_scale,
    ignore_numpy_errors,
    tsplit,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_CIECAM02',
    'TestCIECAM02_to_XYZ',
]


class TestXYZ_to_CIECAM02(unittest.TestCase):
    """
    Defines :func:`colour.appearance.ciecam02.XYZ_to_CIECAM02` definition unit
    tests methods.
    """

    def test_XYZ_to_CIECAM02(self):
        """
        Tests :func:`colour.appearance.ciecam02.XYZ_to_CIECAM02` definition.

        Notes
        -----
        -   The test values have been generated from data of the following file
            by *Fairchild (2013)*:
            http://rit-mcsl.org/fairchild//files/AppModEx.xls
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = InductionFactors_CIECAM02(1, 0.69, 1)
        np.testing.assert_allclose(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array([41.73, 0.1, 219, 2.36, 195.37, 0.11, 278.1, np.nan]),
            rtol=0.01,
            atol=0.01)

        XYZ = np.array([57.06, 43.06, 31.96])
        L_A = 31.83
        np.testing.assert_allclose(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array([65.96, 48.57, 19.6, 52.25, 152.67, 41.67, 399.6,
                      np.nan]),
            rtol=0.01,
            atol=0.01)

        XYZ = np.array([3.53, 6.56, 2.14])
        XYZ_w = np.array([109.85, 100.00, 35.58])
        L_A = 318.31
        np.testing.assert_allclose(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array([21.79, 46.94, 177.1, 58.79, 141.17, 48.8, 220.4,
                      np.nan]),
            rtol=0.01,
            atol=0.01)

        XYZ = np.array([19.01, 20.00, 21.78])
        L_A = 31.83
        np.testing.assert_allclose(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array(
                [42.53, 51.92, 248.9, 60.22, 122.83, 44.54, 305.8, np.nan]),
            rtol=0.01,
            atol=0.01)

        XYZ = np.array([61.45276998, 7.00421901, 82.24067384])
        XYZ_w = np.array([95.05, 100, 108.88])
        L_A = 4.074366543152521
        np.testing.assert_allclose(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array([
                21.72630603341673, 411.5190338631848, 349.12875710099053,
                227.15081998415593, 57.657243286322725, 297.49693233026602,
                375.5788601911363, np.nan
            ]),
            rtol=0.01,
            atol=0.01)

    def test_n_dimensional_XYZ_to_CIECAM02(self):
        """
        Tests :func:`colour.appearance.ciecam02.XYZ_to_CIECAM02` definition
        n-dimensional support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CIECAM02['Average']
        specification = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)

        XYZ = np.tile(XYZ, (6, 1))
        specification = np.tile(specification, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround),
            specification,
            decimal=7)

        XYZ_w = np.tile(XYZ_w, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround),
            specification,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        specification = np.reshape(specification, (2, 3, 8))
        np.testing.assert_almost_equal(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround),
            specification,
            decimal=7)

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_CIECAM02(self):
        """
        Tests :func:`colour.appearance.ciecam02.XYZ_to_CIECAM02` definition
        domain and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CIECAM02['Average']
        specification = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)

        d_r = (
            ('reference', 1, 1),
            ('1', 0.01,
             np.array([
                 1 / 100, 1 / 100, 1 / 360, 1 / 100, 1 / 100, 1 / 100, 1 / 400,
                 np.nan
             ])),
            ('100', 1, np.array([1, 1, 100 / 360, 1, 1, 1, 100 / 400,
                                 np.nan])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_CIECAM02(XYZ * factor_a, XYZ_w * factor_a, L_A, Y_b,
                                    surround),
                    as_float_array(specification) * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_CIECAM02(self):
        """
        Tests :func:`colour.appearance.ciecam02.XYZ_to_CIECAM02` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_w = np.array(case)
            L_A = case[0]
            Y_b = case[0]
            surround = InductionFactors_CIECAM02(case[0], case[0], case[0])
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)


class TestCIECAM02_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.appearance.ciecam02.CIECAM02_to_XYZ` definition unit
    tests methods.
    """

    def test_CIECAM02_to_XYZ(self):
        """
        Tests :func:`colour.appearance.ciecam02.CIECAM02_to_XYZ` definition.
        """

        specification = CAM_Specification_CIECAM02(41.73, 0.1, 219, 2.36,
                                                   195.37, 0.11, 278.1)
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = InductionFactors_CIECAM02(1, 0.69, 1)
        np.testing.assert_allclose(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([19.01, 20.00, 21.78]),
            rtol=0.01,
            atol=0.01)

        specification = CAM_Specification_CIECAM02(
            65.96, 48.57, 19.6, 52.25, 152.67, 41.67, 399.6, np.nan)
        L_A = 31.83
        np.testing.assert_allclose(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([57.06, 43.06, 31.96]),
            rtol=0.01,
            atol=0.01)

        specification = CAM_Specification_CIECAM02(21.79, 46.94, 177.1, 58.79,
                                                   141.17, 48.8, 220.4, np.nan)
        XYZ_w = np.array([109.85, 100.00, 35.58])
        L_A = 318.31
        np.testing.assert_allclose(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([3.53, 6.56, 2.14]),
            rtol=0.01,
            atol=0.01)

        specification = CAM_Specification_CIECAM02(
            42.53, 51.92, 248.9, 60.22, 122.83, 44.54, 305.8, np.nan)
        L_A = 31.83
        np.testing.assert_allclose(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([19.01, 20.00, 21.78]),
            rtol=0.01,
            atol=0.01)

        specification = CAM_Specification_CIECAM02(
            21.72630603341673, 411.5190338631848, 349.12875710099053,
            227.15081998415593, 57.657243286322725, 297.49693233026602,
            375.5788601911363, np.nan)
        XYZ_w = np.array([95.05, 100, 108.88])
        L_A = 4.074366543152521
        np.testing.assert_allclose(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([61.45276998, 7.00421901, 82.24067384]),
            rtol=0.01,
            atol=0.01)

    def test_n_dimensional_CIECAM02_to_XYZ(self):
        """
        Tests :func:`colour.appearance.ciecam02.CIECAM02_to_XYZ` definition
        n-dimensional support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CIECAM02['Average']
        specification = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround)
        XYZ = CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround)

        specification = CAM_Specification_CIECAM02(
            *np.transpose(np.tile(tsplit(specification), (6, 1))).tolist())
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            decimal=7)

        XYZ_w = np.tile(XYZ_w, (6, 1))
        np.testing.assert_almost_equal(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            decimal=7)

        specification = CAM_Specification_CIECAM02(
            *tsplit(np.reshape(specification, (2, 3, 8))).tolist())
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            decimal=7)

    @ignore_numpy_errors
    def test_domain_range_scale_CIECAM02_to_XYZ(self):
        """
        Tests :func:`colour.appearance.ciecam02.CIECAM02_to_XYZ` definition
        domain and range scale support.
        """

        XYZ_i = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CIECAM02['Average']
        specification = XYZ_to_CIECAM02(XYZ_i, XYZ_w, L_A, Y_b, surround)
        XYZ = CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround)

        d_r = (
            ('reference', 1, 1),
            ('1',
             np.array([
                 1 / 100, 1 / 100, 1 / 360, 1 / 100, 1 / 100, 1 / 100, 1 / 400,
                 np.nan
             ]), 0.01),
            ('100', np.array([1, 1, 100 / 360, 1, 1, 1, 100 / 400, np.nan]),
             1),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    CIECAM02_to_XYZ(specification * factor_a, XYZ_w * factor_b,
                                    L_A, Y_b, surround),
                    XYZ * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_raise_exception_CIECAM02_to_XYZ(self):
        """
        Tests :func:`colour.appearance.ciecam02.CIECAM02_to_XYZ` definition
        raised exception.
        """

        self.assertRaises(
            ValueError, CIECAM02_to_XYZ,
            CAM_Specification_CIECAM02(41.731091132513917, None,
                                       219.04843265831178),
            np.array([95.05, 100.00, 108.88]), 318.31, 20.0,
            VIEWING_CONDITIONS_CIECAM02['Average'])

    @ignore_numpy_errors
    def test_nan_CIECAM02_to_XYZ(self):
        """
        Tests :func:`colour.appearance.ciecam02.CIECAM02_to_XYZ` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            J = case[0]
            C = case[0]
            h = case[0]
            XYZ_w = np.array(case)
            L_A = case[0]
            Y_b = case[0]
            surround = InductionFactors_CIECAM02(case[0], case[0], case[0])
            CIECAM02_to_XYZ(
                CAM_Specification_CIECAM02(J, C, h, M=50), XYZ_w, L_A, Y_b,
                surround)
