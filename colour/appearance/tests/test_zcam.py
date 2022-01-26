# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.appearance.zcam` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.appearance import (
    VIEWING_CONDITIONS_ZCAM,
    InductionFactors_ZCAM,
    CAM_Specification_ZCAM,
    XYZ_to_ZCAM,
    ZCAM_to_XYZ,
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
    'TestXYZ_to_ZCAM',
    'TestZCAM_to_XYZ',
]


class TestXYZ_to_ZCAM(unittest.TestCase):
    """
    Defines :func:`colour.appearance.zcam.XYZ_to_ZCAM` definition unit tests
    methods.
    """

    def test_XYZ_to_ZCAM(self):
        """
        Tests :func:`colour.appearance.zcam.XYZ_to_ZCAM` definition.
        """

        XYZ = np.array([185, 206, 163])
        XYZ_w = np.array([256, 264, 202])
        L_a = 264
        Y_b = 100
        surround = VIEWING_CONDITIONS_ZCAM['Average']
        np.testing.assert_allclose(
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround),
            np.array([
                92.2520, 3.0216, 196.3524, 19.1314, 321.3464, 10.5252,
                237.6401, np.nan, 34.7022, 25.2994, 91.6837
            ]),
            rtol=0.025,
            atol=0.025)

        XYZ = np.array([89, 96, 120])
        np.testing.assert_allclose(
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround),
            np.array([
                71.2071, 6.8539, 250.6422, 32.7963, 248.0394, 23.8744,
                307.0595, np.nan, 18.2796, 40.4621, 70.4026
            ]),
            rtol=0.025,
            atol=0.025)

        # NOTE: Hue quadrature :math:`H_z` is significantly different for this
        # test, i.e. 47.748252 vs 43.8258.
        # NOTE: :math:`F_L` as reported in the supplemental document has the
        # same value as for :math:`L_a` = 264 instead of 150. The values seem
        # to be computed for :math:`L_a` = 264 and :math:`Y_b` = 100.
        XYZ = np.array([79, 81, 62])
        # L_a = 150
        # Y_b = 60
        surround = VIEWING_CONDITIONS_ZCAM['Dim']
        np.testing.assert_allclose(
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround),
            np.array([
                68.8890, 0.9774, 58.7532, 12.5916, 196.7686, 2.7918, 43.8258,
                np.nan, 11.0371, 44.4143, 68.8737
            ]),
            rtol=0.025,
            atol=4)

        XYZ = np.array([910, 1114, 500])
        XYZ_w = np.array([2103, 2259, 1401])
        L_a = 359
        Y_b = 16
        surround = VIEWING_CONDITIONS_ZCAM['Dark']
        np.testing.assert_allclose(
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround),
            np.array([
                82.6445, 13.0838, 123.9464, 44.7277, 114.7431, 18.1655,
                178.6422, np.nan, 34.4874, 26.8778, 78.2653
            ]),
            rtol=0.025,
            atol=0.025)

        XYZ = np.array([96, 67, 28])
        np.testing.assert_allclose(
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround),
            np.array([
                33.0139, 19.4070, 389.7720 % 360, 86.1882, 45.8363, 26.9446,
                397.3301, np.nan, 43.6447, 47.9942, 30.2593
            ]),
            rtol=0.025,
            atol=0.025)

    def test_n_dimensional_XYZ_to_ZCAM(self):
        """
        Tests :func:`colour.appearance.zcam.XYZ_to_ZCAM` definition
        n-dimensional support.
        """

        XYZ = np.array([185, 206, 163])
        XYZ_w = np.array([256, 264, 202])
        L_a = 264
        Y_b = 100
        surround = VIEWING_CONDITIONS_ZCAM['Average']
        specification = XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround)

        XYZ = np.tile(XYZ, (6, 1))
        specification = np.tile(specification, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround),
            specification,
            decimal=7)

        XYZ_w = np.tile(XYZ_w, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround),
            specification,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        specification = np.reshape(specification, (2, 3, 11))
        np.testing.assert_almost_equal(
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround),
            specification,
            decimal=7)

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_ZCAM(self):
        """
        Tests :func:`colour.appearance.zcam.XYZ_to_ZCAM` definition
        domain and range scale support.
        """

        XYZ = np.array([185, 206, 163])
        XYZ_w = np.array([256, 264, 202])
        L_a = 264
        Y_b = 100
        surround = VIEWING_CONDITIONS_ZCAM['Average']
        specification = XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround)

        d_r = (
            ('reference', 1, 1),
            ('1', 1,
             np.array([1, 1, 1 / 360, 1, 1, 1, 1 / 400, np.nan, 1, 1, 1])),
            ('100', 100,
             np.array([
                 100, 100, 100 / 360, 100, 100, 100, 100 / 400, np.nan, 100,
                 100, 100
             ])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_ZCAM(XYZ * factor_a, XYZ_w * factor_a, L_a, Y_b,
                                surround),
                    as_float_array(specification) * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_ZCAM(self):
        """
        Tests :func:`colour.appearance.zcam.XYZ_to_ZCAM` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_w = np.array(case)
            L_a = case[0]
            Y_b = 100
            surround = InductionFactors_ZCAM(case[0], case[0], case[0],
                                             case[0])
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround)


class TestZCAM_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.appearance.zcam.ZCAM_to_XYZ` definition unit
    tests methods.
    """

    def test_ZCAM_to_XYZ(self):
        """
        Tests :func:`colour.appearance.zcam.ZCAM_to_XYZ` definition.
        """

        specification = CAM_Specification_ZCAM(
            92.2520, 3.0216, 196.3524, 19.1314, 321.3464, 10.5252, 237.6401,
            np.nan, 34.7022, 25.2994, 91.6837)
        XYZ_w = np.array([256, 264, 202])
        L_a = 264
        Y_b = 100
        surround = VIEWING_CONDITIONS_ZCAM['Average']
        np.testing.assert_allclose(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            np.array([185, 206, 163]),
            atol=0.01,
            rtol=0.01)

        specification = CAM_Specification_ZCAM(
            71.2071, 6.8539, 250.6422, 32.7963, 248.0394, 23.8744, 307.0595,
            np.nan, 18.2796, 40.4621, 70.4026)
        np.testing.assert_allclose(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            np.array([89, 96, 120]),
            atol=0.01,
            rtol=0.01)

        specification = CAM_Specification_ZCAM(
            68.8890, 0.9774, 58.7532, 12.5916, 196.7686, 2.7918, 43.8258,
            np.nan, 11.0371, 44.4143, 68.8737)
        surround = VIEWING_CONDITIONS_ZCAM['Dim']
        np.testing.assert_allclose(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            np.array([79, 81, 62]),
            atol=0.01,
            rtol=0.01)

        specification = CAM_Specification_ZCAM(
            82.6445, 13.0838, 123.9464, 44.7277, 114.7431, 18.1655, 178.6422,
            np.nan, 34.4874, 26.8778, 78.2653)
        XYZ_w = np.array([2103, 2259, 1401])
        L_a = 359
        Y_b = 16
        surround = VIEWING_CONDITIONS_ZCAM['Dark']
        np.testing.assert_allclose(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            np.array([910, 1114, 500]),
            atol=0.01,
            rtol=0.01)

        specification = CAM_Specification_ZCAM(
            33.0139, 19.4070, 389.7720 % 360, 86.1882, 45.8363, 26.9446,
            397.3301, np.nan, 43.6447, 47.9942, 30.2593)
        np.testing.assert_allclose(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            np.array([96, 67, 28]),
            atol=0.01,
            rtol=0.01)

    def test_n_dimensional_ZCAM_to_XYZ(self):
        """
        Tests :func:`colour.appearance.zcam.ZCAM_to_XYZ` definition
        n-dimensional support.
        """

        XYZ = np.array([185, 206, 163])
        XYZ_w = np.array([256, 264, 202])
        L_a = 264
        Y_b = 100
        surround = VIEWING_CONDITIONS_ZCAM['Average']
        specification = XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround)
        XYZ = ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround)

        specification = CAM_Specification_ZCAM(
            *np.transpose(np.tile(tsplit(specification), (6, 1))).tolist())
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            XYZ,
            decimal=7)

        XYZ_w = np.tile(XYZ_w, (6, 1))
        np.testing.assert_almost_equal(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            XYZ,
            decimal=7)

        specification = CAM_Specification_ZCAM(
            *tsplit(np.reshape(specification, (2, 3, 11))).tolist())
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            XYZ,
            decimal=7)

    @ignore_numpy_errors
    def test_domain_range_scale_ZCAM_to_XYZ(self):
        """
        Tests :func:`colour.appearance.zcam.ZCAM_to_XYZ` definition
        domain and range scale support.
        """

        XYZ_i = np.array([185, 206, 163])
        XYZ_w = np.array([256, 264, 202])
        L_a = 264
        Y_b = 100
        surround = VIEWING_CONDITIONS_ZCAM['Average']
        specification = XYZ_to_ZCAM(XYZ_i, XYZ_w, L_a, Y_b, surround)
        XYZ = ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround)

        d_r = (
            ('reference', 1, 1),
            ('1', np.array([1, 1, 1 / 360, 1, 1, 1, 1 / 400, np.nan, 1, 1, 1]),
             1),
            ('100',
             np.array([
                 100, 100, 100 / 360, 100, 100, 100, 100 / 400, np.nan, 100,
                 100, 100
             ]), 100),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    ZCAM_to_XYZ(specification * factor_a, XYZ_w * factor_b,
                                L_a, Y_b, surround),
                    XYZ * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_raise_exception_ZCAM_to_XYZ(self):
        """
        Tests :func:`colour.appearance.zcam.ZCAM_to_XYZ` definition
        raised exception.
        """

        self.assertRaises(
            ValueError, ZCAM_to_XYZ,
            CAM_Specification_ZCAM(
                41.731091132513917,
                None,
                219.04843265831178,
            ), np.array([256, 264, 202]), 318.31, 20.0,
            VIEWING_CONDITIONS_ZCAM['Average'])

    @ignore_numpy_errors
    def test_nan_ZCAM_to_XYZ(self):
        """
        Tests :func:`colour.appearance.zcam.ZCAM_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            J = case[0]
            C = case[0]
            h = case[0]
            XYZ_w = np.array(case)
            L_a = case[0]
            Y_b = 100
            surround = InductionFactors_ZCAM(case[0], case[0], case[0],
                                             case[0])
            ZCAM_to_XYZ(
                CAM_Specification_ZCAM(J, C, h, M=50), XYZ_w, L_a, Y_b,
                surround)
