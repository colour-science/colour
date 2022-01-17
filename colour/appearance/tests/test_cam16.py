# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.appearance.cam16` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.appearance import (
    VIEWING_CONDITIONS_CAM16,
    InductionFactors_CAM16,
    CAM_Specification_CAM16,
    XYZ_to_CAM16,
    CAM16_to_XYZ,
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
    'TestXYZ_to_CAM16',
    'TestCAM16_to_XYZ',
]


class TestXYZ_to_CAM16(unittest.TestCase):
    """
    Defines :func:`colour.appearance.cam16.XYZ_to_CAM16` definition unit
    tests methods.
    """

    def test_XYZ_to_CAM16(self):
        """
        Tests :func:`colour.appearance.cam16.XYZ_to_CAM16` definition.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CAM16['Average']
        np.testing.assert_almost_equal(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array([
                41.73120791, 0.10335574, 217.06795977, 2.34501507,
                195.37170899, 0.10743677, 275.59498615, np.nan
            ]),
            decimal=7)

        XYZ = np.array([57.06, 43.06, 31.96])
        L_A = 31.83
        np.testing.assert_almost_equal(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array([
                65.42828069, 49.67956420, 17.48659243, 52.94308868,
                152.06985268, 42.62473321, 398.03047943, np.nan
            ]),
            decimal=7)

        XYZ = np.array([3.53, 6.56, 2.14])
        XYZ_w = np.array([109.85, 100, 35.58])
        L_A = 318.31
        np.testing.assert_almost_equal(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array([
                21.36052893, 50.99381895, 178.86724266, 61.57953092,
                139.78582768, 53.00732582, 223.01823806, np.nan
            ]),
            decimal=7)

        XYZ = np.array([19.01, 20.00, 21.78])
        L_A = 318.31
        np.testing.assert_almost_equal(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array([
                41.36326063, 52.81154022, 258.88676291, 53.12406914,
                194.52011798, 54.89682038, 311.24768647, np.nan
            ]),
            decimal=7)

        XYZ = np.array([61.45276998, 7.00421901, 82.2406738])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 4.074366543152521
        np.testing.assert_almost_equal(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround),
            np.array([
                21.03801957, 457.78881613, 350.06445098, 241.50642846,
                56.74143988, 330.94646237, 376.43915877, np.nan
            ]),
            decimal=7)

    def test_n_dimensional_XYZ_to_CAM16(self):
        """
        Tests :func:`colour.appearance.cam16.XYZ_to_CAM16` definition
        n-dimensional support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CAM16['Average']
        specification = XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround)

        XYZ = np.tile(XYZ, (6, 1))
        specification = np.tile(specification, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround),
            specification,
            decimal=7)

        XYZ_w = np.tile(XYZ_w, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround),
            specification,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        specification = np.reshape(specification, (2, 3, 8))
        np.testing.assert_almost_equal(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround),
            specification,
            decimal=7)

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_CAM16(self):
        """
        Tests :func:`colour.appearance.cam16.XYZ_to_CAM16` definition domain
        and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CAM16['Average']
        specification = XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround)

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
                    XYZ_to_CAM16(XYZ * factor_a, XYZ_w * factor_a, L_A, Y_b,
                                 surround),
                    as_float_array(specification) * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_CAM16(self):
        """
        Tests :func:`colour.appearance.cam16.XYZ_to_CAM16` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_w = np.array(case)
            L_A = case[0]
            Y_b = case[0]
            surround = InductionFactors_CAM16(case[0], case[0], case[0])
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround)


class TestCAM16_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.appearance.cam16.CAM16_to_XYZ` definition unit tests
    methods.
    """

    def test_CAM16_to_XYZ(self):
        """
        Tests :func:`colour.appearance.cam16.CAM16_to_XYZ` definition.
        """

        specification = CAM_Specification_CAM16(41.73120791, 0.10335574,
                                                217.06795977)
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CAM16['Average']
        np.testing.assert_almost_equal(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([19.01, 20.00, 21.78]),
            decimal=7)

        specification = CAM_Specification_CAM16(65.42828069, 49.67956420,
                                                17.48659243)
        L_A = 31.83
        np.testing.assert_almost_equal(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([57.06, 43.06, 31.96]),
            decimal=7)

        specification = CAM_Specification_CAM16(21.36052893, 50.99381895,
                                                178.86724266)
        XYZ_w = np.array([109.85, 100, 35.58])
        L_A = 318.31
        np.testing.assert_almost_equal(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([3.53, 6.56, 2.14]),
            decimal=7)

        specification = CAM_Specification_CAM16(41.36326063, 52.81154022,
                                                258.88676291)
        L_A = 318.31
        np.testing.assert_almost_equal(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([19.01, 20.00, 21.78]),
            decimal=7)

        specification = CAM_Specification_CAM16(21.03801957, 457.78881613,
                                                350.06445098)
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 4.074366543152521
        np.testing.assert_almost_equal(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            np.array([61.45276998, 7.00421901, 82.2406738]),
            decimal=7)

    def test_n_dimensional_CAM16_to_XYZ(self):
        """
        Tests :func:`colour.appearance.cam16.CAM16_to_XYZ` definition
        n-dimensional support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CAM16['Average']
        specification = XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround)
        XYZ = CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround)

        specification = CAM_Specification_CAM16(
            *np.transpose(np.tile(tsplit(specification), (6, 1))).tolist())
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            decimal=7)

        XYZ_w = np.tile(XYZ_w, (6, 1))
        np.testing.assert_almost_equal(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            decimal=7)

        specification = CAM_Specification_CAM16(
            *tsplit(np.reshape(specification, (2, 3, 8))).tolist())
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            decimal=7)

    @ignore_numpy_errors
    def test_domain_range_scale_CAM16_to_XYZ(self):
        """
        Tests :func:`colour.appearance.cam16.CAM16_to_XYZ` definition domain
        and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CAM16['Average']
        specification = XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround)
        XYZ = CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround)

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
                    CAM16_to_XYZ(specification * factor_a, XYZ_w * factor_b,
                                 L_A, Y_b, surround),
                    XYZ * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_raise_exception_CAM16_to_XYZ(self):
        """
        Tests :func:`colour.appearance.cam16.CAM16_to_XYZ` definition raised
        exception.
        """

        self.assertRaises(
            ValueError, CAM16_to_XYZ,
            CAM_Specification_CAM16(41.731207905126638, None,
                                    217.06795976739301),
            np.array([95.05, 100.00, 108.88]), 318.31, 20.0,
            VIEWING_CONDITIONS_CAM16['Average'])

    @ignore_numpy_errors
    def test_nan_CAM16_to_XYZ(self):
        """
        Tests :func:`colour.appearance.cam16.CAM16_to_XYZ` definition nan
        support.
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
            surround = InductionFactors_CAM16(case[0], case[0], case[0])
            CAM16_to_XYZ(
                CAM_Specification_CAM16(J, C, h, M=50), XYZ_w, L_A, Y_b,
                surround)
