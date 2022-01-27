# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.appearance.kim2009` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.appearance import (
    MEDIA_PARAMETERS_KIM2009,
    VIEWING_CONDITIONS_KIM2009,
    InductionFactors_Kim2009,
    CAM_Specification_Kim2009,
    MediaParameters_Kim2009,
    XYZ_to_Kim2009,
    Kim2009_to_XYZ,
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
    'TestXYZ_to_Kim2009',
    'TestKim2009_to_XYZ',
]


class TestXYZ_to_Kim2009(unittest.TestCase):
    """
    Defines :func:`colour.appearance.kim2009.XYZ_to_Kim2009` definition unit
    tests methods.
    """

    def test_XYZ_to_Kim2009(self):
        """
        Tests :func:`colour.appearance.kim2009.XYZ_to_Kim2009` definition.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_a = 318.31
        media = MEDIA_PARAMETERS_KIM2009['CRT Displays']
        surround = VIEWING_CONDITIONS_KIM2009['Average']
        np.testing.assert_almost_equal(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround),
            np.array([
                28.86190898, 0.55924559, 219.04806678, 9.38377973, 52.71388839,
                0.46417384, 278.06028246, np.nan
            ]),
            decimal=7)

        XYZ = np.array([57.06, 43.06, 31.96])
        L_a = 31.83
        np.testing.assert_almost_equal(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround),
            np.array([
                70.15940419, 57.89295872, 21.27017200, 61.23630434,
                128.14034598, 48.05115573, 1.41841443, np.nan
            ]),
            decimal=7)

        XYZ = np.array([3.53, 6.56, 2.14])
        XYZ_w = np.array([109.85, 100.00, 35.58])
        L_a = 318.31
        np.testing.assert_almost_equal(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround),
            np.array([
                -4.83430022, 37.42013921, 177.12166057, np.nan, -8.82944930,
                31.05871555, 220.36270343, np.nan
            ]),
            decimal=7)

        XYZ = np.array([19.01, 20.00, 21.78])
        L_a = 31.83
        np.testing.assert_almost_equal(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround),
            np.array([
                47.20460719, 56.35723637, 241.04877377, 73.65830083,
                86.21530880, 46.77650619, 301.77516676, np.nan
            ]),
            decimal=7)

    def test_n_dimensional_XYZ_to_Kim2009(self):
        """
        Tests :func:`colour.appearance.kim2009.XYZ_to_Kim2009` definition
        n-dimensional support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_a = 318.31
        media = MEDIA_PARAMETERS_KIM2009['CRT Displays']
        surround = VIEWING_CONDITIONS_KIM2009['Average']
        specification = XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround)

        XYZ = np.tile(XYZ, (6, 1))
        specification = np.tile(specification, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround),
            specification,
            decimal=7)

        XYZ_w = np.tile(XYZ_w, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround),
            specification,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        specification = np.reshape(specification, (2, 3, 8))
        np.testing.assert_almost_equal(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround),
            specification,
            decimal=7)

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_Kim2009(self):
        """
        Tests :func:`colour.appearance.kim2009.XYZ_to_Kim2009` definition
        domain and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_a = 318.31
        media = MEDIA_PARAMETERS_KIM2009['CRT Displays']
        surround = VIEWING_CONDITIONS_KIM2009['Average']
        specification = XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround)

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
                    XYZ_to_Kim2009(XYZ * factor_a, XYZ_w * factor_a, L_a,
                                   media, surround),
                    as_float_array(specification) * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_Kim2009(self):
        """
        Tests :func:`colour.appearance.kim2009.XYZ_to_Kim2009` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_w = np.array(case)
            L_a = case[0]
            media = MediaParameters_Kim2009(case[0])
            surround = InductionFactors_Kim2009(case[0], case[0], case[0])
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround)


class TestKim2009_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.appearance.kim2009.Kim2009_to_XYZ` definition unit
    tests methods.
    """

    def test_Kim2009_to_XYZ(self):
        """
        Tests :func:`colour.appearance.kim2009.Kim2009_to_XYZ` definition.
        """

        specification = CAM_Specification_Kim2009(
            28.86190898, 0.55924559, 219.04806678, 9.38377973, 52.71388839,
            0.46417384, 278.06028246, np.nan)
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_a = 318.31
        media = MEDIA_PARAMETERS_KIM2009['CRT Displays']
        surround = VIEWING_CONDITIONS_KIM2009['Average']
        np.testing.assert_allclose(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            np.array([19.01, 20.00, 21.78]),
            atol=0.01,
            rtol=0.01)

        specification = CAM_Specification_Kim2009(
            70.15940419, 57.89295872, 21.27017200, 61.23630434, 128.14034598,
            48.05115573, 1.41841443, np.nan)
        L_a = 31.83
        np.testing.assert_allclose(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            np.array([57.06, 43.06, 31.96]),
            atol=0.01,
            rtol=0.01)

        specification = CAM_Specification_Kim2009(
            -4.83430022, 37.42013921, 177.12166057, np.nan, -8.82944930,
            31.05871555, 220.36270343, np.nan)
        XYZ_w = np.array([109.85, 100.00, 35.58])
        L_a = 318.31
        np.testing.assert_allclose(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            np.array([3.53, 6.56, 2.14]),
            atol=0.01,
            rtol=0.01)

        specification = CAM_Specification_Kim2009(
            47.20460719, 56.35723637, 241.04877377, 73.65830083, 86.21530880,
            46.77650619, 301.77516676, np.nan)
        L_a = 31.83
        np.testing.assert_allclose(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            np.array([19.01, 20.00, 21.78]),
            atol=0.01,
            rtol=0.01)

    def test_n_dimensional_Kim2009_to_XYZ(self):
        """
        Tests :func:`colour.appearance.kim2009.Kim2009_to_XYZ` definition
        n-dimensional support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_a = 318.31
        media = MEDIA_PARAMETERS_KIM2009['CRT Displays']
        surround = VIEWING_CONDITIONS_KIM2009['Average']
        specification = XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround)
        XYZ = Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround)

        specification = CAM_Specification_Kim2009(
            *np.transpose(np.tile(tsplit(specification), (6, 1))).tolist())
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            XYZ,
            decimal=7)

        XYZ_w = np.tile(XYZ_w, (6, 1))
        np.testing.assert_almost_equal(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            XYZ,
            decimal=7)

        specification = CAM_Specification_Kim2009(
            *tsplit(np.reshape(specification, (2, 3, 8))).tolist())
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            XYZ,
            decimal=7)

    @ignore_numpy_errors
    def test_domain_range_scale_Kim2009_to_XYZ(self):
        """
        Tests :func:`colour.appearance.kim2009.Kim2009_to_XYZ` definition
        domain and range scale support.
        """

        XYZ_i = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_a = 318.31
        media = MEDIA_PARAMETERS_KIM2009['CRT Displays']
        surround = VIEWING_CONDITIONS_KIM2009['Average']
        specification = XYZ_to_Kim2009(XYZ_i, XYZ_w, L_a, media, surround)
        XYZ = Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround)

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
                    Kim2009_to_XYZ(specification * factor_a, XYZ_w * factor_b,
                                   L_a, media, surround),
                    XYZ * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_raise_exception_Kim2009_to_XYZ(self):
        """
        Tests :func:`colour.appearance.kim2009.Kim2009_to_XYZ` definition
        raised exception.
        """

        self.assertRaises(
            ValueError, Kim2009_to_XYZ,
            CAM_Specification_Kim2009(
                41.731091132513917,
                None,
                219.04843265831178,
            ), np.array([95.05, 100.00, 108.88]), 318.31, 20.0,
            VIEWING_CONDITIONS_KIM2009['Average'])

    @ignore_numpy_errors
    def test_nan_Kim2009_to_XYZ(self):
        """
        Tests :func:`colour.appearance.kim2009.Kim2009_to_XYZ` definition nan
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
            media = MediaParameters_Kim2009(case[0])
            surround = InductionFactors_Kim2009(case[0], case[0], case[0])
            Kim2009_to_XYZ(
                CAM_Specification_Kim2009(J, C, h, M=50), XYZ_w, L_a, media,
                surround)
