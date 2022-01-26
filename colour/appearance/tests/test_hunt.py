# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.appearance.hunt` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.appearance import (
    VIEWING_CONDITIONS_HUNT,
    InductionFactors_Hunt,
    XYZ_to_Hunt,
)
from colour.utilities import (
    as_float_array,
    domain_range_scale,
    ignore_numpy_errors,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_Hunt',
]


class TestXYZ_to_Hunt(unittest.TestCase):
    """
    Defines :func:`colour.appearance.hunt.XYZ_to_Hunt` definition unit tests
    methods.
    """

    def test_XYZ_to_Hunt(self):
        """
        Tests :func:`colour.appearance.hunt.XYZ_to_Hunt` definition.

        Notes
        -----
        -   The test values have been generated from data of the following file
            by *Fairchild (2013)*:
            http://rit-mcsl.org/fairchild//files/AppModEx.xls
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        XYZ_b = XYZ_w * np.array([1, 0.2, 1])
        L_A = 318.31
        surround = VIEWING_CONDITIONS_HUNT['Normal Scenes']
        CCT_w = 6504.0
        np.testing.assert_allclose(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            np.array([42.12, 0.16, 269.3, 0.03, 31.92, 0.16, np.nan, np.nan]),
            rtol=0.01,
            atol=0.01)

        XYZ = np.array([57.06, 43.06, 31.96])
        L_A = 31.83
        np.testing.assert_allclose(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            np.array(
                [66.76, 63.89, 18.6, 153.36, 31.22, 58.28, np.nan, np.nan]),
            rtol=0.01,
            atol=0.01)

        XYZ = np.array([3.53, 6.56, 2.14])
        XYZ_w = np.array([109.85, 100.00, 35.58])
        L_A = 318.31
        CCT_w = 2856
        np.testing.assert_allclose(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            np.array([19.56, 74.58, 178.3, 245.4, 18.9, 76.33, np.nan,
                      np.nan]),
            rtol=0.01,
            atol=0.01)

        XYZ = np.array([19.01, 20.00, 21.78])
        L_A = 31.83
        np.testing.assert_allclose(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            np.array(
                [40.27, 73.84, 262.8, 209.29, 22.15, 67.35, np.nan, np.nan]),
            rtol=0.01,
            atol=0.01)

    def test_n_dimensional_XYZ_to_Hunt(self):
        """
        Tests :func:`colour.appearance.hunt.XYZ_to_Hunt` definition
        n-dimensional support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        XYZ_b = XYZ_w * np.array([1, 0.2, 1])
        L_A = 318.31
        surround = VIEWING_CONDITIONS_HUNT['Normal Scenes']
        CCT_w = 6504.0
        specification = XYZ_to_Hunt(
            XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w)

        XYZ = np.tile(XYZ, (6, 1))
        specification = np.tile(specification, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            specification,
            decimal=7)

        XYZ_w = np.tile(XYZ_w, (6, 1))
        XYZ_b = np.tile(XYZ_b, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            specification,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_w = np.reshape(XYZ_w, (2, 3, 3))
        XYZ_b = np.reshape(XYZ_b, (2, 3, 3))
        specification = np.reshape(specification, (2, 3, 8))
        np.testing.assert_almost_equal(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            specification,
            decimal=7)

    def test_domain_range_scale_XYZ_to_Hunt(self):
        """
        Tests :func:`colour.appearance.hunt.XYZ_to_Hunt` definition domain
        and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        XYZ_b = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        surround = VIEWING_CONDITIONS_HUNT['Normal Scenes']
        CCT_w = 6504.0
        specification = XYZ_to_Hunt(
            XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w)

        d_r = (
            ('reference', 1, 1),
            ('1', 0.01, np.array([1, 1, 1 / 360, 1, 1, 1, np.nan, np.nan])),
            ('100', 1, np.array([1, 1, 100 / 360, 1, 1, 1, np.nan, np.nan])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_Hunt(
                        XYZ * factor_a,
                        XYZ_w * factor_a,
                        XYZ_b * factor_a,
                        L_A,
                        surround,
                        CCT_w=CCT_w),
                    as_float_array(specification) * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_raise_exception_XYZ_to_Hunt(self):
        """
        Tests :func:`colour.appearance.hunt.XYZ_to_Hunt` definition raised
        exception.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        XYZ_b = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        surround = VIEWING_CONDITIONS_HUNT['Normal Scenes']
        CCT_w = 6504.0
        S = S_w = 0.5

        try:
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround)
        except ValueError:
            pass

        try:
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w, S=S)
        except ValueError:
            pass

        try:
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w, S_w=S_w)
        except ValueError:
            pass

    @ignore_numpy_errors
    def test_XYZ_p_XYZ_to_Hunt(self):
        """
        Tests :func:`colour.appearance.hunt.XYZ_to_Hunt` definition *XYZ_p*
        argument handling.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        XYZ_b = XYZ_p = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        surround = VIEWING_CONDITIONS_HUNT['Normal Scenes']
        CCT_w = 6504.0

        np.testing.assert_almost_equal(
            XYZ_to_Hunt(
                XYZ,
                XYZ_w,
                XYZ_b,
                L_A,
                surround,
                XYZ_p=XYZ_p,
                CCT_w=CCT_w,
            ),
            np.array([
                30.046267861960700, 0.121050839936350, 269.273759446144600,
                0.019909320692942, 22.209765491265024, 0.123896438259997,
                np.nan, np.nan
            ]),
            decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_Hunt(self):
        """
        Tests :func:`colour.appearance.hunt.XYZ_to_Hunt` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_w = np.array(case)
            XYZ_b = np.array(case)
            L_A = case[0]
            surround = InductionFactors_Hunt(case[0], case[0])
            CCT_w = case[0]
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w)
