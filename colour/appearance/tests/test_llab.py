# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.appearance.llab` module.
"""

import numpy as np
import unittest
from unittest import mock
from itertools import permutations

from colour.appearance import (
    VIEWING_CONDITIONS_LLAB,
    InductionFactors_LLAB,
    XYZ_to_LLAB,
    llab,
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
    'TestXYZ_to_LLAB',
]


class TestXYZ_to_LLAB(unittest.TestCase):
    """
    Defines :func:`colour.appearance.llab.XYZ_to_LLAB` definition unit
    tests methods.
    """

    def test_XYZ_to_LLAB(self):
        """
        Tests :func:`colour.appearance.llab.XYZ_to_LLAB` definition.

        Notes
        -----
        -   The test values have been generated from data of the following file
            by *Fairchild (2013)*:
            http://rit-mcsl.org/fairchild//files/AppModEx.xls
        """

        with mock.patch(
                'colour.appearance.llab.MATRIX_RGB_TO_XYZ_LLAB',
                np.around(
                    np.linalg.inv(llab.MATRIX_XYZ_TO_RGB_LLAB), decimals=4)):

            XYZ = np.array([19.01, 20.00, 21.78])
            XYZ_0 = np.array([95.05, 100.00, 108.88])
            Y_b = 20
            L = 318.31
            surround = VIEWING_CONDITIONS_LLAB[
                'Reference Samples & Images, Average Surround, Subtending < 4']
            np.testing.assert_allclose(
                XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
                np.array([37.37, 0.01, 229.5, 0, 0.02, np.nan, -0.01, -0.01]),
                rtol=0.01,
                atol=0.01)

            XYZ = np.array([57.06, 43.06, 31.96])
            L = 31.83
            np.testing.assert_allclose(
                XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
                np.array(
                    [61.26, 30.51, 22.3, 0.5, 56.55, np.nan, 52.33, 21.43]),
                rtol=0.01,
                atol=0.01)

            XYZ = np.array([3.53, 6.56, 2.14])
            XYZ_0 = np.array([109.85, 100.00, 35.58])
            L = 318.31
            np.testing.assert_allclose(
                XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
                np.array(
                    [16.25, 30.43, 173.8, 1.87, 53.83, np.nan, -53.51, 5.83]),
                rtol=0.01,
                atol=0.01)

            XYZ = np.array([19.01, 20.00, 21.78])
            L = 31.83
            np.testing.assert_allclose(
                XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
                np.array(
                    [39.82, 29.34, 271.9, 0.74, 54.59, np.nan, 1.76, -54.56]),
                rtol=0.01,
                atol=0.01)

    def test_n_dimensional_XYZ_to_LLAB(self):
        """
        Tests :func:`colour.appearance.llab.XYZ_to_LLAB` definition
        n-dimensional support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_0 = np.array([95.05, 100.00, 108.88])
        Y_b = 20
        L = 318.31
        surround = surround = VIEWING_CONDITIONS_LLAB[
            'Reference Samples & Images, Average Surround, Subtending < 4']
        specification = XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround)

        XYZ = np.tile(XYZ, (6, 1))
        specification = np.tile(specification, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
            specification,
            decimal=7)

        XYZ_0 = np.tile(XYZ_0, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
            specification,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_0 = np.reshape(XYZ_0, (2, 3, 3))
        specification = np.reshape(specification, (2, 3, 8))
        np.testing.assert_almost_equal(
            XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
            specification,
            decimal=7)

    def test_colourspace_conversion_matrices_precision(self):
        """
        Tests for loss of precision in conversion between
        *LLAB(l:c)* colour appearance model *CIE XYZ* tristimulus values and
        normalised cone responses matrix.
        """

        start = np.array([1, 1, 1])
        result = np.array(start)
        for _ in range(100000):
            result = llab.MATRIX_RGB_TO_XYZ_LLAB.dot(result)
            result = llab.MATRIX_XYZ_TO_RGB_LLAB.dot(result)
        np.testing.assert_almost_equal(start, result, decimal=7)

    def test_domain_range_scale_XYZ_to_LLAB(self):
        """
        Tests :func:`colour.appearance.llab.XYZ_to_LLAB` definition domain
        and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_0 = np.array([95.05, 100.00, 108.88])
        Y_b = 20
        L = 318.31
        surround = VIEWING_CONDITIONS_LLAB['ref_average_4_minus']
        specification = XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround)

        d_r = (
            ('reference', 1, 1),
            ('1', 0.01, np.array([1, 1, 1 / 360, 1, 1, np.nan, 1, 1])),
            ('100', 1, np.array([1, 1, 100 / 360, 1, 1, np.nan, 1, 1])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_LLAB(XYZ * factor_a, XYZ_0 * factor_a, Y_b, L,
                                surround),
                    as_float_array(specification) * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_LLAB(self):
        """
        Tests :func:`colour.appearance.llab.XYZ_to_LLAB` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_0 = np.array(case)
            Y_b = case[0]
            L = case[0]
            surround = InductionFactors_LLAB(1, case[0], case[0], case[0])
            XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround)
