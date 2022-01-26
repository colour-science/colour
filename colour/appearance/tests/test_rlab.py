# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.appearance.rlab` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.appearance import (
    D_FACTOR_RLAB,
    VIEWING_CONDITIONS_RLAB,
    XYZ_to_RLAB,
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
    'TestXYZ_to_RLAB',
]


class TestXYZ_to_RLAB(unittest.TestCase):
    """
    Defines :func:`colour.appearance.rlab.XYZ_to_RLAB` definition unit
    tests methods.
    """

    def test_XYZ_to_RLAB(self):
        """
        Tests :func:`colour.appearance.rlab.XYZ_to_RLAB` definition.

        Notes
        -----
        -   The test values have been generated from data of the following file
            by *Fairchild (2013)*:
            http://rit-mcsl.org/fairchild//files/AppModEx.xls
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_n = np.array([95.05, 100.00, 108.88])
        Y_n = 318.31
        sigma = 0.4347
        np.testing.assert_allclose(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma),
            np.array([49.67, 0.01, 270, 0, np.nan, 0, -0.01]),
            rtol=0.01,
            atol=0.01)

        XYZ = np.array([57.06, 43.06, 31.96])
        Y_n = 31.83
        np.testing.assert_allclose(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma),
            np.array([69.33, 49.74, 21.3, 0.72, np.nan, 46.33, 18.09]),
            rtol=0.01,
            atol=0.01)

        XYZ = np.array([3.53, 6.56, 2.14])
        XYZ_n = np.array([109.85, 100.00, 35.58])
        Y_n = 318.31
        np.testing.assert_allclose(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma),
            np.array([30.78, 41.02, 176.9, 1.33, np.nan, -40.96, 2.25]),
            rtol=0.01,
            atol=0.01)

        XYZ = np.array([19.01, 20.00, 21.78])
        Y_n = 31.83
        np.testing.assert_allclose(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma),
            np.array([49.83, 54.87, 286.5, 1.1, np.nan, 15.57, -52.61]),
            rtol=0.01,
            atol=0.01)

    def test_n_dimensional_XYZ_to_RLAB(self):
        """
        Tests :func:`colour.appearance.rlab.XYZ_to_RLAB` definition
        n-dimensional support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_n = np.array([95.05, 100.00, 108.88])
        Y_n = 318.31
        sigma = 0.4347
        specification = XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma)

        XYZ = np.tile(XYZ, (6, 1))
        specification = np.tile(specification, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma), specification, decimal=7)

        XYZ_n = np.tile(XYZ_n, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma), specification, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
        specification = np.reshape(specification, (2, 3, 7))
        np.testing.assert_almost_equal(
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma), specification, decimal=7)

    def test_domain_range_scale_XYZ_to_RLAB(self):
        """
        Tests :func:`colour.appearance.rlab.XYZ_to_RLAB` definition domain and
        range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_n = np.array([109.85, 100, 35.58])
        Y_n = 31.83
        sigma = VIEWING_CONDITIONS_RLAB['Average']
        D = D_FACTOR_RLAB['Hard Copy Images']
        specification = XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma, D)

        d_r = (
            ('reference', 1, 1),
            ('1', 0.01, np.array([1, 1, 1 / 360, 1, np.nan, 1, 1])),
            ('100', 1, np.array([1, 1, 100 / 360, 1, np.nan, 1, 1])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_RLAB(XYZ * factor_a, XYZ_n * factor_a, Y_n, sigma,
                                D),
                    as_float_array(specification) * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_RLAB(self):
        """
        Tests :func:`colour.appearance.rlab.XYZ_to_RLAB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_n = np.array(case)
            Y_n = case[0]
            sigma = case[0]
            D = case[0]
            XYZ_to_RLAB(XYZ, XYZ_n, Y_n, sigma, D)
