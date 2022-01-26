# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.appearance.nayatani95` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.appearance import XYZ_to_Nayatani95
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
    'TestXYZ_to_Nayatani95',
]


class TestXYZ_to_Nayatani95(unittest.TestCase):
    """
    Defines :func:`colour.appearance.nayatani95.XYZ_to_Nayatani95` definition
    unit tests methods.
    """

    def test_XYZ_to_Nayatani95(self):
        """
        Tests :func:`colour.appearance.nayatani95.XYZ_to_Nayatani95`
        definition.

        Notes
        -----
        -   The test values have been generated from data of the following file
            by *Fairchild (2013)*:
            http://rit-mcsl.org/fairchild//files/AppModEx.xls
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_n = np.array([95.05, 100.00, 108.88])
        Y_o = 20
        E_o = 5000
        E_or = 1000
        np.testing.assert_allclose(
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or),
            np.array([50, 0.01, 257.5, 0.01, 62.6, 0.02, np.nan, np.nan, 50]),
            rtol=0.01,
            atol=0.01)

        XYZ = np.array([57.06, 43.06, 31.96])
        E_o = 500
        np.testing.assert_allclose(
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or),
            np.array([73, 48.3, 21.6, 37.1, 67.3, 42.9, np.nan, np.nan, 75.9]),
            rtol=0.01,
            atol=0.01)

        XYZ = np.array([3.53, 6.56, 2.14])
        XYZ_n = np.array([109.85, 100.00, 35.58])
        E_o = 5000
        np.testing.assert_allclose(
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or),
            np.array(
                [24.5, 49.3, 190.6, 81.3, 37.5, 62.1, np.nan, np.nan, 29.7]),
            rtol=0.01,
            atol=0.01)

        XYZ = np.array([19.01, 20.00, 21.78])
        E_o = 500
        np.testing.assert_allclose(
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or),
            np.array(
                [49.4, 39.9, 236.3, 40.2, 44.2, 35.8, np.nan, np.nan, 49.4]),
            rtol=0.01,
            atol=0.01)

    def test_n_dimensional_XYZ_to_Nayatani95(self):
        """
        Tests :func:`colour.appearance.nayatani95.XYZ_to_Nayatani95` definition
        n-dimensional support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_n = np.array([95.05, 100.00, 108.88])
        Y_o = 20
        E_o = 5000
        E_or = 1000
        specification = XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or)

        XYZ = np.tile(XYZ, (6, 1))
        specification = np.tile(specification, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or),
            specification,
            decimal=7)

        XYZ_n = np.tile(XYZ_n, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or),
            specification,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
        specification = np.reshape(specification, (2, 3, 9))
        np.testing.assert_almost_equal(
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or),
            specification,
            decimal=7)

    def test_domain_range_scale_XYZ_to_Nayatani95(self):
        """
        Tests :func:`colour.appearance.nayatani95.XYZ_to_Nayatani95` definition
        domain and range scale support.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_n = np.array([95.05, 100.00, 108.88])
        Y_o = 20.0
        E_o = 5000.0
        E_or = 1000.0
        specification = XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or)

        d_r = (
            ('reference', 1, 1),
            ('1', 0.01, np.array([1, 1, 1 / 360, 1, 1, 1, np.nan, np.nan, 1])),
            ('100', 1, np.array([1, 1, 100 / 360, 1, 1, 1, np.nan, np.nan,
                                 1])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_Nayatani95(XYZ * factor_a, XYZ_n * factor_a, Y_o,
                                      E_o, E_or),
                    as_float_array(specification) * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_Nayatani95(self):
        """
        Tests :func:`colour.appearance.nayatani95.XYZ_to_Nayatani95` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_n = np.array(case)
            Y_o = case[0]
            E_o = case[0]
            E_or = case[0]
            XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, E_o, E_or)
