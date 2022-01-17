# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.adaptation.fairchild1990` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.adaptation import chromatic_adaptation_Fairchild1990
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestChromaticAdaptationFairchild1990',
]


class TestChromaticAdaptationFairchild1990(unittest.TestCase):
    """
    Defines :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition unit tests methods.
    """

    def test_chromatic_adaptation_Fairchild1990(self):
        """
        Tests :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition.
        """

        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(
                np.array([19.53, 23.07, 24.97]),
                np.array([111.15, 100.00, 35.20]),
                np.array([94.81, 100.00, 107.30]), 200),
            np.array([23.32526349, 23.32455819, 76.11593750]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(
                np.array([0.14222010, 0.23042768, 0.10495772]) * 100,
                np.array([0.95045593, 1.00000000, 1.08905775]) * 100,
                np.array([1.09846607, 1.00000000, 0.35582280]) * 100, 200),
            np.array([19.28089326, 22.91583715, 3.42923503]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(
                np.array([0.07818780, 0.06157201, 0.28099326]) * 100,
                np.array([0.95045593, 1.00000000, 1.08905775]) * 100,
                np.array([0.99144661, 1.00000000, 0.67315942]) * 100, 200),
            np.array([6.35093475, 6.13061347, 17.36852430]),
            decimal=7)

    def test_n_dimensional_chromatic_adaptation_Fairchild1990(self):
        """
        Tests :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition n-dimensional arrays support.
        """

        XYZ_1 = np.array([19.53, 23.07, 24.97])
        XYZ_n = np.array([111.15, 100.00, 35.20])
        XYZ_r = np.array([94.81, 100.00, 107.30])
        Y_n = 200
        XYZ_c = chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n)

        XYZ_1 = np.tile(XYZ_1, (6, 1))
        XYZ_c = np.tile(XYZ_c, (6, 1))
        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n),
            XYZ_c,
            decimal=7)

        XYZ_n = np.tile(XYZ_n, (6, 1))
        XYZ_r = np.tile(XYZ_r, (6, 1))
        Y_n = np.tile(Y_n, 6)
        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n),
            XYZ_c,
            decimal=7)

        XYZ_1 = np.reshape(XYZ_1, (2, 3, 3))
        XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
        XYZ_r = np.reshape(XYZ_r, (2, 3, 3))
        Y_n = np.reshape(Y_n, (2, 3))
        XYZ_c = np.reshape(XYZ_c, (2, 3, 3))
        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n),
            XYZ_c,
            decimal=7)

    def test_domain_range_scale_chromatic_adaptation_Fairchild1990(self):
        """
        Tests :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition domain and range scale support.
        """

        XYZ_1 = np.array([19.53, 23.07, 24.97])
        XYZ_n = np.array([111.15, 100.00, 35.20])
        XYZ_r = np.array([94.81, 100.00, 107.30])
        Y_n = 200
        XYZ_c = chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n)

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    chromatic_adaptation_Fairchild1990(
                        XYZ_1 * factor, XYZ_n * factor, XYZ_r * factor, Y_n),
                    XYZ_c * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_Fairchild1990(self):
        """
        Tests :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ_1 = np.array(case)
            XYZ_n = np.array(case)
            XYZ_r = np.array(case)
            Y_n = case[0]
            chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n)


if __name__ == '__main__':
    unittest.main()
