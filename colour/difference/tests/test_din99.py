# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.difference.din99` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.difference import delta_E_DIN99

from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestDelta_E_DIN99']


class TestDelta_E_DIN99(unittest.TestCase):
    """
    Defines :func:`colour.difference.din99.delta_E_DIN99` definition unit
    tests methods.
    """

    def test_delta_E_DIN99(self):
        """
        Tests :func:`colour.difference.din99.delta_E_DIN99` definition.
        """

        self.assertAlmostEqual(
            delta_E_DIN99(
                np.array([60.25740000, -34.00990000, 36.26770000]),
                np.array([60.46260000, -34.17510000, 39.43870000])),
            1.177216620111552,
            places=7)

        self.assertAlmostEqual(
            delta_E_DIN99(
                np.array([63.01090000, -31.09610000, -5.86630000]),
                np.array([62.81870000, -29.79460000, -4.08640000])),
            0.987529977993114,
            places=7)

        self.assertAlmostEqual(
            delta_E_DIN99(
                np.array([35.08310000, -44.11640000, 3.79330000]),
                np.array([35.02320000, -40.07160000, 1.59010000])),
            1.535894757971742,
            places=7)

        self.assertAlmostEqual(
            delta_E_DIN99(
                np.array([60.25740000, -34.00990000, 36.26770000]),
                np.array([60.46260000, -34.17510000, 39.43870000]),
                textiles=True),
            1.215652775586509,
            places=7)

        self.assertAlmostEqual(
            delta_E_DIN99(
                np.array([63.01090000, -31.09610000, -5.86630000]),
                np.array([62.81870000, -29.79460000, -4.08640000]),
                textiles=True),
            1.025997138865984,
            places=7)

        self.assertAlmostEqual(
            delta_E_DIN99(
                np.array([35.08310000, -44.11640000, 3.79330000]),
                np.array([35.02320000, -40.07160000, 1.59010000]),
                textiles=True),
            1.539922810033725,
            places=7)

    def test_n_dimensional_delta_E_DIN99(self):
        """
        Tests :func:`colour.difference.din99.delta_E_DIN99` definition
        n-dimensional arrays support.
        """

        Lab_1 = np.array([60.25740000, -34.00990000, 36.26770000])
        Lab_2 = np.array([60.46260000, -34.17510000, 39.43870000])
        delta_E = 1.177216620111552
        np.testing.assert_almost_equal(
            delta_E_DIN99(Lab_1, Lab_2), delta_E, decimal=7)

        Lab_1 = np.tile(Lab_1, (6, 1))
        Lab_2 = np.tile(Lab_2, (6, 1))
        delta_E = np.tile(delta_E, 6)
        np.testing.assert_almost_equal(
            delta_E_DIN99(Lab_1, Lab_2), delta_E, decimal=7)

        Lab_1 = np.reshape(Lab_1, (2, 3, 3))
        Lab_2 = np.reshape(Lab_2, (2, 3, 3))
        delta_E = np.reshape(delta_E, (2, 3))
        np.testing.assert_almost_equal(
            delta_E_DIN99(Lab_1, Lab_2), delta_E, decimal=7)

    def test_domain_range_scale_delta_E_DIN99(self):
        """
        Tests :func:`colour.difference.din99.delta_E_DIN99` definition
        domain and range scale support.
        """

        Lab_1 = np.array([60.25740000, -34.00990000, 36.26770000])
        Lab_2 = np.array([60.46260000, -34.17510000, 39.43870000])
        delta_E = delta_E_DIN99(Lab_1, Lab_2)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    delta_E_DIN99(Lab_1 * factor, Lab_2 * factor),
                    delta_E,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_delta_E_DIN99(self):
        """
        Tests :func:`colour.difference.din99.delta_E_DIN99` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab_1 = np.array(case)
            Lab_2 = np.array(case)
            delta_E_DIN99(Lab_1, Lab_2)


if __name__ == '__main__':
    unittest.main()
