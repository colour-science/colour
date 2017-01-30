#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.hunter_rdab` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.colorimetry import HUNTERLAB_ILLUMINANTS
from colour.models import XYZ_to_Hunter_Rdab

from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_Hunter_Rdab']


class TestXYZ_to_Hunter_Rdab(unittest.TestCase):
    """
    Defines :func:`colour.models.hunter_rdab.XYZ_to_Hunter_Rdab` definition
    unit tests methods.
    """

    def test_XYZ_to_Hunter_Rdab(self):
        """
        Tests :func:`colour.models.hunter_rdab.XYZ_to_Hunter_Rdab` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100),
            np.array([10.08000000, -18.67653764, -3.44329925]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(
                np.array([0.47097710, 0.34950000, 0.11301649]) * 100),
            np.array([34.95000000, 43.14063862, 22.19442342]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(
                np.array([0.25506814, 0.19150000, 0.08849752]) * 100),
            np.array([19.15000000, 33.27585171, 12.90445983]),
            decimal=7)

        h_i = HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        A = h_i['A']
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                A.XYZ_n,
                A.K_ab),
            np.array([10.08000000, -26.39115518, -25.11822727]),
            decimal=7)

        D65 = h_i['D65']
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                D65.XYZ_n,
                D65.K_ab),
            np.array([10.08000000, -17.84427080, 3.39060457]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                D65.XYZ_n,
                K_ab=None),
            np.array([10.08000000, -17.84229822, 3.39006280]),
            decimal=7)

    def test_n_dimensional_XYZ_to_Hunter_Rdab(self):
        """
        Tests :func:`colour.models.hunter_rdab.XYZ_to_Hunter_Rdab` definition
        n-dimensions support.
        """

        h_i = HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        D50 = h_i['D50']

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
        XYZ_n = D50.XYZ_n
        K_ab = D50.K_ab
        R_d_ab = np.array([10.08000000, -18.67653764, -3.44329925])
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab),
            R_d_ab,
            decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        R_d_ab = np.tile(R_d_ab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab),
            R_d_ab,
            decimal=7)

        XYZ_n = np.tile(XYZ_n, (6, 1))
        K_ab = np.tile(K_ab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab),
            R_d_ab,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
        K_ab = np.reshape(K_ab, (2, 3, 2))
        R_d_ab = np.reshape(R_d_ab, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab),
            R_d_ab,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_Hunter_Rdab(self):
        """
        Tests :func:`colour.models.hunter_rdab.XYZ_to_Hunter_Rdab` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_n = np.array(case[0:3])
            K_ab = np.array(case[0:2])
            XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab)


if __name__ == '__main__':
    unittest.main()
