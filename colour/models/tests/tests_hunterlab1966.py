#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.hunterlab1966` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import (
    HUNTERLAB1966_ILLUMINANTS,
    XYZ_to_K_ab_HunterLab1966,
    XYZ_to_HunterLab1966,
    HunterLab1966_to_XYZ)

from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_K_ab_HunterLab1966',
           'TestXYZ_to_HunterLab1966',
           'TestHunterLab1966_to_XYZ']


class TestXYZ_to_K_ab_HunterLab1966(unittest.TestCase):
    """
    Defines :func:`colour.models.hunterlab1966.XYZ_to_K_ab_HunterLab1966`
    definition unit tests methods.
    """

    def test_XYZ_to_K_ab_HunterLab1966(self):
        """
        Tests :func:`colour.models.hunterlab1966.XYZ_to_K_ab_HunterLab1966`
        definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_K_ab_HunterLab1966(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100),
            np.array([46.9256133, 19.9129745]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_K_ab_HunterLab1966(
                np.array([0.47097710, 0.34950000, 0.11301649]) * 100),
            np.array([121.2912993, 21.6529175]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_K_ab_HunterLab1966(
                np.array([0.25506814, 0.19150000, 0.08849752]) * 100),
            np.array([89.260201, 19.1606864]),
            decimal=7)

    def test_n_dimensional_XYZ_to_K_ab_HunterLab1966(self):
        """
        Tests :func:`colour.models.hunterlab1966.XYZ_to_K_ab_HunterLab1966`
        definition n-dimensions support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
        K_ab = 46.9256133, 19.9129745
        np.testing.assert_almost_equal(
            XYZ_to_K_ab_HunterLab1966(XYZ),
            K_ab,
            decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        K_ab = np.tile(K_ab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_K_ab_HunterLab1966(XYZ),
            K_ab,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        K_ab = np.reshape(K_ab, (2, 3, 2))
        np.testing.assert_almost_equal(
            XYZ_to_K_ab_HunterLab1966(XYZ),
            K_ab,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_K_ab_HunterLab1966(self):
        """
        Tests :func:`colour.models.hunterlab1966.XYZ_to_K_ab_HunterLab1966`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ_to_K_ab_HunterLab1966(np.array(case))


class TestXYZ_to_HunterLab1966(unittest.TestCase):
    """
    Defines :func:`colour.models.hunterlab1966.XYZ_to_HunterLab1966` definition
    unit tests methods.
    """

    def test_XYZ_to_HunterLab1966(self):
        """
        Tests :func:`colour.models.hunterlab1966.XYZ_to_HunterLab1966`
        definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_HunterLab1966(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100),
            np.array([31.7490157, -15.1146263, -2.7866076]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_HunterLab1966(
                np.array([0.47097710, 0.34950000, 0.11301649]) * 100),
            np.array([59.118525, 40.8447916, 21.0132865]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_HunterLab1966(
                np.array([0.25506814, 0.19150000, 0.08849752]) * 100),
            np.array([43.760713, 29.0031405, 11.2474916]),
            decimal=7)

        h_i = HUNTERLAB1966_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        A = h_i['A']
        np.testing.assert_almost_equal(
            XYZ_to_HunterLab1966(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                A.XYZ_n,
                A.K_ab),
            np.array([31.7490157, -21.3579441, -20.3277837]),
            decimal=7)

        D65 = h_i['D65']
        np.testing.assert_almost_equal(
            XYZ_to_HunterLab1966(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                D65.XYZ_n,
                D65.K_ab),
            np.array([31.7490157, -14.4410859, 2.7439626]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_HunterLab1966(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                D65.XYZ_n,
                K_ab=None),
            np.array([31.7490157, -14.4394895, 2.7435242]),
            decimal=7)

    def test_n_dimensional_XYZ_to_HunterLab1966(self):
        """
        Tests :func:`colour.models.hunterlab1966.XYZ_to_HunterLab1966`
        definition n-dimensions support.
        """

        h_i = HUNTERLAB1966_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        D50 = h_i['D50']

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
        XYZ_n = D50.XYZ_n
        K_ab = D50.K_ab
        Lab = np.array([31.7490157, -15.1146263, -2.7866076])
        np.testing.assert_almost_equal(
            XYZ_to_HunterLab1966(XYZ, XYZ_n, K_ab),
            Lab,
            decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        Lab = np.tile(Lab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_HunterLab1966(XYZ, XYZ_n, K_ab),
            Lab,
            decimal=7)

        XYZ_n = np.tile(XYZ_n, (6, 1))
        K_ab = np.tile(K_ab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_HunterLab1966(XYZ, XYZ_n, K_ab),
            Lab,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
        K_ab = np.reshape(K_ab, (2, 3, 2))
        Lab = np.reshape(Lab, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_HunterLab1966(XYZ, XYZ_n, K_ab),
            Lab,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_HunterLab1966(self):
        """
        Tests :func:`colour.models.hunterlab1966.XYZ_to_HunterLab1966`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_n = np.array(case[0:3])
            K_ab = np.array(case[0:2])
            XYZ_to_HunterLab1966(XYZ, XYZ_n, K_ab)


class TestHunterLab1966_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.hunterlab1966.HunterLab1966_to_XYZ` definition
    unit tests methods.
    """

    def test_HunterLab1966_to_XYZ(self):
        """
        Tests :func:`colour.models.hunterlab1966.HunterLab1966_to_XYZ`
        definition.
        """

        np.testing.assert_almost_equal(
            HunterLab1966_to_XYZ(
                np.array([31.7490157, -15.1146263, -2.7866076])),
            np.array([7.049534, 10.08, 9.558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            HunterLab1966_to_XYZ(
                np.array([59.118525, 40.8447916, 21.0132865])),
            np.array([47.09771, 34.95, 11.301649]),
            decimal=7)

        np.testing.assert_almost_equal(
            HunterLab1966_to_XYZ(
                np.array([43.760713, 29.0031405, 11.2474916])),
            np.array([25.506814, 19.15, 8.849752]),
            decimal=7)

        h_i = HUNTERLAB1966_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        A = h_i['A']
        np.testing.assert_almost_equal(
            HunterLab1966_to_XYZ(
                np.array([31.7490157, -21.3579441, -20.3277837]),
                A.XYZ_n,
                A.K_ab),
            np.array([7.049534, 10.08, 9.558313]),
            decimal=7)

        D65 = h_i['D65']
        np.testing.assert_almost_equal(
            HunterLab1966_to_XYZ(
                np.array([31.7490157, -14.4410859, 2.7439626]),
                D65.XYZ_n,
                D65.K_ab),
            np.array([7.049534, 10.08, 9.558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            HunterLab1966_to_XYZ(
                np.array([31.7490157, -14.4394895, 2.7435242]),
                D65.XYZ_n,
                K_ab=None),
            np.array([7.049534, 10.08, 9.558313]),
            decimal=7)

    def test_n_dimensional_HunterLab1966_to_XYZ(self):
        """
        Tests :func:`colour.models.hunterlab1966.HunterLab1966_to_XYZ`
        definition n-dimensions support.
        """

        h_i = HUNTERLAB1966_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        D50 = h_i['D50']

        Lab = np.array([31.7490157, -15.1146263, -2.7866076])
        XYZ_n = D50.XYZ_n
        K_ab = D50.K_ab
        XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
        np.testing.assert_almost_equal(
            HunterLab1966_to_XYZ(Lab, XYZ_n, K_ab),
            XYZ,
            decimal=7)

        Lab = np.tile(Lab, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            HunterLab1966_to_XYZ(Lab, XYZ_n, K_ab),
            XYZ,
            decimal=7)

        K_ab = np.tile(K_ab, (6, 1))
        XYZ_n = np.tile(XYZ_n, (6, 1))
        np.testing.assert_almost_equal(
            HunterLab1966_to_XYZ(Lab, XYZ_n, K_ab),
            XYZ,
            decimal=7)

        Lab = np.reshape(Lab, (2, 3, 3))
        XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
        K_ab = np.reshape(K_ab, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            HunterLab1966_to_XYZ(Lab, XYZ_n, K_ab),
            XYZ,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_HunterLab1966_to_XYZ(self):
        """
        Tests :func:`colour.models.hunterlab1966.HunterLab1966_to_XYZ`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab = np.array(case)
            XYZ_n = np.array(case[0:3])
            K_ab = np.array(case[0:2])
            HunterLab1966_to_XYZ(Lab, XYZ_n, K_ab)


if __name__ == '__main__':
    unittest.main()
