#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.hunter_lab` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.colorimetry import HUNTERLAB_ILLUMINANTS
from colour.models import (
    XYZ_to_K_ab_HunterLab1966,
    XYZ_to_Hunter_Lab,
    Hunter_Lab_to_XYZ)

from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_K_ab_HunterLab1966',
           'TestXYZ_to_Hunter_Lab',
           'TestHunter_Lab_to_XYZ']


class TestXYZ_to_K_ab_HunterLab1966(unittest.TestCase):
    """
    Defines :func:`colour.models.hunter_lab.XYZ_to_K_ab_HunterLab1966`
    definition unit tests methods.
    """

    def test_XYZ_to_K_ab_HunterLab1966(self):
        """
        Tests :func:`colour.models.hunter_lab.XYZ_to_K_ab_HunterLab1966`
        definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_K_ab_HunterLab1966(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100),
            np.array([46.92561332, 19.91297447]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_K_ab_HunterLab1966(
                np.array([0.47097710, 0.34950000, 0.11301649]) * 100),
            np.array([121.29129933, 21.65291746]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_K_ab_HunterLab1966(
                np.array([0.25506814, 0.19150000, 0.08849752]) * 100),
            np.array([89.26020100, 19.16068641]),
            decimal=7)

    def test_n_dimensional_XYZ_to_K_ab_HunterLab1966(self):
        """
        Tests :func:`colour.models.hunter_lab.XYZ_to_K_ab_HunterLab1966`
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
        Tests :func:`colour.models.hunter_lab.XYZ_to_K_ab_HunterLab1966`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ_to_K_ab_HunterLab1966(np.array(case))


class TestXYZ_to_Hunter_Lab(unittest.TestCase):
    """
    Defines :func:`colour.models.hunter_lab.XYZ_to_Hunter_Lab` definition unit
    tests methods.
    """

    def test_XYZ_to_Hunter_Lab(self):
        """
        Tests :func:`colour.models.hunter_lab.XYZ_to_Hunter_Lab` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100),
            np.array([31.74901573, -15.11462629, -2.78660758]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(
                np.array([0.47097710, 0.34950000, 0.11301649]) * 100),
            np.array([59.11852502, 40.84479160, 21.01328651]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(
                np.array([0.25506814, 0.19150000, 0.08849752]) * 100),
            np.array([43.76071297, 29.00314048, 11.24749156]),
            decimal=7)

        h_i = HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        A = h_i['A']
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                A.XYZ_n,
                A.K_ab),
            np.array([31.74901573, -21.35794415, -20.32778374]),
            decimal=7)

        D65 = h_i['D65']
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                D65.XYZ_n,
                D65.K_ab),
            np.array([31.74901573, -14.44108591, 2.74396261]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                D65.XYZ_n,
                K_ab=None),
            np.array([31.74901573, -14.43948953, 2.74352417]),
            decimal=7)

    def test_n_dimensional_XYZ_to_Hunter_Lab(self):
        """
        Tests :func:`colour.models.hunter_lab.XYZ_to_Hunter_Lab` definition
        n-dimensions support.
        """

        h_i = HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        D50 = h_i['D50']

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
        XYZ_n = D50.XYZ_n
        K_ab = D50.K_ab
        Lab = np.array([31.74901573, -15.11462629, -2.78660758])
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab),
            Lab,
            decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        Lab = np.tile(Lab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab),
            Lab,
            decimal=7)

        XYZ_n = np.tile(XYZ_n, (6, 1))
        K_ab = np.tile(K_ab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab),
            Lab,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
        K_ab = np.reshape(K_ab, (2, 3, 2))
        Lab = np.reshape(Lab, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab),
            Lab,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_Hunter_Lab(self):
        """
        Tests :func:`colour.models.hunter_lab.XYZ_to_Hunter_Lab` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_n = np.array(case[0:3])
            K_ab = np.array(case[0:2])
            XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab)


class TestHunter_Lab_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.hunter_lab.Hunter_Lab_to_XYZ` definition unit
    tests methods.
    """

    def test_Hunter_Lab_to_XYZ(self):
        """
        Tests :func:`colour.models.hunter_lab.Hunter_Lab_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(
                np.array([31.74901573, -15.11462629, -2.78660758])),
            np.array([7.04953400, 10.08000000, 9.55831300]),
            decimal=7)

        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(
                np.array([59.11852502, 40.84479160, 21.01328651])),
            np.array([47.09771001, 34.95000001, 11.30164900]),
            decimal=7)

        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(
                np.array([43.76071297, 29.00314048, 11.24749156])),
            np.array([25.50681403, 19.15000002, 8.84975199]),
            decimal=7)

        h_i = HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        A = h_i['A']
        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(
                np.array([31.74901573, -21.35794415, -20.32778374]),
                A.XYZ_n,
                A.K_ab),
            np.array([7.04953400, 10.08000000, 9.55831300]),
            decimal=7)

        D65 = h_i['D65']
        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(
                np.array([31.7490157, -14.4410859, 2.7439626]),
                D65.XYZ_n,
                D65.K_ab),
            np.array([7.04953400, 10.08000000, 9.55831300]),
            decimal=7)

        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(
                np.array([31.74901573, -14.43948953, 2.74352417]),
                D65.XYZ_n,
                K_ab=None),
            np.array([7.04953400, 10.08000000, 9.55831300]),
            decimal=7)

    def test_n_dimensional_Hunter_Lab_to_XYZ(self):
        """
        Tests :func:`colour.models.hunter_lab.Hunter_Lab_to_XYZ` definition
        n-dimensions support.
        """

        h_i = HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        D50 = h_i['D50']

        Lab = np.array([31.74901573, -15.11462629, -2.78660758])
        XYZ_n = D50.XYZ_n
        K_ab = D50.K_ab
        XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab),
            XYZ,
            decimal=7)

        Lab = np.tile(Lab, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab),
            XYZ,
            decimal=7)

        K_ab = np.tile(K_ab, (6, 1))
        XYZ_n = np.tile(XYZ_n, (6, 1))
        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab),
            XYZ,
            decimal=7)

        Lab = np.reshape(Lab, (2, 3, 3))
        XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
        K_ab = np.reshape(K_ab, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab),
            XYZ,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_Hunter_Lab_to_XYZ(self):
        """
        Tests :func:`colour.models.hunter_lab.Hunter_Lab_to_XYZ` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab = np.array(case)
            XYZ_n = np.array(case[0:3])
            K_ab = np.array(case[0:2])
            Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab)


if __name__ == '__main__':
    unittest.main()
