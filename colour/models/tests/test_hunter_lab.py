# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.hunter_lab` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.colorimetry import HUNTERLAB_ILLUMINANTS
from colour.models import (XYZ_to_K_ab_HunterLab1966, XYZ_to_Hunter_Lab,
                           Hunter_Lab_to_XYZ)

from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_K_ab_HunterLab1966', 'TestXYZ_to_Hunter_Lab',
    'TestHunter_Lab_to_XYZ'
]


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
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100),
            np.array([80.32152090, 14.59816495]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_K_ab_HunterLab1966(
                np.array([0.14222010, 0.23042768, 0.10495772]) * 100),
            np.array([66.65154834, 20.86664881]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_K_ab_HunterLab1966(
                np.array([0.07818780, 0.06157201, 0.28099326]) * 100),
            np.array([49.41960269, 34.14235426]),
            decimal=7)

    def test_n_dimensional_XYZ_to_K_ab_HunterLab1966(self):
        """
        Tests :func:`colour.models.hunter_lab.XYZ_to_K_ab_HunterLab1966`
        definition n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
        K_ab = XYZ_to_K_ab_HunterLab1966(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        K_ab = np.tile(K_ab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_K_ab_HunterLab1966(XYZ), K_ab, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        K_ab = np.reshape(K_ab, (2, 3, 2))
        np.testing.assert_almost_equal(
            XYZ_to_K_ab_HunterLab1966(XYZ), K_ab, decimal=7)

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
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100),
            np.array([34.92452577, 47.06189858, 14.38615107]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(
                np.array([0.14222010, 0.23042768, 0.10495772]) * 100),
            np.array([48.00288325, -28.98551622, 18.75564181]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(
                np.array([0.07818780, 0.06157201, 0.28099326]) * 100),
            np.array([24.81370791, 14.38300039, -53.25539126]),
            decimal=7)

        h_i = HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        A = h_i['A']
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100, A.XYZ_n,
                A.K_ab),
            np.array([34.92452577, 35.04243086, -2.47688619]),
            decimal=7)

        D65 = h_i['D65']
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
                D65.XYZ_n, D65.K_ab),
            np.array([34.92452577, 47.06189858, 14.38615107]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
                D65.XYZ_n,
                K_ab=None),
            np.array([34.92452577, 47.05669614, 14.38385238]),
            decimal=7)

    def test_n_dimensional_XYZ_to_Hunter_Lab(self):
        """
        Tests :func:`colour.models.hunter_lab.XYZ_to_Hunter_Lab` definition
        n-dimensional support.
        """

        h_i = HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        D65 = h_i['D65']

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        Lab = XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab)

        XYZ = np.tile(XYZ, (6, 1))
        Lab = np.tile(Lab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab), Lab, decimal=7)

        XYZ_n = np.tile(XYZ_n, (6, 1))
        K_ab = np.tile(K_ab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab), Lab, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
        K_ab = np.reshape(K_ab, (2, 3, 2))
        Lab = np.reshape(Lab, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab), Lab, decimal=7)

    def test_domain_range_scale_XYZ_to_Hunter_Lab(self):
        """
        Tests :func:`colour.models.hunter_lab.XYZ_to_Hunter_Lab` definition
        domain and range scale support.
        """

        h_i = HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        D65 = h_i['D65']

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        Lab = XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_Hunter_Lab(XYZ * factor, XYZ_n * factor, K_ab),
                    Lab * factor,
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
                np.array([34.92452577, 47.06189858, 14.38615107])),
            np.array([20.65400800, 12.19722500, 5.13695200]),
            decimal=7)

        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(
                np.array([48.00288325, -28.98551622, 18.75564181])),
            np.array([14.22201000, 23.04276800, 10.49577200]),
            decimal=7)

        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(
                np.array([24.81370791, 14.38300039, -53.25539126])),
            np.array([7.81878000, 6.15720100, 28.09932601]),
            decimal=7)

        h_i = HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        A = h_i['A']
        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(
                np.array([34.92452577, 35.04243086, -2.47688619]), A.XYZ_n,
                A.K_ab),
            np.array([20.65400800, 12.19722500, 5.13695200]),
            decimal=7)

        D65 = h_i['D65']
        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(
                np.array([34.92452577, 47.06189858, 14.38615107]), D65.XYZ_n,
                D65.K_ab),
            np.array([20.65400800, 12.19722500, 5.13695200]),
            decimal=7)

        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(
                np.array([34.92452577, 47.05669614, 14.38385238]),
                D65.XYZ_n,
                K_ab=None),
            np.array([20.65400800, 12.19722500, 5.13695200]),
            decimal=7)

    def test_n_dimensional_Hunter_Lab_to_XYZ(self):
        """
        Tests :func:`colour.models.hunter_lab.Hunter_Lab_to_XYZ` definition
        n-dimensional support.
        """

        h_i = HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        D65 = h_i['D65']

        Lab = np.array([34.92452577, 47.06189858, 14.38615107])
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        XYZ = Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab)

        Lab = np.tile(Lab, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab), XYZ, decimal=7)

        K_ab = np.tile(K_ab, (6, 1))
        XYZ_n = np.tile(XYZ_n, (6, 1))
        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab), XYZ, decimal=7)

        Lab = np.reshape(Lab, (2, 3, 3))
        XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
        K_ab = np.reshape(K_ab, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab), XYZ, decimal=7)

    def test_domain_range_scale_Hunter_Lab_to_XYZ(self):
        """
        Tests :func:`colour.models.hunter_lab.Hunter_Lab_to_XYZ` definition
        domain and range scale support.
        """

        h_i = HUNTERLAB_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']
        D65 = h_i['D65']

        Lab = np.array([34.92452577, 47.06189858, 14.38615107])
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        XYZ = Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    Hunter_Lab_to_XYZ(Lab * factor, XYZ_n * factor, K_ab),
                    XYZ * factor,
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
