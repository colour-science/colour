# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.hunter_rdab` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.colorimetry import TVS_ILLUMINANTS_HUNTERLAB
from colour.models import XYZ_to_Hunter_Rdab, Hunter_Rdab_to_XYZ

from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_Hunter_Rdab',
    'TestHunter_Rdab_to_XYZ',
]


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
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100),
            np.array([12.19722500, 57.12537874, 17.46241341]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(
                np.array([0.14222010, 0.23042768, 0.10495772]) * 100),
            np.array([23.04276800, -32.40057474, 20.96542183]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(
                np.array([0.07818780, 0.06157201, 0.28099326]) * 100),
            np.array([6.15720100, 18.13400284, -67.14408607]),
            decimal=7)

        h_i = TVS_ILLUMINANTS_HUNTERLAB['CIE 1931 2 Degree Standard Observer']
        A = h_i['A']
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100, A.XYZ_n,
                A.K_ab),
            np.array([12.19722500, 42.53572838, -3.00653110]),
            decimal=7)

        D65 = h_i['D65']
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
                D65.XYZ_n, D65.K_ab),
            np.array([12.19722500, 57.12537874, 17.46241341]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
                D65.XYZ_n,
                K_ab=None),
            np.array([12.19722500, 57.11906384, 17.45962317]),
            decimal=7)

    def test_n_dimensional_XYZ_to_Hunter_Rdab(self):
        """
        Tests :func:`colour.models.hunter_rdab.XYZ_to_Hunter_Rdab` definition
        n-dimensional support.
        """

        h_i = TVS_ILLUMINANTS_HUNTERLAB['CIE 1931 2 Degree Standard Observer']
        D65 = h_i['D65']

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        R_d_ab = XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab)

        XYZ = np.tile(XYZ, (6, 1))
        R_d_ab = np.tile(R_d_ab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab), R_d_ab, decimal=7)

        XYZ_n = np.tile(XYZ_n, (6, 1))
        K_ab = np.tile(K_ab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab), R_d_ab, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
        K_ab = np.reshape(K_ab, (2, 3, 2))
        R_d_ab = np.reshape(R_d_ab, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab), R_d_ab, decimal=7)

    def test_domain_range_scale_XYZ_to_Hunter_Rdab(self):
        """
        Tests :func:`colour.models.hunter_lab.XYZ_to_Hunter_Rdab` definition
        domain and range scale support.
        """

        h_i = TVS_ILLUMINANTS_HUNTERLAB['CIE 1931 2 Degree Standard Observer']
        D65 = h_i['D65']

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        R_d_ab = XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab)

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_Hunter_Rdab(XYZ * factor, XYZ_n * factor, K_ab),
                    R_d_ab * factor,
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


class TestHunter_Rdab_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.hunter_rdab.Hunter_Rdab_to_XYZ` definition
    unit tests methods.
    """

    def test_Hunter_Rdab_to_XYZ(self):
        """
        Tests :func:`colour.models.hunter_rdab.Hunter_Rdab_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            Hunter_Rdab_to_XYZ(
                np.array([12.19722500, 57.12537874, 17.46241341])),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            Hunter_Rdab_to_XYZ(
                np.array([23.04276800, -32.40057474, 20.96542183])),
            np.array([0.14222010, 0.23042768, 0.10495772]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            Hunter_Rdab_to_XYZ(
                np.array([6.15720100, 18.13400284, -67.14408607])),
            np.array([0.07818780, 0.06157201, 0.28099326]) * 100,
            decimal=7)

        h_i = TVS_ILLUMINANTS_HUNTERLAB['CIE 1931 2 Degree Standard Observer']
        A = h_i['A']
        np.testing.assert_almost_equal(
            Hunter_Rdab_to_XYZ(
                np.array([12.19722500, 42.53572838, -3.00653110]), A.XYZ_n,
                A.K_ab),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            decimal=7)

        D65 = h_i['D65']
        np.testing.assert_almost_equal(
            Hunter_Rdab_to_XYZ(
                np.array([12.19722500, 57.12537874, 17.46241341]), D65.XYZ_n,
                D65.K_ab),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            Hunter_Rdab_to_XYZ(
                np.array([12.19722500, 57.11906384, 17.45962317]),
                D65.XYZ_n,
                K_ab=None),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            decimal=7)

    def test_n_dimensional_Hunter_Rdab_to_XYZ(self):
        """
        Tests :func:`colour.models.hunter_rdab.Hunter_Rdab_to_XYZ` definition
        n-dimensional support.
        """

        h_i = TVS_ILLUMINANTS_HUNTERLAB['CIE 1931 2 Degree Standard Observer']
        D65 = h_i['D65']

        R_d_ab = np.array([12.19722500, 57.12537874, 17.46241341])
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        XYZ = Hunter_Rdab_to_XYZ(R_d_ab, XYZ_n, K_ab)

        R_d_ab = np.tile(R_d_ab, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            Hunter_Rdab_to_XYZ(R_d_ab, XYZ_n, K_ab), XYZ, decimal=7)

        K_ab = np.tile(K_ab, (6, 1))
        XYZ_n = np.tile(XYZ_n, (6, 1))
        np.testing.assert_almost_equal(
            Hunter_Rdab_to_XYZ(R_d_ab, XYZ_n, K_ab), XYZ, decimal=7)

        R_d_ab = np.reshape(R_d_ab, (2, 3, 3))
        XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
        K_ab = np.reshape(K_ab, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            Hunter_Rdab_to_XYZ(R_d_ab, XYZ_n, K_ab), XYZ, decimal=7)

    def test_domain_range_scale_Hunter_Rdab_to_XYZ(self):
        """
        Tests :func:`colour.models.hunter_lab.Hunter_Rdab_to_XYZ` definition
        domain and range scale support.
        """

        h_i = TVS_ILLUMINANTS_HUNTERLAB['CIE 1931 2 Degree Standard Observer']
        D65 = h_i['D65']

        R_d_ab = np.array([12.19722500, 57.12537874, 17.46241341])
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        XYZ = Hunter_Rdab_to_XYZ(R_d_ab, XYZ_n, K_ab)

        d_r = (('reference', 1), ('1', 0.01), ('100', 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    Hunter_Rdab_to_XYZ(R_d_ab * factor, XYZ_n * factor, K_ab),
                    XYZ * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_Hunter_Rdab_to_XYZ(self):
        """
        Tests :func:`colour.models.hunter_rdab.Hunter_Rdab_to_XYZ` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            R_d_ab = np.array(case)
            XYZ_n = np.array(case[0:3])
            K_ab = np.array(case[0:2])
            Hunter_Rdab_to_XYZ(R_d_ab, XYZ_n, K_ab)


if __name__ == '__main__':
    unittest.main()
