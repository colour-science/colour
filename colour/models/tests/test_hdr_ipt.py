# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.hdr_ipt` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_hdr_IPT, hdr_IPT_to_XYZ
from colour.models.hdr_ipt import exponent_hdr_IPT
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestExponent_hdr_IPT', 'TestXYZ_to_hdr_IPT', 'TestHdr_IPT_to_XYZ']


class TestExponent_hdr_IPT(unittest.TestCase):
    """
    Defines :func:`colour.models.hdr_ipt.exponent_hdr_IPT`
    definition unit tests methods.
    """

    def test_exponent_hdr_IPT(self):
        """
        Tests :func:`colour.models.hdr_ipt.exponent_hdr_IPT` definition.
        """

        self.assertAlmostEqual(
            exponent_hdr_IPT(0.2, 100), 0.482020919845900, places=7)

        self.assertAlmostEqual(
            exponent_hdr_IPT(0.4, 100), 0.667413581325092, places=7)

        self.assertAlmostEqual(
            exponent_hdr_IPT(0.4, 100, method='Fairchild 2010'),
            1.219933220992410,
            places=7)

        self.assertAlmostEqual(
            exponent_hdr_IPT(0.2, 1000), 0.723031379768850, places=7)

    def test_n_dimensional_exponent_hdr_IPT(self):
        """
        Tests :func:`colour.models.hdr_ipt.exponent_hdr_IPT` definition
        n-dimensional arrays support.
        """

        Y_s = 0.2
        Y_abs = 100
        epsilon = exponent_hdr_IPT(Y_s, Y_abs)

        Y_s = np.tile(Y_s, 6)
        Y_abs = np.tile(Y_abs, 6)
        epsilon = np.tile(epsilon, 6)
        np.testing.assert_almost_equal(
            exponent_hdr_IPT(Y_s, Y_abs), epsilon, decimal=7)

        Y_s = np.reshape(Y_s, (2, 3))
        Y_abs = np.reshape(Y_abs, (2, 3))
        epsilon = np.reshape(epsilon, (2, 3))
        np.testing.assert_almost_equal(
            exponent_hdr_IPT(Y_s, Y_abs), epsilon, decimal=7)

        Y_s = np.reshape(Y_s, (2, 3, 1))
        Y_abs = np.reshape(Y_abs, (2, 3, 1))
        epsilon = np.reshape(epsilon, (2, 3, 1))
        np.testing.assert_almost_equal(
            exponent_hdr_IPT(Y_s, Y_abs), epsilon, decimal=7)

    def test_domain_range_scale_exponent_hdr_IPT(self):
        """
        Tests :func:`colour.models.hdr_ipt.exponent_hdr_IPT` definition domain
        and range scale support.
        """

        Y_s = 0.2
        Y_abs = 100
        epsilon = exponent_hdr_IPT(Y_s, Y_abs)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    exponent_hdr_IPT(Y_s * factor, Y_abs), epsilon, decimal=7)

    @ignore_numpy_errors
    def test_nan_exponent_hdr_IPT(self):
        """
        Tests :func:`colour.models.hdr_ipt.exponent_hdr_IPT` definition nan
        support.
        """

        cases = np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        exponent_hdr_IPT(cases, cases)


class TestXYZ_to_hdr_IPT(unittest.TestCase):
    """
    Defines :func:`colour.models.hdr_ipt.XYZ_to_hdr_IPT` definition unit tests
    methods.
    """

    def test_XYZ_to_hdr_IPT(self):
        """
        Tests :func:`colour.models.hdr_ipt.XYZ_to_hdr_IPT` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_hdr_IPT(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([48.39376346, 42.44990202, 22.01954033]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_hdr_IPT(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                method='Fairchild 2010'),
            np.array([30.02873147, 83.93845061, 34.90287382]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_hdr_IPT(
                np.array([0.20654008, 0.12197225, 0.05136952]), Y_s=0.5),
            np.array([20.75088680, 37.98300971, 16.66974299]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_hdr_IPT(
                np.array([0.07818780, 0.06157201, 0.28099326]), Y_abs=1000),
            np.array([23.83205010, -5.98739209, -32.74311745]),
            decimal=7)

    def test_n_dimensional_XYZ_to_hdr_IPT(self):
        """
        Tests :func:`colour.models.hdr_ipt.XYZ_to_hdr_IPT` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Y_s = 0.2
        Y_abs = 100
        IPT_hdr = XYZ_to_hdr_IPT(XYZ, Y_s, Y_abs)

        XYZ = np.tile(XYZ, (6, 1))
        IPT_hdr = np.tile(IPT_hdr, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_hdr_IPT(XYZ, Y_s, Y_abs), IPT_hdr, decimal=7)

        Y_s = np.tile(Y_s, 6)
        Y_abs = np.tile(Y_abs, 6)
        np.testing.assert_almost_equal(
            XYZ_to_hdr_IPT(XYZ, Y_s, Y_abs), IPT_hdr, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        Y_s = np.reshape(Y_s, (2, 3))
        Y_abs = np.reshape(Y_abs, (2, 3))
        IPT_hdr = np.reshape(IPT_hdr, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_hdr_IPT(XYZ, Y_s, Y_abs), IPT_hdr, decimal=7)

    def test_domain_range_scale_XYZ_to_hdr_IPT(self):
        """
        Tests :func:`colour.models.hdr_ipt.XYZ_to_hdr_IPT` definition domain
        and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Y_s = 0.2
        Y_abs = 100
        IPT_hdr = XYZ_to_hdr_IPT(XYZ, Y_s, Y_abs)

        d_r = (('reference', 1, 1), (1, 1, 0.01), (100, 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_hdr_IPT(XYZ * factor_a, Y_s * factor_a, Y_abs),
                    IPT_hdr * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_hdr_IPT(self):
        """
        Tests :func:`colour.models.hdr_ipt.XYZ_to_hdr_IPT` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            Y_s = case[0]
            Y_abs = case[0]
            XYZ_to_hdr_IPT(XYZ, Y_s, Y_abs)


class TestHdr_IPT_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.hdr_ipt.hdr_IPT_to_XYZ` definition unit tests
    methods.
    """

    def test_hdr_IPT_to_XYZ(self):
        """
        Tests :func:`colour.models.hdr_ipt.hdr_IPT_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            hdr_IPT_to_XYZ(np.array([48.39376346, 42.44990202, 22.01954033])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            hdr_IPT_to_XYZ(
                np.array([30.02873147, 83.93845061, 34.90287382]),
                method='Fairchild 2010'),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            hdr_IPT_to_XYZ(
                np.array([20.75088680, 37.98300971, 16.66974299]), Y_s=0.5),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            hdr_IPT_to_XYZ(
                np.array([23.83205010, -5.98739209, -32.74311745]),
                Y_abs=1000),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            decimal=7)

    def test_n_dimensional_hdr_IPT_to_XYZ(self):
        """
        Tests :func:`colour.models.hdr_ipt.hdr_IPT_to_XYZ` definition
        n-dimensional support.
        """

        IPT_hdr = np.array([48.39376346, 42.44990202, 22.01954033])
        Y_s = 0.2
        Y_abs = 100
        XYZ = hdr_IPT_to_XYZ(IPT_hdr, Y_s, Y_abs)

        IPT_hdr = np.tile(IPT_hdr, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            hdr_IPT_to_XYZ(IPT_hdr, Y_s, Y_abs), XYZ, decimal=7)

        Y_s = np.tile(Y_s, 6)
        Y_abs = np.tile(Y_abs, 6)
        np.testing.assert_almost_equal(
            hdr_IPT_to_XYZ(IPT_hdr, Y_s, Y_abs), XYZ, decimal=7)

        IPT_hdr = np.reshape(IPT_hdr, (2, 3, 3))
        Y_s = np.reshape(Y_s, (2, 3))
        Y_abs = np.reshape(Y_abs, (2, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            hdr_IPT_to_XYZ(IPT_hdr, Y_s, Y_abs), XYZ, decimal=7)

    def test_domain_range_scale_hdr_IPT_to_XYZ(self):
        """
        Tests :func:`colour.models.hdr_ipt.hdr_IPT_to_XYZ` definition domain
        and range scale support.
        """

        IPT_hdr = np.array([24.88927680, -11.44574144, 1.63147707])
        Y_s = 0.2
        Y_abs = 100
        XYZ = hdr_IPT_to_XYZ(IPT_hdr, Y_s, Y_abs)

        d_r = (('reference', 1, 1, 1), (1, 0.01, 1, 1), (100, 1, 100, 100))
        for scale, factor_a, factor_b, factor_c in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    hdr_IPT_to_XYZ(IPT_hdr * factor_a, Y_s * factor_b, Y_abs),
                    XYZ * factor_c,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_hdr_IPT_to_XYZ(self):
        """
        Tests :func:`colour.models.hdr_ipt.hdr_IPT_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            IPT_hdr = np.array(case)
            Y_s = case[0]
            Y_abs = case[0]
            hdr_IPT_to_XYZ(IPT_hdr, Y_s, Y_abs)


if __name__ == '__main__':
    unittest.main()
