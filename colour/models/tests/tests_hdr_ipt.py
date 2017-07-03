#!/usr/bin/env python
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
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_hdr_IPT', 'TestHdr_IPT_to_XYZ', 'TestExponent_hdr_IPT']


class TestXYZ_to_hdr_IPT(unittest.TestCase):
    """
    Defines :func:`colour.models.hdr_ipt.TestXYZ_to_hdr_IPT` definition unit
    tests methods.
    """

    def test_XYZ_to_hdr_IPT(self):
        """
        Tests :func:`colour.models.hdr_ipt.XYZ_to_hdr_IPT` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_hdr_IPT(np.array([0.07049534, 0.10080000, 0.09558313])),
            np.array([25.18261761, -22.62111297, 3.18511729]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_hdr_IPT(
                np.array([0.07049534, 0.10080000, 0.09558313]), Y_s=0.5),
            np.array([34.60312115, -15.70974390, 2.26601353]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_hdr_IPT(
                np.array([0.25506814, 0.19150000, 0.08849752]), Y_abs=1000),
            np.array([47.18074546, 32.38073691, 29.13827648]),
            decimal=7)

    def test_n_dimensional_XYZ_to_hdr_IPT(self):
        """
        Tests :func:`colour.models.hdr_ipt.XYZ_to_hdr_IPT` definition
        n-dimensions support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        Y_s = 0.2
        Y_abs = 100
        IPT_hdr = np.array([25.18261761, -22.62111297, 3.18511729])
        np.testing.assert_almost_equal(
            XYZ_to_hdr_IPT(XYZ, Y_s, Y_abs), IPT_hdr, decimal=7)

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
            hdr_IPT_to_XYZ(np.array([25.18261761, -22.62111297, 3.18511729])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            hdr_IPT_to_XYZ(
                np.array([34.60312115, -15.70974390, 2.26601353]), Y_s=0.5),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            hdr_IPT_to_XYZ(
                np.array([47.18074546, 32.38073691, 29.13827648]), Y_abs=1000),
            np.array([0.25506814, 0.19150000, 0.08849752]),
            decimal=7)

    def test_n_dimensional_hdr_IPT_to_XYZ(self):
        """
        Tests :func:`colour.models.hdr_ipt.hdr_IPT_to_XYZ` definition
        n-dimensions support.
        """

        IPT_hdr = np.array([25.18261761, -22.62111297, 3.18511729])
        Y_s = 0.2
        Y_abs = 100
        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        np.testing.assert_almost_equal(
            hdr_IPT_to_XYZ(IPT_hdr, Y_s, Y_abs), XYZ, decimal=7)

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


class TestExponent_hdr_IPT(unittest.TestCase):
    """
    Defines :func:`colour.models.hdr_ipt.exponent_hdr_IPT`
    definition unit tests methods.
    """

    def test_exponent_hdr_IPT(self):
        """
        Tests :func:`colour.models.hdr_ipt.exponent_hdr_IPT`
        definition.
        """

        self.assertAlmostEqual(
            exponent_hdr_IPT(0.2, 100), 1.689138305989492, places=7)

        self.assertAlmostEqual(
            exponent_hdr_IPT(0.4, 100), 1.219933220992410, places=7)

        self.assertAlmostEqual(
            exponent_hdr_IPT(0.2, 1000), 1.126092203992995, places=7)

    def test_n_dimensional_exponent_hdr_IPT(self):
        """
        Tests :func:`colour.models.hdr_ipt.exponent_hdr_IPT`
        definition n-dimensional arrays support.
        """

        Y_s = 0.2
        Y_abs = 100
        e = 1.689138305989492
        np.testing.assert_almost_equal(
            exponent_hdr_IPT(Y_s, Y_abs), e, decimal=7)

        Y_s = np.tile(Y_s, 6)
        Y_abs = np.tile(Y_abs, 6)
        e = np.tile(e, 6)
        np.testing.assert_almost_equal(
            exponent_hdr_IPT(Y_s, Y_abs), e, decimal=7)

        Y_s = np.reshape(Y_s, (2, 3))
        Y_abs = np.reshape(Y_abs, (2, 3))
        e = np.reshape(e, (2, 3))
        np.testing.assert_almost_equal(
            exponent_hdr_IPT(Y_s, Y_abs), e, decimal=7)

        Y_s = np.reshape(Y_s, (2, 3, 1))
        Y_abs = np.reshape(Y_abs, (2, 3, 1))
        e = np.reshape(e, (2, 3, 1))
        np.testing.assert_almost_equal(
            exponent_hdr_IPT(Y_s, Y_abs), e, decimal=7)

    @ignore_numpy_errors
    def test_nan_exponent_hdr_IPT(self):
        """
        Tests :func:`colour.models.hdr_ipt.exponent_hdr_IPT`
        definition nan support.
        """

        cases = np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        exponent_hdr_IPT(cases, cases)


if __name__ == '__main__':
    unittest.main()
