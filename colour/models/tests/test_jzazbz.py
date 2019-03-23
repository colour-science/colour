# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.jzazbz` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_JzAzBz, JzAzBz_to_XYZ
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_JzAzBz', 'TestJzAzBz_to_XYZ']


class TestXYZ_to_JzAzBz(unittest.TestCase):
    """
    Defines :func:`colour.models.jzazbz.TestXYZ_to_JzAzBz` definition unit
    tests methods.
    """

    def test_XYZ_to_JzAzBz(self):
        """
        Tests :func:`colour.models.jzazbz.XYZ_to_JzAzBz` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_JzAzBz(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.00535048, 0.00924302, 0.00526007]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_JzAzBz(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.00619681, -0.00608426, 0.00534077]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_JzAzBz(np.array([0.96907232, 1.00000000, 1.12179215])),
            np.array([0.01766826, 0.00064174, -0.00052906]),
            decimal=7)

    def test_n_dimensional_XYZ_to_JzAzBz(self):
        """
        Tests :func:`colour.models.jzazbz.XYZ_to_JzAzBz` definition
        n-dimensions support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        JzAzBz = np.array([0.00535048, 0.00924302, 0.00526007])
        np.testing.assert_almost_equal(XYZ_to_JzAzBz(XYZ), JzAzBz, decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        JzAzBz = np.tile(JzAzBz, (6, 1))
        np.testing.assert_almost_equal(XYZ_to_JzAzBz(XYZ), JzAzBz, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        JzAzBz = np.reshape(JzAzBz, (2, 3, 3))
        np.testing.assert_almost_equal(XYZ_to_JzAzBz(XYZ), JzAzBz, decimal=7)

    def test_domain_range_scale_XYZ_to_JzAzBz(self):
        """
        Tests :func:`colour.models.jzazbz.XYZ_to_JzAzBz` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        JzAzBz = XYZ_to_JzAzBz(XYZ)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_JzAzBz(XYZ * factor), JzAzBz * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_JzAzBz(self):
        """
        Tests :func:`colour.models.jzazbz.XYZ_to_JzAzBz` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_to_JzAzBz(XYZ)


class TestJzAzBz_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.jzazbz.JzAzBz_to_XYZ` definition unit tests
    methods.
    """

    def test_JzAzBz_to_XYZ(self):
        """
        Tests :func:`colour.models.jzazbz.JzAzBz_to_XYZ` definition.
        """

        np.testing.assert_allclose(
            JzAzBz_to_XYZ(np.array([0.00535048, 0.00924302, 0.00526007])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            rtol=0.000001,
            atol=0.000001)

        np.testing.assert_allclose(
            JzAzBz_to_XYZ(np.array([0.00619681, -0.00608426, 0.00534077])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            rtol=0.000001,
            atol=0.000001)

        np.testing.assert_allclose(
            JzAzBz_to_XYZ(np.array([0.01766826, 0.00064174, -0.00052906])),
            np.array([0.96907232, 1.00000000, 1.12179215]),
            rtol=0.000001,
            atol=0.000001)

    def test_n_dimensional_JzAzBz_to_XYZ(self):
        """
        Tests :func:`colour.models.jzazbz.JzAzBz_to_XYZ` definition
        n-dimensions support.
        """

        JzAzBz = np.array([0.00535048, 0.00924302, 0.00526007])
        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        np.testing.assert_allclose(
            JzAzBz_to_XYZ(JzAzBz), XYZ, rtol=0.000001, atol=0.000001)

        JzAzBz = np.tile(JzAzBz, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            JzAzBz_to_XYZ(JzAzBz), XYZ, rtol=0.000001, atol=0.000001)

        JzAzBz = np.reshape(JzAzBz, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            JzAzBz_to_XYZ(JzAzBz), XYZ, rtol=0.000001, atol=0.000001)

    def test_domain_range_scale_JzAzBz_to_XYZ(self):
        """
        Tests :func:`colour.models.jzazbz.JzAzBz_to_XYZ` definition domain and
        range scale support.
        """

        JzAzBz = np.array([0.00535048, 0.00924302, 0.00526007])
        XYZ = JzAzBz_to_XYZ(JzAzBz)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    JzAzBz_to_XYZ(JzAzBz * factor),
                    XYZ * factor,
                    rtol=0.000001,
                    atol=0.000001)

    @ignore_numpy_errors
    def test_nan_JzAzBz_to_XYZ(self):
        """
        Tests :func:`colour.models.jzazbz.JzAzBz_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            JzAzBz = np.array(case)
            JzAzBz_to_XYZ(JzAzBz)


if __name__ == '__main__':
    unittest.main()
