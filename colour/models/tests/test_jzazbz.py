# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.jzazbz` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.models import (
    XYZ_to_Izazbz,
    Izazbz_to_XYZ,
    XYZ_to_Jzazbz,
    Jzazbz_to_XYZ,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_Izazbz',
    'TestIzazbz_to_XYZ',
    'TestXYZ_to_Jzazbz',
    'TestJzazbz_to_XYZ',
]


class TestXYZ_to_Izazbz(unittest.TestCase):
    """
    Defines :func:`colour.models.jzazbz.TestXYZ_to_Izazbz` definition unit
    tests methods.
    """

    def test_XYZ_to_Izazbz(self):
        """
        Tests :func:`colour.models.jzazbz.XYZ_to_Izazbz` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_Izazbz(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.01207793, 0.00924302, 0.00526007]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Izazbz(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.01397346, -0.00608426, 0.00534077]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Izazbz(np.array([0.96907232, 1.00000000, 1.12179215])),
            np.array([0.03927203, 0.00064174, -0.00052906]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Izazbz(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                method='Safdar 2021'),
            np.array([0.01049146, 0.00924302, 0.00526007]),
            decimal=7)

        np.testing.assert_array_equal(
            XYZ_to_Izazbz(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                method='Safdar 2021'),
            XYZ_to_Izazbz(
                np.array([0.20654008, 0.12197225, 0.05136952]), method='ZCAM'))

    def test_n_dimensional_XYZ_to_Izazbz(self):
        """
        Tests :func:`colour.models.jzazbz.XYZ_to_Izazbz` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Izazbz = XYZ_to_Izazbz(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        Izazbz = np.tile(Izazbz, (6, 1))
        np.testing.assert_almost_equal(XYZ_to_Izazbz(XYZ), Izazbz, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        Izazbz = np.reshape(Izazbz, (2, 3, 3))
        np.testing.assert_almost_equal(XYZ_to_Izazbz(XYZ), Izazbz, decimal=7)

    def test_domain_range_scale_XYZ_to_Izazbz(self):
        """
        Tests :func:`colour.models.jzazbz.XYZ_to_Izazbz` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Izazbz = XYZ_to_Izazbz(XYZ)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_Izazbz(XYZ * factor), Izazbz * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_Izazbz(self):
        """
        Tests :func:`colour.models.jzazbz.XYZ_to_Izazbz` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_to_Izazbz(XYZ)


class TestIzazbz_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.jzazbz.Izazbz_to_XYZ` definition unit tests
    methods.
    """

    def test_Izazbz_to_XYZ(self):
        """
        Tests :func:`colour.models.jzazbz.Izazbz_to_XYZ` definition.
        """

        np.testing.assert_allclose(
            Izazbz_to_XYZ(np.array([0.01207793, 0.00924302, 0.00526007])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            rtol=0.000001,
            atol=0.000001)

        np.testing.assert_allclose(
            Izazbz_to_XYZ(np.array([0.01397346, -0.00608426, 0.00534077])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            rtol=0.000001,
            atol=0.000001)

        np.testing.assert_allclose(
            Izazbz_to_XYZ(np.array([0.03927203, 0.00064174, -0.00052906])),
            np.array([0.96907232, 1.00000000, 1.12179215]),
            rtol=0.000001,
            atol=0.000001)

        np.testing.assert_allclose(
            Izazbz_to_XYZ(np.array([0.03927203, 0.00064174, -0.00052906])),
            np.array([0.96907232, 1.00000000, 1.12179215]),
            rtol=0.000001,
            atol=0.000001)

        np.testing.assert_almost_equal(
            Izazbz_to_XYZ(
                np.array([0.01049146, 0.00924302, 0.00526007]),
                method='Safdar 2021'),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_array_equal(
            Izazbz_to_XYZ(
                np.array([0.01049146, 0.00924302, 0.00526007]),
                method='Safdar 2021'),
            Izazbz_to_XYZ(
                np.array([0.01049146, 0.00924302, 0.00526007]), method='ZCAM'))

    def test_n_dimensional_Izazbz_to_XYZ(self):
        """
        Tests :func:`colour.models.jzazbz.Izazbz_to_XYZ` definition
        n-dimensional support.
        """

        Izazbz = np.array([0.01207793, 0.00924302, 0.00526007])
        XYZ = Izazbz_to_XYZ(Izazbz)

        Izazbz = np.tile(Izazbz, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            Izazbz_to_XYZ(Izazbz), XYZ, rtol=0.000001, atol=0.000001)

        Izazbz = np.reshape(Izazbz, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            Izazbz_to_XYZ(Izazbz), XYZ, rtol=0.000001, atol=0.000001)

    def test_domain_range_scale_Izazbz_to_XYZ(self):
        """
        Tests :func:`colour.models.jzazbz.Izazbz_to_XYZ` definition domain and
        range scale support.
        """

        Izazbz = np.array([0.01207793, 0.00924302, 0.00526007])
        XYZ = Izazbz_to_XYZ(Izazbz)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    Izazbz_to_XYZ(Izazbz * factor),
                    XYZ * factor,
                    rtol=0.000001,
                    atol=0.000001)

    @ignore_numpy_errors
    def test_nan_Izazbz_to_XYZ(self):
        """
        Tests :func:`colour.models.jzazbz.Izazbz_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Izazbz = np.array(case)
            Izazbz_to_XYZ(Izazbz)


class TestXYZ_to_Jzazbz(unittest.TestCase):
    """
    Defines :func:`colour.models.jzazbz.TestXYZ_to_Jzazbz` definition unit
    tests methods.
    """

    def test_XYZ_to_Jzazbz(self):
        """
        Tests :func:`colour.models.jzazbz.XYZ_to_Jzazbz` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_Jzazbz(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.00535048, 0.00924302, 0.00526007]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Jzazbz(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.00619681, -0.00608426, 0.00534077]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Jzazbz(np.array([0.96907232, 1.00000000, 1.12179215])),
            np.array([0.01766826, 0.00064174, -0.00052906]),
            decimal=7)

    def test_n_dimensional_XYZ_to_Jzazbz(self):
        """
        Tests :func:`colour.models.jzazbz.XYZ_to_Jzazbz` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Jzazbz = XYZ_to_Jzazbz(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        Jzazbz = np.tile(Jzazbz, (6, 1))
        np.testing.assert_almost_equal(XYZ_to_Jzazbz(XYZ), Jzazbz, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        Jzazbz = np.reshape(Jzazbz, (2, 3, 3))
        np.testing.assert_almost_equal(XYZ_to_Jzazbz(XYZ), Jzazbz, decimal=7)

    def test_domain_range_scale_XYZ_to_Jzazbz(self):
        """
        Tests :func:`colour.models.jzazbz.XYZ_to_Jzazbz` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Jzazbz = XYZ_to_Jzazbz(XYZ)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_Jzazbz(XYZ * factor), Jzazbz * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_Jzazbz(self):
        """
        Tests :func:`colour.models.jzazbz.XYZ_to_Jzazbz` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_to_Jzazbz(XYZ)


class TestJzazbz_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.jzazbz.Jzazbz_to_XYZ` definition unit tests
    methods.
    """

    def test_Jzazbz_to_XYZ(self):
        """
        Tests :func:`colour.models.jzazbz.Jzazbz_to_XYZ` definition.
        """

        np.testing.assert_allclose(
            Jzazbz_to_XYZ(np.array([0.00535048, 0.00924302, 0.00526007])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            rtol=0.000001,
            atol=0.000001)

        np.testing.assert_allclose(
            Jzazbz_to_XYZ(np.array([0.00619681, -0.00608426, 0.00534077])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            rtol=0.000001,
            atol=0.000001)

        np.testing.assert_allclose(
            Jzazbz_to_XYZ(np.array([0.01766826, 0.00064174, -0.00052906])),
            np.array([0.96907232, 1.00000000, 1.12179215]),
            rtol=0.000001,
            atol=0.000001)

    def test_n_dimensional_Jzazbz_to_XYZ(self):
        """
        Tests :func:`colour.models.jzazbz.Jzazbz_to_XYZ` definition
        n-dimensional support.
        """

        Jzazbz = np.array([0.00535048, 0.00924302, 0.00526007])
        XYZ = Jzazbz_to_XYZ(Jzazbz)

        Jzazbz = np.tile(Jzazbz, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            Jzazbz_to_XYZ(Jzazbz), XYZ, rtol=0.000001, atol=0.000001)

        Jzazbz = np.reshape(Jzazbz, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            Jzazbz_to_XYZ(Jzazbz), XYZ, rtol=0.000001, atol=0.000001)

    def test_domain_range_scale_Jzazbz_to_XYZ(self):
        """
        Tests :func:`colour.models.jzazbz.Jzazbz_to_XYZ` definition domain and
        range scale support.
        """

        Jzazbz = np.array([0.00535048, 0.00924302, 0.00526007])
        XYZ = Jzazbz_to_XYZ(Jzazbz)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    Jzazbz_to_XYZ(Jzazbz * factor),
                    XYZ * factor,
                    rtol=0.000001,
                    atol=0.000001)

    @ignore_numpy_errors
    def test_nan_Jzazbz_to_XYZ(self):
        """
        Tests :func:`colour.models.jzazbz.Jzazbz_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Jzazbz = np.array(case)
            Jzazbz_to_XYZ(Jzazbz)


if __name__ == '__main__':
    unittest.main()
