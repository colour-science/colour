# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.algebra.coordinates.transformations`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.algebra import (cartesian_to_spherical, spherical_to_cartesian,
                            cartesian_to_polar, polar_to_cartesian,
                            cartesian_to_cylindrical, cylindrical_to_cartesian)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestCartesianToSpherical', 'TestSphericalToCartesian',
    'TestCartesianToPolar', 'TestPolarToCartesian',
    'TestCartesianToCylindrical', 'TestCylindricalToCartesian'
]


class TestCartesianToSpherical(unittest.TestCase):
    """
    Defines :func:`colour.algebra.coordinates.transformations.\
cartesian_to_spherical` definition unit tests methods.
    """

    def test_cartesian_to_spherical(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cartesian_to_spherical` definition.
        """

        np.testing.assert_almost_equal(
            cartesian_to_spherical(np.array([3, 1, 6])),
            np.array([6.78232998, 1.08574654, 0.32175055]),
            decimal=7)

        np.testing.assert_almost_equal(
            cartesian_to_spherical(np.array([-1, 9, 16])),
            np.array([18.38477631, 1.05578119, 1.68145355]),
            decimal=7)

        np.testing.assert_almost_equal(
            cartesian_to_spherical(np.array([6.3434, -0.9345, 18.5675])),
            np.array([19.64342307, 1.23829030, -0.14626640]),
            decimal=7)

    def test_n_dimensional_cartesian_to_spherical(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cartesian_to_spherical` definition n-dimensional arrays support.
        """

        a_i = np.array([3, 1, 6])
        a_o = cartesian_to_spherical(a_i)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_almost_equal(
            cartesian_to_spherical(a_i), a_o, decimal=7)

        a_i = np.reshape(a_i, (2, 3, 3))
        a_o = np.reshape(a_o, (2, 3, 3))
        np.testing.assert_almost_equal(
            cartesian_to_spherical(a_i), a_o, decimal=7)

    @ignore_numpy_errors
    def test_nan_cartesian_to_spherical(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cartesian_to_spherical` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            a_i = np.array(case)
            cartesian_to_spherical(a_i)


class TestSphericalToCartesian(unittest.TestCase):
    """
    Defines :func:`colour.algebra.coordinates.transformations.\
spherical_to_cartesian` definition unit tests methods.
    """

    def test_spherical_to_cartesian(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
spherical_to_cartesian` definition.
        """

        np.testing.assert_almost_equal(
            spherical_to_cartesian(
                np.array([6.78232998, 1.08574654, 0.32175055])),
            np.array([3.00000000, 0.99999999, 6.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            spherical_to_cartesian(
                np.array([18.38477631, 1.05578119, 1.68145355])),
            np.array([-1.00000003, 9.00000007, 15.99999996]),
            decimal=7)

        np.testing.assert_almost_equal(
            spherical_to_cartesian(
                np.array([19.64342307, 1.23829030, -0.14626640])),
            np.array([6.34339996, -0.93449999, 18.56750001]),
            decimal=7)

    def test_n_dimensional_spherical_to_cartesian(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
spherical_to_cartesian` definition n-dimensional arrays support.
        """

        a_i = np.array([6.78232998, 1.08574654, 0.32175055])
        a_o = spherical_to_cartesian(a_i)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_almost_equal(
            spherical_to_cartesian(a_i), a_o, decimal=7)

        a_i = np.reshape(a_i, (2, 3, 3))
        a_o = np.reshape(a_o, (2, 3, 3))
        np.testing.assert_almost_equal(
            spherical_to_cartesian(a_i), a_o, decimal=7)

    @ignore_numpy_errors
    def test_nan_spherical_to_cartesian(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
spherical_to_cartesian` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            a_i = np.array(case)
            spherical_to_cartesian(a_i)


class TestCartesianToPolar(unittest.TestCase):
    """
    Defines :func:`colour.algebra.coordinates.transformations.\
cartesian_to_polar` definition unit tests methods.
    """

    def test_cartesian_to_polar(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cartesian_to_polar` definition.
        """

        np.testing.assert_almost_equal(
            cartesian_to_polar(np.array([3, 1])),
            np.array([3.16227766, 0.32175055]),
            decimal=7)

        np.testing.assert_almost_equal(
            cartesian_to_polar(np.array([-1, 9])),
            np.array([9.05538514, 1.68145355]),
            decimal=7)

        np.testing.assert_almost_equal(
            cartesian_to_polar(np.array([6.3434, -0.9345])),
            np.array([6.41186508, -0.14626640]),
            decimal=7)

    def test_n_dimensional_cartesian_to_polar(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cartesian_to_polar` definition n-dimensional arrays support.
        """

        a_i = np.array([3, 1])
        a_o = cartesian_to_polar(a_i)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_almost_equal(cartesian_to_polar(a_i), a_o, decimal=7)

        a_i = np.reshape(a_i, (2, 3, 2))
        a_o = np.reshape(a_o, (2, 3, 2))
        np.testing.assert_almost_equal(cartesian_to_polar(a_i), a_o, decimal=7)

    @ignore_numpy_errors
    def test_nan_cartesian_to_polar(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cartesian_to_polar` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            a_i = np.array(case)
            cartesian_to_polar(a_i)


class TestPolarToCartesian(unittest.TestCase):
    """
    Defines :func:`colour.algebra.coordinates.transformations.\
polar_to_cartesian` definition unit tests methods.
    """

    def test_polar_to_cartesian(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
polar_to_cartesian` definition.
        """

        np.testing.assert_almost_equal(
            polar_to_cartesian(np.array([0.32175055, 1.08574654])),
            np.array([0.15001697, 0.28463718]),
            decimal=7)

        np.testing.assert_almost_equal(
            polar_to_cartesian(np.array([1.68145355, 1.05578119])),
            np.array([0.82819662, 1.46334425]),
            decimal=7)

        np.testing.assert_almost_equal(
            polar_to_cartesian(np.array([-0.14626640, 1.23829030])),
            np.array([-0.04774323, -0.13825500]),
            decimal=7)

    def test_n_dimensional_polar_to_cartesian(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
polar_to_cartesian` definition n-dimensional arrays support.
        """

        a_i = np.array([3.16227766, 0.32175055])
        a_o = polar_to_cartesian(a_i)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_almost_equal(polar_to_cartesian(a_i), a_o, decimal=7)

        a_i = np.reshape(a_i, (2, 3, 2))
        a_o = np.reshape(a_o, (2, 3, 2))
        np.testing.assert_almost_equal(polar_to_cartesian(a_i), a_o, decimal=7)

    @ignore_numpy_errors
    def test_nan_polar_to_cartesian(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
polar_to_cartesian` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            a_i = np.array(case)
            polar_to_cartesian(a_i)


class TestCartesianToCylindrical(unittest.TestCase):
    """
    Defines :func:`colour.algebra.coordinates.transformations.\
cartesian_to_cylindrical` definition unit tests methods.
    """

    def test_cartesian_to_cylindrical(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cartesian_to_cylindrical` definition.
        """

        np.testing.assert_almost_equal(
            cartesian_to_cylindrical(np.array([3, 1, 6])),
            np.array([3.16227766, 0.32175055, 6.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            cartesian_to_cylindrical(np.array([-1, 9, 16])),
            np.array([9.05538514, 1.68145355, 16.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            cartesian_to_cylindrical(np.array([6.3434, -0.9345, 18.5675])),
            np.array([6.41186508, -0.14626640, 18.56750000]),
            decimal=7)

    def test_n_dimensional_cartesian_to_cylindrical(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cartesian_to_cylindrical` definition n-dimensional arrays support.
        """

        a_i = np.array([3, 1, 6])
        a_o = cartesian_to_cylindrical(a_i)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_almost_equal(
            cartesian_to_cylindrical(a_i), a_o, decimal=7)

        a_i = np.reshape(a_i, (2, 3, 3))
        a_o = np.reshape(a_o, (2, 3, 3))
        np.testing.assert_almost_equal(
            cartesian_to_cylindrical(a_i), a_o, decimal=7)

    @ignore_numpy_errors
    def test_nan_cartesian_to_cylindrical(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cartesian_to_cylindrical` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            a_i = np.array(case)
            cartesian_to_cylindrical(a_i)


class TestCylindricalToCartesian(unittest.TestCase):
    """
    Defines :func:`colour.algebra.coordinates.transformations.\
cylindrical_to_cartesian` definition unit tests methods.
    """

    def test_cylindrical_to_cartesian(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cylindrical_to_cartesian` definition.
        """

        np.testing.assert_almost_equal(
            cylindrical_to_cartesian(
                np.array([0.32175055, 1.08574654, 6.78232998])),
            np.array([0.15001697, 0.28463718, 6.78232998]),
            decimal=7)

        np.testing.assert_almost_equal(
            cylindrical_to_cartesian(
                np.array([1.68145355, 1.05578119, 18.38477631])),
            np.array([0.82819662, 1.46334425, 18.38477631]),
            decimal=7)

        np.testing.assert_almost_equal(
            cylindrical_to_cartesian(
                np.array([-0.14626640, 1.23829030, 19.64342307])),
            np.array([-0.04774323, -0.13825500, 19.64342307]),
            decimal=7)

    def test_n_dimensional_cylindrical_to_cartesian(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cylindrical_to_cartesian` definition n-dimensional arrays support.
        """

        a_i = np.array([3.16227766, 0.32175055, 6.00000000])
        a_o = cylindrical_to_cartesian(a_i)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_almost_equal(
            cylindrical_to_cartesian(a_i), a_o, decimal=7)

        a_i = np.reshape(a_i, (2, 3, 3))
        a_o = np.reshape(a_o, (2, 3, 3))
        np.testing.assert_almost_equal(
            cylindrical_to_cartesian(a_i), a_o, decimal=7)

    @ignore_numpy_errors
    def test_nan_cylindrical_to_cartesian(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cylindrical_to_cartesian` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            a_i = np.array(case)
            cylindrical_to_cartesian(a_i)


if __name__ == '__main__':
    unittest.main()
