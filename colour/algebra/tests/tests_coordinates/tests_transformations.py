#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.algebra.coordinates.transformations`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.algebra import (
    cartesian_to_spherical,
    spherical_to_cartesian,
    cartesian_to_cylindrical,
    cylindrical_to_cartesian)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestCartesianToSpherical',
           'TestSphericalToCartesian',
           'TestCartesianToCylindrical',
           'TestCylindricalToCartesian']


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
        a_o = np.array([6.78232998, 1.08574654, 0.32175055])
        np.testing.assert_almost_equal(
            cartesian_to_spherical(a_i),
            a_o,
            decimal=7)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_almost_equal(
            cartesian_to_spherical(a_i),
            a_o,
            decimal=7)

        a_i = np.reshape(a_i, (2, 3, 3))
        a_o = np.reshape(a_o, (2, 3, 3))
        np.testing.assert_almost_equal(
            cartesian_to_spherical(a_i),
            a_o,
            decimal=7)

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
        a_o = np.array([3, 1, 6])
        np.testing.assert_almost_equal(
            spherical_to_cartesian(a_i),
            a_o,
            decimal=7)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_almost_equal(
            spherical_to_cartesian(a_i),
            a_o,
            decimal=7)

        a_i = np.reshape(a_i, (2, 3, 3))
        a_o = np.reshape(a_o, (2, 3, 3))
        np.testing.assert_almost_equal(
            spherical_to_cartesian(a_i),
            a_o,
            decimal=7)

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
            np.array([6.00000000, 0.32175055, 3.16227766]),
            decimal=7)

        np.testing.assert_almost_equal(
            cartesian_to_cylindrical(np.array([-1, 9, 16])),
            np.array([16.00000000, 1.68145355, 9.05538514]),
            decimal=7)

        np.testing.assert_almost_equal(
            cartesian_to_cylindrical(np.array([6.3434, -0.9345, 18.5675])),
            np.array([18.56750000, -0.14626640, 6.41186508]),
            decimal=7)

    def test_n_dimensional_cartesian_to_cylindrical(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cartesian_to_cylindrical` definition n-dimensional arrays support.
        """

        a_i = np.array([3, 1, 6])
        a_o = np.array([6.00000000, 0.32175055, 3.16227766])
        np.testing.assert_almost_equal(
            cartesian_to_cylindrical(a_i),
            a_o,
            decimal=7)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_almost_equal(
            cartesian_to_cylindrical(a_i),
            a_o,
            decimal=7)

        a_i = np.reshape(a_i, (2, 3, 3))
        a_o = np.reshape(a_o, (2, 3, 3))
        np.testing.assert_almost_equal(
            cartesian_to_cylindrical(a_i),
            a_o,
            decimal=7)

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
                np.array([6.78232998, 1.08574654, 0.32175055])),
            np.array([0.15001697, 0.28463718, 6.78232998]),
            decimal=7)

        np.testing.assert_almost_equal(
            cylindrical_to_cartesian(
                np.array([18.38477631, 1.05578119, 1.68145355])),
            np.array([0.82819662, 1.46334425, 18.38477631]),
            decimal=7)

        np.testing.assert_almost_equal(
            cylindrical_to_cartesian(
                np.array([19.64342307, 1.23829030, -0.14626640])),
            np.array([-0.04774323, -0.13825500, 19.64342307]),
            decimal=7)

    def test_n_dimensional_cylindrical_to_cartesian(self):
        """
        Tests :func:`colour.algebra.coordinates.transformations.\
cylindrical_to_cartesian` definition n-dimensional arrays support.
        """

        a_i = np.array([6.00000000, 0.32175055, 3.16227766])
        a_o = np.array([3, 1, 6])
        np.testing.assert_almost_equal(
            cylindrical_to_cartesian(a_i),
            a_o,
            decimal=7)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_almost_equal(
            cylindrical_to_cartesian(a_i),
            a_o,
            decimal=7)

        a_i = np.reshape(a_i, (2, 3, 3))
        a_o = np.reshape(a_o, (2, 3, 3))
        np.testing.assert_almost_equal(
            cylindrical_to_cartesian(a_i),
            a_o,
            decimal=7)

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
