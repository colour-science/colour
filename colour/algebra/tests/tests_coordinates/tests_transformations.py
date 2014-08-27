#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.algebra.coordinates.transformations`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.algebra import cartesian_to_spherical, spherical_to_cartesian
from colour.algebra import cartesian_to_cylindrical, cylindrical_to_cartesian

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
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
    Defines
    :func:`colour.algebra.coordinates.transformations.cartesian_to_spherical`
    definition unit tests methods.
    """

    def test_cartesian_to_spherical(self):
        """
        Tests
        :func:`colour.algebra.coordinates.transformations.cartesian_to_spherical`  # noqa
        definition.
        """

        np.testing.assert_almost_equal(
            cartesian_to_spherical((3, 1, 6)),
            np.array([6.78232998, 1.08574654, 0.32175055]),
            decimal=7)
        np.testing.assert_almost_equal(
            cartesian_to_spherical((-1, 9, 16)),
            np.array([18.38477631, 1.05578119, 1.68145355]),
            decimal=7)
        np.testing.assert_almost_equal(
            cartesian_to_spherical((6.3434, -0.9345, 18.5675)),
            np.array([19.64342307, 1.2382903, -0.1462664]),
            decimal=7)


class TestSphericalToCartesian(unittest.TestCase):
    """
    Defines
    :func:`colour.algebra.coordinates.transformations.spherical_to_cartesian`
    definition unit tests methods.
    """

    def test_spherical_to_cartesian(self):
        """
        Tests
        :func:`colour.algebra.coordinates.transformations.spherical_to_cartesian`  # noqa
        definition.
        """

        np.testing.assert_almost_equal(
            spherical_to_cartesian((6.78232998, 1.08574654, 0.32175055)),
            np.array([3., 0.99999999, 6.]),
            decimal=7)
        np.testing.assert_almost_equal(
            spherical_to_cartesian((18.38477631, 1.05578119, 1.68145355)),
            np.array([-1.00000003, 9.00000007, 15.99999996]),
            decimal=7)
        np.testing.assert_almost_equal(
            spherical_to_cartesian((19.64342307, 1.2382903, -0.1462664)),
            np.array([6.34339996, -0.93449999, 18.56750001]),
            decimal=7)


class TestCartesianToCylindrical(unittest.TestCase):
    """
    Defines
    :func:`colour.algebra.coordinates.transformations.cartesian_to_cylindrical`
    definition unit tests methods.
    """

    def test_cartesian_to_cylindrical(self):
        """
        Tests
        :func:`colour.algebra.coordinates.transformations.cartesian_to_cylindrical`  # noqa
        definition.
        """

        np.testing.assert_almost_equal(
            cartesian_to_cylindrical((3, 1, 6)),
            np.array([6., 0.32175055, 3.16227766]),
            decimal=7)
        np.testing.assert_almost_equal(
            cartesian_to_cylindrical((-1, 9, 16)),
            np.array([16., 1.68145355, 9.05538514]),
            decimal=7)
        np.testing.assert_almost_equal(
            cartesian_to_cylindrical((6.3434, -0.9345, 18.5675)),
            np.array([18.5675, -0.1462664, 6.41186508]),
            decimal=7)


class TestCylindricalToCartesian(unittest.TestCase):
    """
    Defines
    :func:`colour.algebra.coordinates.transformations.cylindrical_to_cartesian`  # noqa
    definition unit tests methods.
    """

    def test_cylindrical_to_cartesian(self):
        """
        Tests
        :func:`colour.algebra.coordinates.transformations.cylindrical_to_cartesian`  # noqa
        definition.
        """

        np.testing.assert_almost_equal(
            cylindrical_to_cartesian((6.78232998, 1.08574654, 0.32175055)),
            np.array([0.15001697, 0.28463718, 6.78232998]),
            decimal=7)
        np.testing.assert_almost_equal(
            cylindrical_to_cartesian((18.38477631, 1.05578119, 1.68145355)),
            np.array([0.82819662, 1.46334425, 18.38477631]),
            decimal=7)
        np.testing.assert_almost_equal(
            cylindrical_to_cartesian((19.64342307, 1.2382903, -0.1462664)),
            np.array([-0.04774323, -0.138255, 19.64342307]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
