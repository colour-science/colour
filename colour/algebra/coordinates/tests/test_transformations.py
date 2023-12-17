"""
Define the unit tests for the
:mod:`colour.algebra.coordinates.transformations` module.
"""

import unittest
from itertools import product

import numpy as np

from colour.algebra import (
    cartesian_to_cylindrical,
    cartesian_to_polar,
    cartesian_to_spherical,
    cylindrical_to_cartesian,
    polar_to_cartesian,
    spherical_to_cartesian,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestCartesianToSpherical",
    "TestSphericalToCartesian",
    "TestCartesianToPolar",
    "TestPolarToCartesian",
    "TestCartesianToCylindrical",
    "TestCylindricalToCartesian",
]


class TestCartesianToSpherical(unittest.TestCase):
    """
    Define :func:`colour.algebra.coordinates.transformations.\
cartesian_to_spherical` definition unit tests methods.
    """

    def test_cartesian_to_spherical(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
cartesian_to_spherical` definition.
        """

        np.testing.assert_allclose(
            cartesian_to_spherical(np.array([3, 1, 6])),
            np.array([6.78232998, 0.48504979, 0.32175055]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            cartesian_to_spherical(np.array([-1, 9, 16])),
            np.array([18.38477631, 0.51501513, 1.68145355]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            cartesian_to_spherical(np.array([6.3434, -0.9345, 18.5675])),
            np.array([19.64342307, 0.33250603, -0.14626640]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_cartesian_to_spherical(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
cartesian_to_spherical` definition n-dimensional arrays support.
        """

        a_i = np.array([3, 1, 6])
        a_o = cartesian_to_spherical(a_i)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_allclose(
            cartesian_to_spherical(a_i), a_o, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a_i = np.reshape(a_i, (2, 3, 3))
        a_o = np.reshape(a_o, (2, 3, 3))
        np.testing.assert_allclose(
            cartesian_to_spherical(a_i), a_o, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_cartesian_to_spherical(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
cartesian_to_spherical` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        cartesian_to_spherical(cases)


class TestSphericalToCartesian(unittest.TestCase):
    """
    Define :func:`colour.algebra.coordinates.transformations.\
spherical_to_cartesian` definition unit tests methods.
    """

    def test_spherical_to_cartesian(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
spherical_to_cartesian` definition.
        """

        np.testing.assert_allclose(
            spherical_to_cartesian(
                np.array([6.78232998, 0.48504979, 0.32175055])
            ),
            np.array([3.00000000, 0.99999999, 6.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            spherical_to_cartesian(
                np.array([18.38477631, 0.51501513, 1.68145355])
            ),
            np.array([-1.00000003, 9.00000007, 15.99999996]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            spherical_to_cartesian(
                np.array([19.64342307, 0.33250603, -0.14626640])
            ),
            np.array([6.34339996, -0.93449999, 18.56750001]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_spherical_to_cartesian(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
spherical_to_cartesian` definition n-dimensional arrays support.
        """

        a_i = np.array([6.78232998, 0.48504979, 0.32175055])
        a_o = spherical_to_cartesian(a_i)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_allclose(
            spherical_to_cartesian(a_i), a_o, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a_i = np.reshape(a_i, (2, 3, 3))
        a_o = np.reshape(a_o, (2, 3, 3))
        np.testing.assert_allclose(
            spherical_to_cartesian(a_i), a_o, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_spherical_to_cartesian(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
spherical_to_cartesian` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        spherical_to_cartesian(cases)


class TestCartesianToPolar(unittest.TestCase):
    """
    Define :func:`colour.algebra.coordinates.transformations.\
cartesian_to_polar` definition unit tests methods.
    """

    def test_cartesian_to_polar(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
cartesian_to_polar` definition.
        """

        np.testing.assert_allclose(
            cartesian_to_polar(np.array([3, 1])),
            np.array([3.16227766, 0.32175055]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            cartesian_to_polar(np.array([-1, 9])),
            np.array([9.05538514, 1.68145355]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            cartesian_to_polar(np.array([6.3434, -0.9345])),
            np.array([6.41186508, -0.14626640]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_cartesian_to_polar(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
cartesian_to_polar` definition n-dimensional arrays support.
        """

        a_i = np.array([3, 1])
        a_o = cartesian_to_polar(a_i)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_allclose(
            cartesian_to_polar(a_i), a_o, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a_i = np.reshape(a_i, (2, 3, 2))
        a_o = np.reshape(a_o, (2, 3, 2))
        np.testing.assert_allclose(
            cartesian_to_polar(a_i), a_o, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_cartesian_to_polar(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
cartesian_to_polar` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        cartesian_to_polar(cases)


class TestPolarToCartesian(unittest.TestCase):
    """
    Define :func:`colour.algebra.coordinates.transformations.\
polar_to_cartesian` definition unit tests methods.
    """

    def test_polar_to_cartesian(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
polar_to_cartesian` definition.
        """

        np.testing.assert_allclose(
            polar_to_cartesian(np.array([0.32175055, 1.08574654])),
            np.array([0.15001697, 0.28463718]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            polar_to_cartesian(np.array([1.68145355, 1.05578119])),
            np.array([0.82819662, 1.46334425]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            polar_to_cartesian(np.array([-0.14626640, 1.23829030])),
            np.array([-0.04774323, -0.13825500]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_polar_to_cartesian(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
polar_to_cartesian` definition n-dimensional arrays support.
        """

        a_i = np.array([3.16227766, 0.32175055])
        a_o = polar_to_cartesian(a_i)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_allclose(
            polar_to_cartesian(a_i), a_o, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a_i = np.reshape(a_i, (2, 3, 2))
        a_o = np.reshape(a_o, (2, 3, 2))
        np.testing.assert_allclose(
            polar_to_cartesian(a_i), a_o, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_polar_to_cartesian(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
polar_to_cartesian` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        polar_to_cartesian(cases)


class TestCartesianToCylindrical(unittest.TestCase):
    """
    Define :func:`colour.algebra.coordinates.transformations.\
cartesian_to_cylindrical` definition unit tests methods.
    """

    def test_cartesian_to_cylindrical(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
cartesian_to_cylindrical` definition.
        """

        np.testing.assert_allclose(
            cartesian_to_cylindrical(np.array([3, 1, 6])),
            np.array([3.16227766, 0.32175055, 6.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            cartesian_to_cylindrical(np.array([-1, 9, 16])),
            np.array([9.05538514, 1.68145355, 16.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            cartesian_to_cylindrical(np.array([6.3434, -0.9345, 18.5675])),
            np.array([6.41186508, -0.14626640, 18.56750000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_cartesian_to_cylindrical(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
cartesian_to_cylindrical` definition n-dimensional arrays support.
        """

        a_i = np.array([3, 1, 6])
        a_o = cartesian_to_cylindrical(a_i)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_allclose(
            cartesian_to_cylindrical(a_i), a_o, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a_i = np.reshape(a_i, (2, 3, 3))
        a_o = np.reshape(a_o, (2, 3, 3))
        np.testing.assert_allclose(
            cartesian_to_cylindrical(a_i), a_o, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_cartesian_to_cylindrical(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
cartesian_to_cylindrical` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        cartesian_to_cylindrical(cases)


class TestCylindricalToCartesian(unittest.TestCase):
    """
    Define :func:`colour.algebra.coordinates.transformations.\
cylindrical_to_cartesian` definition unit tests methods.
    """

    def test_cylindrical_to_cartesian(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
cylindrical_to_cartesian` definition.
        """

        np.testing.assert_allclose(
            cylindrical_to_cartesian(
                np.array([0.32175055, 1.08574654, 6.78232998])
            ),
            np.array([0.15001697, 0.28463718, 6.78232998]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            cylindrical_to_cartesian(
                np.array([1.68145355, 1.05578119, 18.38477631])
            ),
            np.array([0.82819662, 1.46334425, 18.38477631]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            cylindrical_to_cartesian(
                np.array([-0.14626640, 1.23829030, 19.64342307])
            ),
            np.array([-0.04774323, -0.13825500, 19.64342307]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_cylindrical_to_cartesian(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
cylindrical_to_cartesian` definition n-dimensional arrays support.
        """

        a_i = np.array([3.16227766, 0.32175055, 6.00000000])
        a_o = cylindrical_to_cartesian(a_i)

        a_i = np.tile(a_i, (6, 1))
        a_o = np.tile(a_o, (6, 1))
        np.testing.assert_allclose(
            cylindrical_to_cartesian(a_i), a_o, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a_i = np.reshape(a_i, (2, 3, 3))
        a_o = np.reshape(a_o, (2, 3, 3))
        np.testing.assert_allclose(
            cylindrical_to_cartesian(a_i), a_o, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_cylindrical_to_cartesian(self):
        """
        Test :func:`colour.algebra.coordinates.transformations.\
cylindrical_to_cartesian` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        cylindrical_to_cartesian(cases)


if __name__ == "__main__":
    unittest.main()
