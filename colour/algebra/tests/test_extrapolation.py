# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.algebra.extrapolation` module."""

import unittest
from itertools import product

import numpy as np

from colour.algebra import (
    CubicSplineInterpolator,
    Extrapolator,
    LinearInterpolator,
    PchipInterpolator,
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
    "TestExtrapolator",
]


class TestExtrapolator(unittest.TestCase):
    """
    Define :class:`colour.algebra.extrapolation.Extrapolator` class unit
    tests methods.
    """

    def test_required_attributes(self):
        """Test the presence of required attributes."""

        required_attributes = ("interpolator",)

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Extrapolator))

    def test_required_methods(self):
        """Test the presence of required methods."""

        required_methods = ("__init__",)

        for method in required_methods:  # pragma: no cover
            self.assertIn(method, dir(Extrapolator))

    def test_interpolator(self):
        """
        Test :attr:`colour.algebra.extrapolation.Extrapolator.interpolator`
        property.
        """

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([5, 6, 7]), np.array([5, 6, 7]))
        )
        self.assertIsInstance(extrapolator.interpolator, LinearInterpolator)

    def test_method(self):
        """
        Test :attr:`colour.algebra.extrapolation.Extrapolator.method`
        property.
        """

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([5, 6, 7]), np.array([5, 6, 7]))
        )
        self.assertEqual(extrapolator.method, "linear")

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([5, 6, 7]), np.array([5, 6, 7])),
            method="Constant",
        )
        self.assertEqual(extrapolator.method, "constant")

    def test_left(self):
        """
        Test :attr:`colour.algebra.extrapolation.Extrapolator.left`
        property.
        """

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([5, 6, 7]), np.array([5, 6, 7])),
            left=0,
        )
        self.assertEqual(extrapolator.left, 0)

    def test_right(self):
        """
        Test :attr:`colour.algebra.extrapolation.Extrapolator.right`
        property.
        """

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([5, 6, 7]), np.array([5, 6, 7])),
            right=0,
        )
        self.assertEqual(extrapolator.right, 0)

    def test__call__(self):
        """
        Test :meth:`colour.algebra.extrapolation.Extrapolator.__call__`
        method.
        """

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([5, 6, 7]), np.array([5, 6, 7]))
        )
        np.testing.assert_array_equal(extrapolator((4, 8)), (4, 8))
        self.assertEqual(extrapolator(4), 4)

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([3, 4, 5]), np.array([1, 2, 3])),
            method="Constant",
        )
        np.testing.assert_array_equal(
            extrapolator((0.1, 0.2, 8, 9)), (1, 1, 3, 3)
        )
        self.assertEqual(extrapolator(0.1), 1.0)

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([3, 4, 5]), np.array([1, 2, 3])),
            method="Constant",
            left=0,
        )
        np.testing.assert_array_equal(
            extrapolator((0.1, 0.2, 8, 9)), (0, 0, 3, 3)
        )
        self.assertEqual(extrapolator(0.1), 0)

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([3, 4, 5]), np.array([1, 2, 3])),
            method="Constant",
            right=0,
        )
        np.testing.assert_array_equal(
            extrapolator((0.1, 0.2, 8, 9)), (1, 1, 0, 0)
        )
        self.assertEqual(extrapolator(9), 0)

        extrapolator = Extrapolator(
            CubicSplineInterpolator(
                np.array([3, 4, 5, 6]), np.array([1, 2, 3, 4])
            )
        )
        np.testing.assert_allclose(
            extrapolator((0.1, 0.2, 8.0, 9.0)),
            (-1.9, -1.8, 6.0, 7.0),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        self.assertEqual(extrapolator(9), 7)

        extrapolator = Extrapolator(
            PchipInterpolator(np.array([3, 4, 5]), np.array([1, 2, 3]))
        )
        np.testing.assert_allclose(
            extrapolator((0.1, 0.2, 8.0, 9.0)),
            (-1.9, -1.8, 6.0, 7.0),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        self.assertEqual(extrapolator(9), 7.0)

    @ignore_numpy_errors
    def test_nan__call__(self):
        """
        Test :method:`colour.algebra.extrapolation.Extrapolator.__call__`
        method nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            extrapolator = Extrapolator(LinearInterpolator(case, case))
            extrapolator(case[0])


if __name__ == "__main__":
    unittest.main()
