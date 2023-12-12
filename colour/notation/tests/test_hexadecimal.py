# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.notation.hexadecimal` module."""

import unittest
from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.notation.hexadecimal import (
    HEX_to_RGB,
    RGB_to_HEX,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestRGB_to_HEX",
    "TestHEX_to_RGB",
]


class TestRGB_to_HEX(unittest.TestCase):
    """
    Define :func:`colour.notation.hexadecimal.RGB_to_HEX` definition unit
    tests methods.
    """

    def test_RGB_to_HEX(self):
        """Test :func:`colour.notation.hexadecimal.RGB_to_HEX` definition."""

        self.assertEqual(
            RGB_to_HEX(np.array([0.45620519, 0.03081071, 0.04091952])),
            "#74070a",
        )

        self.assertEqual(
            RGB_to_HEX(np.array([0.00000000, 0.00000000, 0.00000000])),
            "#000000",
        )

        self.assertEqual(
            RGB_to_HEX(np.array([1.00000000, 1.00000000, 1.00000000])),
            "#ffffff",
        )

        np.testing.assert_equal(
            RGB_to_HEX(
                np.array(
                    [
                        [10.00000000, 1.00000000, 1.00000000],
                        [1.00000000, 1.00000000, 1.00000000],
                        [0.00000000, 1.00000000, 0.00000000],
                    ]
                )
            ),
            ["#fe0e0e", "#0e0e0e", "#000e00"],
        )

    def test_n_dimensional_RGB_to_HEX(self):
        """
        Test :func:`colour.notation.hexadecimal.RGB_to_HEX` definition
        n-dimensional arrays support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HEX = RGB_to_HEX(RGB)

        RGB = np.tile(RGB, (6, 1))
        HEX = np.tile(HEX, 6)
        self.assertListEqual(RGB_to_HEX(RGB).tolist(), HEX.tolist())

        RGB = np.reshape(RGB, (2, 3, 3))
        HEX = np.reshape(HEX, (2, 3))
        self.assertListEqual(RGB_to_HEX(RGB).tolist(), HEX.tolist())

    def test_domain_range_scale_RGB_to_HEX(self):
        """
        Test :func:`colour.notation.hexadecimal.RGB_to_HEX` definition domain
        and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HEX = RGB_to_HEX(RGB)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                self.assertEqual(RGB_to_HEX(RGB * factor), HEX)

    @ignore_numpy_errors
    def test_nan_RGB_to_HEX(self):
        """
        Test :func:`colour.notation.hexadecimal.RGB_to_HEX` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_HEX(cases)


class TestHEX_to_RGB(unittest.TestCase):
    """
    Define :func:`colour.notation.hexadecimal.HEX_to_RGB` definition unit
    tests methods.
    """

    def test_HEX_to_RGB(self):
        """Test :func:`colour.notation.hexadecimal.HEX_to_RGB` definition."""

        np.testing.assert_allclose(
            HEX_to_RGB("#74070a"),
            np.array([0.45620519, 0.03081071, 0.04091952]),
            atol=1e-1,
        )

        np.testing.assert_allclose(
            HEX_to_RGB("#000000"),
            np.array([0.00000000, 0.00000000, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            HEX_to_RGB("#ffffff"),
            np.array([1.00000000, 1.00000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_HEX_to_RGB(self):
        """
        Test :func:`colour.notation.hexadecimal.HEX_to_RGB` definition
        n-dimensional arrays support.
        """

        HEX = "#74070a"
        RGB = HEX_to_RGB(HEX)

        HEX = np.tile(HEX, 6)
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_allclose(
            HEX_to_RGB(HEX), RGB, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        HEX = np.reshape(HEX, (2, 3))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_allclose(
            HEX_to_RGB(HEX), RGB, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_HEX_to_RGB(self):
        """
        Test :func:`colour.notation.hexadecimal.HEX_to_RGB` definition domain
        and range scale support.
        """

        HEX = "#74070a"
        RGB = HEX_to_RGB(HEX)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_equal(HEX_to_RGB(HEX), RGB * factor)


if __name__ == "__main__":
    unittest.main()
