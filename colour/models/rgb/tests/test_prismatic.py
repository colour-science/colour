# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.rgb.prismatic` module."""

import unittest
from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb import Prismatic_to_RGB, RGB_to_Prismatic
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestRGB_to_Prismatic",
    "TestPrismatic_to_RGB",
]


class TestRGB_to_Prismatic(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.prismatic.TestRGB_to_Prismatic` definition
    unit tests methods.
    """

    def test_RGB_to_Prismatic(self):
        """Test :func:`colour.models.rgb.prismatic.RGB_to_Prismatic` definition."""

        np.testing.assert_allclose(
            RGB_to_Prismatic(np.array([0.0, 0.0, 0.0])),
            np.array([0.0, 0.0, 0.0, 0.0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            RGB_to_Prismatic(np.array([0.25, 0.50, 0.75])),
            np.array([0.7500000, 0.1666667, 0.3333333, 0.5000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_to_Prismatic(self):
        """
        Test :func:`colour.models.rgb.prismatic.RGB_to_Prismatic` definition
        n-dimensional support.
        """

        RGB = np.array([0.25, 0.50, 0.75])
        Lrgb = RGB_to_Prismatic(RGB)

        RGB = np.tile(RGB, (6, 1))
        Lrgb = np.tile(Lrgb, (6, 1))
        np.testing.assert_allclose(
            RGB_to_Prismatic(RGB), Lrgb, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        RGB = np.reshape(RGB, (2, 3, 3))
        Lrgb = np.reshape(Lrgb, (2, 3, 4))
        np.testing.assert_allclose(
            RGB_to_Prismatic(RGB), Lrgb, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_RGB_to_Prismatic(self):
        """
        Test :func:`colour.models.rgb.prismatic.RGB_to_Prismatic` definition
        domain and range scale support.
        """

        RGB = np.array([0.25, 0.50, 0.75])
        Lrgb = RGB_to_Prismatic(RGB)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    RGB_to_Prismatic(RGB * factor),
                    Lrgb * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_Prismatic(self):
        """
        Test :func:`colour.models.rgb.prismatic.RGB_to_Prismatic` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_Prismatic(cases)


class TestPrismatic_to_RGB(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.prismatic.Prismatic_to_RGB` definition
    unit tests methods.
    """

    def test_Prismatic_to_RGB(self):
        """Test :func:`colour.models.rgb.prismatic.Prismatic_to_RGB` definition."""

        np.testing.assert_allclose(
            Prismatic_to_RGB(np.array([0.0, 0.0, 0.0, 0.0])),
            np.array([0.0, 0.0, 0.0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Prismatic_to_RGB(
                np.array([0.7500000, 0.1666667, 0.3333333, 0.5000000])
            ),
            np.array([0.25, 0.50, 0.75]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Prismatic_to_RGB(self):
        """
        Test :func:`colour.models.rgb.prismatic.Prismatic_to_RGB` definition
        n-dimensional support.
        """

        Lrgb = np.array([0.7500000, 0.1666667, 0.3333333, 0.5000000])
        RGB = Prismatic_to_RGB(Lrgb)

        Lrgb = np.tile(Lrgb, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_allclose(
            Prismatic_to_RGB(Lrgb), RGB, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Lrgb = np.reshape(Lrgb, (2, 3, 4))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_allclose(
            Prismatic_to_RGB(Lrgb), RGB, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_Prismatic_to_RGB(self):
        """
        Test :func:`colour.models.rgb.prismatic.Prismatic_to_RGB` definition
        domain and range scale support.
        """

        Lrgb = np.array([0.7500000, 0.1666667, 0.3333333, 0.5000000])
        RGB = Prismatic_to_RGB(Lrgb)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    Prismatic_to_RGB(Lrgb * factor),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Prismatic_to_RGB(self):
        """
        Test :func:`colour.models.rgb.prismatic.Prismatic_to_RGB` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Prismatic_to_RGB(cases)


if __name__ == "__main__":
    unittest.main()
