"""Define the unit tests for the :mod:`colour.notation.hexadecimal` module."""

from __future__ import annotations

import typing
from itertools import product

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.notation.hexadecimal import HEX_to_RGB, RGB_to_HEX
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_assert_equal,
    xp_reshape,
)

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


class TestRGB_to_HEX:
    """
    Define :func:`colour.notation.hexadecimal.RGB_to_HEX` definition unit
    tests methods.
    """

    def test_RGB_to_HEX(self, xp: ModuleType) -> None:
        """Test :func:`colour.notation.hexadecimal.RGB_to_HEX` definition."""

        assert (
            RGB_to_HEX(xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp))
            == "#74070a"
        )

        assert (
            RGB_to_HEX(xp_as_array([0.00000000, 0.00000000, 0.00000000], xp=xp))
            == "#000000"
        )

        assert (
            RGB_to_HEX(xp_as_array([1.00000000, 1.00000000, 1.00000000], xp=xp))
            == "#ffffff"
        )

        xp_assert_equal(
            RGB_to_HEX(
                xp_as_array(
                    [
                        [10.00000000, 1.00000000, 1.00000000],
                        [1.00000000, 1.00000000, 1.00000000],
                        [0.00000000, 1.00000000, 0.00000000],
                    ],
                    xp=xp,
                )
            ),
            ["#fe0e0e", "#0e0e0e", "#000e00"],
        )

    def test_n_dimensional_RGB_to_HEX(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.notation.hexadecimal.RGB_to_HEX` definition
        n-dimensional arrays support.
        """

        RGB = xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)
        HEX = RGB_to_HEX(RGB)

        RGB = xp_as_array(np.tile(as_ndarray(RGB), (6, 1)), xp=xp)
        HEX = np.tile(HEX, 6)
        assert RGB_to_HEX(RGB).tolist() == HEX.tolist()

        RGB = xp_reshape(RGB, (2, 3, 3), xp=xp)
        HEX = np.reshape(HEX, (2, 3))
        assert RGB_to_HEX(RGB).tolist() == HEX.tolist()

    def test_domain_range_scale_RGB_to_HEX(
        self,
        xp: ModuleType,  # noqa: ARG002
    ) -> None:
        """
        Test :func:`colour.notation.hexadecimal.RGB_to_HEX` definition domain
        and range scale support.
        """

        RGB = np.array([0.45620519, 0.03081071, 0.04091952])
        HEX = RGB_to_HEX(RGB)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                assert RGB_to_HEX(RGB * factor) == HEX

    @ignore_numpy_errors
    def test_nan_RGB_to_HEX(self) -> None:
        """
        Test :func:`colour.notation.hexadecimal.RGB_to_HEX` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_HEX(cases)


class TestHEX_to_RGB:
    """
    Define :func:`colour.notation.hexadecimal.HEX_to_RGB` definition unit
    tests methods.
    """

    def test_HEX_to_RGB(self) -> None:
        """Test :func:`colour.notation.hexadecimal.HEX_to_RGB` definition."""

        xp_assert_close(
            HEX_to_RGB("#74070a"),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS * 1e06,
        )

        xp_assert_close(
            HEX_to_RGB("#000000"),
            [0.00000000, 0.00000000, 0.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            HEX_to_RGB("#ffffff"),
            [1.00000000, 1.00000000, 1.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_HEX_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.notation.hexadecimal.HEX_to_RGB` definition
        n-dimensional arrays support.
        """

        HEX = "#74070a"
        RGB = xp_as_array(HEX_to_RGB(HEX), xp=xp)

        HEX = np.tile(HEX, 6)
        RGB = xp_as_array(np.tile(as_ndarray(RGB), (6, 1)), xp=xp)
        xp_assert_close(HEX_to_RGB(HEX), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        HEX = np.reshape(HEX, (2, 3))
        RGB = xp_reshape(RGB, (2, 3, 3), xp=xp)
        xp_assert_close(HEX_to_RGB(HEX), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_HEX_to_RGB(
        self,
        xp: ModuleType,  # noqa: ARG002
    ) -> None:
        """
        Test :func:`colour.notation.hexadecimal.HEX_to_RGB` definition domain
        and range scale support.
        """

        HEX = "#74070a"
        RGB = HEX_to_RGB(HEX)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_equal(HEX_to_RGB(HEX), RGB * factor)
