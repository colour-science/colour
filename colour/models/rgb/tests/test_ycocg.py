"""Define the unit tests for the :mod:`colour.models.rgb.ycocg` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb import RGB_to_YCoCg, YCoCg_to_RGB
from colour.utilities import (
    as_ndarray,
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
__status__ = "Development"

__all__ = [
    "TestRGB_to_YCoCg",
    "TestYCoCg_to_RGB",
]


class TestRGB_to_YCoCg:
    """
    Define :func:`colour.models.rgb.ycocg.RGB_to_YCoCg` definition unit tests
    methods.
    """

    def test_RGB_to_YCoCg(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ycocg.RGB_to_YCoCg` definition."""

        xp_assert_equal(
            RGB_to_YCoCg(xp_as_array([0.75, 0.75, 0.0], xp=xp)),
            [0.5625, 0.375, 0.1875],
        )

        xp_assert_equal(
            RGB_to_YCoCg(xp_as_array([0.25, 0.5, 0.75], xp=xp)),
            [0.5, -0.25, 0.0],
        )

        xp_assert_equal(
            RGB_to_YCoCg(xp_as_array([0.0, 0.75, 0.75], xp=xp)),
            [0.5625, -0.375, 0.1875],
        )

    def test_n_dimensional_RGB_to_YCoCg(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ycocg.RGB_to_YCoCg` definition
        n-dimensional arrays support.
        """

        RGB = xp_as_array([0.75, 0.75, 0.0], xp=xp)
        YCoCg = as_ndarray(RGB_to_YCoCg(RGB))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 3), xp=xp)
        YCoCg = xp.tile(xp_as_array(YCoCg, xp=xp), (4,))
        YCoCg = xp_reshape(xp_as_array(YCoCg, xp=xp), (4, 3), xp=xp)
        xp_assert_close(RGB_to_YCoCg(RGB), YCoCg, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 3), xp=xp)
        YCoCg = xp.tile(xp_as_array(YCoCg, xp=xp), (4,))
        YCoCg = xp_reshape(xp_as_array(YCoCg, xp=xp), (4, 4, 3), xp=xp)
        xp_assert_close(RGB_to_YCoCg(RGB), YCoCg, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 4, 3), xp=xp)
        YCoCg = xp.tile(xp_as_array(YCoCg, xp=xp), (4,))
        YCoCg = xp_reshape(xp_as_array(YCoCg, xp=xp), (4, 4, 4, 3), xp=xp)
        xp_assert_close(RGB_to_YCoCg(RGB), YCoCg, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_RGB_to_YCoCg(self) -> None:
        """
        Test :func:`colour.models.rgb.ycocg.RGB_to_YCoCg` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_YCoCg(cases)


class TestYCoCg_to_RGB:
    """
    Define :func:`colour.models.rgb.ycocg.YCoCg_to_RGB` definition unit tests
    methods.
    """

    def test_YCoCg_to_RGB(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.ycocg.YCoCg_to_RGB` definition."""

        xp_assert_equal(
            YCoCg_to_RGB(xp_as_array([0.5625, 0.375, 0.1875], xp=xp)),
            [0.75, 0.75, 0.0],
        )

        xp_assert_equal(
            YCoCg_to_RGB(xp_as_array([0.5, -0.25, 0.0], xp=xp)),
            [0.25, 0.5, 0.75],
        )

        xp_assert_equal(
            YCoCg_to_RGB(xp_as_array([0.5625, -0.375, 0.1875], xp=xp)),
            [0.0, 0.75, 0.75],
        )

    def test_n_dimensional_YCoCg_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.ycocg.YCoCg_to_RGB` definition
        n-dimensional arrays support.
        """

        YCoCg = xp_as_array([0.5625, 0.375, 0.1875], xp=xp)
        RGB = as_ndarray(YCoCg_to_RGB(YCoCg))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 3), xp=xp)
        YCoCg = xp.tile(xp_as_array(YCoCg, xp=xp), (4,))
        YCoCg = xp_reshape(xp_as_array(YCoCg, xp=xp), (4, 3), xp=xp)
        xp_assert_close(YCoCg_to_RGB(YCoCg), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 3), xp=xp)
        YCoCg = xp.tile(xp_as_array(YCoCg, xp=xp), (4,))
        YCoCg = xp_reshape(xp_as_array(YCoCg, xp=xp), (4, 4, 3), xp=xp)
        xp_assert_close(YCoCg_to_RGB(YCoCg), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (4,))
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (4, 4, 4, 3), xp=xp)
        YCoCg = xp.tile(xp_as_array(YCoCg, xp=xp), (4,))
        YCoCg = xp_reshape(xp_as_array(YCoCg, xp=xp), (4, 4, 4, 3), xp=xp)
        xp_assert_close(YCoCg_to_RGB(YCoCg), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_YCoCg_to_RGB(self) -> None:
        """
        Test :func:`colour.models.rgb.ycocg.YCoCg_to_RGB` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        YCoCg_to_RGB(cases)
