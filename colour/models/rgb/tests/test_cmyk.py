"""Define the unit tests for the :mod:`colour.models.rgb.cmyk` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.cmyk import CMY_to_CMYK, CMY_to_RGB, CMYK_to_CMY, RGB_to_CMY
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestRGB_to_CMY",
    "TestCMY_to_RGB",
    "TestCMY_to_CMYK",
    "TestCMYK_to_CMY",
]


class TestRGB_to_CMY:
    """
    Define :func:`colour.models.rgb.cmyk.RGB_to_CMY` definition unit tests
    methods.
    """

    def test_RGB_to_CMY(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.cmyk.RGB_to_CMY` definition."""

        xp_assert_close(
            RGB_to_CMY(xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)),
            [0.54379481, 0.96918929, 0.95908048],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_CMY(xp_as_array([0.00000000, 0.00000000, 0.00000000], xp=xp)),
            [1.00000000, 1.00000000, 1.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB_to_CMY(xp_as_array([1.00000000, 1.00000000, 1.00000000], xp=xp)),
            [0.00000000, 0.00000000, 0.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_to_CMY(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cmyk.RGB_to_CMY` definition
        n-dimensional arrays support.
        """

        RGB = xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)
        CMY = as_ndarray(RGB_to_CMY(RGB))

        RGB = xp.tile(xp_as_array(RGB, xp=xp), (6, 1))
        CMY = xp.tile(xp_as_array(CMY, xp=xp), (6, 1))
        xp_assert_close(RGB_to_CMY(RGB), CMY, atol=TOLERANCE_ABSOLUTE_TESTS)

        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (2, 3, 3), xp=xp)
        CMY = xp_reshape(xp_as_array(CMY, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(RGB_to_CMY(RGB), CMY, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_RGB_to_CMY(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cmyk.RGB_to_CMY` definition domain and
        range scale support.
        """

        RGB = xp_as_array([0.45620519, 0.03081071, 0.04091952], xp=xp)
        CMY = as_ndarray(RGB_to_CMY(RGB))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    RGB_to_CMY(RGB * factor),
                    CMY * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_RGB_to_CMY(self) -> None:
        """
        Test :func:`colour.models.rgb.cmyk.RGB_to_CMY` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        RGB_to_CMY(cases)


class TestCMY_to_RGB:
    """
    Define :func:`colour.models.rgb.cmyk.CMY_to_RGB` definition unit tests
    methods.
    """

    def test_CMY_to_RGB(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.cmyk.CMY_to_RGB` definition."""

        xp_assert_close(
            CMY_to_RGB(xp_as_array([0.54379481, 0.96918929, 0.95908048], xp=xp)),
            [0.45620519, 0.03081071, 0.04091952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CMY_to_RGB(xp_as_array([1.00000000, 1.00000000, 1.00000000], xp=xp)),
            [0.00000000, 0.00000000, 0.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CMY_to_RGB(xp_as_array([0.00000000, 0.00000000, 0.00000000], xp=xp)),
            [1.00000000, 1.00000000, 1.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CMY_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cmyk.CMY_to_RGB` definition
        n-dimensional arrays support.
        """

        CMY = xp_as_array([0.54379481, 0.96918929, 0.95908048], xp=xp)
        RGB = as_ndarray(CMY_to_RGB(CMY))

        CMY = xp.tile(xp_as_array(CMY, xp=xp), (6, 1))
        RGB = xp.tile(xp_as_array(RGB, xp=xp), (6, 1))
        xp_assert_close(CMY_to_RGB(CMY), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

        CMY = xp_reshape(xp_as_array(CMY, xp=xp), (2, 3, 3), xp=xp)
        RGB = xp_reshape(xp_as_array(RGB, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(CMY_to_RGB(CMY), RGB, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_CMY_to_RGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cmyk.CMY_to_RGB` definition domain and
        range scale support.
        """

        CMY = xp_as_array([0.54379481, 0.96918929, 0.95908048], xp=xp)
        RGB = as_ndarray(CMY_to_RGB(CMY))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    CMY_to_RGB(CMY * factor),
                    RGB * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_CMY_to_RGB(self) -> None:
        """Test :func:`colour.models.rgb.cmyk.CMY_to_RGB` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        CMY_to_RGB(cases)


class TestCMY_to_CMYK:
    """
    Define :func:`colour.models.rgb.cmyk.CMY_to_CMYK` definition unit tests
    methods.
    """

    def test_CMY_to_CMYK(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.cmyk.CMY_to_CMYK` definition."""

        xp_assert_close(
            CMY_to_CMYK(xp_as_array([0.54379481, 0.96918929, 0.95908048], xp=xp)),
            [0.00000000, 0.93246304, 0.91030457, 0.54379481],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CMY_to_CMYK(xp_as_array([0.15000000, 1.00000000, 1.00000000], xp=xp)),
            [0.00000000, 1.00000000, 1.00000000, 0.15000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CMY_to_CMYK(xp_as_array([0.15000000, 0.00000000, 0.00000000], xp=xp)),
            [0.15000000, 0.00000000, 0.00000000, 0.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CMY_to_CMYK(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cmyk.CMY_to_CMYK` definition
        n-dimensional arrays support.
        """

        CMY = xp_as_array([0.54379481, 0.96918929, 0.95908048], xp=xp)
        CMYK = as_ndarray(CMY_to_CMYK(CMY))

        CMY = xp.tile(xp_as_array(CMY, xp=xp), (6, 1))
        CMYK = xp.tile(xp_as_array(CMYK, xp=xp), (6, 1))
        xp_assert_close(CMY_to_CMYK(CMY), CMYK, atol=TOLERANCE_ABSOLUTE_TESTS)

        CMY = xp_reshape(xp_as_array(CMY, xp=xp), (2, 3, 3), xp=xp)
        CMYK = xp_reshape(xp_as_array(CMYK, xp=xp), (2, 3, 4), xp=xp)
        xp_assert_close(CMY_to_CMYK(CMY), CMYK, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_CMY_to_CMYK(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cmyk.CMY_to_CMYK` definition domain and
        range scale support.
        """

        CMY = xp_as_array([0.54379481, 0.96918929, 0.95908048], xp=xp)
        CMYK = as_ndarray(CMY_to_CMYK(CMY))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    CMY_to_CMYK(CMY * factor),
                    CMYK * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_CMY_to_CMYK(self) -> None:
        """
        Test :func:`colour.models.rgb.cmyk.CMY_to_CMYK` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        CMY_to_CMYK(cases)


class TestCMYK_to_CMY:
    """
    Define :func:`colour.models.rgb.cmyk.CMYK_to_CMY` definition unit tests
    methods.
    """

    def test_CMYK_to_CMY(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.cmyk.CMYK_to_CMY` definition."""

        xp_assert_close(
            CMYK_to_CMY(
                xp_as_array([0.00000000, 0.93246304, 0.91030457, 0.54379481], xp=xp)
            ),
            [0.54379481, 0.96918929, 0.95908048],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CMYK_to_CMY(
                xp_as_array([0.00000000, 1.00000000, 1.00000000, 0.15000000], xp=xp)
            ),
            [0.15000000, 1.00000000, 1.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CMYK_to_CMY(
                xp_as_array([0.15000000, 0.00000000, 0.00000000, 0.00000000], xp=xp)
            ),
            [0.15000000, 0.00000000, 0.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CMYK_to_CMY(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cmyk.CMYK_to_CMY` definition
        n-dimensional arrays support.
        """

        CMYK = xp_as_array([0.00000000, 0.93246304, 0.91030457, 0.54379481], xp=xp)
        CMY = as_ndarray(CMYK_to_CMY(CMYK))

        CMYK = xp.tile(xp_as_array(CMYK, xp=xp), (6, 1))
        CMY = xp.tile(xp_as_array(CMY, xp=xp), (6, 1))
        xp_assert_close(CMYK_to_CMY(CMYK), CMY, atol=TOLERANCE_ABSOLUTE_TESTS)

        CMYK = xp_reshape(xp_as_array(CMYK, xp=xp), (2, 3, 4), xp=xp)
        CMY = xp_reshape(xp_as_array(CMY, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(CMYK_to_CMY(CMYK), CMY, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_CMYK_to_CMY(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.cmyk.CMYK_to_CMY` definition domain and
        range scale support.
        """

        CMYK = xp_as_array([0.00000000, 0.93246304, 0.91030457, 0.54379481], xp=xp)
        CMY = as_ndarray(CMYK_to_CMY(CMYK))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    CMYK_to_CMY(CMYK * factor),
                    CMY * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_CMYK_to_CMY(self) -> None:
        """
        Test :func:`colour.models.rgb.cmyk.CMYK_to_CMY` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=4))))
        CMYK_to_CMY(cases)
