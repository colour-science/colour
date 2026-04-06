"""Define the unit tests for the :mod:`colour.temperature.cie_d` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import CCT_to_xy_CIE_D, xy_to_CCT_CIE_D
from colour.utilities import (
    as_ndarray,
    ignore_numpy_errors,
    is_scipy_installed,
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
    "TestXy_to_CCT_CIE_D",
    "TestCCT_to_xy_CIE_D",
]


class TestXy_to_CCT_CIE_D:
    """
    Define :func:`colour.temperature.cie_d.xy_to_CCT_CIE_D` definition unit
    tests methods.
    """

    def test_xy_to_CCT_CIE_D(self, xp: ModuleType) -> None:
        """Test :func:`colour.temperature.cie_d.xy_to_CCT_CIE_D` definition."""

        xp_assert_close(
            xy_to_CCT_CIE_D(
                xp_as_array([0.382343625000000, 0.383766261015578], xp=xp),
            ),
            4000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_CCT_CIE_D(
                xp_as_array([0.305357431486880, 0.321646345474552], xp=xp),
            ),
            7000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_CCT_CIE_D(
                xp_as_array([0.24985367, 0.254799464210944], xp=xp),
            ),
            25000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xy_to_CCT_CIE_D(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.cie_d.xy_to_CCT_CIE_D` definition
        n-dimensional arrays support.
        """

        if not is_scipy_installed():  # pragma: no cover
            return

        xy = xp_as_array([0.382343625000000, 0.383766261015578], xp=xp)
        CCT = as_ndarray(xy_to_CCT_CIE_D(xy))

        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        CCT = xp.tile(xp_as_array(CCT, xp=xp), (6,))
        xp_assert_close(xy_to_CCT_CIE_D(xy), CCT, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        CCT = xp_reshape(xp_as_array(CCT, xp=xp), (2, 3), xp=xp)
        xp_assert_close(xy_to_CCT_CIE_D(xy), CCT, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_xy_to_CCT_CIE_D(self) -> None:
        """
        Test :func:`colour.temperature.cie_d.xy_to_CCT_CIE_D` definition nan
        support.
        """

        if not is_scipy_installed():  # pragma: no cover
            return

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_CCT_CIE_D(cases)


class TestCCT_to_xy_CIE_D:
    """
    Define :func:`colour.temperature.cie_d.CCT_to_xy_CIE_D` definition
    unit tests methods.
    """

    def test_CCT_to_xy_CIE_D(self, xp: ModuleType) -> None:
        """Test :func:`colour.temperature.cie_d.CCT_to_xy_CIE_D` definition."""

        xp_assert_close(
            CCT_to_xy_CIE_D(xp_as_array([4000], xp=xp)),
            [[0.382343625000000, 0.383766261015578]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_xy_CIE_D(xp_as_array([7000], xp=xp)),
            [[0.305357431486880, 0.321646345474552]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_xy_CIE_D(xp_as_array([25000], xp=xp)),
            [[0.24985367, 0.254799464210944]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CCT_to_xy_CIE_D(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.cie_d.CCT_to_xy_CIE_D` definition
        n-dimensional arrays support.
        """

        CCT = 4000
        xy = as_ndarray(CCT_to_xy_CIE_D(CCT))

        CCT = xp.tile(xp_as_array(CCT, xp=xp), (6,))
        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        xp_assert_close(CCT_to_xy_CIE_D(CCT), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

        CCT = xp_reshape(xp_as_array(CCT, xp=xp), (2, 3), xp=xp)
        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(CCT_to_xy_CIE_D(CCT), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_CCT_to_xy_CIE_D(self) -> None:
        """
        Test :func:`colour.temperature.cie_d.CCT_to_xy_CIE_D` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        CCT_to_xy_CIE_D(cases)
