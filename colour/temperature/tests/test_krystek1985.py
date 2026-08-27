"""Define the unit tests for the :mod:`colour.temperature.krystek1985` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import CCT_to_uv_Krystek1985, uv_to_CCT_Krystek1985
from colour.utilities import (
    as_ndarray,
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
    "TestUv_to_CCT_Krystek1985",
]


class TestUv_to_CCT_Krystek1985:
    """
    Define :func:`colour.temperature.krystek1985.uv_to_CCT_Krystek1985`
    definition unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1e-2)
    def test_uv_to_CCT_Krystek1985(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.krystek1985.uv_to_CCT_Krystek1985`
        definition.
        """

        xp_assert_close(
            uv_to_CCT_Krystek1985(
                xp_as_array([0.448087794140145, 0.354731965027727], xp=xp),
            ),
            1000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            uv_to_CCT_Krystek1985(
                xp_as_array([0.198152565091092, 0.307023596915037], xp=xp),
            ),
            7000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            uv_to_CCT_Krystek1985(
                xp_as_array([0.185675876767054, 0.282233658593898], xp=xp),
            ),
            15000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_uv_to_CCT_Krystek1985(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.krystek1985.uv_to_CCT_Krystek1985`
        definition n-dimensional arrays support.
        """

        uv = xp_as_array([0.198152565091092, 0.307023596915037], xp=xp)
        CCT = as_ndarray(uv_to_CCT_Krystek1985(uv))

        uv = xp.tile(xp_as_array(uv, xp=xp), (6, 1))
        CCT = xp.tile(xp_as_array(CCT, xp=xp), (6,))
        xp_assert_close(uv_to_CCT_Krystek1985(uv), CCT, atol=TOLERANCE_ABSOLUTE_TESTS)

        uv = xp_reshape(xp_as_array(uv, xp=xp), (2, 3, 2), xp=xp)
        CCT = xp_reshape(xp_as_array(CCT, xp=xp), (2, 3), xp=xp)
        xp_assert_close(uv_to_CCT_Krystek1985(uv), CCT, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_uv_to_CCT_Krystek1985(self) -> None:
        """
        Test :func:`colour.temperature.krystek1985.uv_to_CCT_Krystek1985`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        uv_to_CCT_Krystek1985(cases)


class TestCCT_to_uv_Krystek1985:
    """
    Define :func:`colour.temperature.krystek1985.CCT_to_uv_Krystek1985`
    definition unit tests methods.
    """

    def test_CCT_to_uv_Krystek1985(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.krystek1985.CCT_to_uv_Krystek1985`
        definition.
        """

        xp_assert_close(
            CCT_to_uv_Krystek1985(xp_as_array([1000], xp=xp)),
            [[0.448087794140145, 0.354731965027727]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_uv_Krystek1985(xp_as_array([7000], xp=xp)),
            [[0.198152565091092, 0.307023596915037]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_uv_Krystek1985(xp_as_array([15000], xp=xp)),
            [[0.185675876767054, 0.282233658593898]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CCT_to_uv_Krystek1985(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.krystek1985.CCT_to_uv_Krystek1985`
        definition n-dimensional arrays support.
        """

        CCT = 7000
        uv = as_ndarray(CCT_to_uv_Krystek1985(CCT))

        CCT = xp.tile(xp_as_array(CCT, xp=xp), (6,))
        uv = xp.tile(xp_as_array(uv, xp=xp), (6, 1))
        xp_assert_close(CCT_to_uv_Krystek1985(CCT), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

        CCT = xp_reshape(xp_as_array(CCT, xp=xp), (2, 3), xp=xp)
        uv = xp_reshape(xp_as_array(uv, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(CCT_to_uv_Krystek1985(CCT), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_CCT_to_uv_Krystek1985(self) -> None:
        """
        Test :func:`colour.temperature.krystek1985.CCT_to_uv_Krystek1985`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        CCT_to_uv_Krystek1985(cases)
