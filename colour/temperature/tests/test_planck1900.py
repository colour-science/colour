"""Define the unit tests for the :mod:`colour.temperature.planck1900` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import CCT_to_uv_Planck1900, uv_to_CCT_Planck1900
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
    "TestUv_to_CCT_Planck1900",
    "TestCCT_to_uv_Planck1900",
]


class TestUv_to_CCT_Planck1900:
    """
    Define :func:`colour.temperature.planck1900.uv_to_CCT_Planck1900`
    definition unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1)
    def test_uv_to_CCT_Planck1900(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.planck1900.uv_to_CCT_Planck1900`
        definition.
        """

        xp_assert_close(
            uv_to_CCT_Planck1900(
                xp_as_array([0.225109670227493, 0.334387366663923], xp=xp),
            ),
            4000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            uv_to_CCT_Planck1900(
                xp_as_array([0.198126929048352, 0.307025980523306], xp=xp),
            ),
            7000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            uv_to_CCT_Planck1900(
                xp_as_array([0.182932683590136, 0.274073232217536], xp=xp),
            ),
            25000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_uv_to_CCT_Planck1900(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.planck1900.uv_to_CCT_Planck1900`
        definition n-dimensional arrays support.
        """

        uv = xp_as_array([0.225109670227493, 0.334387366663923], xp=xp)
        CCT = as_ndarray(uv_to_CCT_Planck1900(uv))

        uv = xp.tile(xp_as_array(uv, xp=xp), (6, 1))
        CCT = xp.tile(xp_as_array(CCT, xp=xp), (6,))
        xp_assert_close(uv_to_CCT_Planck1900(uv), CCT, atol=TOLERANCE_ABSOLUTE_TESTS)

        uv = xp_reshape(xp_as_array(uv, xp=xp), (2, 3, 2), xp=xp)
        CCT = xp_reshape(xp_as_array(CCT, xp=xp), (2, 3), xp=xp)
        xp_assert_close(uv_to_CCT_Planck1900(uv), CCT, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_uv_to_CCT_Planck1900(self) -> None:
        """
        Test :func:`colour.temperature.planck1900.uv_to_CCT_Planck1900`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        uv_to_CCT_Planck1900(cases)


class TestCCT_to_uv_Planck1900:
    """
    Define :func:`colour.temperature.planck1900.CCT_to_uv_Planck1900` definition
    unit tests methods.
    """

    def test_CCT_to_uv_Planck1900(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.planck1900.CCT_to_uv_Planck1900`
        definition.
        """

        xp_assert_close(
            CCT_to_uv_Planck1900(xp_as_array([4000], xp=xp)),
            [[0.225109670227493, 0.334387366663923]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_uv_Planck1900(xp_as_array([7000], xp=xp)),
            [[0.198126929048352, 0.307025980523306]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_uv_Planck1900(xp_as_array([25000], xp=xp)),
            [[0.182932683590136, 0.274073232217536]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CCT_to_uv_Planck1900(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.planck1900.CCT_to_uv_Planck1900` definition
        n-dimensional arrays support.
        """

        CCT = 4000
        uv = as_ndarray(CCT_to_uv_Planck1900(CCT))

        CCT = xp.tile(xp_as_array(CCT, xp=xp), (6,))
        uv = xp.tile(xp_as_array(uv, xp=xp), (6, 1))
        xp_assert_close(CCT_to_uv_Planck1900(CCT), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

        CCT = xp_reshape(xp_as_array(CCT, xp=xp), (2, 3), xp=xp)
        uv = xp_reshape(xp_as_array(uv, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(CCT_to_uv_Planck1900(CCT), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_CCT_to_uv_Planck1900(self) -> None:
        """
        Test :func:`colour.temperature.planck1900.CCT_to_uv_Planck1900` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        CCT_to_uv_Planck1900(cases)
