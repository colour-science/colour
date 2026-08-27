"""Define the unit tests for the :mod:`colour.volume.macadam_limits` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_ndarray,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)
from colour.volume import is_within_macadam_limits

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestIsWithinMacadamLimits",
]


class TestIsWithinMacadamLimits:
    """
    Define :func:`colour.volume.macadam_limits.is_within_macadam_limits`
    definition unit tests methods.
    """

    def test_is_within_macadam_limits(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.volume.macadam_limits.is_within_macadam_limits`
        definition.
        """

        assert is_within_macadam_limits(
            xp_as_array([0.3205, 0.4131, 0.5100], xp=xp), "A"
        )

        assert not is_within_macadam_limits(
            xp_as_array([0.0005, 0.0031, 0.0010], xp=xp), "A"
        )

        assert is_within_macadam_limits(
            xp_as_array([0.4325, 0.3788, 0.1034], xp=xp), "C"
        )

        assert not is_within_macadam_limits(
            xp_as_array([0.0025, 0.0088, 0.0340], xp=xp), "C"
        )

    def test_n_dimensional_is_within_macadam_limits(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.volume.macadam_limits.is_within_macadam_limits`
        definition n-dimensional arrays support.
        """

        a = xp_as_array([0.3205, 0.4131, 0.5100], xp=xp)
        b = as_ndarray(is_within_macadam_limits(a, "A"))

        a = xp.tile(xp_as_array(a, xp=xp), (6, 1))
        b = xp.tile(xp_as_array(b, xp=xp), (6,))
        xp_assert_close(
            is_within_macadam_limits(a, "A"),
            b,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3, 3), xp=xp)
        b = xp_reshape(xp_as_array(b, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            is_within_macadam_limits(a, "A"),
            b,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_is_within_macadam_limits(self) -> None:
        """
        Test :func:`colour.volume.macadam_limits.is_within_macadam_limits`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        is_within_macadam_limits(cases, "A")
