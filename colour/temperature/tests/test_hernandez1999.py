"""Define the unit tests for the :mod:`colour.temperature.hernandez1999` module."""

from __future__ import annotations

import typing
from contextlib import nullcontext

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import CCT_to_xy_Hernandez1999, xy_to_CCT_Hernandez1999
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
    "Testxy_to_CCT_Hernandez1999",
    "TestCCT_to_xy_Hernandez1999",
]


class Testxy_to_CCT_Hernandez1999:
    """
    Define :func:`colour.temperature.hernandez1999.xy_to_CCT_Hernandez1999`
    definition unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1e-1)
    def test_xy_to_CCT_Hernandez1999(
        self, xp: ModuleType, request: pytest.FixtureRequest
    ) -> None:
        """
        Test :func:`colour.temperature.hernandez1999.xy_to_CCT_McCamy1992`
        definition.
        """

        xp_assert_close(
            xy_to_CCT_Hernandez1999(xp_as_array([0.31270, 0.32900], xp=xp)),
            6500.74204318,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_CCT_Hernandez1999(xp_as_array([0.44757, 0.40745], xp=xp)),
            2790.64222533,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_CCT_Hernandez1999(
                xp_as_array([0.244162248213914, 0.240333674758318], xp=xp)
            ),
            64448.11092565,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # The first chromaticity is the integrated 200000 K Planckian
        # radiator. Both cases overflow an unweighted float32 exponential,
        # even though their final CCTs and derivatives are representable.
        for xy, expected in (
            ([0.24119282938183745, 0.23598464668748229], 208646.4896016103),
            ([0.25, 0.23], 188999.45429354507),
        ):
            native = xp_as_array(xy, xp=xp)
            with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
                result = xy_to_CCT_Hernandez1999(native)
            assert xp.isfinite(result)
            if hasattr(native, "device"):
                assert result.device == native.device

            n = (native[0] - 0.3356) / (native[1] - 0.1691)
            control = (
                36284.48953
                + xp.exp(np.log(0.00228) - n / 0.07861)
                + xp.exp(np.log(5.4535e-36) - n / 0.01543)
            )
            xp_assert_close(result, control, atol=TOLERANCE_ABSOLUTE_TESTS)
            if float(xp.finfo(native.dtype).eps) < np.finfo(np.float32).eps:
                xp_assert_close(result, expected, atol=TOLERANCE_ABSOLUTE_TESTS)

            if xp.__name__ != "numpy":
                autodiff = request.getfixturevalue("autodiff")
                autodiff(xy_to_CCT_Hernandez1999, xy)

    def test_n_dimensional_xy_to_CCT_Hernandez1999(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.hernandez1999.xy_to_CCT_Hernandez1999`
        definition n-dimensional arrays support.
        """

        xy = xp_as_array([0.31270, 0.32900], xp=xp)
        CCT = as_ndarray(xy_to_CCT_Hernandez1999(xy))

        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        CCT = xp.tile(xp_as_array(CCT, xp=xp), (6,))
        xp_assert_close(xy_to_CCT_Hernandez1999(xy), CCT, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        CCT = xp_reshape(xp_as_array(CCT, xp=xp), (2, 3), xp=xp)
        xp_assert_close(xy_to_CCT_Hernandez1999(xy), CCT, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_xy_to_CCT_Hernandez1999(self) -> None:
        """
        Test :func:`colour.temperature.hernandez1999.xy_to_CCT_Hernandez1999`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_CCT_Hernandez1999(cases)


class TestCCT_to_xy_Hernandez1999:
    """
    Define :func:`colour.temperature.hernandez1999.CCT_to_xy_Hernandez1999`
    definition unit tests methods.
    """

    def test_CCT_to_xy_Hernandez1999(self) -> None:
        """
        Test :func:`colour.temperature.hernandez1999.CCT_to_xy_Hernandez1999`
        definition.
        """

        xp_assert_close(
            CCT_to_xy_Hernandez1999(6500.74204318),
            [0.31270000, 0.32900000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_xy_Hernandez1999(2790.64222533),
            [0.39242193, 0.29118533],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_xy_Hernandez1999(64448.11092565),
            [0.24382815, 0.24059395],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CCT_to_xy_Hernandez1999(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.hernandez1999.CCT_to_xy_Hernandez1999`
        definition n-dimensional arrays support.
        """

        CCT = 6500.74204318
        xy = as_ndarray(CCT_to_xy_Hernandez1999(CCT))

        CCT = xp.tile(xp_as_array(CCT, xp=xp), (6,))
        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        xp_assert_close(CCT_to_xy_Hernandez1999(CCT), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

        CCT = xp_reshape(xp_as_array(CCT, xp=xp), (2, 3), xp=xp)
        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(CCT_to_xy_Hernandez1999(CCT), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_CCT_to_xy_Hernandez1999(self) -> None:
        """
        Test :func:`colour.temperature.hernandez1999.CCT_to_xy_Hernandez1999`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        CCT_to_xy_Hernandez1999(cases)
