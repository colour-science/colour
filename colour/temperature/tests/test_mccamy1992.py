"""Define the unit tests for the :mod:`colour.temperature.mccamy1992` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import CCT_to_xy_McCamy1992, xy_to_CCT_McCamy1992
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
    "Testxy_to_CCT_McCamy1992",
    "TestCCT_to_xy_McCamy1992",
]


class Testxy_to_CCT_McCamy1992:
    """
    Define :func:`colour.temperature.mccamy1992.xy_to_CCT_McCamy1992`
    definition unit tests methods.
    """

    def test_xy_to_CCT_McCamy1992(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.mccamy1992.xy_to_CCT_McCamy1992`
        definition.
        """

        xp_assert_close(
            xy_to_CCT_McCamy1992(xp_as_array([0.31270, 0.32900], xp=xp)),
            6505.08059131,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_CCT_McCamy1992(xp_as_array([0.44757, 0.40745], xp=xp)),
            2857.28961266,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_CCT_McCamy1992(
                xp_as_array([0.252520939374083, 0.252220883926284], xp=xp)
            ),
            19501.61953130,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xy_to_CCT_McCamy1992(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.mccamy1992.xy_to_CCT_McCamy1992`
        definition n-dimensional arrays support.
        """

        xy = xp_as_array([0.31270, 0.32900], xp=xp)
        CCT = as_ndarray(xy_to_CCT_McCamy1992(xy))

        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        CCT = np.tile(CCT, 6)
        xp_assert_close(xy_to_CCT_McCamy1992(xy), CCT, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        CCT = np.reshape(CCT, (2, 3))
        xp_assert_close(xy_to_CCT_McCamy1992(xy), CCT, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_xy_to_CCT_McCamy1992(self) -> None:
        """
        Test :func:`colour.temperature.mccamy1992.xy_to_CCT_McCamy1992`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_CCT_McCamy1992(cases)


class TestCCT_to_xy_McCamy1992:
    """
    Define :func:`colour.temperature.mccamy1992.CCT_to_xy_McCamy1992`
    definition unit tests methods.
    """

    def test_CCT_to_xy_McCamy1992(self) -> None:
        """
        Test :func:`colour.temperature.mccamy1992.CCT_to_xy_McCamy1992`
        definition.
        """

        xp_assert_close(
            CCT_to_xy_McCamy1992(6505.08059131),
            [0.31270000, 0.32900000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_xy_McCamy1992(2857.28961266),
            [0.38658009, 0.29047836],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_xy_McCamy1992(19501.61953130),
            [0.25017434, 0.25418195],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CCT_to_xy_McCamy1992(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.mccamy1992.CCT_to_xy_McCamy1992`
        definition n-dimensional arrays support.
        """

        CCT = 6505.08059131
        xy = as_ndarray(CCT_to_xy_McCamy1992(CCT))

        CCT = np.tile(CCT, 6)
        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        xp_assert_close(CCT_to_xy_McCamy1992(CCT), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

        CCT = np.reshape(CCT, (2, 3))
        xy = xp_reshape(xy, (2, 3, 2), xp=xp)
        xp_assert_close(CCT_to_xy_McCamy1992(CCT), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_CCT_to_xy_McCamy1992(self) -> None:
        """
        Test :func:`colour.temperature.mccamy1992.CCT_to_xy_McCamy1992`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        CCT_to_xy_McCamy1992(cases)
