"""Define the unit tests for the :mod:`colour.temperature` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType


from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import CCT_to_xy, xy_to_CCT
from colour.utilities import xp_as_array, xp_assert_close

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXy_to_CCT",
    "TestCCT_to_xy",
]


class TestXy_to_CCT:
    """
    Define :func:`colour.temperature.xy_to_CCT` definition unit tests methods.
    """

    def test_xy_to_CCT(self, xp: ModuleType) -> None:
        """Test :func:`colour.temperature.xy_to_CCT` definition."""

        xy = xp_as_array([0.31270, 0.32900], xp=xp)

        # Test default method (CIE Illuminant D Series)
        xp_assert_close(
            xy_to_CCT(xy),
            6508.1175148,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

        # Test Hernandez 1999 method
        xp_assert_close(
            xy_to_CCT(xy, "Hernandez 1999"),
            6500.7420431,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

        # Test McCamy 1992 method
        xp_assert_close(
            xy_to_CCT(xy, "McCamy 1992"),
            6505.08059131,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )


class TestCCT_to_xy:
    """
    Define :func:`colour.temperature.CCT_to_xy` definition unit tests methods.
    """

    def test_CCT_to_xy(self, xp: ModuleType) -> None:
        """Test :func:`colour.temperature.CCT_to_xy` definition."""

        # Test default method (CIE Illuminant D Series)
        xp_assert_close(
            CCT_to_xy(xp_as_array([6500], xp=xp)),
            [[0.31277888, 0.3291835]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test explicit CIE Illuminant D Series method
        xp_assert_close(
            CCT_to_xy(xp_as_array([6500], xp=xp), method="CIE Illuminant D Series"),
            [[0.31277888, 0.3291835]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test Hernandez 1999 method
        xp_assert_close(
            CCT_to_xy(xp_as_array([6500], xp=xp), "Hernandez 1999"),
            [[0.31271354, 0.32900208]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
