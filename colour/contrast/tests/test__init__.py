"""Define the unit tests for the :mod:`colour.contrast` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType


from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.contrast import contrast_sensitivity_function
from colour.utilities import xp_as_array, xp_assert_close

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestContrastSensitivityFunction",
]


class TestContrastSensitivityFunction:
    """
    Define :func:`colour.contrast.contrast_sensitivity_function` definition
    unit tests methods.
    """

    def test_contrast_sensitivity_function(self, xp: ModuleType) -> None:
        """Test :func:`colour.contrast.contrast_sensitivity_function` definition."""

        _a = lambda v: xp_as_array([v], xp=xp)  # noqa: E731

        # Test default method (Barten 1999)
        xp_assert_close(
            contrast_sensitivity_function(
                u=_a(4),
                sigma=_a(0.01),
                E=_a(65),
                X_0=_a(60),
                X_max=_a(12),
                Y_0=_a(60),
                Y_max=_a(12),
                p=_a(1.2e6),
            ),
            [352.761342126727020],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test explicit Barten 1999 method with different parameters
        xp_assert_close(
            contrast_sensitivity_function(
                "Barten 1999",
                u=_a(8),
                sigma=_a(0.01),
                E=_a(65),
                X_0=_a(60),
                X_max=_a(12),
                Y_0=_a(60),
                Y_max=_a(12),
                p=_a(1.2e6),
            ),
            [177.706338840717340],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test with another set of parameters
        xp_assert_close(
            contrast_sensitivity_function(
                u=_a(20),
                sigma=_a(0.01),
                E=_a(65),
                X_0=_a(60),
                X_max=_a(12),
                Y_0=_a(60),
                Y_max=_a(12),
                p=_a(1.2e6),
            ),
            [37.455090830648620],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
