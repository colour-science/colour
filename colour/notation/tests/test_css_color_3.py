"""Define the unit tests for the :mod:`colour.notation.css_color_3` module."""

from __future__ import annotations

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.notation import keyword_to_RGB_CSSColor3
from colour.utilities import xp_assert_close, xp_assert_equal

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestKeywordToRGBCSSColor3",
]


class TestKeywordToRGBCSSColor3:
    """
    Define :func:`colour.notation.css_color_3.keyword_to_RGB_CSSColor3`
    definition unit tests methods.
    """

    def test_keyword_to_RGB_CSSColor3(self) -> None:
        """
        Test :func:`colour.notation.css_color_3.keyword_to_RGB_CSSColor3`
        definition.
        """

        xp_assert_equal(keyword_to_RGB_CSSColor3("black"), [0, 0, 0])

        xp_assert_equal(keyword_to_RGB_CSSColor3("white"), [1, 1, 1])

        xp_assert_close(
            keyword_to_RGB_CSSColor3("aliceblue"),
            [0.94117647, 0.97254902, 1.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
