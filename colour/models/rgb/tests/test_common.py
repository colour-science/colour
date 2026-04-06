"""Define the unit tests for the :mod:`colour.models.rgb.common` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType


from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import XYZ_to_sRGB, sRGB_to_XYZ
from colour.utilities import xp_as_array, xp_assert_close

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_sRGB",
    "TestsRGB_to_XYZ",
]


class TestXYZ_to_sRGB:
    """
    Define :func:`colour.models.rgb.common.XYZ_to_sRGB` definition unit tests
    methods.
    """

    def test_XYZ_to_sRGB(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.common.XYZ_to_sRGB` definition."""

        xp_assert_close(
            XYZ_to_sRGB(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            [0.70573936, 0.19248266, 0.22354169],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_sRGB(xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            [0.25847003, 0.58276102, 0.29718877],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_sRGB(
                xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp),
                [0.34570, 0.35850],
            ),
            [0.09838967, 0.25404426, 0.65130925],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_sRGB(
                xp_as_array([0.00000000, 0.00000000, 0.00000000], xp=xp),
                [0.44757, 0.40745],
            ),
            [0.00000000, 0.00000000, 0.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_sRGB(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                [0.44757, 0.40745],
                chromatic_adaptation_transform="Bradford",
            ),
            [0.60873814, 0.23259548, 0.43714892],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_sRGB(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                apply_cctf_encoding=False,
            ),
            [0.45620520, 0.03081070, 0.04091953],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestsRGB_to_XYZ:
    """
    Define :func:`colour.models.rgb.common.sRGB_to_XYZ` definition unit tests
    methods.
    """

    def test_sRGB_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.rgb.common.sRGB_to_XYZ` definition."""

        xp_assert_close(
            sRGB_to_XYZ(xp_as_array([0.70573936, 0.19248266, 0.22354169], xp=xp)),
            [0.20654290, 0.12197943, 0.05137140],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sRGB_to_XYZ(xp_as_array([0.25847003, 0.58276102, 0.29718877], xp=xp)),
            [0.14222582, 0.23043727, 0.10496290],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sRGB_to_XYZ(
                xp_as_array([0.09838967, 0.25404426, 0.65130925], xp=xp),
                [0.34570, 0.35850],
            ),
            [0.07819162, 0.06157356, 0.28099475],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sRGB_to_XYZ(
                xp_as_array([0.00000000, 0.00000000, 0.00000000], xp=xp),
                [0.44757, 0.40745],
            ),
            [0.00000000, 0.00000000, 0.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sRGB_to_XYZ(
                xp_as_array([0.60873814, 0.23259548, 0.43714892], xp=xp),
                [0.44757, 0.40745],
                chromatic_adaptation_transform="Bradford",
            ),
            [0.20654449, 0.12197792, 0.05137030],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sRGB_to_XYZ(
                xp_as_array([0.45620520, 0.03081070, 0.04091953], xp=xp),
                apply_cctf_decoding=False,
            ),
            [0.20654291, 0.12197943, 0.05137141],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
