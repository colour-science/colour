"""Define the unit tests for the :mod:`colour.io.ocio` module."""

from __future__ import annotations

import os
import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.io import process_image_OpenColorIO
from colour.utilities import full, is_opencolorio_installed, xp_assert_close

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "ROOT_RESOURCES",
    "TestProcessImageOpenColorIO",
]

ROOT_RESOURCES: str = os.path.join(os.path.dirname(__file__), "resources")


class TestProcessImageOpenColorIO:
    """
    Define :func:`colour.io.ocio.process_image_OpenColorIO` definition unit
    tests methods.
    """

    def test_process_image_OpenColorIO(self) -> None:
        """Test :func:`colour.io.ocio.process_image_OpenColorIO` definition."""

        # TODO: Remove when "Pypi" wheel compatible with "ARM" on "macOS" is
        # released.
        if not is_opencolorio_installed():  # pragma: no cover
            return

        import PyOpenColorIO as ocio_module  # noqa: PLC0415

        ocio = typing.cast("typing.Any", ocio_module)

        config = os.path.join(ROOT_RESOURCES, "config-aces-reference.ocio.yaml")

        a = full([4, 2, 3], 0.18)

        xp_assert_close(
            process_image_OpenColorIO(
                a, "ACES - ACES2065-1", "ACES - ACEScct", config=config
            ),
            [
                [
                    [0.41358781, 0.41358781, 0.41358781],
                    [0.41358781, 0.41358781, 0.41358781],
                ],
                [
                    [0.41358781, 0.41358781, 0.41358781],
                    [0.41358781, 0.41358781, 0.41358781],
                ],
                [
                    [0.41358781, 0.41358781, 0.41358781],
                    [0.41358781, 0.41358781, 0.41358781],
                ],
                [
                    [0.41358781, 0.41358781, 0.41358781],
                    [0.41358781, 0.41358781, 0.41358781],
                ],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            process_image_OpenColorIO(
                a,
                "ACES - ACES2065-1",
                "Display - sRGB",
                "Output - SDR Video - ACES 1.0",
                ocio.TRANSFORM_DIR_FORWARD,
                config=config,
            ),
            [
                [
                    [0.35595229, 0.35595256, 0.35595250],
                    [0.35595229, 0.35595256, 0.35595250],
                ],
                [
                    [0.35595229, 0.35595256, 0.35595250],
                    [0.35595229, 0.35595256, 0.35595250],
                ],
                [
                    [0.35595229, 0.35595256, 0.35595250],
                    [0.35595229, 0.35595256, 0.35595250],
                ],
                [
                    [0.35595229, 0.35595256, 0.35595250],
                    [0.35595229, 0.35595256, 0.35595250],
                ],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test scalar input
        a_scalar = np.array(0.18)
        result_scalar = process_image_OpenColorIO(
            a_scalar, "ACES - ACES2065-1", "ACES - ACEScct", config=config
        )
        assert isinstance(result_scalar, (float, np.floating))
        xp_assert_close(result_scalar, 0.41358781, atol=TOLERANCE_ABSOLUTE_TESTS)

        # Test single-channel input
        a_single = np.array([0.18])
        result_single = process_image_OpenColorIO(
            a_single, "ACES - ACES2065-1", "ACES - ACEScct", config=config
        )
        assert result_single.shape == (1,)
        xp_assert_close(result_single, [0.41358781], atol=TOLERANCE_ABSOLUTE_TESTS)
