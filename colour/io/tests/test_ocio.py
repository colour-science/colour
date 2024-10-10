"""Define the unit tests for the :mod:`colour.io.ocio` module."""

from __future__ import annotations

import os

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.io import process_image_OpenColorIO
from colour.utilities import full, is_opencolorio_installed, tstack

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

    def test_process_image_OpenColorIO(self):
        """Test :func:`colour.io.ocio.process_image_OpenColorIO` definition."""

        # TODO: Remove when "Pypi" wheel compatible with "ARM" on "macOS" is
        # released.
        if not is_opencolorio_installed():  # pragma: no cover
            return

        import PyOpenColorIO as ocio

        config = os.path.join(ROOT_RESOURCES, "config-aces-reference.ocio.yaml")

        a = full([4, 2, 3], 0.18)

        np.testing.assert_allclose(
            process_image_OpenColorIO(
                a, "ACES - ACES2065-1", "ACES - ACEScct", config=config
            ),
            np.array(
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
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            process_image_OpenColorIO(
                a,
                "ACES - ACES2065-1",
                "Display - sRGB",
                "Output - SDR Video - ACES 1.0",
                ocio.TRANSFORM_DIR_FORWARD,
                config=config,
            ),
            np.array(
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
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            process_image_OpenColorIO(
                tstack(([0.2, 0.4, 0.6], [0.2, 0.4, 0.6], [0.2, 0.4, 0.6])),
                "ACES - ACES2065-1",
                "ACES - ACEScct",
                config=config,
            ),
            np.array(
                [
                    [0.42226437, 0.42226437, 0.42226437],
                    [0.47934198, 0.47934198, 0.47934198],
                    [0.51273096, 0.51273096, 0.51273096],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
