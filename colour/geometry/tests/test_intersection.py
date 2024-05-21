"""Define the unit tests for the :mod:`colour.geometry.intersection` module."""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.geometry import (
    extend_line_segment,
    intersect_line_segments,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestExtendLineSegment",
    "TestIntersectLineSegments",
]


class TestExtendLineSegment:
    """
    Define :func:`colour.geometry.intersection.extend_line_segment` definition unit
    tests methods.
    """

    def test_extend_line_segment(self):
        """Test :func:`colour.geometry.intersection.extend_line_segment` definition."""

        np.testing.assert_allclose(
            extend_line_segment(
                np.array([0.95694934, 0.13720932]),
                np.array([0.28382835, 0.60608318]),
            ),
            np.array([-0.5367248, 1.17765341]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            extend_line_segment(
                np.array([0.95694934, 0.13720932]),
                np.array([0.28382835, 0.60608318]),
                5,
            ),
            np.array([-3.81893739, 3.46393435]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            extend_line_segment(
                np.array([0.95694934, 0.13720932]),
                np.array([0.28382835, 0.60608318]),
                -1,
            ),
            np.array([1.1043815, 0.03451295]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestIntersectLineSegments:
    """
    Define :func:`colour.geometry.intersection.intersect_line_segments`
    definition unit tests methods.
    """

    def test_intersect_line_segments(self):
        """
        Test :func:`colour.geometry.intersection.intersect_line_segments`
        definition.
        """

        l_1 = np.array(
            [
                [[0.15416284, 0.7400497], [0.26331502, 0.53373939]],
                [[0.01457496, 0.91874701], [0.90071485, 0.03342143]],
            ]
        )
        l_2 = np.array(
            [
                [[0.95694934, 0.13720932], [0.28382835, 0.60608318]],
                [[0.94422514, 0.85273554], [0.00225923, 0.52122603]],
                [[0.55203763, 0.48537741], [0.76813415, 0.16071675]],
                [[0.01457496, 0.91874701], [0.90071485, 0.03342143]],
            ]
        )

        s = intersect_line_segments(l_1, l_2)

        np.testing.assert_allclose(
            s.xy,
            np.array(
                [
                    [
                        [np.nan, np.nan],
                        [0.22791841, 0.60064309],
                        [np.nan, np.nan],
                        [np.nan, np.nan],
                    ],
                    [
                        [0.42814517, 0.50555685],
                        [0.30560559, 0.62798382],
                        [0.7578749, 0.17613012],
                        [np.nan, np.nan],
                    ],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_array_equal(
            s.intersect,
            np.array([[False, True, False, False], [True, True, True, False]]),
        )

        np.testing.assert_array_equal(
            s.parallel,
            np.array([[False, False, False, False], [False, False, False, True]]),
        )

        np.testing.assert_array_equal(
            s.coincident,
            np.array([[False, False, False, False], [False, False, False, True]]),
        )
