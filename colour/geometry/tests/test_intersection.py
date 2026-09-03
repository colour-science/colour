"""Define the unit tests for the :mod:`colour.geometry.intersection` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.geometry import (
    extend_line_segment,
    intersect_line_segments,
    intersect_ray_circle_2d,
)
from colour.utilities import as_ndarray, xp_as_array, xp_assert_close, xp_assert_equal

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestExtendLineSegment",
    "TestIntersectLineSegments",
    "TestIntersectRayCircle2D",
]


class TestExtendLineSegment:
    """
    Define :func:`colour.geometry.intersection.extend_line_segment` definition unit
    tests methods.
    """

    def test_extend_line_segment(self, xp: ModuleType) -> None:
        """Test :func:`colour.geometry.intersection.extend_line_segment` definition."""

        xp_assert_close(
            extend_line_segment(
                xp_as_array([0.95694934, 0.13720932], xp=xp),
                xp_as_array([0.28382835, 0.60608318], xp=xp),
            ),
            [-0.5367248, 1.17765341],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            extend_line_segment(
                xp_as_array([0.95694934, 0.13720932], xp=xp),
                xp_as_array([0.28382835, 0.60608318], xp=xp),
                5,
            ),
            [-3.81893739, 3.46393435],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            extend_line_segment(
                xp_as_array([0.95694934, 0.13720932], xp=xp),
                xp_as_array([0.28382835, 0.60608318], xp=xp),
                -1,
            ),
            [1.1043815, 0.03451295],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestIntersectLineSegments:
    """
    Define :func:`colour.geometry.intersection.intersect_line_segments`
    definition unit tests methods.
    """

    def test_intersect_line_segments(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.geometry.intersection.intersect_line_segments`
        definition.
        """

        l_1 = xp_as_array(
            [
                [[0.15416284, 0.7400497], [0.26331502, 0.53373939]],
                [[0.01457496, 0.91874701], [0.90071485, 0.03342143]],
            ],
            xp=xp,
        )
        l_2 = xp_as_array(
            [
                [[0.95694934, 0.13720932], [0.28382835, 0.60608318]],
                [[0.94422514, 0.85273554], [0.00225923, 0.52122603]],
                [[0.55203763, 0.48537741], [0.76813415, 0.16071675]],
                [[0.01457496, 0.91874701], [0.90071485, 0.03342143]],
            ],
            xp=xp,
        )

        s = intersect_line_segments(l_1, l_2)

        xp_assert_close(
            s.xy,
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
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_equal(
            s.intersect,
            [[False, True, False, False], [True, True, True, False]],
        )

        xp_assert_equal(
            s.parallel,
            [[False, False, False, False], [False, False, False, True]],
        )

        xp_assert_equal(
            s.coincident,
            [[False, False, False, False], [False, False, False, True]],
        )

    def test_intersect_line_segments_autodiff(
        self, xp: ModuleType, autodiff: typing.Callable
    ) -> None:
        """Test inactive parallel segments do not contaminate gradients."""

        l_2 = xp_as_array(
            [
                [[0.0, 1.0], [2.0, 1.0]],
                [[1.0, -1.0], [1.0, 1.0]],
            ],
            xp=xp,
        )

        def function(l_1: typing.Any) -> typing.Any:
            """Return the selected non-parallel intersection."""

            return intersect_line_segments(l_1, l_2).xy[0, 1]

        intersection, (gradient,), _inputs = autodiff(function, [[0.0, 0.0, 2.0, 0.0]])

        xp_assert_close(intersection, [1.0, 0.0])
        assert xp.isfinite(gradient).all()
        assert xp.any(gradient != 0)


class TestIntersectRayCircle2D:
    """
    Define :func:`colour.geometry.intersection.intersect_ray_circle_2d`
    definition unit tests methods.
    """

    def test_intersect_ray_circle_2d(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.geometry.intersection.\
intersect_ray_circle_2d` definition.
        """

        # Ray pointing up from inside a circle.
        distance = float(
            as_ndarray(
                intersect_ray_circle_2d(
                    xp_as_array([0.0, 5.0], xp=xp),
                    xp_as_array([0.0, 1.0], xp=xp),
                    10.0,
                )
            )
        )
        xp_assert_close(distance, 5.0, atol=TOLERANCE_ABSOLUTE_TESTS * 0.001)

        # Ray pointing down from inside, should hit far side.
        distance = float(
            as_ndarray(
                intersect_ray_circle_2d(
                    xp_as_array([0.0, 5.0], xp=xp),
                    xp_as_array([0.0, -1.0], xp=xp),
                    10.0,
                )
            )
        )
        xp_assert_close(distance, 15.0, atol=TOLERANCE_ABSOLUTE_TESTS * 0.001)

        # No forward intersection (outside, pointing away).
        distance = float(
            as_ndarray(
                intersect_ray_circle_2d(
                    xp_as_array([0.0, 15.0], xp=xp),
                    xp_as_array([0.0, 1.0], xp=xp),
                    10.0,
                )
            )
        )
        assert np.isnan(distance)

        # Horizontal ray from offset origin (3,0) -> hits circle r=5 at x=5.
        distance = float(
            as_ndarray(
                intersect_ray_circle_2d(
                    xp_as_array([3.0, 0.0], xp=xp),
                    xp_as_array([1.0, 0.0], xp=xp),
                    5.0,
                )
            )
        )
        xp_assert_close(distance, 2.0, atol=TOLERANCE_ABSOLUTE_TESTS * 0.001)

        # Tangent (touch only): no forward intersection.
        distance = float(
            as_ndarray(
                intersect_ray_circle_2d(
                    xp_as_array([0.0, 10.0], xp=xp),
                    xp_as_array([1.0, 0.0], xp=xp),
                    10.0,
                )
            )
        )
        assert np.isnan(distance)

        # Ray from origin pointing outward.
        distance = float(
            as_ndarray(
                intersect_ray_circle_2d(
                    xp_as_array([0.0, 0.0], xp=xp),
                    xp_as_array([1.0, 0.0], xp=xp),
                    5.0,
                )
            )
        )
        xp_assert_close(distance, 5.0, atol=TOLERANCE_ABSOLUTE_TESTS * 0.001)
