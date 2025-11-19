"""Define the unit tests for the :mod:`colour.geometry` module."""

from __future__ import annotations

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.geometry import (
    primitive,
    primitive_cube,
    primitive_grid,
    primitive_vertices,
    primitive_vertices_cube_mpl,
    primitive_vertices_grid_mpl,
    primitive_vertices_quad_mpl,
    primitive_vertices_sphere,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPrimitive",
    "TestPrimitiveVertices",
]


class TestPrimitive:
    """
    Define :func:`colour.geometry.primitive` definition
    unit tests methods.
    """

    def test_primitive(self) -> None:
        """Test :func:`colour.geometry.primitive` definition."""

        # Test Grid method
        grid_result = primitive("Grid")
        grid_expected = primitive_grid()
        assert len(grid_result) == len(grid_expected)
        for i in range(len(grid_result)):
            np.testing.assert_array_equal(grid_result[i], grid_expected[i])

        # Test Cube method
        cube_result = primitive("Cube")
        cube_expected = primitive_cube()
        assert len(cube_result) == len(cube_expected)
        for i in range(len(cube_result)):
            np.testing.assert_array_equal(cube_result[i], cube_expected[i])


class TestPrimitiveVertices:
    """
    Define :func:`colour.geometry.primitive_vertices` definition
    unit tests methods.
    """

    def test_primitive_vertices(self) -> None:
        """Test :func:`colour.geometry.primitive_vertices` definition."""

        np.testing.assert_allclose(
            primitive_vertices("Quad MPL"),
            primitive_vertices_quad_mpl(),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            primitive_vertices("Grid MPL"),
            primitive_vertices_grid_mpl(),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            primitive_vertices("Cube MPL"),
            primitive_vertices_cube_mpl(),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            primitive_vertices("Sphere"),
            primitive_vertices_sphere(),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
