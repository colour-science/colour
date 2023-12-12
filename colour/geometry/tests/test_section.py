# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.geometry.section` module."""

import unittest

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.geometry import hull_section, primitive_cube
from colour.geometry.section import (
    close_chord,
    edges_to_chord,
    unique_vertices,
)
from colour.utilities import is_trimesh_installed

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestEdgesToChord",
    "TestCloseChord",
    "TestUniqueVertices",
    "TestHullSection",
]


class TestEdgesToChord(unittest.TestCase):
    """
    Define :func:`colour.geometry.section.edges_to_chord` definition unit
    tests methods.
    """

    def test_edges_to_chord(self):
        """Test :func:`colour.geometry.section.edges_to_chord` definition."""

        edges = np.array(
            [
                [[0.0, -0.5, 0.0], [0.5, -0.5, 0.0]],
                [[-0.5, -0.5, 0.0], [0.0, -0.5, 0.0]],
                [[0.5, 0.5, 0.0], [0.0, 0.5, 0.0]],
                [[0.0, 0.5, 0.0], [-0.5, 0.5, 0.0]],
                [[-0.5, 0.0, 0.0], [-0.5, -0.5, 0.0]],
                [[-0.5, 0.5, 0.0], [-0.5, 0.0, 0.0]],
                [[0.5, -0.5, 0.0], [0.5, 0.0, 0.0]],
                [[0.5, 0.0, 0.0], [0.5, 0.5, 0.0]],
            ]
        )

        np.testing.assert_allclose(
            edges_to_chord(edges),
            np.array(
                [
                    [0.0, -0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [0.5, -0.5, -0.0],
                    [0.5, 0.0, -0.0],
                    [0.5, 0.0, -0.0],
                    [0.5, 0.5, -0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [-0.5, 0.5, 0.0],
                    [-0.5, 0.5, -0.0],
                    [-0.5, 0.0, -0.0],
                    [-0.5, 0.0, -0.0],
                    [-0.5, -0.5, -0.0],
                    [-0.5, -0.5, 0.0],
                    [0.0, -0.5, 0.0],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            edges_to_chord(edges, 5),
            np.array(
                [
                    [-0.5, 0.5, 0.0],
                    [-0.5, 0.0, 0.0],
                    [-0.5, 0.0, 0.0],
                    [-0.5, -0.5, 0.0],
                    [-0.5, -0.5, 0.0],
                    [0.0, -0.5, 0.0],
                    [0.0, -0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [-0.5, 0.5, 0.0],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestCloseChord(unittest.TestCase):
    """
    Define :func:`colour.geometry.section.close_chord` definition unit tests
    methods.
    """

    def test_close_chord(self):
        """Test :func:`colour.geometry.section.close_chord` definition."""

        np.testing.assert_allclose(
            close_chord(np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])),
            np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestUniqueVertices(unittest.TestCase):
    """
    Define :func:`colour.geometry.section.unique_vertices` definition unit
    tests methods.
    """

    def test_unique_vertices(self):
        """Test :func:`colour.geometry.section.unique_vertices` definition."""

        np.testing.assert_allclose(
            unique_vertices(
                np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]])
            ),
            np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            unique_vertices(
                np.array(
                    [[0.0, 0.51, 0.0], [0.0, 0.0, 0.51], [0.0, 0.52, 0.0]]
                ),
                1,
            ),
            np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestHullSection(unittest.TestCase):
    """
    Define :func:`colour.geometry.section.hull_section` definition unit tests
    methods.
    """

    def test_hull_section(self):
        """Test :func:`colour.geometry.section.hull_section` definition."""

        if not is_trimesh_installed():  # pragma: no cover
            return

        import trimesh

        vertices, faces, _outline = primitive_cube(1, 1, 1, 2, 2, 2)
        hull = trimesh.Trimesh(vertices["position"], faces, process=False)

        np.testing.assert_allclose(
            hull_section(hull, origin=0),
            np.array(
                [
                    [0.0, -0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.0, 0.5, 0.0],
                    [-0.5, 0.5, 0.0],
                    [-0.5, 0.0, 0.0],
                    [-0.5, -0.5, 0.0],
                    [0.0, -0.5, 0.0],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            hull_section(hull, axis="+x", origin=0),
            np.array(
                [
                    [0.0, 0.0, -0.5],
                    [0.0, 0.5, -0.5],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.0, 0.0, 0.5],
                    [0.0, -0.5, 0.5],
                    [0.0, -0.5, 0.0],
                    [0.0, -0.5, -0.5],
                    [0.0, 0.0, -0.5],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            hull_section(hull, axis="+y", origin=0),
            np.array(
                [
                    [0.0, 0.0, -0.5],
                    [-0.5, 0.0, -0.5],
                    [-0.5, 0.0, 0.0],
                    [-0.5, 0.0, 0.5],
                    [0.0, 0.0, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.0, 0.0],
                    [0.5, 0.0, -0.5],
                    [0.0, 0.0, -0.5],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        hull.vertices = (hull.vertices + 0.5) * 2
        np.testing.assert_allclose(
            hull_section(hull, origin=0.5, normalise=True),
            np.array(
                [
                    [1.0, 0.0, 1.0],
                    [2.0, 0.0, 1.0],
                    [2.0, 1.0, 1.0],
                    [2.0, 2.0, 1.0],
                    [1.0, 2.0, 1.0],
                    [0.0, 2.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        self.assertRaises(ValueError, hull_section, hull, origin=-1)


if __name__ == "__main__":
    unittest.main()
