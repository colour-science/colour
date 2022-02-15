"""Defines the unit tests for the :mod:`colour.geometry.section` module."""

import numpy as np
import unittest

from colour.geometry.section import (
    edges_to_chord,
    close_chord,
    unique_vertices,
)
from colour.geometry import primitive_cube, hull_section
from colour.utilities import is_trimesh_installed

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
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

        np.testing.assert_almost_equal(
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
        )

        np.testing.assert_almost_equal(
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
        )


class TestCloseChord(unittest.TestCase):
    """
    Define :func:`colour.geometry.section.close_chord` definition unit tests
    methods.
    """

    def test_close_chord(self):
        """Test :func:`colour.geometry.section.close_chord` definition."""

        np.testing.assert_almost_equal(
            close_chord(np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])),
            np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]]),
        )


class TestUniqueVertices(unittest.TestCase):
    """
    Define :func:`colour.geometry.section.unique_vertices` definition unit
    tests methods.
    """

    def test_unique_vertices(self):
        """Test :func:`colour.geometry.section.unique_vertices` definition."""

        np.testing.assert_almost_equal(
            unique_vertices(
                np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.0, 0.5, 0.0]])
            ),
            np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
        )

        np.testing.assert_almost_equal(
            unique_vertices(
                np.array(
                    [[0.0, 0.51, 0.0], [0.0, 0.0, 0.51], [0.0, 0.52, 0.0]]
                ),
                1,
            ),
            np.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]),
        )


class TestHullSection(unittest.TestCase):
    """
    Define :func:`colour.geometry.section.hull_section` definition unit tests
    methods.
    """

    def test_hull_section(self):
        """Test :func:`colour.geometry.section.hull_section` definition."""

        if not is_trimesh_installed:  # pragma: no cover
            return

        import trimesh

        vertices, faces, _outline = primitive_cube(1, 1, 1, 2, 2, 2)
        hull = trimesh.Trimesh(vertices["position"], faces, process=False)

        np.testing.assert_almost_equal(
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
        )

        np.testing.assert_almost_equal(
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
        )

        np.testing.assert_almost_equal(
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
        )

        hull.vertices = (hull.vertices + 0.5) * 2
        np.testing.assert_almost_equal(
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
        )

        self.assertRaises(ValueError, hull_section, hull, origin=-1)


if __name__ == "__main__":
    unittest.main()
