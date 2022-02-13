"""Defines the unit tests for the :mod:`colour.plotting.volume` module."""

import numpy as np
import unittest
from matplotlib.pyplot import Axes, Figure

from colour.plotting import plot_RGB_colourspaces_gamuts, plot_RGB_scatter
from colour.plotting.volume import nadir_grid, RGB_identity_cube

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestNadirGrid",
    "TestRGBIdentityCube",
    "TestPlotRGBColourspacesGamuts",
    "TestPlotRGBScatter",
]


class TestNadirGrid(unittest.TestCase):
    """
    Define :func:`colour.plotting.volume.nadir_grid` definition unit tests
    methods.
    """

    def test_nadir_grid(self):
        """Test :func:`colour.plotting.volume.nadir_grid` definition."""

        quads, faces_colours, edges_colours = nadir_grid(segments=1)

        np.testing.assert_almost_equal(
            quads,
            np.array(
                [
                    [
                        [-1.00000000, -1.00000000, 0.00000000],
                        [1.00000000, -1.00000000, 0.00000000],
                        [1.00000000, 1.00000000, 0.00000000],
                        [-1.00000000, 1.00000000, 0.00000000],
                    ],
                    [
                        [-1.00000000, -1.00000000, 0.00000000],
                        [0.00000000, -1.00000000, 0.00000000],
                        [0.00000000, 0.00000000, 0.00000000],
                        [-1.00000000, 0.00000000, 0.00000000],
                    ],
                    [
                        [-1.00000000, 0.00000000, 0.00000000],
                        [0.00000000, 0.00000000, 0.00000000],
                        [0.00000000, 1.00000000, 0.00000000],
                        [-1.00000000, 1.00000000, 0.00000000],
                    ],
                    [
                        [0.00000000, -1.00000000, 0.00000000],
                        [1.00000000, -1.00000000, 0.00000000],
                        [1.00000000, 0.00000000, 0.00000000],
                        [0.00000000, 0.00000000, 0.00000000],
                    ],
                    [
                        [0.00000000, 0.00000000, 0.00000000],
                        [1.00000000, 0.00000000, 0.00000000],
                        [1.00000000, 1.00000000, 0.00000000],
                        [0.00000000, 1.00000000, 0.00000000],
                    ],
                    [
                        [-1.00000000, -0.00100000, 0.00000000],
                        [1.00000000, -0.00100000, 0.00000000],
                        [1.00000000, 0.00100000, 0.00000000],
                        [-1.00000000, 0.00100000, 0.00000000],
                    ],
                    [
                        [-0.00100000, -1.00000000, 0.00000000],
                        [0.00100000, -1.00000000, 0.00000000],
                        [0.00100000, 1.00000000, 0.00000000],
                        [-0.00100000, 1.00000000, 0.00000000],
                    ],
                ]
            ),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            faces_colours,
            np.array(
                [
                    [0.25000000, 0.25000000, 0.25000000, 0.10000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 0.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000],
                ]
            ),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            edges_colours,
            np.array(
                [
                    [0.50000000, 0.50000000, 0.50000000, 0.50000000],
                    [0.75000000, 0.75000000, 0.75000000, 0.25000000],
                    [0.75000000, 0.75000000, 0.75000000, 0.25000000],
                    [0.75000000, 0.75000000, 0.75000000, 0.25000000],
                    [0.75000000, 0.75000000, 0.75000000, 0.25000000],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000],
                    [0.00000000, 0.00000000, 0.00000000, 1.00000000],
                ]
            ),
            decimal=7,
        )


class TestRGBIdentityCube(unittest.TestCase):
    """
    Define :func:`colour.plotting.volume.RGB_identity_cube` definition unit
    tests methods.
    """

    def test_RGB_identity_cube(self):
        """Test :func:`colour.plotting.volume.RGB_identity_cube` definition."""

        vertices, RGB = RGB_identity_cube(1, 1, 1)

        np.testing.assert_almost_equal(
            vertices,
            np.array(
                [
                    [
                        [0.00000000, 0.00000000, 0.00000000],
                        [1.00000000, 0.00000000, 0.00000000],
                        [1.00000000, 1.00000000, 0.00000000],
                        [0.00000000, 1.00000000, 0.00000000],
                    ],
                    [
                        [0.00000000, 0.00000000, 1.00000000],
                        [1.00000000, 0.00000000, 1.00000000],
                        [1.00000000, 1.00000000, 1.00000000],
                        [0.00000000, 1.00000000, 1.00000000],
                    ],
                    [
                        [0.00000000, 0.00000000, 0.00000000],
                        [1.00000000, 0.00000000, 0.00000000],
                        [1.00000000, 0.00000000, 1.00000000],
                        [0.00000000, 0.00000000, 1.00000000],
                    ],
                    [
                        [0.00000000, 1.00000000, 0.00000000],
                        [1.00000000, 1.00000000, 0.00000000],
                        [1.00000000, 1.00000000, 1.00000000],
                        [0.00000000, 1.00000000, 1.00000000],
                    ],
                    [
                        [0.00000000, 0.00000000, 0.00000000],
                        [0.00000000, 1.00000000, 0.00000000],
                        [0.00000000, 1.00000000, 1.00000000],
                        [0.00000000, 0.00000000, 1.00000000],
                    ],
                    [
                        [1.00000000, 0.00000000, 0.00000000],
                        [1.00000000, 1.00000000, 0.00000000],
                        [1.00000000, 1.00000000, 1.00000000],
                        [1.00000000, 0.00000000, 1.00000000],
                    ],
                ]
            ),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            RGB,
            np.array(
                [
                    [0.50000000, 0.50000000, 0.00000000],
                    [0.50000000, 0.50000000, 1.00000000],
                    [0.50000000, 0.00000000, 0.50000000],
                    [0.50000000, 1.00000000, 0.50000000],
                    [0.00000000, 0.50000000, 0.50000000],
                    [1.00000000, 0.50000000, 0.50000000],
                ]
            ),
            decimal=7,
        )


class TestPlotRGBColourspacesGamuts(unittest.TestCase):
    """
    Define :func:`colour.plotting.volume.plot_RGB_colourspaces_gamuts`
    definition unit tests methods.
    """

    def test_plot_RGB_colourspaces_gamuts(self):
        """
        Test :func:`colour.plotting.volume.plot_RGB_colourspaces_gamuts`
        definition.
        """

        figure, axes = plot_RGB_colourspaces_gamuts(
            ["ITU-R BT.709", "ACEScg", "S-Gamut"],
            show_spectral_locus=True,
            face_colours=[0.18, 0.18, 0.18],
            chromatically_adapt=True,
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotRGBScatter(unittest.TestCase):
    """
    Define :func:`colour.plotting.volume.plot_RGB_scatter` definition unit
    tests methods.
    """

    def test_plot_RGB_scatter(self):
        """Test :func:`colour.plotting.volume.plot_RGB_scatter` definition."""

        figure, axes = plot_RGB_scatter(
            np.random.random((128, 128, 3)), "ITU-R BT.709"
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == "__main__":
    unittest.main()
