"""Define the unit tests for the :mod:`colour.plotting.volume` module."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.plotting import plot_RGB_colourspaces_gamuts, plot_RGB_scatter
from colour.plotting.volume import RGB_identity_cube, nadir_grid
from colour.utilities import xp_assert_close

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestNadirGrid",
    "TestRGBIdentityCube",
    "TestPlotRGBColourspacesGamuts",
    "TestPlotRGBScatter",
]


class TestNadirGrid:
    """
    Define :func:`colour.plotting.volume.nadir_grid` definition unit tests
    methods.
    """

    def test_nadir_grid(self) -> None:
        """Test :func:`colour.plotting.volume.nadir_grid` definition."""

        quads, faces_colours, edges_colours = nadir_grid(segments=1)

        xp_assert_close(
            quads,
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
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            faces_colours,
            [
                [0.25000000, 0.25000000, 0.25000000, 0.10000000],
                [0.00000000, 0.00000000, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.00000000, 1.00000000],
                [0.00000000, 0.00000000, 0.00000000, 1.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            edges_colours,
            [
                [0.50000000, 0.50000000, 0.50000000, 0.50000000],
                [0.75000000, 0.75000000, 0.75000000, 0.25000000],
                [0.75000000, 0.75000000, 0.75000000, 0.25000000],
                [0.75000000, 0.75000000, 0.75000000, 0.25000000],
                [0.75000000, 0.75000000, 0.75000000, 0.25000000],
                [0.00000000, 0.00000000, 0.00000000, 1.00000000],
                [0.00000000, 0.00000000, 0.00000000, 1.00000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestRGBIdentityCube:
    """
    Define :func:`colour.plotting.volume.RGB_identity_cube` definition unit
    tests methods.
    """

    def test_RGB_identity_cube(self) -> None:
        """Test :func:`colour.plotting.volume.RGB_identity_cube` definition."""

        vertices, RGB = RGB_identity_cube(1, 1, 1)

        xp_assert_close(
            vertices,
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
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            RGB,
            [
                [0.50000000, 0.50000000, 0.00000000],
                [0.50000000, 0.50000000, 1.00000000],
                [0.50000000, 0.00000000, 0.50000000],
                [0.50000000, 1.00000000, 0.50000000],
                [0.00000000, 0.50000000, 0.50000000],
                [1.00000000, 0.50000000, 0.50000000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestPlotRGBColourspacesGamuts:
    """
    Define :func:`colour.plotting.volume.plot_RGB_colourspaces_gamuts`
    definition unit tests methods.
    """

    def test_plot_RGB_colourspaces_gamuts(self) -> None:
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

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotRGBScatter:
    """
    Define :func:`colour.plotting.volume.plot_RGB_scatter` definition unit
    tests methods.
    """

    def test_plot_RGB_scatter(self) -> None:
        """Test :func:`colour.plotting.volume.plot_RGB_scatter` definition."""

        figure, axes = plot_RGB_scatter(np.random.random((128, 128, 3)), "ITU-R BT.709")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)
