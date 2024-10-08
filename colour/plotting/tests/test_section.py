"""Define the unit tests for the :mod:`colour.plotting.section` module."""

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from colour.geometry import primitive_cube
from colour.models import RGB_COLOURSPACE_sRGB, RGB_to_XYZ
from colour.plotting import (
    plot_RGB_colourspace_section,
    plot_visible_spectrum_section,
)
from colour.plotting.section import (
    plot_hull_section_colours,
    plot_hull_section_contour,
)
from colour.utilities import is_trimesh_installed

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotHullSectionColours",
    "TestPlotHullSectionContour",
    "TestPlotVisibleSpectrumSection",
    "TestPlotRGBColourspaceSection",
]


class TestPlotHullSectionColours:
    """
    Define :func:`colour.plotting.section.plot_hull_section_colours`
    definition unit tests methods.
    """

    def test_plot_hull_section_colours(self):
        """
        Test :func:`colour.plotting.section.plot_hull_section_colours`
        definition.
        """

        if not is_trimesh_installed():  # pragma: no cover
            return

        import trimesh

        vertices, faces, _outline = primitive_cube(1, 1, 1, 64, 64, 64)
        XYZ_vertices = RGB_to_XYZ(vertices["position"] + 0.5, RGB_COLOURSPACE_sRGB)
        hull = trimesh.Trimesh(XYZ_vertices, faces, process=False)

        figure, axes = plot_hull_section_colours(hull)

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_hull_section_colours(hull, axis="+x")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_hull_section_colours(hull, axis="+y")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotHullSectionContour:
    """
    Define :func:`colour.plotting.section.plot_hull_section_contour`
    definition unit tests methods.
    """

    def test_plot_hull_section_contour(self):
        """
        Test :func:`colour.plotting.section.plot_hull_section_contour`
        definition.
        """

        if not is_trimesh_installed():  # pragma: no cover
            return

        import trimesh

        vertices, faces, _outline = primitive_cube(1, 1, 1, 64, 64, 64)
        XYZ_vertices = RGB_to_XYZ(vertices["position"] + 0.5, RGB_COLOURSPACE_sRGB)
        hull = trimesh.Trimesh(XYZ_vertices, faces, process=False)

        figure, axes = plot_hull_section_contour(hull)

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotVisibleSpectrumSection:
    """
    Define :func:`colour.plotting.section.plot_visible_spectrum_section`
    definition unit tests methods.
    """

    def test_plot_visible_spectrum_section(self):
        """
        Test :func:`colour.plotting.section.plot_visible_spectrum_section`
        definition.
        """

        if not is_trimesh_installed():  # pragma: no cover
            return

        figure, axes = plot_visible_spectrum_section()

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotRGBColourspaceSection:
    """
    Define :func:`colour.plotting.section.plot_RGB_colourspace_section`
    definition unit tests methods.
    """

    def test_plot_RGB_colourspace_section(self):
        """
        Test :func:`colour.plotting.section.plot_RGB_colourspace_section`
        definition.
        """

        if not is_trimesh_installed():  # pragma: no cover
            return

        figure, axes = plot_RGB_colourspace_section("sRGB")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)
