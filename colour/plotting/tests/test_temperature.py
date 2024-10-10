"""Define the unit tests for the :mod:`colour.plotting.temperature` module."""

import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from colour.plotting import (
    plot_planckian_locus_in_chromaticity_diagram_CIE1931,
    plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS,
)
from colour.plotting.temperature import (
    lines_daylight_locus,
    lines_planckian_locus,
    plot_daylight_locus,
    plot_planckian_locus,
    plot_planckian_locus_in_chromaticity_diagram,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLinesPlanckianLocus",
    "TestPlotDaylightLocus",
    "TestLinesPlanckianLocus",
    "TestPlotPlanckianLocus",
    "TestPlotPlanckianLocusInChromaticityDiagram",
    "TestPlotPlanckianLocusInChromaticityDiagramCIE1931",
    "TestPlotPlanckianLocusInChromaticityDiagramCIE1960UCS",
]


class TestLinesDaylightLocus:
    """
    Define :func:`colour.plotting.diagrams.lines_daylight_locus` definition
    unit tests methods.
    """

    def test_lines_daylight_locus(self):
        """
        Test :func:`colour.plotting.diagrams.lines_daylight_locus` definition.
        """

        assert len(lines_daylight_locus()) == 1


class TestPlotDaylightLocus:
    """
    Define :func:`colour.plotting.temperature.plot_daylight_locus` definition
    unit tests methods.
    """

    def test_plot_daylight_locus(self):
        """
        Test :func:`colour.plotting.temperature.plot_daylight_locus`
        definition.
        """

        figure, axes = plot_daylight_locus()

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        pytest.raises(ValueError, lambda: plot_daylight_locus(method="Undefined"))

        figure, axes = plot_daylight_locus(method="CIE 1976 UCS")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_daylight_locus(planckian_locus_colours="RGB")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestLinesPlanckianLocus:
    """
    Define :func:`colour.plotting.diagrams.lines_planckian_locus` definition
    unit tests methods.
    """

    def test_lines_planckian_locus(self):
        """
        Test :func:`colour.plotting.diagrams.lines_planckian_locus` definition.
        """

        assert len(lines_planckian_locus()) == 2


class TestPlotPlanckianLocus:
    """
    Define :func:`colour.plotting.temperature.plot_planckian_locus` definition
    unit tests methods.
    """

    def test_plot_planckian_locus(self):
        """
        Test :func:`colour.plotting.temperature.plot_planckian_locus`
        definition.
        """

        figure, axes = plot_planckian_locus()

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        pytest.raises(ValueError, lambda: plot_planckian_locus(method="Undefined"))

        figure, axes = plot_planckian_locus(method="CIE 1976 UCS")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_planckian_locus(planckian_locus_colours="RGB")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_planckian_locus(planckian_locus_labels=[5500, 6500])

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotPlanckianLocusInChromaticityDiagram:
    """
    Define :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram` definition unit tests methods.
    """

    def test_plot_planckian_locus_in_chromaticity_diagram(self):
        """
        Test :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram` definition.
        """

        figure, axes = plot_planckian_locus_in_chromaticity_diagram(
            ["A", "B", "C"],
            annotate_kwargs={"arrowprops": {"width": 10}},
            plot_kwargs={
                "markersize": 15,
            },
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_planckian_locus_in_chromaticity_diagram(
            ["A", "B", "C"],
            annotate_kwargs=[{"arrowprops": {"width": 10}}] * 3,
            plot_kwargs=[
                {
                    "markersize": 15,
                }
            ]
            * 3,
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        pytest.raises(
            ValueError,
            lambda: plot_planckian_locus_in_chromaticity_diagram(
                ["A", "B", "C"],
                chromaticity_diagram_callable=lambda **x: x,
                planckian_locus_callable=lambda **x: x,
                method="Undefined",
            ),
        )


class TestPlotPlanckianLocusInChromaticityDiagramCIE1931:
    """
    Define :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram_CIE1931` definition unit tests
    methods.
    """

    def test_plot_planckian_locus_in_chromaticity_diagram_CIE1931(self):
        """
        Test :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram_CIE1931` definition.
        """

        figure, axes = plot_planckian_locus_in_chromaticity_diagram_CIE1931(
            ["A", "B", "C"]
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotPlanckianLocusInChromaticityDiagramCIE1960UCS:
    """
    Define :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS` definition unit tests
    methods.
    """

    def test_plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(self):
        """
        Test :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS` definition.
        """

        figure, axes = plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(
            ["A", "B", "C"]
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)
