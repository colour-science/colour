"""Defines the unit tests for the :mod:`colour.plotting.temperature` module."""

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.plotting import (
    plot_planckian_locus_in_chromaticity_diagram_CIE1931,
    plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS,
)
from colour.plotting.temperature import (
    plot_planckian_locus,
    plot_planckian_locus_in_chromaticity_diagram,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotPlanckianLocus",
    "TestPlotPlanckianLocusInChromaticityDiagram",
    "TestPlotPlanckianLocusInChromaticityDiagramCIE1931",
    "TestPlotPlanckianLocusInChromaticityDiagramCIE1960UCS",
]


class TestPlotPlanckianLocus(unittest.TestCase):
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

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError, lambda: plot_planckian_locus(method="Undefined")
        )

        figure, axes = plot_planckian_locus(method="CIE 1976 UCS")

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_planckian_locus(planckian_locus_colours="RGB")

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_planckian_locus(
            planckian_locus_labels=[5500, 6500]
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotPlanckianLocusInChromaticityDiagram(unittest.TestCase):
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

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

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

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError,
            lambda: plot_planckian_locus_in_chromaticity_diagram(
                ["A", "B", "C"],
                chromaticity_diagram_callable=lambda **x: x,
                planckian_locus_callable=lambda **x: x,
                method="Undefined",
            ),
        )


class TestPlotPlanckianLocusInChromaticityDiagramCIE1931(unittest.TestCase):
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

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotPlanckianLocusInChromaticityDiagramCIE1960UCS(unittest.TestCase):
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

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == "__main__":
    unittest.main()
