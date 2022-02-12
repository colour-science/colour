"""Defines the unit tests for the :mod:`colour.plotting.diagrams` module."""

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.colorimetry import (
    MSDS_CMFS,
    SDS_ILLUMINANTS,
    SpectralShape,
    reshape_msds,
)
from colour.plotting import (
    plot_chromaticity_diagram_CIE1931,
    plot_chromaticity_diagram_CIE1960UCS,
    plot_chromaticity_diagram_CIE1976UCS,
    plot_sds_in_chromaticity_diagram_CIE1931,
    plot_sds_in_chromaticity_diagram_CIE1960UCS,
    plot_sds_in_chromaticity_diagram_CIE1976UCS,
)
from colour.plotting.diagrams import (
    plot_spectral_locus,
    plot_chromaticity_diagram_colours,
    plot_chromaticity_diagram,
    plot_sds_in_chromaticity_diagram,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotSpectralLocus",
    "TestPlotChromaticityDiagramColours",
    "TestPlotChromaticityDiagram",
    "TestPlotChromaticityDiagramCIE1931",
    "TestPlotChromaticityDiagramCIE1960UCS",
    "TestPlotChromaticityDiagramCIE1976UCS",
    "TestPlotSdsInChromaticityDiagram",
    "TestPlotSdsInChromaticityDiagramCIE1931",
    "TestPlotSdsInChromaticityDiagramCIE1960UCS",
    "TestPlotSdsInChromaticityDiagramCIE1976UCS",
]


class TestPlotSpectralLocus(unittest.TestCase):
    """
    Define :func:`colour.plotting.diagrams.plot_spectral_locus` definition
    unit tests methods.
    """

    def test_plot_spectral_locus(self):
        """Test :func:`colour.plotting.diagrams.plot_spectral_locus` definition."""

        figure, axes = plot_spectral_locus()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_spectral_locus(spectral_locus_colours="RGB")

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_spectral_locus(
            method="CIE 1960 UCS", spectral_locus_colours="RGB"
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_spectral_locus(
            method="CIE 1976 UCS", spectral_locus_colours="RGB"
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        # pylint: disable=E1102
        figure, axes = plot_spectral_locus(
            reshape_msds(
                MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
                SpectralShape(400, 700, 10),
            )
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError, lambda: plot_spectral_locus(method="Undefined")
        )


class TestPlotChromaticityDiagramColours(unittest.TestCase):
    """
    Define :func:`colour.plotting.diagrams.plot_chromaticity_diagram_colours`
    definition unit tests methods.
    """

    def test_plot_chromaticity_diagram_colours(self):
        """
        Test :func:`colour.plotting.diagrams.plot_chromaticity_diagram_colours`
        definition.
        """

        figure, axes = plot_chromaticity_diagram_colours()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError,
            lambda: plot_chromaticity_diagram_colours(method="Undefined"),
        )

        figure, axes = plot_chromaticity_diagram_colours(diagram_colours="RGB")

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotChromaticityDiagram(unittest.TestCase):
    """
    Define :func:`colour.plotting.diagrams.plot_chromaticity_diagram`
    definition unit tests methods.
    """

    def test_plot_chromaticity_diagram(self):
        """
        Test :func:`colour.plotting.diagrams.plot_chromaticity_diagram`
        definition.
        """

        figure, axes = plot_chromaticity_diagram()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_chromaticity_diagram(method="CIE 1960 UCS")

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_chromaticity_diagram(method="CIE 1976 UCS")

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError,
            lambda: plot_chromaticity_diagram(
                method="Undefined",
                show_diagram_colours=False,
                show_spectral_locus=False,
            ),
        )


class TestPlotChromaticityDiagramCIE1931(unittest.TestCase):
    """
    Define :func:`colour.plotting.diagrams.plot_chromaticity_diagram_CIE1931`
    definition unit tests methods.
    """

    def test_plot_chromaticity_diagram_CIE1931(self):
        """
        Test :func:`colour.plotting.diagrams.\
plot_chromaticity_diagram_CIE1931` definition.
        """

        figure, axes = plot_chromaticity_diagram_CIE1931()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotChromaticityDiagramCIE1960UCS(unittest.TestCase):
    """
    Define :func:`colour.plotting.diagrams.\
plot_chromaticity_diagram_CIE1960UCS` definition unit tests methods.
    """

    def test_plot_chromaticity_diagram_CIE1960UCS(self):
        """
        Test :func:`colour.plotting.diagrams.\
plot_chromaticity_diagram_CIE1960UCS` definition.
        """

        figure, axes = plot_chromaticity_diagram_CIE1960UCS()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotChromaticityDiagramCIE1976UCS(unittest.TestCase):
    """
    Define :func:`colour.plotting.diagrams.\
plot_chromaticity_diagram_CIE1976UCS` definition unit tests methods.
    """

    def test_plot_chromaticity_diagram_CIE1976UCS(self):
        """
        Test :func:`colour.plotting.diagrams.\
plot_chromaticity_diagram_CIE1976UCS` definition.
        """

        figure, axes = plot_chromaticity_diagram_CIE1976UCS()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSdsInChromaticityDiagram(unittest.TestCase):
    """
    Define :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram` definition unit tests methods.
    """

    def test_plot_sds_in_chromaticity_diagram(self):
        """
        Test :func:`colour.plotting.diagrams.plot_sds_in_chromaticity_diagram`
        definition.
        """

        figure, axes = plot_sds_in_chromaticity_diagram(
            [SDS_ILLUMINANTS["A"], SDS_ILLUMINANTS["D65"]],
            annotate_kwargs={"arrowprops": {"width": 10}},
            plot_kwargs={"normalise_sd_colours": True, "use_sd_colours": True},
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_sds_in_chromaticity_diagram(
            [SDS_ILLUMINANTS["A"], SDS_ILLUMINANTS["D65"]],
            annotate_kwargs=[{"arrowprops": {"width": 10}}] * 2,
            plot_kwargs=[
                {"normalise_sd_colours": True, "use_sd_colours": True}
            ]
            * 2,
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError,
            lambda: plot_sds_in_chromaticity_diagram(
                [SDS_ILLUMINANTS["A"], SDS_ILLUMINANTS["D65"]],
                chromaticity_diagram_callable=lambda **x: x,
                method="Undefined",
            ),
        )


class TestPlotSdsInChromaticityDiagramCIE1931(unittest.TestCase):
    """
    Define :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram_CIE1931` definition unit tests methods.
    """

    def test_plot_sds_in_chromaticity_diagram_CIE1931(self):
        """
        Test :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram_CIE1931` definition.
        """

        figure, axes = plot_sds_in_chromaticity_diagram_CIE1931(
            [SDS_ILLUMINANTS["A"], SDS_ILLUMINANTS["D65"]]
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSdsInChromaticityDiagramCIE1960UCS(unittest.TestCase):
    """
    Define :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram_CIE1960UCS` definition unit tests methods.
    """

    def test_plot_sds_in_chromaticity_diagram_CIE1960UCS(self):
        """
        Test :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram_CIE1960UCS` definition.
        """

        figure, axes = plot_sds_in_chromaticity_diagram_CIE1960UCS(
            [SDS_ILLUMINANTS["A"], SDS_ILLUMINANTS["D65"]]
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSdsInChromaticityDiagramCIE1976UCS(unittest.TestCase):
    """
    Define :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram_CIE1976UCS` definition unit tests methods.
    """

    def test_plot_sds_in_chromaticity_diagram_CIE1976UCS(self):
        """
        Test :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram_CIE1976UCS` definition.
        """

        figure, axes = plot_sds_in_chromaticity_diagram_CIE1976UCS(
            [SDS_ILLUMINANTS["A"], SDS_ILLUMINANTS["D65"]]
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == "__main__":
    unittest.main()
