"""Define the unit tests for the :mod:`colour.plotting.diagrams` module."""


import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from colour.colorimetry import (
    MSDS_CMFS,
    SDS_ILLUMINANTS,
    SpectralShape,
    reshape_msds,
)
from colour.plotting import (
    lines_spectral_locus,
    plot_chromaticity_diagram_CIE1931,
    plot_chromaticity_diagram_CIE1960UCS,
    plot_chromaticity_diagram_CIE1976UCS,
    plot_sds_in_chromaticity_diagram_CIE1931,
    plot_sds_in_chromaticity_diagram_CIE1960UCS,
    plot_sds_in_chromaticity_diagram_CIE1976UCS,
)
from colour.plotting.diagrams import (
    plot_chromaticity_diagram,
    plot_chromaticity_diagram_colours,
    plot_sds_in_chromaticity_diagram,
    plot_spectral_locus,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLinesSpectralLocus",
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


class TestLinesSpectralLocus:
    """
    Define :func:`colour.plotting.diagrams.lines_spectral_locus` definition
    unit tests methods.
    """

    def test_lines_spectral_locus(self):
        """
        Test :func:`colour.plotting.diagrams.lines_spectral_locus`
        definition.
        """

        assert len(lines_spectral_locus()) == 2


class TestPlotSpectralLocus:
    """
    Define :func:`colour.plotting.diagrams.plot_spectral_locus` definition
    unit tests methods.
    """

    def test_plot_spectral_locus(self):
        """
        Test :func:`colour.plotting.diagrams.plot_spectral_locus` definition.
        """

        figure, axes = plot_spectral_locus()

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_spectral_locus(spectral_locus_colours="RGB")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_spectral_locus(
            method="CIE 1960 UCS", spectral_locus_colours="RGB"
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_spectral_locus(
            method="CIE 1976 UCS", spectral_locus_colours="RGB"
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_spectral_locus(
            reshape_msds(
                MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
                SpectralShape(400, 700, 10),
            )
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        pytest.raises(ValueError, lambda: plot_spectral_locus(method="Undefined"))


class TestPlotChromaticityDiagramColours:
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

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        pytest.raises(
            ValueError,
            lambda: plot_chromaticity_diagram_colours(method="Undefined"),
        )

        figure, axes = plot_chromaticity_diagram_colours(diagram_colours="RGB")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotChromaticityDiagram:
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

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_chromaticity_diagram(method="CIE 1960 UCS")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_chromaticity_diagram(method="CIE 1976 UCS")

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        pytest.raises(
            ValueError,
            lambda: plot_chromaticity_diagram(
                method="Undefined",
                show_diagram_colours=False,
                show_spectral_locus=False,
            ),
        )


class TestPlotChromaticityDiagramCIE1931:
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

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotChromaticityDiagramCIE1960UCS:
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

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotChromaticityDiagramCIE1976UCS:
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

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotSdsInChromaticityDiagram:
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

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        figure, axes = plot_sds_in_chromaticity_diagram(
            [SDS_ILLUMINANTS["A"], SDS_ILLUMINANTS["D65"]],
            annotate_kwargs=[{"arrowprops": {"width": 10}}] * 2,
            plot_kwargs=[{"normalise_sd_colours": True, "use_sd_colours": True}] * 2,
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        pytest.raises(
            ValueError,
            lambda: plot_sds_in_chromaticity_diagram(
                [SDS_ILLUMINANTS["A"], SDS_ILLUMINANTS["D65"]],
                chromaticity_diagram_callable=lambda **x: x,
                method="Undefined",
            ),
        )


class TestPlotSdsInChromaticityDiagramCIE1931:
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

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotSdsInChromaticityDiagramCIE1960UCS:
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

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotSdsInChromaticityDiagramCIE1976UCS:
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

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)
