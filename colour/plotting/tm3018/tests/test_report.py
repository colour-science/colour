"""Define the unit tests for the :mod:`colour.plotting.tm3018.report` module."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from colour.colorimetry import SDS_ILLUMINANTS
from colour.plotting.tm3018.report import (
    plot_single_sd_colour_rendition_report,
    plot_single_sd_colour_rendition_report_full,
    plot_single_sd_colour_rendition_report_intermediate,
    plot_single_sd_colour_rendition_report_simple,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotSingleSdColourRenditionReportFull",
    "TestPlotSingleSdColourRenditionReportIntermediate",
    "TestPlotSingleSdColourRenditionReportSimple",
    "TestPlotSingleSdColourRenditionReport",
]


class TestPlotSingleSdColourRenditionReportFull:
    """
    Define :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report_full` definition unit tests methods.
    """

    def test_plot_single_sd_colour_rendition_report_full(self) -> None:
        """
        Test :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report_full` definition.
        """

        figure, axes = plot_single_sd_colour_rendition_report_full(
            SDS_ILLUMINANTS["FL2"]
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotSingleSdColourRenditionReportIntermediate:
    """
    Define :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report_intermediate` definition unit tests
    methods.
    """

    def test_plot_single_sd_colour_rendition_report_intermediate(self) -> None:
        """
        Test :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report_intermediate` definition.
        """

        figure, axes = plot_single_sd_colour_rendition_report_intermediate(
            SDS_ILLUMINANTS["FL2"]
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotSingleSdColourRenditionReportSimple:
    """
    Define :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report_simple` definition unit tests methods.
    """

    def test_plot_color_vector_graphic(self) -> None:
        """
        Test :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report_simple` definition.
        """

        figure, axes = plot_single_sd_colour_rendition_report_simple(
            SDS_ILLUMINANTS["FL2"]
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotSingleSdColourRenditionReport:
    """
    Define :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report` definition unit tests methods.
    """

    def test_plot_single_sd_colour_rendition_report(self) -> None:
        """
        Test :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report` definition.
        """

        # Test default method (full)
        figure, axes = plot_single_sd_colour_rendition_report(SDS_ILLUMINANTS["FL2"])

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        # Test intermediate method
        figure, axes = plot_single_sd_colour_rendition_report(
            SDS_ILLUMINANTS["FL2"], method="intermediate"
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)

        # Test simple method
        figure, axes = plot_single_sd_colour_rendition_report(
            SDS_ILLUMINANTS["FL2"], method="simple"
        )

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)
