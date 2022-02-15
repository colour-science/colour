"""Defines the unit tests for the :mod:`colour.plotting.quality` module."""

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.colorimetry import (
    SDS_ILLUMINANTS,
    SDS_LIGHT_SOURCES,
    SpectralShape,
    reshape_sd,
)
from colour.plotting import (
    plot_single_sd_colour_rendering_index_bars,
    plot_multi_sds_colour_rendering_indexes_bars,
    plot_single_sd_colour_quality_scale_bars,
    plot_multi_sds_colour_quality_scales_bars,
)
from colour.plotting.quality import plot_colour_quality_bars
from colour.quality import colour_quality_scale

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotColourQualityBars",
    "TestPlotSingleSdColourRenderingIndexBars",
    "TestPlotMultiSdsColourRenderingIndexesBars",
    "TestPlotSingleSdColourQualityScaleBars",
    "TestPlotMultiSdsColourQualityScalesBars",
]


class TestPlotColourQualityBars(unittest.TestCase):
    """
    Define :func:`colour.plotting.quality.plot_colour_quality_bars` definition
    unit tests methods.
    """

    def test_plot_colour_quality_bars(self):
        """
        Test :func:`colour.plotting.quality.plot_colour_quality_bars`
        definition.
        """

        illuminant = SDS_ILLUMINANTS["FL2"]
        light_source = SDS_LIGHT_SOURCES["Kinoton 75P"]
        light_source = reshape_sd(light_source, SpectralShape(360, 830, 1))
        cqs_i = colour_quality_scale(illuminant, additional_data=True)
        cqs_l = colour_quality_scale(light_source, additional_data=True)

        figure, axes = plot_colour_quality_bars([cqs_i, cqs_l])

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSingleSdColourRenderingIndexBars(unittest.TestCase):
    """
    Define :func:`colour.plotting.quality.\
plot_single_sd_colour_rendering_index_bars` definition unit tests methods.
    """

    def test_plot_single_sd_colour_rendering_index_bars(self):
        """
        Test :func:`colour.plotting.quality.\
plot_single_sd_colour_rendering_index_bars` definition.
        """

        figure, axes = plot_single_sd_colour_rendering_index_bars(
            SDS_ILLUMINANTS["FL2"]
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiSdsColourRenderingIndexesBars(unittest.TestCase):
    """
    Define :func:`colour.plotting.quality.\
plot_multi_sds_colour_rendering_indexes_bars` definition unit tests methods.
    """

    def test_plot_multi_sds_colour_rendering_indexes_bars(self):
        """
        Test :func:`colour.plotting.quality.\
plot_multi_sds_colour_rendering_indexes_bars` definition.
        """

        figure, axes = plot_multi_sds_colour_rendering_indexes_bars(
            [SDS_ILLUMINANTS["FL2"], SDS_LIGHT_SOURCES["Kinoton 75P"]]
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSingleSdColourQualityScaleBars(unittest.TestCase):
    """
    Define :func:`colour.plotting.quality.\
plot_single_sd_colour_quality_scale_bars` definition unit tests methods.
    """

    def test_plot_single_sd_colour_quality_scale_bars(self):
        """
        Test :func:`colour.plotting.quality.\
plot_single_sd_colour_quality_scale_bars` definition.
        """

        figure, axes = plot_single_sd_colour_quality_scale_bars(
            SDS_ILLUMINANTS["FL2"]
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiSdsColourQualityScalesBars(unittest.TestCase):
    """
    Define :func:`colour.plotting.quality.\
plot_multi_sds_colour_quality_scales_bars` definition unit tests methods.
    """

    def test_plot_multi_sds_colour_quality_scales_bars(self):
        """
        Test :func:`colour.plotting.quality.\
plot_multi_sds_colour_quality_scales_bars` definition.
        """

        figure, axes = plot_multi_sds_colour_quality_scales_bars(
            [SDS_ILLUMINANTS["FL2"], SDS_LIGHT_SOURCES["Kinoton 75P"]]
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == "__main__":
    unittest.main()
