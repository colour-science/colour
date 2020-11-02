# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.plotting.tm3018.report` module.
"""

from __future__ import division, unicode_literals

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.colorimetry import SDS_ILLUMINANTS
from colour.plotting.tm3018.report import (
    plot_single_sd_colour_rendition_report_full,
    plot_single_sd_colour_rendition_report_intermediate,
    plot_single_sd_colour_rendition_report_simple,
    plot_single_sd_colour_rendition_report)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestPlotSingleSdColourRenditionReportFull',
    'TestPlotSingleSdColourRenditionReportIntermediate',
    'TestPlotSingleSdColourRenditionReportSimple',
    'TestPlotSingleSdColourRenditionReport'
]


class TestPlotSingleSdColourRenditionReportFull(unittest.TestCase):
    """
    Defines :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report_full` definition unit tests methods.
    """

    def test_plot_single_sd_colour_rendition_report_full(self):
        """
        Tests :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report_full` definition.
        """

        figure, axes = plot_single_sd_colour_rendition_report_full(
            SDS_ILLUMINANTS['FL2'])

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSingleSdColourRenditionReportIntermediate(unittest.TestCase):
    """
    Defines :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report_intermediate` definition unit tests
    methods.
    """

    def test_plot_single_sd_colour_rendition_report_intermediate(self):
        """
        Tests :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report_intermediate` definition.
        """

        figure, axes = plot_single_sd_colour_rendition_report_intermediate(
            SDS_ILLUMINANTS['FL2'])

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSingleSdColourRenditionReportSimple(unittest.TestCase):
    """
    Defines :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report_simple` definition unit tests methods.
    """

    def test_plot_color_vector_graphic(self):
        """
        Tests :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report_simple` definition.
        """

        figure, axes = plot_single_sd_colour_rendition_report_simple(
            SDS_ILLUMINANTS['FL2'])

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSingleSdColourRenditionReport(unittest.TestCase):
    """
    Defines :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report` definition unit tests methods.
    """

    def test_plot_single_sd_colour_rendition_report(self):
        """
        Tests :func:`colour.plotting.tm3018.report.\
plot_single_sd_colour_rendition_report` definition.
        """

        figure, axes = plot_single_sd_colour_rendition_report(
            SDS_ILLUMINANTS['FL2'])

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == '__main__':
    unittest.main()
