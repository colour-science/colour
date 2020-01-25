# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.plotting.common` module.
"""

from __future__ import division, unicode_literals

import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import tempfile
import unittest
from functools import partial
from matplotlib.pyplot import Axes, Figure

import colour
from colour.colorimetry import ILLUMINANTS_SDS
from colour.io import read_image
from colour.models import RGB_COLOURSPACES, XYZ_to_sRGB, gamma_function
from colour.plotting import ColourSwatch
from colour.plotting import (
    colour_style, override_style, XYZ_to_plotting_colourspace, colour_cycle,
    artist, camera, render, label_rectangles, uniform_axes3d,
    filter_passthrough, filter_RGB_colourspaces, filter_cmfs,
    filter_illuminants, filter_colour_checkers, plot_single_colour_swatch,
    plot_multi_colour_swatches, plot_single_function, plot_multi_functions,
    plot_image)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestColourStyle', 'TestOverrideStyle', 'TestXyzToPlottingColourspace',
    'TestColourCycle', 'TestArtist', 'TestCamera', 'TestRender',
    'TestLabelRectangles', 'TestUniformAxes3d', 'TestFilterPassthrough',
    'TestFilterRgbColourspaces', 'TestFilterCmfs', 'TestFilterIlluminants',
    'TestFilterColourCheckers', 'TestPlotSingleColourSwatch',
    'TestPlotMultiColourSwatches', 'TestPlotSingleFunction',
    'TestPlotMultiFunctions', 'TestPlotImage'
]


class TestColourStyle(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.colour_style` definition unit tests
    methods.
    """

    def test_colour_style(self):
        """
        Tests :func:`colour.plotting.common.colour_style` definition.
        """

        self.assertIsInstance(colour_style(use_style=False), dict)


class TestOverrideStyle(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.override_style` definition unit tests
    methods.
    """

    def test_override_style(self):
        """
        Tests :func:`colour.plotting.common.override_style` definition.
        """

        text_color = plt.rcParams['text.color']
        try:

            @override_style(**{'text.color': 'red'})
            def test_text_color_override():
                """
                Tests :func:`colour.plotting.common.override_style` definition.
                """

                assert plt.rcParams['text.color'] == 'red'

            test_text_color_override()
        finally:
            plt.rcParams['text.color'] = text_color


class TestXyzToPlottingColourspace(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.XYZ_to_plotting_colourspace`
    definition unit tests methods.
    """

    def test_XYZ_to_plotting_colourspace(self):
        """
        Tests :func:`colour.plotting.common.XYZ_to_plotting_colourspace`
        definition.
        """

        XYZ = np.random.random(3)
        np.testing.assert_almost_equal(
            XYZ_to_sRGB(XYZ), XYZ_to_plotting_colourspace(XYZ), decimal=7)


class TestColourCycle(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.colour_cycle` definition unit tests
    methods.
    """

    def test_colour_cycle(self):
        """
        Tests :func:`colour.plotting.common.colour_cycle` definition.
        """

        cycler = colour_cycle()

        np.testing.assert_almost_equal(
            next(cycler),
            np.array([0.95686275, 0.26274510, 0.21176471, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            next(cycler),
            np.array([0.61582468, 0.15423299, 0.68456747, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            next(cycler),
            np.array([0.25564014, 0.31377163, 0.70934256, 1.00000000]),
            decimal=7)

        cycler = colour_cycle(colour_cycle_map='viridis')

        np.testing.assert_almost_equal(
            next(cycler),
            np.array([0.26700400, 0.00487400, 0.32941500, 1.00000000]),
            decimal=7)


class TestArtist(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.artist` definition unit tests
    methods.
    """

    def test_artist(self):
        """
        Tests :func:`colour.plotting.common.artist` definition.
        """

        figure_1, axes_1 = artist()

        self.assertIsInstance(figure_1, Figure)
        self.assertIsInstance(axes_1, Axes)

        _figure_2, axes_2 = artist(axes=axes_1, uniform=True)
        self.assertIs(axes_1, axes_2)

        figure_3, _axes_3 = artist(uniform=True)
        self.assertEqual(figure_3.get_figwidth(), figure_3.get_figheight())


class TestCamera(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.camera` definition unit tests
    methods.
    """

    def test_camera(self):
        """
        Tests :func:`colour.plotting.common.camera` definition.
        """

        figure, _axes = artist()
        axes = figure.add_subplot(111, projection='3d')

        _figure, axes = camera(axes=axes, elevation=45, azimuth=90)

        self.assertEqual(axes.elev, 45)
        self.assertEqual(axes.azim, 90)


class TestRender(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.render` definition unit tests
    methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._temporary_directory = tempfile.mkdtemp()

    def tearDown(self):
        """
        After tests actions.
        """

        shutil.rmtree(self._temporary_directory)

    def test_render(self):
        """
        Tests :func:`colour.plotting.common.render` definition.
        """

        figure, axes = artist()

        render(
            figure=figure,
            axes=axes,
            standalone=False,
            aspect='equal',
            axes_visible=True,
            bounding_box=[0, 1, 0, 1],
            tight_layout=False,
            legend=True,
            legend_columns=2,
            transparent_background=False,
            title='Render Unit Test',
            wrap_title=True,
            x_label='x Label',
            y_label='y Label',
            x_ticker=False,
            y_ticker=False,
        )

        render(standalone=True)

        render(
            filename=os.path.join(self._temporary_directory, 'render.png'),
            axes_visible=False)


class TestLabelRectangles(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.label_rectangles` definition unit
    tests methods.
    """

    def test_label_rectangles(self):
        """
        Tests :func:`colour.plotting.common.label_rectangles` definition.
        """

        figure, axes = artist()

        samples = np.linspace(0, 1, 10)

        _figure, axes = label_rectangles(
            samples, axes.bar(samples, 1), figure=figure, axes=axes)

        self.assertEqual(len(axes.texts), len(samples))


class TestUniformAxes3d(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.uniform_axes3d` definition unit tests
    methods.
    """

    def test_uniform_axes3d(self):
        """
        Tests :func:`colour.plotting.common.uniform_axes3d` definition.
        """

        figure, _axes = artist()
        axes = figure.add_subplot(111, projection='3d')

        uniform_axes3d(axes=axes)

        self.assertEqual(axes.get_xlim(), axes.get_ylim())
        self.assertEqual(axes.get_xlim(), axes.get_zlim())


class TestFilterPassthrough(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.filter_passthrough` definition unit
    tests methods.
    """

    def test_filter_passthrough(self):
        """
        Tests :func:`colour.plotting.common.filter_passthrough` definition.
        """

        self.assertListEqual(
            sorted([
                colourspace.name for colourspace in filter_passthrough(
                    RGB_COLOURSPACES, ['^ACES.*']).values()
            ]), ['ACES2065-1', 'ACEScc', 'ACEScct', 'ACEScg', 'ACESproxy'])

        self.assertListEqual(
            sorted(filter_passthrough(RGB_COLOURSPACES, ['^ACEScc$']).keys()),
            ['ACEScc'])

        self.assertListEqual(
            sorted(filter_passthrough(RGB_COLOURSPACES, ['^acescc$']).keys()),
            ['ACEScc'])

        self.assertDictEqual(
            filter_passthrough(
                ILLUMINANTS_SDS, [ILLUMINANTS_SDS['D65'], {
                    'Is': 'Excluded'
                }],
                allow_non_siblings=False), {'D65': ILLUMINANTS_SDS['D65']})

        self.assertDictEqual(
            filter_passthrough(
                ILLUMINANTS_SDS, [ILLUMINANTS_SDS['D65'], {
                    'Is': 'Included'
                }],
                allow_non_siblings=True), {
                    'D65': ILLUMINANTS_SDS['D65'],
                    'Is': 'Included'
                })

        self.assertListEqual(
            sorted([
                element for element in filter_passthrough({
                    'John': 'Doe',
                    'Luke': 'Skywalker'
                }, ['John']).values()
            ]), ['Doe', 'John'])


class TestFilterRgbColourspaces(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.filter_RGB_colourspaces` definition
    unit tests methods.
    """

    def test_filter_RGB_colourspaces(self):
        """
        Tests :func:`colour.plotting.common.filter_RGB_colourspaces`
        definition.
        """

        self.assertListEqual(
            sorted([
                colourspace.name for colourspace in filter_RGB_colourspaces(
                    ['^ACES.*']).values()
            ]), ['ACES2065-1', 'ACEScc', 'ACEScct', 'ACEScg', 'ACESproxy'])


class TestFilterCmfs(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.filter_cmfs` definition unit tests
    methods.
    """

    def test_filter_cmfs(self):
        """
        Tests :func:`colour.plotting.common.filter_cmfs` definition.
        """

        self.assertListEqual(
            sorted([
                cmfs.name for cmfs in filter_cmfs(['.*2 Degree.*']).values()
            ]), [
                'CIE 1931 2 Degree Standard Observer',
                'CIE 2012 2 Degree Standard Observer',
                'Stiles & Burch 1955 2 Degree RGB CMFs',
                'Stockman & Sharpe 2 Degree Cone Fundamentals',
                'Wright & Guild 1931 2 Degree RGB CMFs'
            ])


class TestFilterIlluminants(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.filter_illuminants` definition unit
    tests methods.
    """

    def test_filter_illuminants(self):
        """
        Tests :func:`colour.plotting.common.filter_illuminants` definition.
        """

        self.assertListEqual(
            sorted(filter_illuminants(['^D.*']).keys()),
            ['D50', 'D55', 'D60', 'D65', 'D75', 'Daylight FL'])


class TestFilterColourCheckers(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.filter_colour_checkers` definition
    unit tests methods.
    """

    def test_filter_colour_checkers(self):
        """
        Tests :func:`colour.plotting.common.filter_colour_checkers` definition.
        """

        self.assertListEqual(
            sorted([
                colour_checker.name for colour_checker in
                filter_colour_checkers(['.*24.*']).values()
            ]), [
                'ColorChecker24 - After November 2014',
                'ColorChecker24 - Before November 2014'
            ])


class TestPlotSingleColourSwatch(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.plot_single_colour_swatch` definition
    unit tests methods.
    """

    def test_plot_single_colour_swatch(self):
        """
        Tests :func:`colour.plotting.common.plot_single_colour_swatch`
        definition.
        """

        figure, axes = plot_single_colour_swatch(
            ColourSwatch(RGB=(0.45620519, 0.03081071, 0.04091952)))

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiColourSwatches(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.plot_multi_colour_swatches`
    definition unit tests methods.
    """

    def test_plot_multi_colour_swatches(self):
        """
        Tests :func:`colour.plotting.common.plot_multi_colour_swatches`
        definition.
        """

        figure, axes = plot_multi_colour_swatches([
            ColourSwatch(RGB=(0.45293517, 0.31732158, 0.26414773)),
            ColourSwatch(RGB=(0.77875824, 0.57726450, 0.50453169))
        ])

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSingleFunction(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.plot_single_function` definition unit
    tests methods.
    """

    def test_plot_single_function(self):
        """
        Tests :func:`colour.plotting.common.plot_single_function` definition.
        """

        figure, axes = plot_single_function(
            partial(gamma_function, exponent=1 / 2.2))

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiFunctions(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.plot_multi_functions` definition unit
    tests methods.
    """

    def test_plot_multi_functions(self):
        """
        Tests :func:`colour.plotting.common.plot_multi_functions` definition.
        """

        functions = functions = {
            'Gamma 2.2': lambda x: x ** (1 / 2.2),
            'Gamma 2.4': lambda x: x ** (1 / 2.4),
            'Gamma 2.6': lambda x: x ** (1 / 2.6),
        }
        figure, axes = plot_multi_functions(functions)

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_multi_functions(functions, log_x=10, log_y=10)

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_multi_functions(functions, log_x=10)

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_multi_functions(functions, log_y=10)

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotImage(unittest.TestCase):
    """
    Defines :func:`colour.plotting.common.plot_image` definition unit tests
    methods.
    """

    def test_plot_image(self):
        """
        Tests :func:`colour.plotting.common.plot_image` definition.
        """

        path = os.path.join(colour.__path__[0], '..', 'docs', '_static',
                            'Logo_Medium_001.png')

        # Distribution does not ship the documentation thus we are skipping
        # this unit test if the image does not exist.
        if not os.path.exists(path):  # noqa
            return

        figure, axes = plot_image(read_image(path))

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == '__main__':
    unittest.main()
