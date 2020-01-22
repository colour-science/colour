# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.plotting.colorimetry` module.
"""

from __future__ import division, unicode_literals

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.colorimetry import SpectralDistribution
from colour.plotting import (
    plot_single_sd, plot_multi_sds, plot_single_cmfs, plot_multi_cmfs,
    plot_single_illuminant_sd, plot_multi_illuminant_sds,
    plot_visible_spectrum, plot_single_lightness_function,
    plot_multi_lightness_functions, plot_single_luminance_function,
    plot_multi_luminance_functions, plot_blackbody_spectral_radiance,
    plot_blackbody_colours)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestPlotSingleSd', 'TestPlotMultiSds', 'TestPlotSingleCmfs',
    'TestPlotMultiCmfs', 'TestPlotSingleIlluminantSd',
    'TestPlotMultiIlluminantSds', 'TestPlotVisibleSpectrum',
    'TestPlotSingleLightnessFunction', 'TestPlotMultiLightnessFunctions',
    'TestPlotSingleLuminanceFunction', 'TestPlotMultiLuminanceFunctions',
    'TestPlotBlackbodySpectralRadiance', 'TestPlotBlackbodyColours'
]


class TestPlotSingleSd(unittest.TestCase):
    """
    Defines :func:`colour.plotting.colorimetry.plot_single_sd` definition unit
    tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.colorimetry.plot_single_sd` definition.
        """

        sd = SpectralDistribution(
            {
                500: 0.004900,
                510: 0.009300,
                520: 0.063270,
                530: 0.165500,
                540: 0.290400,
                550: 0.433450,
                560: 0.594500
            },
            name='Custom 1')

        figure, axes = plot_single_sd(
            sd,
            out_of_gamut_clipping=False,
            modulate_colours_with_sd_amplitude=True,
            equalize_sd_amplitude=True)

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiSds(unittest.TestCase):
    """
    Defines :func:`colour.plotting.colorimetry.plot_multi_sds` definition unit
    tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.colorimetry.plot_multi_sds` definition.
        """

        sd_1 = SpectralDistribution(
            {
                500: 0.004900,
                510: 0.009300,
                520: 0.063270,
                530: 0.165500,
                540: 0.290400,
                550: 0.433450,
                560: 0.594500
            },
            name='Custom 1')
        sd_2 = SpectralDistribution(
            {
                500: 0.323000,
                510: 0.503000,
                520: 0.710000,
                530: 0.862000,
                540: 0.954000,
                550: 0.994950,
                560: 0.995000
            },
            name='Custom 2')

        figure, axes = plot_multi_sds(
            [sd_1, sd_2], use_sds_colours=True, normalise_sds_colours=True)

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSingleCmfs(unittest.TestCase):
    """
    Defines :func:`colour.plotting.colorimetry.plot_single_cmfs` definition
    unit tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.colorimetry.plot_single_cmfs` definition.
        """

        figure, axes = plot_single_cmfs()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiCmfs(unittest.TestCase):
    """
    Defines :func:`colour.plotting.colorimetry.plot_multi_cmfs` definition unit
    tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.colorimetry.plot_multi_cmfs` definition.
        """

        figure, axes = plot_multi_cmfs()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSingleIlluminantSd(unittest.TestCase):
    """
    Defines :func:`colour.plotting.colorimetry.plot_single_illuminant_sd`
    definition unit tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.colorimetry.plot_single_illuminant_sd`
        definition.
        """

        figure, axes = plot_single_illuminant_sd()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiIlluminantSds(unittest.TestCase):
    """
    Defines :func:`colour.plotting.colorimetry.plot_multi_illuminant_sds`
    definition unit tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.colorimetry.plot_multi_illuminant_sds`
        definition.
        """

        figure, axes = plot_multi_illuminant_sds()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotVisibleSpectrum(unittest.TestCase):
    """
    Defines :func:`colour.plotting.colorimetry.plot_visible_spectrum`
    definition unit tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.colorimetry.plot_visible_spectrum`
        definition.
        """

        figure, axes = plot_visible_spectrum()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSingleLightnessFunction(unittest.TestCase):
    """
    Defines :func:`colour.plotting.colorimetry.plot_single_lightness_function`
    definition unit tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.colorimetry.\
plot_single_lightness_function` definition.
        """

        figure, axes = plot_single_lightness_function()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiLightnessFunctions(unittest.TestCase):
    """
    Defines :func:`colour.plotting.colorimetry.plot_multi_lightness_functions`
    definition unit tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.colorimetry.\
plot_multi_lightness_functions` definition.
        """

        figure, axes = plot_multi_lightness_functions()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSingleLuminanceFunction(unittest.TestCase):
    """
    Defines :func:`colour.plotting.colorimetry.plot_single_luminance_function`
    definition unit tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.colorimetry.\
plot_single_luminance_function` definition.
        """

        figure, axes = plot_single_luminance_function()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiLuminanceFunctions(unittest.TestCase):
    """
    Defines :func:`colour.plotting.colorimetry.plot_multi_luminance_functions`
    definition unit tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.colorimetry.\
plot_multi_luminance_functions` definition.
        """

        figure, axes = plot_multi_luminance_functions()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotBlackbodySpectralRadiance(unittest.TestCase):
    """
    Defines :func:`colour.plotting.colorimetry.\
plot_blackbody_spectral_radiance` definition unit tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.colorimetry.\
plot_blackbody_spectral_radiance` definition.
        """

        figure, axes = plot_blackbody_spectral_radiance()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotBlackbodyColours(unittest.TestCase):
    """
    Defines :func:`colour.plotting.colorimetry.plot_blackbody_colours`
    definition unit tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.colorimetry.plot_blackbody_colours`
        definition.
        """

        figure, axes = plot_blackbody_colours()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == '__main__':
    unittest.main()
