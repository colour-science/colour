"""Defines the unit tests for the :mod:`colour.plotting.phenomena` module."""

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.plotting import (
    plot_single_sd_rayleigh_scattering,
    plot_the_blue_sky,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotSingleSdRayleighScattering",
    "TestPlotTheBlueSky",
]


class TestPlotSingleSdRayleighScattering(unittest.TestCase):
    """
    Define :func:`colour.plotting.phenomena.\
plot_single_sd_rayleigh_scattering` definition unit tests methods.
    """

    def test_plot_single_sd_rayleigh_scattering(self):
        """
        Test :func:`colour.plotting.phenomena.\
plot_single_sd_rayleigh_scattering` definition.
        """

        figure, axes = plot_single_sd_rayleigh_scattering()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotTheBlueSky(unittest.TestCase):
    """
    Define :func:`colour.plotting.phenomena.plot_the_blue_sky` definition unit
    tests methods.
    """

    def test_plot_the_blue_sky(self):
        """Test :func:`colour.plotting.phenomena.plot_the_blue_sky` definition."""

        figure, axes = plot_the_blue_sky()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == "__main__":
    unittest.main()
