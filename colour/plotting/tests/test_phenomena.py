# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.plotting.phenomena` module.
"""

from __future__ import division, unicode_literals

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.plotting import (plot_single_sd_rayleigh_scattering,
                             plot_the_blue_sky)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestPlotSingleSdRayleighScattering', 'TestPlotTheBlueSky']


class TestPlotSingleSdRayleighScattering(unittest.TestCase):
    """
    Defines :func:`colour.plotting.phenomena.\
plot_single_sd_rayleigh_scattering` definition unit tests methods.
    """

    def test_plot_single_sd_rayleigh_scattering(self):
        """
        Tests :func:`colour.plotting.phenomena.\
plot_single_sd_rayleigh_scattering` definition.
        """

        figure, axes = plot_single_sd_rayleigh_scattering()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotTheBlueSky(unittest.TestCase):
    """
    Defines :func:`colour.plotting.phenomena.plot_the_blue_sky` definition unit
    tests methods.
    """

    def test_plot_the_blue_sky(self):
        """
        Tests :func:`colour.plotting.phenomena.plot_the_blue_sky` definition.
        """

        figure, axes = plot_the_blue_sky()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == '__main__':
    unittest.main()
