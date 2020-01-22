# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.plotting.characterisation` module.
"""

from __future__ import division, unicode_literals

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.plotting import (plot_single_colour_checker,
                             plot_multi_colour_checkers)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestPlotSingleColourChecker', 'TestPlotMultiColourCheckers']


class TestPlotSingleColourChecker(unittest.TestCase):
    """
    Defines :func:`colour.plotting.characterisation.plot_single_colour_checker`
    definition unit tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Tests :func:`colour.plotting.characterisation.\
plot_single_colour_checker` definition.
        """

        figure, axes = plot_single_colour_checker()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiColourCheckers(unittest.TestCase):
    """
    Defines :func:`colour.plotting.characterisation.plot_multi_colour_checkers`
    definition unit tests methods.
    """

    def test_plot_multi_colour_checkers(self):
        """
        Tests :func:`colour.plotting.characterisation.\
plot_multi_colour_checkers` definition.
        """

        figure, axes = plot_multi_colour_checkers()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == '__main__':
    unittest.main()
