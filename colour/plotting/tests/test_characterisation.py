"""Defines the unit tests for the :mod:`colour.plotting.characterisation` module."""

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.plotting import (
    plot_single_colour_checker,
    plot_multi_colour_checkers,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotSingleColourChecker",
    "TestPlotMultiColourCheckers",
]


class TestPlotSingleColourChecker(unittest.TestCase):
    """
    Define :func:`colour.plotting.characterisation.plot_single_colour_checker`
    definition unit tests methods.
    """

    def test_plot_single_colour_checker(self):
        """
        Test :func:`colour.plotting.characterisation.\
plot_single_colour_checker` definition.
        """

        figure, axes = plot_single_colour_checker()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiColourCheckers(unittest.TestCase):
    """
    Define :func:`colour.plotting.characterisation.plot_multi_colour_checkers`
    definition unit tests methods.
    """

    def test_plot_multi_colour_checkers(self):
        """
        Test :func:`colour.plotting.characterisation.\
plot_multi_colour_checkers` definition.
        """

        figure, axes = plot_multi_colour_checkers(
            ["ColorChecker 1976", "ColorChecker 2005"]
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == "__main__":
    unittest.main()
