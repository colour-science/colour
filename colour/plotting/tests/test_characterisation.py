# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.plotting.characterisation` module."""


from matplotlib.axes import Axes
from matplotlib.figure import Figure

from colour.plotting import (
    plot_multi_colour_checkers,
    plot_single_colour_checker,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotSingleColourChecker",
    "TestPlotMultiColourCheckers",
]


class TestPlotSingleColourChecker:
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

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)


class TestPlotMultiColourCheckers:
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

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)
