# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.plotting.notation` module."""

import unittest

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from colour.plotting import (
    plot_multi_munsell_value_functions,
    plot_single_munsell_value_function,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotSingleMunsellValueFunction",
    "TestPlotMultiMunsellValueFunctions",
]


class TestPlotSingleMunsellValueFunction(unittest.TestCase):
    """
    Define :func:`colour.plotting.notation.plot_single_munsell_value_function`
    definition unit tests methods.
    """

    def test_plot_single_munsell_value_function(self):
        """
        Test :func:`colour.plotting.notation.\
plot_single_munsell_value_function` definition.
        """

        figure, axes = plot_single_munsell_value_function("ASTM D1535")

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiMunsellValueFunctions(unittest.TestCase):
    """
    Define :func:`colour.plotting.notation.plot_multi_munsell_value_functions`
    definition unit tests methods.
    """

    def test_plot_multi_munsell_value_functions(self):
        """
        Test :func:`colour.plotting.notation.\
plot_multi_munsell_value_functions` definition.
        """

        figure, axes = plot_multi_munsell_value_functions(
            ["ASTM D1535", "McCamy 1987"]
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == "__main__":
    unittest.main()
