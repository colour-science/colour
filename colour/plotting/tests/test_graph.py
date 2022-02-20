"""Defines the unit tests for the :mod:`colour.plotting.graph` module."""
import platform
import tempfile
import unittest

from colour.plotting import plot_automatic_colour_conversion_graph
from colour.utilities import is_networkx_installed

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotAutomaticColourConversionGraph",
]


class TestPlotAutomaticColourConversionGraph(unittest.TestCase):
    """
    Define :func:`colour.plotting.graph.\
plot_automatic_colour_conversion_graph` definition unit tests methods.
    """

    def test_plot_automatic_colour_conversion_graph(self):
        """
        Test :func:`colour.plotting.graph.\
plot_automatic_colour_conversion_graph` definition.
        """

        if not is_networkx_installed() or platform.system() in (
            "Windows",
            "Microsoft",
        ):  # pragma: no cover
            return

        plot_automatic_colour_conversion_graph(  # pragma: no cover
            f"{tempfile.mkstemp()[-1]}.png"
        )


if __name__ == "__main__":
    unittest.main()
