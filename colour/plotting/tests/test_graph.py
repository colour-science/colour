"""Define the unit tests for the :mod:`colour.plotting.graph` module."""

import tempfile

from colour.plotting import plot_automatic_colour_conversion_graph
from colour.utilities import is_graphviz_installed, is_networkx_installed

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotAutomaticColourConversionGraph",
]


class TestPlotAutomaticColourConversionGraph:
    """
    Define :func:`colour.plotting.graph.\
plot_automatic_colour_conversion_graph` definition unit tests methods.
    """

    def test_plot_automatic_colour_conversion_graph(self):
        """
        Test :func:`colour.plotting.graph.\
plot_automatic_colour_conversion_graph` definition.
        """

        if (
            not is_graphviz_installed() or not is_networkx_installed()
        ):  # pragma: no cover
            return

        plot_automatic_colour_conversion_graph(  # pragma: no cover
            f"{tempfile.mkstemp()[-1]}.png"
        )
