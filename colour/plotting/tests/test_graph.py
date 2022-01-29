# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.plotting.graph` module.
"""
import platform
import tempfile
import unittest

from colour.plotting import plot_automatic_colour_conversion_graph
from colour.utilities import is_networkx_installed

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013-2021 - Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotAutomaticColourConversionGraph",
]


class TestPlotAutomaticColourConversionGraph(unittest.TestCase):
    """
    Defines :func:`colour.plotting.graph.\
plot_automatic_colour_conversion_graph` definition unit tests methods.
    """

    def test_plot_automatic_colour_conversion_graph(self):
        """
        Tests :func:`colour.plotting.graph.\
plot_automatic_colour_conversion_graph` definition.
        """

        if not is_networkx_installed() or platform.system() in (
            "Windows",
            "Microsoft",
        ):  # pragma: no cover
            return

        plot_automatic_colour_conversion_graph(  # pragma: no cover
            "{0}.png".format(tempfile.mkstemp()[-1])
        )


if __name__ == "__main__":
    unittest.main()
