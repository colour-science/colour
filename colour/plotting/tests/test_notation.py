# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.plotting.notation` module.
"""

from __future__ import division, unicode_literals

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.plotting import (plot_single_munsell_value_function,
                             plot_multi_munsell_value_functions)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestPlotSingleMunsellValueFunction', 'TestPlotMultiMunsellValueFunctions'
]


class TestPlotSingleMunsellValueFunction(unittest.TestCase):
    """
    Defines :func:`colour.plotting.notation.plot_single_munsell_value_function`
    definition unit tests methods.
    """

    def test_plot_single_munsell_value_function(self):
        """
        Tests :func:`colour.plotting.notation.\
plot_single_munsell_value_function` definition.
        """

        figure, axes = plot_single_munsell_value_function()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiMunsellValueFunctions(unittest.TestCase):
    """
    Defines :func:`colour.plotting.notation.plot_multi_munsell_value_functions`
    definition unit tests methods.
    """

    def test_plot_multi_munsell_value_functions(self):
        """
        Tests :func:`colour.plotting.notation.\
plot_multi_munsell_value_functions` definition.
        """

        figure, axes = plot_multi_munsell_value_functions()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == '__main__':
    unittest.main()
