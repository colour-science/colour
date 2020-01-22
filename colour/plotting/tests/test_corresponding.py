# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.plotting.corresponding` module.
"""

from __future__ import division, unicode_literals

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.plotting import plot_corresponding_chromaticities_prediction

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestPlotCorrespondingChromaticitiesPrediction']


class TestPlotCorrespondingChromaticitiesPrediction(unittest.TestCase):
    """
    Defines :func:`colour.plotting.corresponding.\
plot_corresponding_chromaticities_prediction` definition unit tests methods.
    """

    def test_plot_corresponding_chromaticities_prediction(self):
        """
        Tests :func:`colour.plotting.corresponding.\
plot_corresponding_chromaticities_prediction` definition.
        """

        figure, axes = plot_corresponding_chromaticities_prediction()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == '__main__':
    unittest.main()
