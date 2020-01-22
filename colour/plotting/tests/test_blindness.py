# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.plotting.blindness` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from matplotlib.pyplot import Axes, Figure

from colour.plotting import plot_cvd_simulation_Machado2009

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestPlotCvdSimulationMachado2009']


class TestPlotCvdSimulationMachado2009(unittest.TestCase):
    """
    Defines :func:`colour.plotting.blindness.plot_cvd_simulation_Machado2009`
    definition unit tests methods.
    """

    def test_plot_cvd_simulation_Machado2009(self):
        """
        Tests :func:`colour.plotting.blindness.plot_cvd_simulation_Machado2009`
        definition.
        """

        figure, axes = plot_cvd_simulation_Machado2009(
            np.random.rand(32, 32, 3))

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == '__main__':
    unittest.main()
