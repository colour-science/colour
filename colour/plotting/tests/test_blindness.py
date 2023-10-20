# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.plotting.blindness` module."""

import unittest

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from colour.plotting import plot_cvd_simulation_Machado2009

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotCvdSimulationMachado2009",
]


class TestPlotCvdSimulationMachado2009(unittest.TestCase):
    """
    Define :func:`colour.plotting.blindness.plot_cvd_simulation_Machado2009`
    definition unit tests methods.
    """

    def test_plot_cvd_simulation_Machado2009(self):
        """
        Test :func:`colour.plotting.blindness.plot_cvd_simulation_Machado2009`
        definition.
        """

        figure, axes = plot_cvd_simulation_Machado2009(
            np.random.rand(32, 32, 3)
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == "__main__":
    unittest.main()
