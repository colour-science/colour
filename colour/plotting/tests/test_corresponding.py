# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.plotting.corresponding` module."""


from matplotlib.axes import Axes
from matplotlib.figure import Figure

from colour.plotting import plot_corresponding_chromaticities_prediction

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotCorrespondingChromaticitiesPrediction",
]


class TestPlotCorrespondingChromaticitiesPrediction:
    """
    Define :func:`colour.plotting.corresponding.\
plot_corresponding_chromaticities_prediction` definition unit tests methods.
    """

    def test_plot_corresponding_chromaticities_prediction(self):
        """
        Test :func:`colour.plotting.corresponding.\
plot_corresponding_chromaticities_prediction` definition.
        """

        figure, axes = plot_corresponding_chromaticities_prediction()

        assert isinstance(figure, Figure)
        assert isinstance(axes, Axes)
