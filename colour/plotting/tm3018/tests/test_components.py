"""Defines the unit tests for the :mod:`colour.plotting.tm3018.components` module."""

from __future__ import annotations

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.colorimetry import SDS_ILLUMINANTS
from colour.hints import cast
from colour.quality import (
    ColourQuality_Specification_ANSIIESTM3018,
    colour_fidelity_index_ANSIIESTM3018,
)
from colour.plotting.tm3018.components import (
    plot_spectra_ANSIIESTM3018,
    plot_colour_vector_graphic,
    plot_16_bin_bars,
    plot_local_chroma_shifts,
    plot_local_hue_shifts,
    plot_local_colour_fidelities,
    plot_colour_fidelity_indexes,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlotSpectraANSIIESTM3018",
    "TestPlotColourVectorGraphic",
    "TestPlot16BinBars",
    "TestPlotLocalChromaShifts",
    "TestPlotLocalHueShifts",
    "TestPlotLocalColourFidelities",
    "TestPlotColourFidelityIndexes",
]

SPECIFICATION_ANSIIESTM3018: ColourQuality_Specification_ANSIIESTM3018 = cast(
    ColourQuality_Specification_ANSIIESTM3018,
    colour_fidelity_index_ANSIIESTM3018(SDS_ILLUMINANTS["FL2"], True),
)


class TestPlotSpectraANSIIESTM3018(unittest.TestCase):
    """
        Define :func:`colour.plotting.tm3018.components.
    plot_spectra_ANSIIESTM3018` definition unit tests methods.
    """

    def test_plot_spectra_ANSIIESTM3018(self):
        """
        Test :func:`colour.plotting.tm3018.components.\
plot_spectra_ANSIIESTM3018` definition.
        """

        figure, axes = plot_spectra_ANSIIESTM3018(SPECIFICATION_ANSIIESTM3018)

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotColourVectorGraphic(unittest.TestCase):
    """
    Define :func:`colour.plotting.tm3018.components.\
plot_colour_vector_graphic` definition unit tests methods.
    """

    def test_plot_colour_vector_graphic(self):
        """
        Test :func:`colour.plotting.tm3018.components.\
plot_colour_vector_graphic` definition.
        """

        figure, axes = plot_colour_vector_graphic(SPECIFICATION_ANSIIESTM3018)

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlot16BinBars(unittest.TestCase):
    """
    Define :func:`colour.plotting.tm3018.components.plot_16_bin_bars`
    definition unit tests methods.
    """

    def test_plot_16_bin_bars(self):
        """
        Test :func:`colour.plotting.tm3018.components.plot_16_bin_bars`
        definition.
        """

        figure, axes = plot_16_bin_bars(range(16), "{0}")

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotLocalChromaShifts(unittest.TestCase):
    """
    Define :func:`colour.plotting.tm3018.components.plot_local_chroma_shifts`
    definition unit tests methods.
    """

    def test_plot_local_chroma_shifts(self):
        """
        Test :func:`colour.plotting.tm3018.components.\
plot_local_chroma_shifts` definition.
        """

        figure, axes = plot_local_chroma_shifts(SPECIFICATION_ANSIIESTM3018)

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotLocalHueShifts(unittest.TestCase):
    """
    Define :func:`colour.plotting.tm3018.components.plot_local_hue_shifts`
    definition unit tests methods.
    """

    def test_plot_local_hue_shifts(self):
        """
        Test :func:`colour.plotting.tm3018.components.\
plot_local_hue_shifts` definition.
        """

        figure, axes = plot_local_hue_shifts(SPECIFICATION_ANSIIESTM3018)

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotLocalColourFidelities(unittest.TestCase):
    """
        Define :func:`colour.plotting.tm3018.components.
    plot_local_colour_fidelities` definition unit tests methods.
    """

    def test_plot_local_colour_fidelities(self):
        """
        Test :func:`colour.plotting.tm3018.components.\
plot_local_colour_fidelities` definition.
        """

        figure, axes = plot_local_colour_fidelities(
            SPECIFICATION_ANSIIESTM3018
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotColourFidelityIndexes(unittest.TestCase):
    """
    Define :func:`colour.plotting.tm3018.components.\
plot_colour_fidelity_indexes` definition unit tests methods.
    """

    def test_plot_colour_fidelity_indexes(self):
        """
        Test :func:`colour.plotting.tm3018.components.\
plot_colour_fidelity_indexes` definition.
        """

        figure, axes = plot_colour_fidelity_indexes(
            SPECIFICATION_ANSIIESTM3018
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == "__main__":
    unittest.main()
