# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.plotting.diagrams` module.
"""

from __future__ import division, unicode_literals

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.colorimetry import (ILLUMINANTS_SDS, SpectralShape,
                                STANDARD_OBSERVERS_CMFS)
from colour.plotting import (plot_chromaticity_diagram_CIE1931,
                             plot_chromaticity_diagram_CIE1960UCS,
                             plot_chromaticity_diagram_CIE1976UCS,
                             plot_sds_in_chromaticity_diagram_CIE1931,
                             plot_sds_in_chromaticity_diagram_CIE1960UCS,
                             plot_sds_in_chromaticity_diagram_CIE1976UCS)
from colour.plotting.diagrams import (
    plot_spectral_locus, plot_chromaticity_diagram_colours,
    plot_chromaticity_diagram, plot_sds_in_chromaticity_diagram)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestPlotSpectralLocus', 'TestPlotChromaticityDiagramColours',
    'TestPlotChromaticityDiagram', 'TestPlotChromaticityDiagramCIE1931',
    'TestPlotChromaticityDiagramCIE1960UCS',
    'TestPlotChromaticityDiagramCIE1976UCS',
    'TestPlotSdsInChromaticityDiagram',
    'TestPlotSdsInChromaticityDiagramCIE1931',
    'TestPlotSdsInChromaticityDiagramCIE1960UCS',
    'TestPlotSdsInChromaticityDiagramCIE1976UCS'
]


class TestPlotSpectralLocus(unittest.TestCase):
    """
    Defines :func:`colour.plotting.diagrams.plot_spectral_locus` definition
    unit tests methods.
    """

    def test_plot_spectral_locus(self):
        """
        Tests :func:`colour.plotting.diagrams.plot_spectral_locus` definition.
        """

        figure, axes = plot_spectral_locus()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_spectral_locus(spectral_locus_colours='RGB')

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_spectral_locus(
            method='CIE 1960 UCS', spectral_locus_colours='RGB')

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_spectral_locus(
            method='CIE 1976 UCS', spectral_locus_colours='RGB')

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_spectral_locus(STANDARD_OBSERVERS_CMFS[
            'CIE 1931 2 Degree Standard Observer'].copy().align(
                SpectralShape(400, 700, 10)))

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError, lambda: plot_spectral_locus(method='Undefined'))


class TestPlotChromaticityDiagramColours(unittest.TestCase):
    """
    Defines :func:`colour.plotting.diagrams.plot_chromaticity_diagram_colours`
    definition unit tests methods.
    """

    def test_plot_chromaticity_diagram_colours(self):
        """
        Tests :func:`colour.plotting.diagrams.\
plot_chromaticity_diagram_colours` definition.
        """

        figure, axes = plot_chromaticity_diagram_colours()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError,
            lambda: plot_chromaticity_diagram_colours(method='Undefined'))


class TestPlotChromaticityDiagram(unittest.TestCase):
    """
    Defines :func:`colour.plotting.diagrams.plot_chromaticity_diagram`
    definition unit tests methods.
    """

    def test_plot_chromaticity_diagram(self):
        """
        Tests :func:`colour.plotting.diagrams.plot_chromaticity_diagram`
        definition.
        """

        figure, axes = plot_chromaticity_diagram()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_chromaticity_diagram(method='CIE 1960 UCS')

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_chromaticity_diagram(method='CIE 1976 UCS')

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError, lambda: plot_chromaticity_diagram(
                method='Undefined',
                show_diagram_colours=False,
                show_spectral_locus=False)
        )


class TestPlotChromaticityDiagramCIE1931(unittest.TestCase):
    """
    Defines :func:`colour.plotting.diagrams.plot_chromaticity_diagram_CIE1931`
    definition unit tests methods.
    """

    def test_plot_chromaticity_diagram_CIE1931(self):
        """
        Tests :func:`colour.plotting.diagrams.\
plot_chromaticity_diagram_CIE1931` definition.
        """

        figure, axes = plot_chromaticity_diagram_CIE1931()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotChromaticityDiagramCIE1960UCS(unittest.TestCase):
    """
    Defines :func:`colour.plotting.diagrams.\
plot_chromaticity_diagram_CIE1960UCS` definition unit tests methods.
    """

    def test_plot_chromaticity_diagram_CIE1960UCS(self):
        """
        Tests :func:`colour.plotting.diagrams.\
plot_chromaticity_diagram_CIE1960UCS` definition.
        """

        figure, axes = plot_chromaticity_diagram_CIE1960UCS()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotChromaticityDiagramCIE1976UCS(unittest.TestCase):
    """
    Defines :func:`colour.plotting.diagrams.\
plot_chromaticity_diagram_CIE1976UCS` definition unit tests methods.
    """

    def test_plot_chromaticity_diagram_CIE1976UCS(self):
        """
        Tests :func:`colour.plotting.diagrams.\
plot_chromaticity_diagram_CIE1976UCS` definition.
        """

        figure, axes = plot_chromaticity_diagram_CIE1976UCS()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSdsInChromaticityDiagram(unittest.TestCase):
    """
    Defines :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram` definition unit tests methods.
    """

    def test_plot_sds_in_chromaticity_diagram(self):
        """
        Tests :func:`colour.plotting.diagrams.plot_sds_in_chromaticity_diagram`
        definition.
        """

        figure, axes = plot_sds_in_chromaticity_diagram(
            [ILLUMINANTS_SDS['A'], ILLUMINANTS_SDS['D65']],
            annotate_parameters={'arrowprops': {
                'width': 10
            }})

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_sds_in_chromaticity_diagram(
            [ILLUMINANTS_SDS['A'], ILLUMINANTS_SDS['D65']],
            annotate_parameters=[
                {
                    'arrowprops': {
                        'width': 10
                    }
                },
                {
                    'arrowprops': {
                        'width': 10
                    }
                },
            ])

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError,
            lambda: plot_sds_in_chromaticity_diagram(
                [ILLUMINANTS_SDS['A'], ILLUMINANTS_SDS['D65']],
                chromaticity_diagram_callable=lambda **x: x,
                method='Undefined')
        )


class TestPlotSdsInChromaticityDiagramCIE1931(unittest.TestCase):
    """
    Defines :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram_CIE1931` definition unit tests methods.
    """

    def test_plot_sds_in_chromaticity_diagram_CIE1931(self):
        """
        Tests :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram_CIE1931` definition.
        """

        figure, axes = plot_sds_in_chromaticity_diagram_CIE1931(
            [ILLUMINANTS_SDS['A'], ILLUMINANTS_SDS['D65']])

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSdsInChromaticityDiagramCIE1960UCS(unittest.TestCase):
    """
    Defines :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram_CIE1960UCS` definition unit tests methods.
    """

    def test_plot_sds_in_chromaticity_diagram_CIE1960UCS(self):
        """
        Tests :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram_CIE1960UCS` definition.
        """

        figure, axes = plot_sds_in_chromaticity_diagram_CIE1960UCS(
            [ILLUMINANTS_SDS['A'], ILLUMINANTS_SDS['D65']])

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSdsInChromaticityDiagramCIE1976UCS(unittest.TestCase):
    """
    Defines :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram_CIE1976UCS` definition unit tests methods.
    """

    def test_plot_sds_in_chromaticity_diagram_CIE1976UCS(self):
        """
        Tests :func:`colour.plotting.diagrams.\
plot_sds_in_chromaticity_diagram_CIE1976UCS` definition.
        """

        figure, axes = plot_sds_in_chromaticity_diagram_CIE1976UCS(
            [ILLUMINANTS_SDS['A'], ILLUMINANTS_SDS['D65']])

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == '__main__':
    unittest.main()
