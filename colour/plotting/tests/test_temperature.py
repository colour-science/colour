# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.plotting.temperature` module.
"""

from __future__ import division, unicode_literals

import unittest
from matplotlib.pyplot import Axes, Figure

from colour.plotting import (
    plot_planckian_locus_in_chromaticity_diagram_CIE1931,
    plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS)
from colour.plotting.temperature import (
    plot_planckian_locus, plot_planckian_locus_CIE1931,
    plot_planckian_locus_CIE1960UCS,
    plot_planckian_locus_in_chromaticity_diagram)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestPlotPlanckianLocus', 'TestPlotPlanckianLocusCIE1931',
    'TestPlotPlanckianLocusCIE1960UCS',
    'TestPlotPlanckianLocusInChromaticityDiagram',
    'TestPlotPlanckianLocusInChromaticityDiagramCIE1931',
    'TestPlotPlanckianLocusInChromaticityDiagramCIE1960UCS'
]


class TestPlotPlanckianLocus(unittest.TestCase):
    """
    Defines :func:`colour.plotting.temperature.plot_planckian_locus` definition
    unit tests methods.
    """

    def test_plot_planckian_locus(self):
        """
        Tests :func:`colour.plotting.temperature.plot_planckian_locus`
        definition.
        """

        figure, axes = plot_planckian_locus()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError, lambda: plot_planckian_locus(method='Undefined'))


class TestPlotPlanckianLocusCIE1931(unittest.TestCase):
    """
    Defines :func:`colour.plotting.temperature.plot_planckian_locus_CIE1931`
    definition unit tests methods.
    """

    def test_plot_planckian_locus(self):
        """
        Tests :func:`colour.plotting.temperature.plot_planckian_locus_CIE1931`
        definition.
        """

        figure, axes = plot_planckian_locus_CIE1931()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotPlanckianLocusCIE1960UCS(unittest.TestCase):
    """
    Defines :func:`colour.plotting.temperature.plot_planckian_locus_CIE1960UCS`
    definition unit tests methods.
    """

    def test_plot_planckian_locus(self):
        """
        Tests :func:`colour.plotting.temperature.\
plot_planckian_locus_CIE1960UCS` definition.
        """

        figure, axes = plot_planckian_locus_CIE1960UCS()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotPlanckianLocusInChromaticityDiagram(unittest.TestCase):
    """
    Defines :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram` definition unit tests methods.
    """

    def test_plot_planckian_locus_in_chromaticity_diagram(self):
        """
        Tests :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram` definition.
        """

        figure, axes = plot_planckian_locus_in_chromaticity_diagram(
            annotate_parameters={'arrowprops': {
                'width': 10
            }})

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_planckian_locus_in_chromaticity_diagram(
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
                {
                    'arrowprops': {
                        'width': 10
                    }
                },
            ])

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError, lambda: plot_planckian_locus_in_chromaticity_diagram(
                chromaticity_diagram_callable=lambda **x: x,
                planckian_locus_callable=lambda **x: x,
                method='Undefined'))


class TestPlotPlanckianLocusInChromaticityDiagramCIE1931(unittest.TestCase):
    """
    Defines :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram_CIE1931` definition unit tests
    methods.
    """

    def test_plot_planckian_locus_in_chromaticity_diagram_CIE1931(self):
        """
        Tests :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram_CIE1931` definition.
        """

        figure, axes = plot_planckian_locus_in_chromaticity_diagram_CIE1931()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotPlanckianLocusInChromaticityDiagramCIE1960UCS(unittest.TestCase):
    """
    Defines :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS` definition unit tests
    methods.
    """

    def test_plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(self):
        """
        Tests :func:`colour.plotting.temperature.\
plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS` definition.
        """

        figure, axes = plot_planckian_locus_in_chromaticity_diagram_CIE1960UCS(
        )

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == '__main__':
    unittest.main()
