# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.plotting.models` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from matplotlib.pyplot import Axes, Figure

from colour.plotting import (
    common_colourspace_model_axis_reorder, plot_pointer_gamut,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS,
    plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS,
    plot_single_cctf, plot_multi_cctfs, plot_constant_hue_loci)
from colour.plotting.models import (
    plot_RGB_colourspaces_in_chromaticity_diagram,
    plot_RGB_chromaticities_in_chromaticity_diagram, ellipses_MacAdam1942,
    plot_ellipses_MacAdam1942_in_chromaticity_diagram)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestCommonColourspaceModelAxisReorder', 'TestPlotPointerGamut',
    'TestPlotRGBColourspacesInChromaticityDiagram',
    'TestPlotRGBColourspacesInChromaticityDiagramCIE1931',
    'TestPlotRGBColourspacesInChromaticityDiagramCIE1960UCS',
    'TestPlotRGBColourspacesInChromaticityDiagramCIE1976UCS',
    'TestPlotRGBChromaticitiesInChromaticityDiagram',
    'TestPlotRGBChromaticitiesInChromaticityDiagramCIE1931',
    'TestPlotRGBChromaticitiesInChromaticityDiagramCIE1960UCS',
    'TestPlotRGBChromaticitiesInChromaticityDiagramCIE1976UCS',
    'TestPlotEllipsesMacAdam1942InChromaticityDiagram',
    'TestPlotEllipsesMacAdam1942InChromaticityDiagramCIE1931',
    'TestPlotEllipsesMacAdam1942InChromaticityDiagramCIE1960UCS',
    'TestPlotEllipsesMacAdam1942InChromaticityDiagramCIE1976UCS',
    'TestPlotSingleCctf', 'TestPlotMultiCctfs', 'TestPlotConstantHueLoci'
]


class TestCommonColourspaceModelAxisReorder(unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.\
common_colourspace_model_axis_reorder` definition unit tests methods.
    """

    def test_common_colourspace_model_axis_reorder(self):
        """
        Tests :func:`colour.plotting.models.\
common_colourspace_model_axis_reorder` definition.
        """

        a = np.array([0, 1, 2])

        np.testing.assert_almost_equal(
            common_colourspace_model_axis_reorder(a),
            np.array([0, 1, 2]),
            decimal=7)

        np.testing.assert_almost_equal(
            common_colourspace_model_axis_reorder(a, 'CIE Lab'),
            np.array([1, 2, 0]),
            decimal=7)

        np.testing.assert_almost_equal(
            common_colourspace_model_axis_reorder(a, 'CIE LCHab'),
            np.array([1, 2, 0]),
            decimal=7)

        np.testing.assert_almost_equal(
            common_colourspace_model_axis_reorder(a, 'CIE Luv'),
            np.array([1, 2, 0]),
            decimal=7)

        np.testing.assert_almost_equal(
            common_colourspace_model_axis_reorder(a, 'CIE LCHab'),
            np.array([1, 2, 0]),
            decimal=7)

        np.testing.assert_almost_equal(
            common_colourspace_model_axis_reorder(a, 'DIN 99'),
            np.array([1, 2, 0]),
            decimal=7)

        np.testing.assert_almost_equal(
            common_colourspace_model_axis_reorder(a, 'Hunter Lab'),
            np.array([1, 2, 0]),
            decimal=7)

        np.testing.assert_almost_equal(
            common_colourspace_model_axis_reorder(a, 'Hunter Rdab'),
            np.array([1, 2, 0]),
            decimal=7)

        np.testing.assert_almost_equal(
            common_colourspace_model_axis_reorder(a, 'IPT'),
            np.array([1, 2, 0]),
            decimal=7)

        np.testing.assert_almost_equal(
            common_colourspace_model_axis_reorder(a, 'JzAzBz'),
            np.array([1, 2, 0]),
            decimal=7)

        np.testing.assert_almost_equal(
            common_colourspace_model_axis_reorder(a, 'OSA UCS'),
            np.array([1, 2, 0]),
            decimal=7)

        np.testing.assert_almost_equal(
            common_colourspace_model_axis_reorder(a, 'hdr-CIELAB'),
            np.array([1, 2, 0]),
            decimal=7)

        np.testing.assert_almost_equal(
            common_colourspace_model_axis_reorder(a, 'hdr-IPT'),
            np.array([1, 2, 0]),
            decimal=7)


class TestPlotPointerGamut(unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.plot_pointer_gamut` definition unit
    tests methods.
    """

    def test_plot_pointer_gamut(self):
        """
        Tests :func:`colour.plotting.models.plot_pointer_gamut` definition.
        """

        figure, axes = plot_pointer_gamut()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_pointer_gamut(method='CIE 1960 UCS')

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_pointer_gamut(method='CIE 1976 UCS')

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError, lambda: plot_pointer_gamut(method='Undefined'))


class TestPlotRGBColourspacesInChromaticityDiagram(unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram` definition unit tests methods.
    """

    def test_plot_RGB_colourspaces_in_chromaticity_diagram(self):
        """
        Tests :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram` definition.
        """

        figure, axes = plot_RGB_colourspaces_in_chromaticity_diagram(
            show_pointer_gamut=True)

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        self.assertRaises(
            ValueError,
            lambda: plot_RGB_colourspaces_in_chromaticity_diagram(
                chromaticity_diagram_callable=lambda **x: x,
                method='Undefined')
        )


class TestPlotRGBColourspacesInChromaticityDiagramCIE1931(unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931` definition unit tests
    methods.
    """

    def test_plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931(self):
        """
        Tests :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931` definition.
        """

        figure, axes = plot_RGB_colourspaces_in_chromaticity_diagram_CIE1931()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotRGBColourspacesInChromaticityDiagramCIE1960UCS(
        unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS` definition unit tests
    methods.
    """

    def test_plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS(self):
        """
        Tests :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS` definition.
        """

        figure, axes = (
            plot_RGB_colourspaces_in_chromaticity_diagram_CIE1960UCS())

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotRGBColourspacesInChromaticityDiagramCIE1976UCS(
        unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS` definition unit tests
    methods.
    """

    def test_plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS(self):
        """
        Tests :func:`colour.plotting.models.\
plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS` definition.
        """

        figure, axes = (
            plot_RGB_colourspaces_in_chromaticity_diagram_CIE1976UCS())

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotRGBChromaticitiesInChromaticityDiagram(unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.\
plot_RGB_chromaticities_in_chromaticity_diagram` definition unit tests methods.
    """

    def test_plot_RGB_chromaticities_in_chromaticity_diagram(self):
        """
        Tests :func:`colour.plotting.models.\
plot_RGB_chromaticities_in_chromaticity_diagram` definition.
        """

        figure, axes = plot_RGB_chromaticities_in_chromaticity_diagram(
            np.random.random((128, 128, 3)),
            scatter_parameters={'marker': 'v'})

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotRGBChromaticitiesInChromaticityDiagramCIE1931(unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.\
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931` definition unit tests
    methods.
    """

    def test_plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(self):
        """
        Tests :func:`colour.plotting.models.\
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931` definition.
        """

        figure, axes = plot_RGB_chromaticities_in_chromaticity_diagram_CIE1931(
            np.random.random((128, 128, 3)))

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotRGBChromaticitiesInChromaticityDiagramCIE1960UCS(
        unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.\
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS` definition unit
    tests methods.
    """

    def test_plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS(self):
        """
        Tests :func:`colour.plotting.models.\
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS` definition.
        """

        figure, axes = (
            plot_RGB_chromaticities_in_chromaticity_diagram_CIE1960UCS(
                np.random.random((128, 128, 3))))

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotRGBChromaticitiesInChromaticityDiagramCIE1976UCS(
        unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.\
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS` definition unit
    tests methods.
    """

    def test_plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS(self):
        """
        Tests :func:`colour.plotting.models.\
plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS` definition.
        """

        figure, axes = (
            plot_RGB_chromaticities_in_chromaticity_diagram_CIE1976UCS(
                np.random.random((128, 128, 3))))

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestEllipsesMacAdam1942(unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.ellipses_MacAdam1942` definition unit
    tests methods.
    """

    def test_ellipses_MacAdam1942(self):
        """
        Tests :func:`colour.plotting.models.ellipses_MacAdam1942` definition.
        """

        self.assertEqual(len(ellipses_MacAdam1942()), 25)

        self.assertRaises(
            ValueError, lambda: ellipses_MacAdam1942(method='Undefined'))


class TestPlotEllipsesMacAdam1942InChromaticityDiagram(unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram` definition unit tests
    methods.
    """

    def test_plot_ellipses_MacAdam1942_in_chromaticity_diagram(self):
        """
        Tests :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram` definition.
        """

        figure, axes = plot_ellipses_MacAdam1942_in_chromaticity_diagram(
            chromaticity_diagram_clipping=True,
            ellipse_parameters={'color': 'k'})

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)

        figure, axes = plot_ellipses_MacAdam1942_in_chromaticity_diagram(
            chromaticity_diagram_clipping=True,
            ellipse_parameters=[{
                'color': 'k'
            }] * 25)

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotEllipsesMacAdam1942InChromaticityDiagramCIE1931(
        unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931` definition unit
    tests methods.
    """

    def test_plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931(self):
        """
        Tests :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931` definition.
        """

        figure, axes = (
            plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1931())

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotEllipsesMacAdam1942InChromaticityDiagramCIE1960UCS(
        unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS` definition unit
    tests methods.
    """

    def test_plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS(
            self):
        """
        Tests :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS` definition.
        """

        figure, axes = (
            plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1960UCS())

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotEllipsesMacAdam1942InChromaticityDiagramCIE1976UCS(
        unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS` definition unit
    tests methods.
    """

    def test_plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS(
            self):
        """
        Tests :func:`colour.plotting.models.\
plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS` definition.
        """

        figure, axes = (
            plot_ellipses_MacAdam1942_in_chromaticity_diagram_CIE1976UCS())

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotSingleCctf(unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.plot_single_cctf` definition unit
    tests methods.
    """

    def test_plot_single_cctf(self):
        """
        Tests :func:`colour.plotting.models.plot_single_cctf` definition.
        """

        figure, axes = plot_single_cctf()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotMultiCctfs(unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.plot_multi_cctfs` definition unit
    tests methods.
    """

    def test_plot_multi_cctfs(self):
        """
        Tests :func:`colour.plotting.models.plot_multi_cctfs` definition.
        """

        figure, axes = plot_multi_cctfs()

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


class TestPlotConstantHueLoci(unittest.TestCase):
    """
    Defines :func:`colour.plotting.models.plot_constant_hue_loci` definition
    unit tests methods.
    """

    def test_plot_constant_hue_loci(self):
        """
        Tests :func:`colour.plotting.models.plot_constant_hue_loci` definition.
        """

        data = np.array([
            [
                None,
                np.array([0.95010000, 1.00000000, 1.08810000]),
                np.array([0.40920000, 0.28120000, 0.30600000]),
                np.array([
                    [0.02495100, 0.01908600, 0.02032900],
                    [0.10944300, 0.06235900, 0.06788100],
                    [0.27186500, 0.18418700, 0.19565300],
                    [0.48898900, 0.40749400, 0.44854600],
                ]),
                None,
            ],
            [
                None,
                np.array([0.95010000, 1.00000000, 1.08810000]),
                np.array([0.30760000, 0.48280000, 0.42770000]),
                np.array([
                    [0.02108000, 0.02989100, 0.02790400],
                    [0.06194700, 0.11251000, 0.09334400],
                    [0.15255800, 0.28123300, 0.23234900],
                    [0.34157700, 0.56681300, 0.47035300],
                ]),
                None,
            ],
            [
                None,
                np.array([0.95010000, 1.00000000, 1.08810000]),
                np.array([0.39530000, 0.28120000, 0.18450000]),
                np.array([
                    [0.02436400, 0.01908600, 0.01468800],
                    [0.10331200, 0.06235900, 0.02854600],
                    [0.26311900, 0.18418700, 0.12109700],
                    [0.43158700, 0.40749400, 0.39008600],
                ]),
                None,
            ],
            [
                None,
                np.array([0.95010000, 1.00000000, 1.08810000]),
                np.array([0.20510000, 0.18420000, 0.57130000]),
                np.array([
                    [0.03039800, 0.02989100, 0.06123300],
                    [0.08870000, 0.08498400, 0.21843500],
                    [0.18405800, 0.18418700, 0.40111400],
                    [0.32550100, 0.34047200, 0.50296900],
                    [0.53826100, 0.56681300, 0.80010400],
                ]),
                None,
            ],
            [
                None,
                np.array([0.95010000, 1.00000000, 1.08810000]),
                np.array([0.35770000, 0.28120000, 0.11250000]),
                np.array([
                    [0.03678100, 0.02989100, 0.01481100],
                    [0.17127700, 0.11251000, 0.01229900],
                    [0.30080900, 0.28123300, 0.21229800],
                    [0.52976000, 0.40749400, 0.11720000],
                ]),
                None,
            ],
        ])

        figure, axes = plot_constant_hue_loci(
            data, 'IPT', scatter_parameters={'marker': 'v'})

        self.assertIsInstance(figure, Figure)
        self.assertIsInstance(axes, Axes)


if __name__ == '__main__':
    unittest.main()
