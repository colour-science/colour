# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.algebra.interpolation` module.

References
----------
-   :cite:`Fairman1985b` : Fairman, H. S. (1985). The calculation of weight
    factors for tristimulus integration. Color Research & Application, 10(4),
    199-203. doi:10.1002/col.5080100407
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import unittest
from itertools import permutations

from colour.algebra.interpolation import vertices_and_relative_coordinates
from colour.algebra import (
    kernel_nearest_neighbour, kernel_linear, kernel_sinc, kernel_lanczos,
    kernel_cardinal_spline, KernelInterpolator, LinearInterpolator,
    SpragueInterpolator, CubicSplineInterpolator, PchipInterpolator,
    NullInterpolator, lagrange_coefficients, table_interpolation_trilinear,
    table_interpolation_tetrahedral)
from colour.algebra import random_triplet_generator
from colour.io import read_LUT
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'POINTS_DATA_A', 'LINEAR_INTERPOLATED_POINTS_DATA_A_10_SAMPLES',
    'SPRAGUE_INTERPOLATED_POINTS_DATA_A_10_SAMPLES',
    'CUBIC_SPLINE_INTERPOLATED_POINTS_DATA_A_X2_SAMPLES',
    'LAGRANGE_COEFFICIENTS_A', 'LAGRANGE_COEFFICIENTS_B', 'LUT_TABLE',
    'TestKernelNearestNeighbour', 'TestKernelLinear', 'TestKernelSinc',
    'TestKernelLanczos', 'TestKernelCardinalSpline', 'TestKernelInterpolator',
    'TestLinearInterpolator', 'TestSpragueInterpolator',
    'TestCubicSplineInterpolator', 'TestPchipInterpolator',
    'TestNullInterpolator', 'TestLagrangeCoefficients',
    'TestVerticesAndRelativeCoordinates', 'TestTableInterpolationTrilinear',
    'TestTableInterpolationTetrahedral'
]

POINTS_DATA_A = (9.3700, 12.3200, 12.4600, 9.5100, 5.9200, 4.3300, 4.2900,
                 3.8800, 4.5100, 10.9200, 27.5000, 49.6700, 69.5900, 81.7300,
                 88.1900, 86.0500)

LINEAR_INTERPOLATED_POINTS_DATA_A_10_SAMPLES = (
    9.370, 9.665, 9.960, 10.255, 10.550, 10.845, 11.140, 11.435, 11.730,
    12.025, 12.320, 12.334, 12.348, 12.362, 12.376, 12.390, 12.404, 12.418,
    12.432, 12.446, 12.460, 12.165, 11.870, 11.575, 11.280, 10.985, 10.690,
    10.395, 10.100, 9.805, 9.510, 9.151, 8.792, 8.433, 8.074, 7.715, 7.356,
    6.997, 6.638, 6.279, 5.920, 5.761, 5.602, 5.443, 5.284, 5.125, 4.966,
    4.807, 4.648, 4.489, 4.330, 4.326, 4.322, 4.318, 4.314, 4.310, 4.306,
    4.302, 4.298, 4.294, 4.290, 4.249, 4.208, 4.167, 4.126, 4.085, 4.044,
    4.003, 3.962, 3.921, 3.880, 3.943, 4.006, 4.069, 4.132, 4.195, 4.258,
    4.321, 4.384, 4.447, 4.510, 5.151, 5.792, 6.433, 7.074, 7.715, 8.356,
    8.997, 9.638, 10.279, 10.920, 12.578, 14.236, 15.894, 17.552, 19.210,
    20.868, 22.526, 24.184, 25.842, 27.500, 29.717, 31.934, 34.151, 36.368,
    38.585, 40.802, 43.019, 45.236, 47.453, 49.670, 51.662, 53.654, 55.646,
    57.638, 59.630, 61.622, 63.614, 65.606, 67.598, 69.590, 70.804, 72.018,
    73.232, 74.446, 75.660, 76.874, 78.088, 79.302, 80.516, 81.730, 82.376,
    83.022, 83.668, 84.314, 84.960, 85.606, 86.252, 86.898, 87.544, 88.190,
    87.976, 87.762, 87.548, 87.334, 87.120, 86.906, 86.692, 86.478, 86.264,
    86.050)

SPRAGUE_INTERPOLATED_POINTS_DATA_A_10_SAMPLES = (
    9.37000000, 9.72075073, 10.06936191, 10.41147570, 10.74302270, 11.06022653,
    11.35960827, 11.63799100, 11.89250427, 12.12058860, 12.32000000,
    12.48873542, 12.62489669, 12.72706530, 12.79433478, 12.82623598,
    12.82266243, 12.78379557, 12.71003009, 12.60189921, 12.46000000,
    12.28440225, 12.07404800, 11.82976500, 11.55443200, 11.25234375,
    10.92857600, 10.58835050, 10.23640000, 9.87633325, 9.51000000, 9.13692962,
    8.75620800, 8.36954763, 7.98097600, 7.59601562, 7.22086400, 6.86157362,
    6.52323200, 6.20914162, 5.92000000, 5.65460200, 5.41449600, 5.20073875,
    5.01294400, 4.84968750, 4.70891200, 4.58833225, 4.48584000, 4.39990900,
    4.33000000, 4.27757887, 4.24595200, 4.23497388, 4.24099200, 4.25804688,
    4.27907200, 4.29709387, 4.30643200, 4.30389887, 4.29000000, 4.26848387,
    4.24043200, 4.20608887, 4.16603200, 4.12117188, 4.07275200, 4.02234887,
    3.97187200, 3.92356387, 3.88000000, 3.84319188, 3.81318400, 3.79258487,
    3.78691200, 3.80367187, 3.85144000, 3.93894087, 4.07412800, 4.26326387,
    4.51000000, 4.81362075, 5.17028800, 5.58225150, 6.05776000, 6.60890625,
    7.24947200, 7.99277300, 8.84950400, 9.82558375, 10.92000000, 12.12700944,
    13.44892800, 14.88581406, 16.43283200, 18.08167969, 19.82201600,
    21.64288831, 23.53416000, 25.48793794, 27.50000000, 29.57061744,
    31.69964800, 33.88185481, 36.10777600, 38.36511719, 40.64014400,
    42.91907456, 45.18947200, 47.44163694, 49.67000000, 51.87389638,
    54.05273600, 56.20157688, 58.31198400, 60.37335938, 62.37427200,
    64.30378787, 66.15280000, 67.91535838, 69.59000000, 71.17616669,
    72.66283200, 74.04610481, 75.33171200, 76.53183594, 77.66195200,
    78.73766606, 79.77155200, 80.76998919, 81.73000000, 82.64375688,
    83.51935227, 84.35919976, 85.15567334, 85.89451368, 86.55823441,
    87.12952842, 87.59467414, 87.94694187, 88.19000000, 88.33345751,
    88.37111372, 88.30221714, 88.13600972, 87.88846516, 87.57902706,
    87.22734720, 86.85002373, 86.45733945, 86.05000000)

CUBIC_SPLINE_INTERPOLATED_POINTS_DATA_A_X2_SAMPLES = (
    9.37000000, 11.08838189, 12.26359953, 12.78808025, 12.55425139,
    11.50391691, 9.87473603, 8.01707329, 6.30369624, 5.08664365, 4.43550284,
    4.25438019, 4.29206798, 4.21753374, 3.98875865, 3.79691327, 4.02534907,
    5.23223510, 8.08816250, 13.36306794, 21.19519815, 30.89350026, 41.64531611,
    52.53540869, 62.65180882, 71.10713687, 77.46889540, 82.31355134,
    86.05208477, 88.28078752, 88.45998434, 86.05000000)

LAGRANGE_COEFFICIENTS_A = np.array([
    [0.92625, 0.09750, -0.02375],
    [0.85500, 0.19000, -0.04500],
    [0.78625, 0.27750, -0.06375],
    [0.72000, 0.36000, -0.08000],
    [0.65625, 0.43750, -0.09375],
    [0.59500, 0.51000, -0.10500],
    [0.53625, 0.57750, -0.11375],
    [0.48000, 0.64000, -0.12000],
    [0.42625, 0.69750, -0.12375],
    [0.37500, 0.75000, -0.12500],
    [0.32625, 0.79750, -0.12375],
    [0.28000, 0.84000, -0.12000],
    [0.23625, 0.87750, -0.11375],
    [0.19500, 0.91000, -0.10500],
    [0.15625, 0.93750, -0.09375],
    [0.12000, 0.96000, -0.08000],
    [0.08625, 0.97750, -0.06375],
    [0.05500, 0.99000, -0.04500],
    [0.02625, 0.99750, -0.02375],
])

LAGRANGE_COEFFICIENTS_B = np.array([
    [-0.0154375, 0.9725625, 0.0511875, -0.0083125],
    [-0.0285000, 0.9405000, 0.1045000, -0.0165000],
    [-0.0393125, 0.9041875, 0.1595625, -0.0244375],
    [-0.0480000, 0.8640000, 0.2160000, -0.0320000],
    [-0.0546875, 0.8203125, 0.2734375, -0.0390625],
    [-0.0595000, 0.7735000, 0.3315000, -0.0455000],
    [-0.0625625, 0.7239375, 0.3898125, -0.0511875],
    [-0.0640000, 0.6720000, 0.4480000, -0.0560000],
    [-0.0639375, 0.6180625, 0.5056875, -0.0598125],
    [-0.0625000, 0.5625000, 0.5625000, -0.0625000],
    [-0.0598125, 0.5056875, 0.6180625, -0.0639375],
    [-0.0560000, 0.4480000, 0.6720000, -0.0640000],
    [-0.0511875, 0.3898125, 0.7239375, -0.0625625],
    [-0.0455000, 0.3315000, 0.7735000, -0.0595000],
    [-0.0390625, 0.2734375, 0.8203125, -0.0546875],
    [-0.0320000, 0.2160000, 0.8640000, -0.0480000],
    [-0.0244375, 0.1595625, 0.9041875, -0.0393125],
    [-0.0165000, 0.1045000, 0.9405000, -0.0285000],
    [-0.0083125, 0.0511875, 0.9725625, -0.0154375],
])

LUT_TABLE = read_LUT(
    os.path.join(
        os.path.dirname(__file__), '..', '..', 'io', 'luts', 'tests',
        'resources', 'iridas_cube', 'ColourCorrect.cube')).table


class TestKernelNearestNeighbour(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.kernel_nearest_neighbour`
    definition units tests methods.
    """

    def test_kernel_nearest(self):
        """
        Tests :func:`colour.algebra.interpolation.kernel_nearest_neighbour`
        definition.
        """

        np.testing.assert_almost_equal(
            kernel_nearest_neighbour(np.linspace(-5, 5, 25)),
            np.array([
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0
            ]),
            decimal=7)


class TestKernelLinear(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.kernel_linear` definition
    units tests methods.
    """

    def test_kernel_linear(self):
        """
        Tests :func:`colour.algebra.interpolation.kernel_linear` definition.
        """

        np.testing.assert_almost_equal(
            kernel_linear(np.linspace(-5, 5, 25)),
            np.array([
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                0.16666667, 0.58333333, 1.00000000, 0.58333333, 0.16666667,
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
            ]),
            decimal=7)


class TestKernelSinc(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.kernel_sinc` definition
    units tests methods.
    """

    def test_kernel_sinc(self):
        """
        Tests :func:`colour.algebra.interpolation.kernel_sinc` definition.
        """

        np.testing.assert_almost_equal(
            kernel_sinc(np.linspace(-5, 5, 25)),
            np.array([
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                0.02824617, 0.12732395, 0.03954464, -0.16539867, -0.18006326,
                0.19098593, 0.73791298, 1.00000000, 0.73791298, 0.19098593,
                -0.18006326, -0.16539867, 0.03954464, 0.12732395, 0.02824617,
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            kernel_sinc(np.linspace(-5, 5, 25), 1),
            np.array([
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                0.19098593, 0.73791298, 1.00000000, 0.73791298, 0.19098593,
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
            ]),
            decimal=7)


class TestKernelLanczos(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.kernel_lanczos` definition
    units tests methods.
    """

    def test_kernel_lanczos(self):
        """
        Tests :func:`colour.algebra.interpolation.kernel_lanczos` definition.
        """

        np.testing.assert_almost_equal(
            kernel_lanczos(np.linspace(-5, 5, 25)),
            np.array([
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                0.00000000e+00, 8.06009483e-04, 2.43170841e-02, 1.48478897e-02,
                -9.33267411e-02, -1.32871018e-01, 1.67651704e-01,
                7.14720157e-01, 1.00000000e+00, 7.14720157e-01, 1.67651704e-01,
                -1.32871018e-01, -9.33267411e-02, 1.48478897e-02,
                2.43170841e-02, 8.06009483e-04, 0.00000000e+00, 0.00000000e+00,
                0.00000000e+00, 0.00000000e+00, 0.00000000e+00
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            kernel_lanczos(np.linspace(-5, 5, 25), 1),
            np.array([
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                0.03647563, 0.54451556, 1.00000000, 0.54451556, 0.03647563,
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
            ]),
            decimal=7)


class TestKernelCardinalSpline(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.kernel_cardinal_spline`
    definition units tests methods.
    """

    def test_kernel_cardinal_spline(self):
        """
        Tests :func:`colour.algebra.interpolation.kernel_cardinal_spline`
        definition.
        """

        np.testing.assert_almost_equal(
            kernel_cardinal_spline(np.linspace(-5, 5, 25)),
            np.array([
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                0.00000000, 0.00000000, 0.00000000, -0.03703704, -0.0703125,
                0.13194444, 0.67447917, 1.00000000, 0.67447917, 0.13194444,
                -0.0703125, -0.03703704, 0.00000000, 0.00000000, 0.00000000,
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            kernel_cardinal_spline(np.linspace(-5, 5, 25), 0, 1),
            np.array([
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000,
                0.00000000, 0.00000000, 0.00000000, 0.00617284, 0.0703125,
                0.26157407, 0.52922454, 0.66666667, 0.52922454, 0.26157407,
                0.0703125, 0.00617284, 0.00000000, 0.00000000, 0.00000000,
                0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000
            ]),
            decimal=7)


class TestKernelInterpolator(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.KernelInterpolator` class units
    tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('x', 'y', 'window', 'kernel', 'kernel_args',
                               'padding_args')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(KernelInterpolator))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ()

        for method in required_methods:
            self.assertIn(method, dir(KernelInterpolator))

    def test___call__(self):
        """
        Tests :func:`colour.algebra.interpolation.KernelInterpolator.__call__`
        method.
        """

        x = np.arange(11, 26, 1)
        y = np.sin(x / len(x) * np.pi * 6) / (x / len(x)) + np.pi
        x_i = np.linspace(11, 25, 25)

        kernel_interpolator = KernelInterpolator(x, y)
        np.testing.assert_almost_equal(
            kernel_interpolator(x_i),
            np.array([
                4.43848790, 4.26286480, 3.64640076, 2.77982023, 2.13474499,
                2.08206794, 2.50585862, 3.24992692, 3.84593162, 4.06289704,
                3.80825633, 3.21068994, 2.65177161, 2.32137382, 2.45995375,
                2.88799997, 3.43843598, 3.79504892, 3.79937086, 3.47673343,
                2.99303182, 2.59305006, 2.47805594, 2.82957843, 3.14159265
            ]),
            decimal=7)

        kernel_interpolator = KernelInterpolator(x, y, kernel=kernel_sinc)
        np.testing.assert_almost_equal(
            kernel_interpolator(x_i),
            np.array([
                4.43848790, 4.47570010, 3.84353906, 3.05959493, 2.53514958,
                2.19916874, 2.93225625, 3.32187855, 4.09458791, 4.23088094,
                3.92591447, 3.53263071, 2.65177161, 2.73541557, 2.65740315,
                3.17077616, 3.69624479, 3.87159620, 4.06433758, 3.56283868,
                3.28312289, 2.79652091, 2.62481419, 3.22117115, 3.14159265
            ]),
            decimal=7)

        kernel_interpolator = KernelInterpolator(x, y, window=1)
        np.testing.assert_almost_equal(
            kernel_interpolator(x_i),
            np.array([
                4.43848790, 4.96712277, 4.09584229, 3.23991575, 2.80418924,
                2.28470276, 3.20024753, 3.41120944, 4.46416970, 4.57878168,
                4.15371498, 3.92841633, 2.65177161, 3.02110187, 2.79812654,
                3.44218674, 4.00032377, 4.01356870, 4.47633386, 3.70912627,
                3.58365067, 3.14325415, 2.88247572, 3.37531662, 3.14159265
            ]),
            decimal=7)

        kernel_interpolator = KernelInterpolator(
            x, y, window=1, kernel_args={'a': 1})
        np.testing.assert_almost_equal(
            kernel_interpolator(x_i),
            np.array([
                4.43848790, 3.34379320, 3.62463711, 2.34585418, 2.04767083,
                2.09444849, 2.13349835, 3.10304927, 3.29553153, 3.59884738,
                3.48484031, 2.72974983, 2.65177161, 2.03850468, 2.29470194,
                2.76179863, 2.80189050, 3.75979450, 2.98422257, 3.48444099,
                2.49208997, 2.46516442, 2.42336082, 2.25975903, 3.14159265
            ]),
            decimal=7)

        kernel_interpolator = KernelInterpolator(
            x, y, padding_args={
                'pad_width': (3, 3),
                'mode': 'mean'
            })
        np.testing.assert_almost_equal(
            kernel_interpolator(x_i),
            np.array([
                4.4384879, 4.35723245, 3.62918155, 2.77471295, 2.13474499,
                2.08206794, 2.50585862, 3.24992692, 3.84593162, 4.06289704,
                3.80825633, 3.21068994, 2.65177161, 2.32137382, 2.45995375,
                2.88799997, 3.43843598, 3.79504892, 3.79937086, 3.47673343,
                2.99303182, 2.59771985, 2.49380017, 2.76339043, 3.14159265
            ]),
            decimal=7)

        x_1 = np.arange(1, 10, 1)
        x_2 = x_1 * 10
        x_3 = x_1 / 10
        y = np.sin(x_1 / len(x_1) * np.pi * 6) / (x_1 / len(x_1))
        x_i = np.linspace(1, 9, 25)

        np.testing.assert_almost_equal(
            KernelInterpolator(x_1, y)(x_i),
            KernelInterpolator(x_2, y)(x_i * 10),
            decimal=7)

        np.testing.assert_almost_equal(
            KernelInterpolator(x_1, y)(x_i),
            KernelInterpolator(x_3, y)(x_i / 10),
            decimal=7)

    @ignore_numpy_errors
    def test_nan__call__(self):
        """
        Tests :func:`colour.algebra.interpolation.KernelInterpolator.__call__`
        method nan support.
        """

        # NOTE: As the "x" independent variable must be uniform, it cannot
        # contain NaNs.
        # TODO: Revisit if the interpolator can be applied on non-uniform "x"
        # independent variable.

        pass


class TestLinearInterpolator(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.LinearInterpolator` class units
    tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('x', 'y')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(LinearInterpolator))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ()

        for method in required_methods:
            self.assertIn(method, dir(LinearInterpolator))

    def test___call__(self):
        """
        Tests :func:`colour.algebra.interpolation.LinearInterpolator.__call__`
        method.
        """

        interval = 0.1
        x = np.arange(len(POINTS_DATA_A))
        linear_interpolator = LinearInterpolator(x, POINTS_DATA_A)

        for i, value in enumerate(
                np.arange(0,
                          len(POINTS_DATA_A) - 1 + interval, interval)):
            self.assertAlmostEqual(
                LINEAR_INTERPOLATED_POINTS_DATA_A_10_SAMPLES[i],
                linear_interpolator(value),
                places=7)

        np.testing.assert_almost_equal(
            linear_interpolator(
                np.arange(0,
                          len(POINTS_DATA_A) - 1 + interval, interval)),
            LINEAR_INTERPOLATED_POINTS_DATA_A_10_SAMPLES)

    @ignore_numpy_errors
    def test_nan__call__(self):
        """
        Tests :func:`colour.algebra.interpolation.LinearInterpolator.__call__`
        method nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            try:
                linear_interpolator = LinearInterpolator(
                    np.array(case), np.array(case))
                linear_interpolator(case[0])
            except ValueError:
                pass


class TestSpragueInterpolator(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.SpragueInterpolator` class
    unit tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('x', 'y')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(SpragueInterpolator))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ()

        for method in required_methods:
            self.assertIn(method, dir(SpragueInterpolator))

    def test___call__(self):
        """
        Tests :func:`colour.algebra.interpolation.SpragueInterpolator.__call__`
        method.
        """

        interval = 0.1
        x = np.arange(len(POINTS_DATA_A))
        sprague_interpolator = SpragueInterpolator(x, POINTS_DATA_A)

        for i, value in enumerate(
                np.arange(0,
                          len(POINTS_DATA_A) - 1 + interval, interval)):
            self.assertAlmostEqual(
                SPRAGUE_INTERPOLATED_POINTS_DATA_A_10_SAMPLES[i],
                sprague_interpolator(value),
                places=7)

        np.testing.assert_almost_equal(
            sprague_interpolator(
                np.arange(0,
                          len(POINTS_DATA_A) - 1 + interval, interval)),
            SPRAGUE_INTERPOLATED_POINTS_DATA_A_10_SAMPLES)

    @ignore_numpy_errors
    def test_nan__call__(self):
        """
        Tests :func:`colour.algebra.interpolation.SpragueInterpolator.__call__`
        method nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            try:
                sprague_interpolator = SpragueInterpolator(
                    np.array(case), np.array(case))
                sprague_interpolator(case[0])
            except AssertionError:
                pass


class TestCubicSplineInterpolator(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.CubicSplineInterpolator` class
    unit tests methods.
    """

    def test___call__(self):
        """
        Tests :func:`colour.algebra.interpolation.\
CubicSplineInterpolator.__call__` method.

        Notes
        -----
        -   This class is a wrapper around *scipy.interpolate.interp1d* class
            and is assumed to be unit tested thoroughly.
        """

        np.testing.assert_almost_equal(
            CubicSplineInterpolator(
                np.linspace(0, 1, len(POINTS_DATA_A)),
                POINTS_DATA_A)(np.linspace(0, 1,
                                           len(POINTS_DATA_A) * 2)),
            CUBIC_SPLINE_INTERPOLATED_POINTS_DATA_A_X2_SAMPLES)


class TestPchipInterpolator(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.PchipInterpolator` class
    unit tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('x', 'y')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(PchipInterpolator))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ()

        for method in required_methods:
            self.assertIn(method, dir(PchipInterpolator))


class TestNullInterpolator(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.NullInterpolator` class
    unit tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('x', 'y')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(NullInterpolator))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ()

        for method in required_methods:
            self.assertIn(method, dir(NullInterpolator))

    def test___call__(self):
        """
        Tests :func:`colour.algebra.interpolation.NullInterpolator.__call__`
        method.
        """

        x = np.arange(len(POINTS_DATA_A))
        null_interpolator = NullInterpolator(x, POINTS_DATA_A)
        np.testing.assert_almost_equal(
            null_interpolator(np.array([0.75, 2.0, 3.0, 4.75])),
            np.array([np.nan, 12.46, 9.51, np.nan]))

        null_interpolator = NullInterpolator(x, POINTS_DATA_A, 0.25, 0.25)
        np.testing.assert_almost_equal(
            null_interpolator(np.array([0.75, 2.0, 3.0, 4.75])),
            np.array([12.32, 12.46, 9.51, 4.33]))

    @ignore_numpy_errors
    def test_nan__call__(self):
        """
        Tests :func:`colour.algebra.interpolation.NullInterpolator.__call__`
        method nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            try:
                null_interpolator = NullInterpolator(
                    np.array(case), np.array(case))
                null_interpolator(case[0])
            except ValueError:
                pass


class TestLagrangeCoefficients(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.lagrange_coefficients`
    definition unit tests methods.
    """

    def test_lagrange_coefficients(self):
        """
        Tests :func:`colour.algebra.interpolation.lagrange_coefficients`
        definition.

        Notes
        -----
        -   :attr:`LAGRANGE_COEFFICIENTS_A` and :attr:`LAGRANGE_COEFFICIENTS_B`
            attributes data is matching :cite:`Fairman1985b`.

        References
        ----------
        :cite:`Fairman1985b`
        """

        lc = [lagrange_coefficients(i, 3) for i in np.linspace(0.05, 0.95, 19)]
        np.testing.assert_almost_equal(lc, LAGRANGE_COEFFICIENTS_A, decimal=7)

        lc = [lagrange_coefficients(i, 4) for i in np.linspace(1.05, 1.95, 19)]
        np.testing.assert_almost_equal(lc, LAGRANGE_COEFFICIENTS_B, decimal=7)


class TestVerticesAndRelativeCoordinates(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.\
vertices_and_relative_coordinates` definition unit tests methods.
    """

    def test_vertices_and_relative_coordinates(self):
        """
        Tests :func:`colour.algebra.interpolation.\
vertices_and_relative_coordinates` definition.
        """

        prng = np.random.RandomState(4)

        V_xyz = random_triplet_generator(4, random_state=prng)
        vertices, V_xyzr = vertices_and_relative_coordinates(V_xyz, LUT_TABLE)

        np.testing.assert_almost_equal(
            vertices,
            np.array([
                [
                    [0.58919500, 0.58919500, 0.13916400],
                    [0.33333300, 0.00000000, 0.33333300],
                    [0.83331100, 0.83331100, 0.83331100],
                    [0.79789400, -0.03541200, -0.03541200],
                ],
                [
                    [0.59460100, 0.59460100, 0.36958600],
                    [0.39062300, 0.00000000, 0.78124600],
                    [0.83331100, 0.83331100, 1.24996300],
                    [0.75276700, -0.02847900, 0.36214400],
                ],
                [
                    [0.66343200, 0.93018800, 0.12992000],
                    [0.41665500, 0.41665500, 0.41665500],
                    [0.70710200, 1.11043500, 0.70710200],
                    [0.63333300, 0.31666700, 0.00000000],
                ],
                [
                    [0.68274900, 0.99108200, 0.37441600],
                    [0.41665500, 0.41665500, 0.83330800],
                    [0.51971400, 0.74472900, 0.74472900],
                    [0.73227800, 0.31562600, 0.31562600],
                ],
                [
                    [0.89131800, 0.61982300, 0.07683300],
                    [0.75276700, -0.02847900, 0.36214400],
                    [1.06561000, 0.64895700, 0.64895700],
                    [1.19684100, -0.05311700, -0.05311700],
                ],
                [
                    [0.95000000, 0.63333300, 0.31666700],
                    [0.66666700, 0.00000000, 0.66666700],
                    [1.00000000, 0.66666700, 1.00000000],
                    [1.16258800, -0.05037200, 0.35394800],
                ],
                [
                    [0.88379200, 0.88379200, 0.20874600],
                    [0.73227800, 0.31562600, 0.31562600],
                    [0.89460600, 0.89460600, 0.66959000],
                    [1.03843900, 0.31089900, -0.05287000],
                ],
                [
                    [0.88919900, 0.88919900, 0.43916800],
                    [0.66666700, 0.33333300, 0.66666700],
                    [1.24996600, 1.24996600, 1.24996600],
                    [1.13122500, 0.29792000, 0.29792000],
                ],
            ]))
        np.testing.assert_almost_equal(
            V_xyzr,
            np.array([
                [0.90108952, 0.09318647, 0.75894709],
                [0.64169675, 0.64826849, 0.30437460],
                [0.91805308, 0.92882336, 0.33814877],
                [0.14444798, 0.01869077, 0.59305522],
            ]))


class TestTableInterpolationTrilinear(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.\
table_interpolation_trilinear` definition unit tests methods.
    """

    def test_interpolation_trilinear(self):
        """
        Tests :func:`colour.algebra.interpolation.\
table_interpolation_trilinear` definition.
        """

        prng = np.random.RandomState(4)

        V_xyz = random_triplet_generator(16, random_state=prng)

        np.testing.assert_almost_equal(
            table_interpolation_trilinear(V_xyz, LUT_TABLE),
            np.array([
                [1.07937594, -0.02773926, 0.55498254],
                [0.53983424, 0.37099516, 0.13994561],
                [1.13449122, -0.00305380, 0.13792909],
                [0.73411897, 1.00141020, 0.59348239],
                [0.74066176, 0.44679540, 0.55030394],
                [0.20634750, 0.84797880, 0.55905579],
                [0.92348649, 0.73112515, 0.42362820],
                [0.03639248, 0.70357649, 0.52375041],
                [0.29215488, 0.19697840, 0.44603879],
                [0.47793470, 0.08696360, 0.70288463],
                [0.88883354, 0.68680856, 0.87404642],
                [0.21430977, 0.16796653, 0.19634247],
                [0.82118989, 0.69239283, 0.39932389],
                [1.06679072, 0.37974319, 0.49759377],
                [0.17856230, 0.44755467, 0.62045271],
                [0.59220355, 0.93136492, 0.30063692],
            ]))


class TestTableInterpolationTetrahedral(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.\
table_interpolation_tetrahedral` definition unit tests methods.
    """

    def test_interpolation_tetrahedral(self):
        """
        Tests :func:`colour.algebra.interpolation.\
table_interpolation_tetrahedral` definition.
        """

        prng = np.random.RandomState(4)

        V_xyz = random_triplet_generator(16, random_state=prng)

        np.testing.assert_almost_equal(
            table_interpolation_tetrahedral(V_xyz, LUT_TABLE),
            np.array([
                [1.08039215, -0.02840092, 0.55855303],
                [0.52208945, 0.35297753, 0.13599555],
                [1.14373467, -0.00422138, 0.13413290],
                [0.71384967, 0.98420883, 0.57982724],
                [0.76771576, 0.46280975, 0.55106736],
                [0.20861663, 0.85077712, 0.57102264],
                [0.90398698, 0.72351675, 0.41151955],
                [0.03749453, 0.70226823, 0.52614254],
                [0.29626758, 0.21645072, 0.47615873],
                [0.46729624, 0.07494851, 0.68892548],
                [0.85907681, 0.67744258, 0.84410486],
                [0.24335535, 0.20896545, 0.21996717],
                [0.79244027, 0.66930773, 0.39213595],
                [1.08383608, 0.37985897, 0.49011919],
                [0.14683649, 0.43624903, 0.58706947],
                [0.61272658, 0.92799297, 0.29650424],
            ]))


if __name__ == '__main__':
    unittest.main()
