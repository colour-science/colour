#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.algebra.interpolation` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.algebra import (
    LinearInterpolator,
    SpragueInterpolator,
    PchipInterpolator,
    lagrange_coefficients)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['POINTS_DATA_A',
           'LINEAR_INTERPOLATED_POINTS_DATA_A_10_SAMPLES',
           'SPRAGUE_INTERPOLATED_POINTS_DATA_A_10_SAMPLES',
           'LAGRANGE_COEFFICIENTS_A',
           'LAGRANGE_COEFFICIENTS_B',
           'TestLinearInterpolator',
           'TestSpragueInterpolator',
           'TestPchipInterpolator',
           'TestLagrangeCoefficients']

POINTS_DATA_A = (
    9.3700,
    12.3200,
    12.4600,
    9.5100,
    5.9200,
    4.3300,
    4.2900,
    3.8800,
    4.5100,
    10.9200,
    27.5000,
    49.6700,
    69.5900,
    81.7300,
    88.1900,
    86.0500)

LINEAR_INTERPOLATED_POINTS_DATA_A_10_SAMPLES = (
    9.370,
    9.665,
    9.960,
    10.255,
    10.550,
    10.845,
    11.140,
    11.435,
    11.730,
    12.025,
    12.320,
    12.334,
    12.348,
    12.362,
    12.376,
    12.390,
    12.404,
    12.418,
    12.432,
    12.446,
    12.460,
    12.165,
    11.870,
    11.575,
    11.280,
    10.985,
    10.690,
    10.395,
    10.100,
    9.805,
    9.510,
    9.151,
    8.792,
    8.433,
    8.074,
    7.715,
    7.356,
    6.997,
    6.638,
    6.279,
    5.920,
    5.761,
    5.602,
    5.443,
    5.284,
    5.125,
    4.966,
    4.807,
    4.648,
    4.489,
    4.330,
    4.326,
    4.322,
    4.318,
    4.314,
    4.310,
    4.306,
    4.302,
    4.298,
    4.294,
    4.290,
    4.249,
    4.208,
    4.167,
    4.126,
    4.085,
    4.044,
    4.003,
    3.962,
    3.921,
    3.880,
    3.943,
    4.006,
    4.069,
    4.132,
    4.195,
    4.258,
    4.321,
    4.384,
    4.447,
    4.510,
    5.151,
    5.792,
    6.433,
    7.074,
    7.715,
    8.356,
    8.997,
    9.638,
    10.279,
    10.920,
    12.578,
    14.236,
    15.894,
    17.552,
    19.210,
    20.868,
    22.526,
    24.184,
    25.842,
    27.500,
    29.717,
    31.934,
    34.151,
    36.368,
    38.585,
    40.802,
    43.019,
    45.236,
    47.453,
    49.670,
    51.662,
    53.654,
    55.646,
    57.638,
    59.630,
    61.622,
    63.614,
    65.606,
    67.598,
    69.590,
    70.804,
    72.018,
    73.232,
    74.446,
    75.660,
    76.874,
    78.088,
    79.302,
    80.516,
    81.730,
    82.376,
    83.022,
    83.668,
    84.314,
    84.960,
    85.606,
    86.252,
    86.898,
    87.544,
    88.190,
    87.976,
    87.762,
    87.548,
    87.334,
    87.120,
    86.906,
    86.692,
    86.478,
    86.264,
    86.050)

SPRAGUE_INTERPOLATED_POINTS_DATA_A_10_SAMPLES = (
    9.37000000,
    9.72075073,
    10.06936191,
    10.41147570,
    10.74302270,
    11.06022653,
    11.35960827,
    11.63799100,
    11.89250427,
    12.12058860,
    12.32000000,
    12.48873542,
    12.62489669,
    12.72706530,
    12.79433478,
    12.82623598,
    12.82266243,
    12.78379557,
    12.71003009,
    12.60189921,
    12.46000000,
    12.28440225,
    12.07404800,
    11.82976500,
    11.55443200,
    11.25234375,
    10.92857600,
    10.58835050,
    10.23640000,
    9.87633325,
    9.51000000,
    9.13692962,
    8.75620800,
    8.36954763,
    7.98097600,
    7.59601562,
    7.22086400,
    6.86157362,
    6.52323200,
    6.20914162,
    5.92000000,
    5.65460200,
    5.41449600,
    5.20073875,
    5.01294400,
    4.84968750,
    4.70891200,
    4.58833225,
    4.48584000,
    4.39990900,
    4.33000000,
    4.27757887,
    4.24595200,
    4.23497388,
    4.24099200,
    4.25804688,
    4.27907200,
    4.29709387,
    4.30643200,
    4.30389887,
    4.29000000,
    4.26848387,
    4.24043200,
    4.20608887,
    4.16603200,
    4.12117188,
    4.07275200,
    4.02234887,
    3.97187200,
    3.92356387,
    3.88000000,
    3.84319188,
    3.81318400,
    3.79258487,
    3.78691200,
    3.80367187,
    3.85144000,
    3.93894087,
    4.07412800,
    4.26326387,
    4.51000000,
    4.81362075,
    5.17028800,
    5.58225150,
    6.05776000,
    6.60890625,
    7.24947200,
    7.99277300,
    8.84950400,
    9.82558375,
    10.92000000,
    12.12700944,
    13.44892800,
    14.88581406,
    16.43283200,
    18.08167969,
    19.82201600,
    21.64288831,
    23.53416000,
    25.48793794,
    27.50000000,
    29.57061744,
    31.69964800,
    33.88185481,
    36.10777600,
    38.36511719,
    40.64014400,
    42.91907456,
    45.18947200,
    47.44163694,
    49.67000000,
    51.87389638,
    54.05273600,
    56.20157688,
    58.31198400,
    60.37335938,
    62.37427200,
    64.30378787,
    66.15280000,
    67.91535838,
    69.59000000,
    71.17616669,
    72.66283200,
    74.04610481,
    75.33171200,
    76.53183594,
    77.66195200,
    78.73766606,
    79.77155200,
    80.76998919,
    81.73000000,
    82.64375688,
    83.51935227,
    84.35919976,
    85.15567334,
    85.89451368,
    86.55823441,
    87.12952842,
    87.59467414,
    87.94694187,
    88.19000000,
    88.33345751,
    88.37111372,
    88.30221714,
    88.13600972,
    87.88846516,
    87.57902706,
    87.22734720,
    86.85002373,
    86.45733945,
    86.05000000)

LAGRANGE_COEFFICIENTS_A = np.array(
    [[0.92625, 0.09750, -0.02375],
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
     [0.02625, 0.99750, -0.02375]])

LAGRANGE_COEFFICIENTS_B = np.array(
    [[-0.0154375, 0.9725625, 0.0511875, -0.0083125],
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
     [-0.0083125, 0.0511875, 0.9725625, -0.0154375]])


class TestLinearInterpolator(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.LinearInterpolator` class units
    tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('x',
                               'y')

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
                np.arange(0, len(POINTS_DATA_A) - 1 + interval, interval)):
            self.assertAlmostEqual(
                LINEAR_INTERPOLATED_POINTS_DATA_A_10_SAMPLES[i],
                linear_interpolator(value),
                places=7)

        np.testing.assert_almost_equal(
            linear_interpolator(
                np.arange(0, len(POINTS_DATA_A) - 1 + interval, interval)),
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
                import traceback
                from colour.utilities import warning

                warning(traceback.format_exc())


class TestSpragueInterpolator(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.SpragueInterpolator` class
    unit tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('x',
                               'y')

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
                np.arange(0, len(POINTS_DATA_A) - 1 + interval, interval)):
            self.assertAlmostEqual(
                SPRAGUE_INTERPOLATED_POINTS_DATA_A_10_SAMPLES[i],
                sprague_interpolator(value),
                places=7)

        np.testing.assert_almost_equal(
            sprague_interpolator(
                np.arange(0, len(POINTS_DATA_A) - 1 + interval, interval)),
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
                import traceback
                from colour.utilities import warning

                warning(traceback.format_exc())


class TestPchipInterpolator(unittest.TestCase):
    """
    Defines :func:`colour.algebra.interpolation.PchipInterpolator` class
    unit tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('x',
                               'y')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(PchipInterpolator))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ()

        for method in required_methods:
            self.assertIn(method, dir(PchipInterpolator))


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
        :attr:`LAGRANGE_COEFFICIENTS_A` and :attr:`LAGRANGE_COEFFICIENTS_B`
        attributes data is matching [1]_.

        References
        ----------
        .. [1]  Fairman, H. S. (1985). The calculation of weight factors for
                tristimulus integration. Color Research & Application, 10(4),
                199–203. doi:10.1002/col.5080100407
        """

        lc = [lagrange_coefficients(i, 3)
              for i in np.linspace(0.05, 0.95, 19)]
        np.testing.assert_almost_equal(lc, LAGRANGE_COEFFICIENTS_A, decimal=7)

        lc = [lagrange_coefficients(i, 4)
              for i in np.linspace(1.05, 1.95, 19)]
        np.testing.assert_almost_equal(lc, LAGRANGE_COEFFICIENTS_B, decimal=7)


if __name__ == '__main__':
    unittest.main()
