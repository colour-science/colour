#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.algebra.geometry` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.algebra import (
    normalise_vector,
    euclidean_distance,
    line_segments_intersections)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestNormaliseVector',
           'TestEuclideanDistance',
           'TestLineSegmentsIntersections']


class TestNormaliseVector(unittest.TestCase):
    """
    Defines :func:`colour.algebra.geometry.normalise_vector` definition unit
    tests methods.
    """

    def test_normalise_vector(self):
        """
        Tests :func:`colour.algebra.geometry.normalise_vector` definition.
        """

        np.testing.assert_almost_equal(
            normalise_vector(np.array([0.07049534, 0.10080000, 0.09558313])),
            np.array([0.4525411, 0.6470803, 0.6135908]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise_vector(np.array([0.47097710, 0.34950000, 0.11301649])),
            np.array([0.7885376, 0.5851535, 0.1892189]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise_vector(np.array([0.25506814, 0.19150000, 0.08849752])),
            np.array([0.7705887, 0.5785424, 0.2673607]),
            decimal=7)


class TestEuclideanDistance(unittest.TestCase):
    """
    Defines :func:`colour.algebra.geometry.euclidean_distance` definition unit
    tests methods.
    """

    def test_euclidean_distance(self):
        """
        Tests :func:`colour.algebra.geometry.euclidean_distance` definition.
        """

        self.assertAlmostEqual(
            euclidean_distance(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 426.67945353, 72.39590835])),
            451.713301974,
            places=7)

        self.assertAlmostEqual(
            euclidean_distance(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193])),
            52.6498611564,
            places=7)

        self.assertAlmostEqual(
            euclidean_distance(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716])),
            346.064891718,
            places=7)

    def test_n_dimensional_euclidean_distance(self):
        """
        Tests :func:`colour.algebra.geometry.euclidean_distance` definition
        n-dimensional arrays support.
        """

        xy_1 = np.array([100.00000000, 21.57210357, 272.22819350])
        xy_2 = np.array([100.00000000, 426.67945353, 72.39590835])
        distance = 451.71330197359117
        np.testing.assert_almost_equal(
            euclidean_distance(xy_1, xy_2),
            distance,
            decimal=7)

        xy_1 = np.tile(xy_1, (6, 1))
        xy_2 = np.tile(xy_2, (6, 1))
        distance = np.tile(distance, 6)
        np.testing.assert_almost_equal(
            euclidean_distance(xy_1, xy_2),
            distance,
            decimal=7)

        xy_1 = np.reshape(xy_1, (2, 3, 3))
        xy_2 = np.reshape(xy_2, (2, 3, 3))
        distance = np.reshape(distance, (2, 3))
        np.testing.assert_almost_equal(
            euclidean_distance(xy_1, xy_2),
            distance,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_euclidean_distance(self):
        """
        Tests :func:`colour.algebra.geometry.euclidean_distance` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            xy_1 = np.array(case)
            xy_2 = np.array(case)
            euclidean_distance(xy_1, xy_2)


class TestLineSegmentsIntersections(unittest.TestCase):
    """
    Defines :func:`colour.algebra.geometry.line_segments_intersections`
    definition unit tests methods.
    """

    def test_line_segments_intersections(self):
        """
        Tests :func:`colour.algebra.geometry.line_segments_intersections`
        definition.
        """

        l_1 = np.array([[[0.15416284, 0.7400497],
                         [0.26331502, 0.53373939]],
                        [[0.01457496, 0.91874701],
                         [0.90071485, 0.03342143]]])
        l_2 = np.array([[[0.95694934, 0.13720932],
                         [0.28382835, 0.60608318]],
                        [[0.94422514, 0.85273554],
                         [0.00225923, 0.52122603]],
                        [[0.55203763, 0.48537741],
                         [0.76813415, 0.16071675]],
                        [[0.01457496, 0.91874701],
                         [0.90071485, 0.03342143]]])

        s = line_segments_intersections(l_1, l_2)

        np.testing.assert_almost_equal(
            s.xy,
            np.array([[[np.nan, np.nan],
                       [0.22791841, 0.60064309],
                       [np.nan, np.nan],
                       [np.nan, np.nan]],

                      [[0.42814517, 0.50555685],
                       [0.30560559, 0.62798382],
                       [0.7578749, 0.17613012],
                       [np.nan, np.nan]]]),
            decimal=7)

        np.testing.assert_array_equal(
            s.intersect,
            np.array([[False, True, False, False],
                      [True, True, True, False]], dtype=bool))

        np.testing.assert_array_equal(
            s.parallel,
            np.array([[False, False, False, False],
                      [False, False, False, True]], dtype=bool))

        np.testing.assert_array_equal(
            s.coincident,
            np.array([[False, False, False, False],
                      [False, False, False, True]], dtype=bool))


if __name__ == '__main__':
    unittest.main()
