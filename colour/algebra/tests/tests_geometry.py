#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.algebra.geometry` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.algebra import normalise_vector, line_segments_intersections

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestNormaliseVector',
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
