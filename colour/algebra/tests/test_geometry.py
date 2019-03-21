# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.algebra.geometry` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.algebra import (
    normalise_vector, euclidean_distance, extend_line_segment,
    intersect_line_segments, ellipse_coefficients_general_form,
    ellipse_coefficients_canonical_form, point_at_angle_on_ellipse,
    ellipse_fitting_Halir1998)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestNormaliseVector', 'TestEuclideanDistance', 'TestExtendLineSegment',
    'TestIntersectLineSegments', 'TestEllipseCoefficientsCanonicalForm',
    'TestEllipseCoefficientsGeneralForm', 'TestPointAtAngleOnEllipse',
    'TestEllipseFittingHalir1998'
]


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
            normalise_vector(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.84197033, 0.49722560, 0.20941026]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise_vector(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.48971705, 0.79344877, 0.36140872]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise_vector(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.26229003, 0.20655044, 0.94262445]),
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
            451.71330197,
            places=7)

        self.assertAlmostEqual(
            euclidean_distance(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193])),
            52.64986116,
            places=7)

        self.assertAlmostEqual(
            euclidean_distance(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716])),
            346.06489172,
            places=7)

    def test_n_dimensional_euclidean_distance(self):
        """
        Tests :func:`colour.algebra.geometry.euclidean_distance` definition
        n-dimensional arrays support.
        """

        a = np.array([100.00000000, 21.57210357, 272.22819350])
        b = np.array([100.00000000, 426.67945353, 72.39590835])
        distance = 451.71330197
        np.testing.assert_almost_equal(
            euclidean_distance(a, b), distance, decimal=7)

        a = np.tile(a, (6, 1))
        b = np.tile(b, (6, 1))
        distance = np.tile(distance, 6)
        np.testing.assert_almost_equal(
            euclidean_distance(a, b), distance, decimal=7)

        a = np.reshape(a, (2, 3, 3))
        b = np.reshape(b, (2, 3, 3))
        distance = np.reshape(distance, (2, 3))
        np.testing.assert_almost_equal(
            euclidean_distance(a, b), distance, decimal=7)

    @ignore_numpy_errors
    def test_nan_euclidean_distance(self):
        """
        Tests :func:`colour.algebra.geometry.euclidean_distance` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            a = np.array(case)
            b = np.array(case)
            euclidean_distance(a, b)


class TestExtendLineSegment(unittest.TestCase):
    """
    Defines :func:`colour.algebra.geometry.extend_line_segment` definition unit
    tests methods.
    """

    def test_extend_line_segment(self):
        """
        Tests :func:`colour.algebra.geometry.extend_line_segment` definition.
        """

        np.testing.assert_almost_equal(
            extend_line_segment(
                np.array([0.95694934, 0.13720932]),
                np.array([0.28382835, 0.60608318])),
            np.array([-0.5367248, 1.17765341]),
            decimal=7)

        np.testing.assert_almost_equal(
            extend_line_segment(
                np.array([0.95694934, 0.13720932]),
                np.array([0.28382835, 0.60608318]), 5),
            np.array([-3.81893739, 3.46393435]),
            decimal=7)

        np.testing.assert_almost_equal(
            extend_line_segment(
                np.array([0.95694934, 0.13720932]),
                np.array([0.28382835, 0.60608318]), -1),
            np.array([1.1043815, 0.03451295]),
            decimal=7)


class TestIntersectLineSegments(unittest.TestCase):
    """
    Defines :func:`colour.algebra.geometry.intersect_line_segments` definition
    unit tests methods.
    """

    def test_intersect_line_segments(self):
        """
        Tests :func:`colour.algebra.geometry.intersect_line_segments`
        definition.
        """

        l_1 = np.array([
            [[0.15416284, 0.7400497], [0.26331502, 0.53373939]],
            [[0.01457496, 0.91874701], [0.90071485, 0.03342143]],
        ])
        l_2 = np.array([
            [[0.95694934, 0.13720932], [0.28382835, 0.60608318]],
            [[0.94422514, 0.85273554], [0.00225923, 0.52122603]],
            [[0.55203763, 0.48537741], [0.76813415, 0.16071675]],
            [[0.01457496, 0.91874701], [0.90071485, 0.03342143]],
        ])

        s = intersect_line_segments(l_1, l_2)

        np.testing.assert_almost_equal(
            s.xy,
            np.array([[[np.nan, np.nan], [0.22791841, 0.60064309],
                       [np.nan, np.nan], [np.nan, np.nan]],
                      [[0.42814517, 0.50555685], [0.30560559, 0.62798382],
                       [0.7578749, 0.17613012], [np.nan, np.nan]]]),
            decimal=7)

        np.testing.assert_array_equal(
            s.intersect,
            np.array([[False, True, False, False], [True, True, True, False]]))

        np.testing.assert_array_equal(
            s.parallel,
            np.array([[False, False, False, False],
                      [False, False, False, True]]))

        np.testing.assert_array_equal(
            s.coincident,
            np.array([[False, False, False, False],
                      [False, False, False, True]]))


class TestEllipseCoefficientsCanonicalForm(unittest.TestCase):
    """
    Defines :func:`colour.algebra.geometry.ellipse_coefficients_canonical_form`
    definition unit tests methods.
    """

    def test_ellipse_coefficients_canonical_form(self):
        """
        Tests :func:`colour.algebra.geometry.\
ellipse_coefficients_canonical_form` definition.
        """

        np.testing.assert_almost_equal(
            ellipse_coefficients_canonical_form(
                np.array([2.5, -3.0, 2.5, -1.0, -1.0, -3.5])),
            np.array([0.5, 0.5, 2, 1, 45]),
            decimal=7)

        np.testing.assert_almost_equal(
            ellipse_coefficients_canonical_form(
                np.array([1.0, 0.0, 1.0, 0.0, 0.0, -1.0])),
            np.array([0.0, 0.0, 1, 1, 0]),
            decimal=7)


class TestEllipseCoefficientsGeneralForm(unittest.TestCase):
    """
    Defines :func:`colour.algebra.geometry.ellipse_coefficients_general_form`
    definition unit tests methods.
    """

    def test_ellipse_coefficients_general_form(self):
        """
        Tests :func:`colour.algebra.geometry.ellipse_coefficients_general_form`
        definition.
        """

        np.testing.assert_almost_equal(
            ellipse_coefficients_general_form(np.array([0.5, 0.5, 2, 1, 45])),
            np.array([2.5, -3.0, 2.5, -1.0, -1.0, -3.5]),
            decimal=7)

        np.testing.assert_almost_equal(
            ellipse_coefficients_general_form(np.array([0.0, 0.0, 1, 1, 0])),
            np.array([1.0, 0.0, 1.0, 0.0, 0.0, -1.0]),
            decimal=7)


class TestPointAtAngleOnEllipse(unittest.TestCase):
    """
    Defines :func:`colour.algebra.geometry.point_at_angle_on_ellipse`
    definition unit tests methods.
    """

    def test_point_at_angle_on_ellipse(self):
        """
        Tests :func:`colour.algebra.geometry.point_at_angle_on_ellipse`
        definition.
        """

        np.testing.assert_almost_equal(
            point_at_angle_on_ellipse(
                np.array([0, 90, 180, 270]), np.array([0.0, 0.0, 2, 1, 0])),
            np.array([[2, 0], [0, 1], [-2, 0], [0, -1]]),
            decimal=7)

        np.testing.assert_almost_equal(
            point_at_angle_on_ellipse(
                np.linspace(0, 360, 10), np.array([0.5, 0.5, 2, 1, 45])),
            np.array([
                [1.91421356, 1.91421356],
                [1.12883096, 2.03786992],
                [0.04921137, 1.44193985],
                [-0.81947922, 0.40526565],
                [-1.07077081, -0.58708129],
                [-0.58708129, -1.07077081],
                [0.40526565, -0.81947922],
                [1.44193985, 0.04921137],
                [2.03786992, 1.12883096],
                [1.91421356, 1.91421356],
            ]),
            decimal=7)


class TestEllipseFittingHalir1998(unittest.TestCase):
    """
    Defines :func:`colour.algebra.geometry.ellipse_fitting_Halir1998`
    definition unit tests methods.
    """

    def test_ellipse_fitting_Halir1998(self):
        """
        Tests :func:`colour.algebra.geometry.ellipse_fitting_Halir1998`
        definition.
        """

        np.testing.assert_almost_equal(
            ellipse_fitting_Halir1998(
                np.array([[2, 0], [0, 1], [-2, 0], [0, -1]])),
            np.array([
                0.24253563,
                0.00000000,
                0.97014250,
                0.00000000,
                0.00000000,
                -0.97014250,
            ]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
