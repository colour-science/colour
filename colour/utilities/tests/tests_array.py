#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.utilities.array` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.utilities import (
    as_numeric,
    closest,
    normalise,
    steps,
    is_uniform,
    in_array,
    tstack,
    tsplit,
    row_as_diagonal,
    dot_vector,
    dot_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestAsNumeric',
           'TestClosest',
           'TestNormalise',
           'TestSteps',
           'TestIsUniform',
           'TestInArray',
           'TestTstack',
           'TestTsplit',
           'TestRowAsDiagonal',
           'TestDotVector',
           'TestDotMatrix']


class TestAsNumeric(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.as_numeric` definition unit tests
    methods.
    """

    def test_as_numeric(self):
        """
        Tests :func:`colour.utilities.array.as_numeric` definition.
        """

        self.assertEqual(as_numeric(1), 1)

        self.assertEqual(as_numeric(np.array([1])), 1)

        np.testing.assert_almost_equal(as_numeric(np.array([1, 2, 3])),
                                       np.array([1, 2, 3]))


class TestClosest(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.closest` definition unit tests
    methods.
    """

    def test_closest(self):
        """
        Tests :func:`colour.utilities.array.closest` definition.
        """

        y = np.array([24.31357115,
                      63.62396289,
                      55.71528816,
                      62.70988028,
                      46.84480573,
                      25.40026416])

        self.assertEqual(closest(y, 63.05), 62.70988028)

        self.assertEqual(closest(y, 24.90), 25.40026416)

        self.assertEqual(closest(y, 51.15), 46.84480573)


class TestNormalise(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.normalise` definition units
    tests methods.
    """

    def test_normalise(self):
        """
        Tests :func:`colour.utilities.array.normalise` definition.
        """

        np.testing.assert_almost_equal(
            normalise(np.array([0.1151847498, 0.1008000000, 0.0508937252])),
            np.array([1., 0.87511585, 0.4418443]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise(np.array([[0.1151847498, 0.1008000000, 0.0508937252],
                                [0.0704953400, 0.1008000000, 0.0955831300],
                                [0.1750135800, 0.3881879500, 0.3216195500]])),
            np.array([[0.29672418, 0.25966803, 0.13110589],
                      [0.18160105, 0.25966803, 0.246229],
                      [0.45084753, 1., 0.82851503]]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise(np.array([[0.1151847498, 0.1008000000, 0.0508937252],
                                [0.0704953400, 0.1008000000, 0.0955831300],
                                [0.1750135800, 0.3881879500, 0.3216195500]]),
                      axis=-1),
            np.array([[1., 0.87511585, 0.4418443],
                      [0.69935853, 1., 0.94824534],
                      [0.45084753, 1., 0.82851503]]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise(np.array([0.1151847498, 0.1008000000, 0.0508937252]),
                      factor=10),
            np.array([10., 8.75115848, 4.418443]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise(np.array([-0.1151847498, -0.1008000000, 0.0508937252])),
            np.array([0., 0., 1.]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise(np.array([-0.1151847498, -0.1008000000, 0.0508937252]),
                      clip=False),
            np.array([-2.26324069, -1.9805978, 1.]),
            decimal=7)


class TestSteps(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.steps` definition unit tests
    methods.
    """

    def test_steps(self):
        """
        Tests :func:`colour.utilities.array.steps` definition.
        """

        np.testing.assert_almost_equal(steps(range(0, 10, 2)), np.array([2]))

        np.testing.assert_almost_equal(
            steps([1, 2, 3, 4, 6, 6.5]),
            np.array([0.5, 1, 2]))


class TestIsUniform(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.is_uniform` definition unit tests
    methods.
    """

    def test_is_uniform(self):
        """
        Tests :func:`colour.utilities.array.is_uniform` definition.
        """

        self.assertTrue(is_uniform(range(0, 10, 2)))

        self.assertFalse(is_uniform([1, 2, 3, 4, 6]))


class TestInArray(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.in_array` definition unit tests
    methods.
    """

    def test_in_array(self):
        """
        Tests :func:`colour.utilities.array.in_array` definition.
        """

        self.assertTrue(
            np.array_equal(
                in_array(np.array([0.50, 0.60]),
                         np.linspace(0, 10, 101)),
                np.array([True, True])))

        self.assertFalse(
            np.array_equal(
                in_array(np.array([0.50, 0.61]),
                         np.linspace(0, 10, 101)),
                np.array([True, True])))

        self.assertTrue(
            np.array_equal(
                in_array(np.array([[0.50], [0.60]]),
                         np.linspace(0, 10, 101)),
                np.array([[True], [True]])))

    def test_n_dimensional_in_array(self):
        """
        Tests :func:`colour.utilities.array.in_array` definition n-dimensions
        support.
        """

        np.testing.assert_almost_equal(
            in_array(np.array([0.50, 0.60]),
                     np.linspace(0, 10, 101)).shape,
            np.array([2]))

        np.testing.assert_almost_equal(
            in_array(np.array([[0.50, 0.60]]),
                     np.linspace(0, 10, 101)).shape,
            np.array([1, 2]))

        np.testing.assert_almost_equal(
            in_array(np.array([[0.50], [0.60]]),
                     np.linspace(0, 10, 101)).shape,
            np.array([2, 1]))


class TestTstack(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.tstack` definition unit tests
    methods.
    """

    def test_tstack(self):
        """
        Tests :func:`colour.utilities.array.tstack` definition.
        """

        a = 0
        np.testing.assert_almost_equal(
            tstack((a, a, a)),
            np.array([0, 0, 0]))

        a = np.arange(0, 6)
        np.testing.assert_almost_equal(
            tstack((a, a, a)),
            np.array([[0, 0, 0],
                      [1, 1, 1],
                      [2, 2, 2],
                      [3, 3, 3],
                      [4, 4, 4],
                      [5, 5, 5]]))

        a = np.reshape(a, (1, 6))
        np.testing.assert_almost_equal(
            tstack((a, a, a)),
            np.array([[[0, 0, 0],
                       [1, 1, 1],
                       [2, 2, 2],
                       [3, 3, 3],
                       [4, 4, 4],
                       [5, 5, 5]]]))

        a = np.reshape(a, (1, 2, 3))
        np.testing.assert_almost_equal(
            tstack((a, a, a)),
            np.array([[[[0, 0, 0],
                        [1, 1, 1],
                        [2, 2, 2]],
                       [[3, 3, 3],
                        [4, 4, 4],
                        [5, 5, 5]]]]))


class TestTsplit(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.tsplit` definition unit tests
    methods.
    """

    def test_tsplit(self):
        """
        Tests :func:`colour.utilities.array.tsplit` definition.
        """

        a = np.array([0, 0, 0])
        np.testing.assert_almost_equal(tsplit(a),
                                       np.array([0, 0, 0]))
        a = np.array([[0, 0, 0],
                      [1, 1, 1],
                      [2, 2, 2],
                      [3, 3, 3],
                      [4, 4, 4],
                      [5, 5, 5]])
        np.testing.assert_almost_equal(
            tsplit(a),
            np.array([[0, 1, 2, 3, 4, 5],
                      [0, 1, 2, 3, 4, 5],
                      [0, 1, 2, 3, 4, 5]]))

        a = np.array([[[0, 0, 0],
                       [1, 1, 1],
                       [2, 2, 2],
                       [3, 3, 3],
                       [4, 4, 4],
                       [5, 5, 5]]])
        np.testing.assert_almost_equal(
            tsplit(a),
            np.array([[[0, 1, 2, 3, 4, 5]],
                      [[0, 1, 2, 3, 4, 5]],
                      [[0, 1, 2, 3, 4, 5]]]))

        a = np.array([[[[0, 0, 0],
                        [1, 1, 1],
                        [2, 2, 2]],
                       [[3, 3, 3],
                        [4, 4, 4],
                        [5, 5, 5]]]])
        np.testing.assert_almost_equal(
            tsplit(a),
            np.array([[[[0, 1, 2],
                        [3, 4, 5]]],
                      [[[0, 1, 2],
                        [3, 4, 5]]],
                      [[[0, 1, 2],
                        [3, 4, 5]]]]))


class TestRowAsDiagonal(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.row_as_diagonal` definition unit
    tests methods.
    """

    def test_row_as_diagonal(self):
        """
        Tests :func:`colour.utilities.array.row_as_diagonal` definition.
        """

        np.testing.assert_almost_equal(
            row_as_diagonal(np.array([[0.25891593, 0.07299478, 0.36586996],
                                      [0.30851087, 0.37131459, 0.16274825],
                                      [0.71061831, 0.67718718, 0.09562581],
                                      [0.71588836, 0.76772047, 0.15476079],
                                      [0.92985142, 0.22263399, 0.88027331]])),
            np.array([[[0.25891593, 0., 0.],
                       [0., 0.07299478, 0.],
                       [0., 0., 0.36586996]],
                      [[0.30851087, 0., 0.],
                       [0., 0.37131459, 0.],
                       [0., 0., 0.16274825]],
                      [[0.71061831, 0., 0.],
                       [0., 0.67718718, 0.],
                       [0., 0., 0.09562581]],
                      [[0.71588836, 0., 0.],
                       [0., 0.76772047, 0.],
                       [0., 0., 0.15476079]],
                      [[0.92985142, 0., 0.],
                       [0., 0.22263399, 0.],
                       [0., 0., 0.88027331]]]))


class TestDotVector(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.dot_vector` definition unit tests
    methods.
    """

    def test_dot_vector(self):
        """
        Tests :func:`colour.utilities.array.dot_vector` definition.
        """

        m = np.array([[0.7328, 0.4296, -0.1624],
                      [-0.7036, 1.6975, 0.0061],
                      [0.0030, 0.0136, 0.9834]])
        m = np.reshape(np.tile(m, (6, 1)), (6, 3, 3))

        v = np.array([0.07049534, 0.10080000, 0.09558313])
        v = np.tile(v, (6, 1))

        np.testing.assert_almost_equal(
            dot_vector(m, v),
            np.array([[0.07943996, 0.12209054, 0.09557882],
                      [0.07943996, 0.12209054, 0.09557882],
                      [0.07943996, 0.12209054, 0.09557882],
                      [0.07943996, 0.12209054, 0.09557882],
                      [0.07943996, 0.12209054, 0.09557882],
                      [0.07943996, 0.12209054, 0.09557882]]),
            decimal=7)


class TestDotMatrix(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.dot_matrix` definition unit tests
    methods.
    """

    def test_dot_matrix(self):
        """
        Tests :func:`colour.utilities.array.dot_matrix` definition.
        """

        a = np.array([[0.7328, 0.4296, -0.1624],
                      [-0.7036, 1.6975, 0.0061],
                      [0.0030, 0.0136, 0.9834]])
        a = np.reshape(np.tile(a, (6, 1)), (6, 3, 3))

        b = a

        np.testing.assert_almost_equal(
            dot_matrix(a, b),
            np.array([[[0.23424208, 1.04184824, -0.27609032],
                       [-1.70994078, 2.57932265, 0.13061813],
                       [-0.00442036, 0.03774904, 0.96667132]],
                      [[0.23424208, 1.04184824, -0.27609032],
                       [-1.70994078, 2.57932265, 0.13061813],
                       [-0.00442036, 0.03774904, 0.96667132]],
                      [[0.23424208, 1.04184824, -0.27609032],
                       [-1.70994078, 2.57932265, 0.13061813],
                       [-0.00442036, 0.03774904, 0.96667132]],
                      [[0.23424208, 1.04184824, -0.27609032],
                       [-1.70994078, 2.57932265, 0.13061813],
                       [-0.00442036, 0.03774904, 0.96667132]],
                      [[0.23424208, 1.04184824, -0.27609032],
                       [-1.70994078, 2.57932265, 0.13061813],
                       [-0.00442036, 0.03774904, 0.96667132]],
                      [[0.23424208, 1.04184824, -0.27609032],
                       [-1.70994078, 2.57932265, 0.13061813],
                       [-0.00442036, 0.03774904, 0.96667132]]]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
