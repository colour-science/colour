#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.utilities.array` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.utilities import (
    as_array,
    as_numeric,
    as_stack,
    as_shape,
    auto_axis,
    closest,
    normalise,
    steps,
    is_uniform,
    row_as_diagonal)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestAsArray',
           'TestAsNumeric',
           'TestAsStack',
           'TestAsShape',
           'TestAutoAxis',
           'TestClosest',
           'TestNormalise',
           'TestSteps',
           'TestIsUniform',
           'TestRowAsDiagonal']


class TestAsArray(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.as_array` definition unit tests
    methods.
    """

    def test_as_array(self):
        """
        Tests :func:`colour.utilities.array.as_array` definition.
        """

        self.assertEqual(as_array(1), np.array([1]))
        self.assertEqual(as_array([1]), np.array([1]))
        self.assertEqual(as_array((1,)), np.array((1,)))
        self.assertEqual(as_array(np.array([1])), np.array([1]))
        self.assertTupleEqual(
            as_array(np.array([1, 2, 3]), shape=(3, 1)).shape,
            np.array([[1], [2], [3]]).shape)


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


class TestAsStack(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.as_stack` definition unit tests
    methods.
    """

    def test_as_stack(self):
        """
        Tests :func:`colour.utilities.array.as_stack` definition.
        """

        x = np.array([1, 2, 3])
        y = np.array([4, 5, 6])
        np.testing.assert_almost_equal(as_stack((x, y)),
                                       np.array([[[1, 4],
                                                  [2, 5],
                                                  [3, 6]]]))
        np.testing.assert_almost_equal(as_stack((x, y)), as_stack((x, y), 'D'))
        np.testing.assert_almost_equal(as_stack((x, y), 'Horizontal'),
                                       np.array([1, 2, 3, 4, 5, 6]))
        np.testing.assert_almost_equal(as_stack((x, y), 'Horizontal'),
                                       as_stack((x, y), 'H'))
        np.testing.assert_almost_equal(as_stack((x, y), 'Vertical'),
                                       np.array([[1, 2, 3],
                                                 [4, 5, 6]]))

        np.testing.assert_almost_equal(as_stack((x, y)),
                                       np.array([[[1, 4],
                                                  [2, 5],
                                                  [3, 6]]]))
        np.testing.assert_almost_equal(as_stack((x, y), shape=(2, 3)),
                                       np.array([[1, 4, 2],
                                                 [5, 3, 6]]))


class TestAsShape(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.as_shape` definition unit tests
    methods.
    """

    def test_as_shape(self):
        """
        Tests :func:`colour.utilities.array.as_shape` definition.
        """

        self.assertTupleEqual(as_shape(np.array([1, 2, 3])), (3,))
        self.assertTupleEqual(as_shape(1), (1,))
        self.assertTupleEqual(as_shape(None), (1,))
        self.assertTupleEqual(as_shape('Nemo'), (1,))
        self.assertTupleEqual(as_shape([[[1, 2, 3], [4, 5, 6]]]), (1, 2, 3))


class TestAutoAxis(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.auto_axis` definition unit tests
    methods.
    """

    def test_auto_axis(self):
        """
        Tests :func:`colour.utilities.array.auto_axis` definition.
        """

        self.assertIsNone(auto_axis(None))
        self.assertTupleEqual(auto_axis((3, 3)), (3, -1))
        self.assertTupleEqual(auto_axis((3,)), (-1,))
        self.assertTupleEqual(auto_axis((3, 3), 'L'), (3, -1))
        self.assertTupleEqual(auto_axis((3,), 'L'), (-1,))
        self.assertTupleEqual(auto_axis((3, 3), 'First'), (-1, 3))
        self.assertTupleEqual(auto_axis((3,), 'First'), (-1,))
        self.assertTupleEqual(auto_axis((3, 3), 'F'), (-1, 3))
        self.assertTupleEqual(auto_axis((3,), 'F'), (-1,))


class TestClosest(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.closest` definition unit tests
    methods.
    """

    def test_closest(self):
        """
        Tests :func:`colour.utilities.array.closest` definition.
        """

        y = np.array(
            [24.31357115,
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
            normalise(np.array([0.1151847498, 0.1008, 0.0508937252])),
            np.array([1., 0.87511585, 0.4418443]),
            decimal=7)
        np.testing.assert_almost_equal(
            normalise(np.array([0.1151847498, 0.1008, 0.0508937252]),
                      factor=10),
            np.array([10., 8.75115848, 4.418443]),
            decimal=7)
        np.testing.assert_almost_equal(
            normalise(np.array([-0.1151847498, -0.1008, 0.0508937252])),
            np.array([0., 0., 1.]),
            decimal=7)
        np.testing.assert_almost_equal(
            normalise(np.array([-0.1151847498, -0.1008, 0.0508937252]),
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

        self.assertTupleEqual(steps(range(0, 10, 2)), (2,))
        self.assertTupleEqual(
            tuple(sorted(steps([1, 2, 3, 4, 6, 6.5]))),
            (0.5, 1, 2))


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


if __name__ == '__main__':
    unittest.main()
