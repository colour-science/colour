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
    as_numeric,
    closest,
    normalise,
    steps,
    is_uniform,
    tstack,
    tsplit,
    row_as_diagonal)

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
           'TestTstack',
           'TestTsplit',
           'TestRowAsDiagonal']


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


if __name__ == '__main__':
    unittest.main()
