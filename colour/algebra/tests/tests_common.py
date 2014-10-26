#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.algebra.common` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.algebra import (
    steps,
    closest,
    as_array,
    is_uniform,
    is_iterable,
    is_numeric,
    is_integer,
    normalise)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestSteps',
           'TestClosest',
           'TestAsArray',
           'TestIsUniform',
           'TestIsIterable',
           'TestIsNumeric',
           'TestIsInteger',
           'TestNormalise']


class TestSteps(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.steps` definition unit tests
    methods.
    """

    def test_steps(self):
        """
        Tests :func:`colour.algebra.common.steps` definition.
        """

        self.assertTupleEqual(steps(range(0, 10, 2)), (2,))
        self.assertTupleEqual(
            tuple(sorted(steps([1, 2, 3, 4, 6, 6.5]))),
            (0.5, 1, 2))


class TestClosest(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.closest` definition unit tests
    methods.
    """

    def test_closest(self):
        """
        Tests :func:`colour.algebra.common.closest` definition.
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


class TestAsArray(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.as_array` definition unit tests
    methods.
    """

    def test_as_array(self):
        """
        Tests :func:`colour.algebra.common.as_array` definition.
        """

        self.assertEqual(as_array(1), np.array([1]))
        self.assertEqual(as_array([1]), np.array([1]))
        self.assertEqual(as_array((1,)), np.array((1,)))
        self.assertEqual(as_array(np.array([1])), np.array([1]))
        self.assertTupleEqual(
            as_array(np.array([1, 2, 3]), shape=(3, 1)).shape,
            np.array([[1], [2], [3]]).shape)


class TestIsUniform(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.is_uniform` definition unit tests
    methods.
    """

    def test_is_uniform(self):
        """
        Tests :func:`colour.algebra.common.is_uniform` definition.
        """

        self.assertTrue(is_uniform(range(0, 10, 2)))
        self.assertFalse(is_uniform([1, 2, 3, 4, 6]))


class TestIsIterable(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.is_iterable` definition unit tests
    methods.
    """

    def test_is_iterable(self):
        """
        Tests :func:`colour.algebra.common.is_iterable` definition.
        """

        self.assertTrue(is_iterable(''))
        self.assertTrue(is_iterable(()))
        self.assertTrue(is_iterable([]))
        self.assertTrue(is_iterable(dict()))
        self.assertTrue(is_iterable(np.array([])))
        self.assertFalse(is_iterable(1))
        self.assertFalse(is_iterable(2))


class TestIsNumeric(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.is_numeric` definition unit tests
    methods.
    """

    def test_is_numeric(self):
        """
        Tests :func:`colour.algebra.common.is_numeric` definition.
        """

        self.assertTrue(is_numeric(1))
        self.assertTrue(is_numeric(1))
        self.assertTrue(is_numeric(complex(1)))
        self.assertFalse(is_numeric((1,)))
        self.assertFalse(is_numeric([1]))
        self.assertFalse(is_numeric('1'))


class TestIsInteger(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.is_integer` definition units
    tests methods.
    """

    def test_is_integer(self):
        """
        Tests :func:`colour.algebra.common.is_integer` definition.
        """

        self.assertTrue(is_integer(1))
        self.assertTrue(is_integer(1.001))
        self.assertFalse(is_integer(1.01))


class TestNormalise(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.normalise` definition units
    tests methods.
    """

    def test_normalise(self):
        """
        Tests :func:`colour.algebra.common.normalise` definition.
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


if __name__ == '__main__':
    unittest.main()
