#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.utilities.common` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from collections import OrderedDict

from colour.utilities import (batch, is_iterable, is_string, is_numeric,
                              is_integer, filter_kwargs, first_item)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestBatch', 'TestIsIterable', 'TestIsString', 'TestIsNumeric',
    'TestIsInteger', 'TestFilterKwargs', 'TestFirstItem'
]


class TestBatch(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.batch` definition unit tests
    methods.
    """

    def test_batch(self):
        """
        Tests :func:`colour.utilities.common.batch` definition.
        """

        self.assertListEqual(
            list(batch(tuple(range(10)))),
            [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)])  # yapf: disable

        self.assertListEqual(
            list(batch(tuple(range(10)), 5)),
            [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)])  # yapf: disable

        self.assertListEqual(
            list(batch(tuple(range(10)), 1)),
            [(0,), (1,), (2,), (3,), (4,),
             (5,), (6,), (7,), (8,), (9,)])  # yapf: disable


class TestIsIterable(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.is_iterable` definition unit tests
    methods.
    """

    def test_is_iterable(self):
        """
        Tests :func:`colour.utilities.common.is_iterable` definition.
        """

        self.assertTrue(is_iterable(''))

        self.assertTrue(is_iterable(()))

        self.assertTrue(is_iterable([]))

        self.assertTrue(is_iterable(dict()))

        self.assertTrue(is_iterable(set()))

        self.assertTrue(is_iterable(np.array([])))

        self.assertFalse(is_iterable(1))

        self.assertFalse(is_iterable(2))

        generator = (a for a in range(10))
        self.assertTrue(is_iterable(generator))
        self.assertEqual(len(list(generator)), 10)


class TestIsString(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.is_string` definition unit tests
    methods.
    """

    def test_is_string(self):
        """
        Tests :func:`colour.utilities.common.is_string` definition.
        """

        self.assertTrue(is_string(str('Hello World!')))

        self.assertTrue(is_string('Hello World!'))

        self.assertTrue(is_string(r'Hello World!'))

        self.assertFalse(is_string(1))

        self.assertFalse(is_string([1]))

        self.assertFalse(is_string({1: None}))


class TestIsNumeric(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.is_numeric` definition unit tests
    methods.
    """

    def test_is_numeric(self):
        """
        Tests :func:`colour.utilities.common.is_numeric` definition.
        """

        self.assertTrue(is_numeric(1))

        self.assertTrue(is_numeric(1))

        self.assertTrue(is_numeric(complex(1)))

        self.assertFalse(is_numeric((1, )))

        self.assertFalse(is_numeric([1]))

        self.assertFalse(is_numeric('1'))


class TestIsInteger(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.is_integer` definition units
    tests methods.
    """

    def test_is_integer(self):
        """
        Tests :func:`colour.utilities.common.is_integer` definition.
        """

        self.assertTrue(is_integer(1))

        self.assertTrue(is_integer(1.001))

        self.assertFalse(is_integer(1.01))


class TestFilterKwargs(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.filter_kwargs` definition units
    tests methods.
    """

    def test_filter_kwargs(self):
        """
        Tests :func:`colour.utilities.common.filter_kwargs` definition.
        """

        def fn_a(a):
            """
            :func:`filter_kwargs` unit tests :func:`fn_a`.
            """
            return a

        def fn_b(a, b=0):
            """
            :func:`filter_kwargs` unit tests :func:`fn_b`.
            """

            return a, b

        def fn_c(a, b=0, c=0):
            """
            :func:`filter_kwargs` unit tests :func:`fn_c`.
            """

            return a, b, c

        self.assertEqual(1, fn_a(1, **filter_kwargs(fn_a, b=2, c=3)))

        self.assertTupleEqual((1, 2), fn_b(1, **filter_kwargs(fn_b, b=2, c=3)))

        self.assertTupleEqual((1, 2, 3),
                              fn_c(1, **filter_kwargs(fn_c, b=2, c=3)))


class TestFirstItem(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.first_item` definition units
    tests methods.
    """

    def test_first_item(self):
        """
        Tests :func:`colour.utilities.common.first_item` definition.
        """

        self.assertEqual(first_item(range(10)), 0)

        dictionary = OrderedDict([(0, 'a'), (1, 'b'), (2, 'c')])
        self.assertEqual(first_item(dictionary.items()), (0, 'a'))

        self.assertEqual(first_item(dictionary.values()), 'a')


if __name__ == '__main__':
    unittest.main()
