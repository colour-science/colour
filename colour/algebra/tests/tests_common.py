# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_common.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`colour.algebra.common` module.

**Others:**

"""

from __future__ import unicode_literals

import sys
import numpy

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import colour.algebra.common

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestGetSteps",
           "TestGetClosest",
           "TestToNdarray",
           "TestIsUniform",
           "TestIsIterable",
           "TestIsNumber",
           "TestIsEvenInteger"]


class TestGetSteps(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.get_steps` definition units tests methods.
    """

    def test_get_steps(self):
        """
        Tests :func:`colour.algebra.common.get_steps` definition.
        """

        self.assertTupleEqual(colour.algebra.common.get_steps(range(0, 10, 2)), (2,))
        self.assertTupleEqual(tuple(sorted(colour.algebra.common.get_steps([1, 2, 3, 4, 6, 6.5]))), (0.5, 1, 2))


class TestGetClosest(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.get_closest` definition units tests methods.
    """

    def test_get_closest(self):
        """
        Tests :func:`colour.algebra.common.get_closest` definition.
        """

        y = numpy.array([24.31357115, 63.62396289, 55.71528816, 62.70988028, 46.84480573, 25.40026416])
        self.assertEqual(colour.algebra.common.get_closest(y, 63.05), 62.70988028)
        self.assertEqual(colour.algebra.common.get_closest(y, 24.90), 25.40026416)
        self.assertEqual(colour.algebra.common.get_closest(y, 51.15), 46.84480573)


class TestToNdarray(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.to_ndarray` definition units tests methods.
    """

    def test_to_ndarray(self):
        """
        Tests :func:`colour.algebra.common.to_ndarray` definition.
        """

        self.assertEqual(colour.algebra.common.to_ndarray(1), numpy.array([1]))
        self.assertEqual(colour.algebra.common.to_ndarray([1]), numpy.array([1]))
        self.assertEqual(colour.algebra.common.to_ndarray((1,)), numpy.array((1,)))
        self.assertEqual(colour.algebra.common.to_ndarray(numpy.array([1])), numpy.array([1]))


class TestIsUniform(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.is_uniform` definition units tests methods.
    """

    def test_is_uniform(self):
        """
        Tests :func:`colour.algebra.common.is_uniform` definition.
        """

        self.assertTrue(colour.algebra.common.is_uniform(range(0, 10, 2)))
        self.assertFalse(colour.algebra.common.is_uniform([1, 2, 3, 4, 6]))


class TestIsIterable(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.is_iterable` definition units tests methods.
    """

    def test_is_iterable(self):
        """
        Tests :func:`colour.algebra.common.is_iterable` definition.
        """

        self.assertTrue(colour.algebra.common.is_iterable(""))
        self.assertTrue(colour.algebra.common.is_iterable(()))
        self.assertTrue(colour.algebra.common.is_iterable([]))
        self.assertTrue(colour.algebra.common.is_iterable(dict()))
        self.assertTrue(colour.algebra.common.is_iterable(numpy.array([])))
        self.assertFalse(colour.algebra.common.is_iterable(1))
        self.assertFalse(colour.algebra.common.is_iterable(2.))


class TestIsNumber(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.is_number` definition units tests methods.
    """

    def test_is_number(self):
        """
        Tests :func:`colour.algebra.common.is_number` definition.
        """

        self.assertTrue(colour.algebra.common.is_number(1))
        self.assertTrue(colour.algebra.common.is_number(1.))
        self.assertTrue(colour.algebra.common.is_number(long(1)))
        self.assertTrue(colour.algebra.common.is_number(complex(1)))
        self.assertFalse(colour.algebra.common.is_number((1,)))
        self.assertFalse(colour.algebra.common.is_number([1]))
        self.assertFalse(colour.algebra.common.is_number("1"))


class TestIsEvenInteger(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.is_even_integer` definition units tests methods.
    """

    def test_is_even_integer(self):
        """
        Tests :func:`colour.algebra.common.is_even_integer` definition.
        """

        self.assertTrue(colour.algebra.common.is_even_integer(1))
        self.assertTrue(colour.algebra.common.is_even_integer(1.001))
        self.assertFalse(colour.algebra.common.is_even_integer(1.01))


if __name__ == "__main__":
    unittest.main()
