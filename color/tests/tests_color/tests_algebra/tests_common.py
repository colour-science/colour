# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_common.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`color.algebra.common` module.

**Others:**

"""

from __future__ import unicode_literals

import sys

import numpy


if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import color.algebra.common

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestIsUniform"]


class TestGetSteps(unittest.TestCase):
    """
    Defines :func:`color.algebra.common.get_steps` definition units tests methods.
    """

    def test_get_steps(self):
        """
        Tests :func:`color.algebra.common.get_steps` definition.
        """

        self.assertTupleEqual(color.algebra.common.get_steps(range(0, 10, 2)), (2,))
        self.assertTupleEqual(tuple(sorted(color.algebra.common.get_steps([1, 2, 3, 4, 6, 6.5]))), (0.5, 1, 2))


class TestIsUniform(unittest.TestCase):
    """
    Defines :func:`color.algebra.common.is_uniform` definition units tests methods.
    """

    def test_is_uniform(self):
        """
        Tests :func:`color.algebra.common.is_uniform` definition.
        """

        self.assertTrue(color.algebra.common.is_uniform(range(0, 10, 2)))
        self.assertFalse(color.algebra.common.is_uniform([1, 2, 3, 4, 6]))


class TestGetClosest(unittest.TestCase):
    """
    Defines :func:`color.algebra.common.get_closest` definition units tests methods.
    """

    def test_get_closest(self):
        """
        Tests :func:`color.algebra.common.get_closest` definition.
        """

        y = numpy.array([24.31357115, 63.62396289, 55.71528816, 62.70988028, 46.84480573, 25.40026416])
        self.assertEqual(color.algebra.common.get_closest(y, 63.05), 62.70988028)
        self.assertEqual(color.algebra.common.get_closest(y, 24.90), 25.40026416)
        self.assertEqual(color.algebra.common.get_closest(y, 51.15), 46.84480573)


if __name__ == "__main__":
    unittest.main()
