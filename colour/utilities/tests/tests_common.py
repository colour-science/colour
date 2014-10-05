#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.utilities.common` module.
"""

from __future__ import division, unicode_literals

import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import colour.utilities.common

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2008 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestBatch',
           'TestIsString']


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
            list(colour.utilities.common.batch(tuple(range(10)))),
            [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9,)])
        self.assertListEqual(
            list(colour.utilities.common.batch(tuple(range(10)), 5)),
            [(0, 1, 2, 3, 4), (5, 6, 7, 8, 9)])
        self.assertListEqual(
            list(colour.utilities.common.batch(tuple(range(10)), 1)),
            [(0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,)])


class TestIsString(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.is_string` definition unit tests
    methods.
    """

    def test_is_string(self):
        """
        Tests :func:`colour.utilities.common.is_string` definition.
        """

        self.assertTrue(colour.utilities.common.is_string(str('Hello World!')))
        self.assertTrue(colour.utilities.common.is_string('Hello World!'))
        self.assertTrue(colour.utilities.common.is_string(r'Hello World!'))
        self.assertFalse(colour.utilities.common.is_string(1))
        self.assertFalse(colour.utilities.common.is_string([1]))
        self.assertFalse(colour.utilities.common.is_string({1: None}))


if __name__ == '__main__':
    unittest.main()
