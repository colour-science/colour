#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_common.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`color.utilities.common` module.

**Others:**

"""

from __future__ import unicode_literals

import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import color.utilities.common

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestIsUniform"]


class TestGetSteps(unittest.TestCase):
    """
    Defines :func:`color.utilities.common.get_steps` definition units tests methods.
    """

    def test_get_steps(self):
        """
        Tests :func:`color.utilities.common.get_steps` definition.
        """

        self.assertTupleEqual(color.utilities.common.get_steps(range(0, 10, 2)), (2,))
        self.assertTupleEqual(tuple(sorted(color.utilities.common.get_steps([1, 2, 3, 4, 6, 6.5]))), (0.5, 1, 2))


class TestIsUniform(unittest.TestCase):
    """
    Defines :func:`color.utilities.common.is_uniform` definition units tests methods.
    """

    def test_is_uniform(self):
        """
        Tests :func:`color.utilities.common.is_uniform` definition.
        """

        self.assertTrue(color.utilities.common.is_uniform(range(0, 10, 2)))
        self.assertFalse(color.utilities.common.is_uniform([1, 2, 3, 4, 6]))


if __name__ == "__main__":
    unittest.main()
