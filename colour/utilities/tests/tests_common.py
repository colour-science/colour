# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_common.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`colour.utilities.common` module.

**Others:**

"""

from __future__ import unicode_literals

import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import colour.utilities.common

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2008 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestIsString"]


class TestIsString(unittest.TestCase):
    """
    Defines :func:`colour.utilities.common.is_string` definition units tests
    methods.
    """

    def test_is_string(self):
        """
        Tests :func:`colour.utilities.common.is_string` definition.
        """

        self.assertTrue(colour.utilities.common.is_string(str("Hello World!")))
        self.assertTrue(colour.utilities.common.is_string("Hello World!"))
        self.assertTrue(colour.utilities.common.is_string(r"Hello World!"))
        self.assertFalse(colour.utilities.common.is_string(1))
        self.assertFalse(colour.utilities.common.is_string([1]))
        self.assertFalse(colour.utilities.common.is_string({1: None}))


if __name__ == "__main__":
    unittest.main()
