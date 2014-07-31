# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_matrix.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`colour.algebra.matrix` module.

**Others:**

"""

from __future__ import unicode_literals

import sys

import numpy as np


if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.algebra import is_identity

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestIsIdentity"]


class TestIsIdentity(unittest.TestCase):
    """
    Defines :func:`colour.algebra.matrix.is_identity` definition units tests methods.
    """

    def test_is_identity(self):
        """
        Tests :func:`colour.algebra.matrix.is_identity` definition.
        """

        self.assertTrue(is_identity(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)))
        self.assertFalse(is_identity(np.array([1, 2, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)))
        self.assertTrue(is_identity(np.array([1, 0, 0, 1]).reshape(2, 2), n=2))
        self.assertFalse(is_identity(np.array([1, 2, 0, 1]).reshape(2, 2), n=2))


if __name__ == "__main__":
    unittest.main()
