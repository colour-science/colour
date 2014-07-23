# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_lightness.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`colour.computation.lightness` module.

**Others:**

"""

from __future__ import unicode_literals

import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import colour.computation.lightness

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestLightness1958",
           "TestLightness1964",
           "TestLightness1976"]


class TestLightness1958(unittest.TestCase):
    """
    Defines :func:`colour.computation.lightness.lightness_glasser1958` definition units tests methods.
    """

    def test_lightness_glasser1958(self):
        """
        Tests :func:`colour.computation.lightness.lightness_glasser1958` definition.
        """

        self.assertAlmostEqual(colour.computation.lightness.lightness_glasser1958(10.08), 36.2505626458, places=7)
        self.assertAlmostEqual(colour.computation.lightness.lightness_glasser1958(56.76), 78.8117999039, places=7)
        self.assertAlmostEqual(colour.computation.lightness.lightness_glasser1958(98.32), 98.3447052593, places=7)


class TestLightness1964(unittest.TestCase):
    """
    Defines :func:`colour.computation.lightness.lightness_wyszecki1964` definition units tests methods.
    """

    def test_lightness_wyszecki1964(self):
        """
        Tests :func:`colour.computation.lightness.lightness_wyszecki1964` definition.
        """

        self.assertAlmostEqual(colour.computation.lightness.lightness_wyszecki1964(10.08), 37.0041149128, places=7)
        self.assertAlmostEqual(colour.computation.lightness.lightness_wyszecki1964(56.76), 79.0773031869, places=7)
        self.assertAlmostEqual(colour.computation.lightness.lightness_wyszecki1964(98.32), 98.3862250488, places=7)


class TestLightness1976(unittest.TestCase):
    """
    Defines :func:`colour.computation.lightness.lightness_1976` definition units tests methods.
    """

    def test_lightness_1976(self):
        """
        Tests :func:`colour.computation.lightness.lightness_1976` definition.
        """

        self.assertAlmostEqual(colour.computation.lightness.lightness_1976(10.08, 100.), 37.9856290977, places=7)
        self.assertAlmostEqual(colour.computation.lightness.lightness_1976(56.76, 100.), 80.0444155585, places=7)
        self.assertAlmostEqual(colour.computation.lightness.lightness_1976(98.32, 100.), 99.3467279026, places=7)


if __name__ == "__main__":
    unittest.main()
