# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_munsell.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`colour.computation.munsell` module.

**Others:**

"""

from __future__ import unicode_literals

import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import colour.computation.munsell

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestMunsellValue1920",
           "TestMunsellValue1933",
           "TestMunsellValue1943",
           "TestMunsellValue1944",
           "TestMunsellValue1955"]


class TestMunsellValue1920(unittest.TestCase):
    """
    Defines :func:`colour.computation.munsell.munsell_value_1920` definition units tests methods.
    """

    def test_munsell_value_1920(self):
        """
        Tests :func:`colour.computation.munsell.munsell_value_1920` definition.
        """

        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1920(10.08), 3.17490157328, places=7)
        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1920(56.76), 7.53392328073, places=7)
        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1920(98.32), 9.91564420499, places=7)


class TestMunsellValue1933(unittest.TestCase):
    """
    Defines :func:`colour.computation.munsell.munsell_value_1933` definition units tests methods.
    """

    def test_munsell_value_1933(self):
        """
        Tests :func:`colour.computation.munsell.munsell_value_1933` definition.
        """

        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1933(10.08), 3.79183555086, places=7)
        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1933(56.76), 8.27013181776, places=7)
        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1933(98.32), 9.95457710587, places=7)


class TestMunsellValue1943(unittest.TestCase):
    """
    Defines :func:`colour.computation.munsell.munsell_value_1943` definition units tests methods.
    """

    def test_munsell_value_1943(self):
        """
        Tests :func:`colour.computation.munsell.munsell_value_1943` definition.
        """

        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1943(10.08), 3.74629715382, places=7)
        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1943(56.76), 7.8225814259, places=7)
        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1943(98.32), 9.88538236116, places=7)


class TestMunsellValue1944(unittest.TestCase):
    """
    Defines :func:`colour.computation.munsell.munsell_value_1944` definition units tests methods.
    """

    def test_munsell_value_1944(self):
        """
        Tests :func:`colour.computation.munsell.munsell_value_1944` definition.
        """

        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1944(10.08), 3.68650805994, places=7)
        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1944(56.76), 7.89881184275, places=7)
        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1944(98.32), 9.85197100995, places=7)


class TestMunsellValue1955(unittest.TestCase):
    """
    Defines :func:`colour.computation.munsell.munsell_value_1955` definition units tests methods.
    """

    def test_munsell_value_1955(self):
        """
        Tests :func:`colour.computation.munsell.munsell_value_1955` definition.
        """

        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1955(10.08), 3.69528622419, places=7)
        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1955(56.76), 7.84875137062, places=7)
        self.assertAlmostEqual(colour.computation.munsell.munsell_value_1955(98.32), 9.75492813681, places=7)


if __name__ == "__main__":
    unittest.main()
