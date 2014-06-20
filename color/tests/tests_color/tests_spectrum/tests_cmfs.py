# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_cmfs.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`color.spectrum.cmfs` module.

**Others:**

"""

from __future__ import unicode_literals

import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from color.spectrum.cmfs import RGB_ColorMatchingFunctions
from color.spectrum.cmfs import XYZ_ColorMatchingFunctions

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestRGB_ColorMatchingFunctions",
           "TestXYZ_ColorMatchingFunctions"]


class TestRGB_ColorMatchingFunctions(unittest.TestCase):
    """
    Defines :class:`color.spectrum.cmfs.RGB_ColorMatchingFunctions` class units tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ("name",
                               "mapping",
                               "labels",
                               "triad",
                               "x",
                               "y",
                               "z",
                               "wavelengths",
                               "values",
                               "shape",
                               "r_bar",
                               "g_bar",
                               "b_bar")

        for attribute in required_attributes:
            self.assertIn(attribute, dir(RGB_ColorMatchingFunctions))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ("get",
                            "extrapolate",
                            "interpolate",
                            "align",
                            "zeros",
                            "clone")

        for method in required_methods:
            self.assertIn(method, dir(RGB_ColorMatchingFunctions))


class TestXYZ_ColorMatchingFunctions(unittest.TestCase):
    """
    Defines :class:`color.spectrum.cmfs.XYZ_ColorMatchingFunctions` class units tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ("name",
                               "mapping",
                               "labels",
                               "triad",
                               "x",
                               "y",
                               "z",
                               "wavelengths",
                               "values",
                               "shape",
                               "x_bar",
                               "y_bar",
                               "z_bar")

        for attribute in required_attributes:
            self.assertIn(attribute, dir(XYZ_ColorMatchingFunctions))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ("get",
                            "extrapolate",
                            "interpolate",
                            "align",
                            "zeros",
                            "clone")

        for method in required_methods:
            self.assertIn(method, dir(XYZ_ColorMatchingFunctions))


if __name__ == "__main__":
    unittest.main()
