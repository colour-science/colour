# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_cfs.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`color.spectrum.cfs` module.

**Others:**

"""

from __future__ import unicode_literals

import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from color.spectrum.cfs import LMS_ConeFundamentals

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestLMS_ConeFundamentals"]


class TestLMS_ConeFundamentals(unittest.TestCase):
    """
    Defines :class:`color.spectrum.cfs.LMS_ConeFundamentals` class units tests methods.
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
                               "l_bar",
                               "m_bar",
                               "s_bar")

        for attribute in required_attributes:
            self.assertIn(attribute, dir(LMS_ConeFundamentals))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ("get",
                            "extrapolate",
                            "interpolate",
                            "align",
                            "zeros",
                            "normalize",
                            "clone")

        for method in required_methods:
            self.assertIn(method, dir(LMS_ConeFundamentals))


if __name__ == "__main__":
    unittest.main()
