# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_cmfs.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`color.computation.cmfs` module.

**Others:**

"""

from __future__ import unicode_literals

import numpy
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import color.computation.cmfs
import color.dataset.cmfs
from color.computation.cmfs import LMS_ConeFundamentals
from color.computation.cmfs import RGB_ColorMatchingFunctions
from color.computation.cmfs import XYZ_ColorMatchingFunctions

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestLMS_ConeFundamentals",
           "TestRGB_ColorMatchingFunctions",
           "TestXYZ_ColorMatchingFunctions",
           "TestRGB_2_degree_cmfs_to_XYZ_2_degree_cmfs",
           "TestRGB_10_degree_cmfs_to_XYZ_10_degree_cmfs"]


class TestLMS_ConeFundamentals(unittest.TestCase):
    """
    Defines :class:`color.computation.cmfs.LMS_ConeFundamentals` class units tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ("name",
                               "mapping",
                               "labels",
                               "data",
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


class TestRGB_ColorMatchingFunctions(unittest.TestCase):
    """
    Defines :class:`color.computation.cmfs.RGB_ColorMatchingFunctions` class units tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ("name",
                               "mapping",
                               "labels",
                               "data",
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
                            "normalize",
                            "clone")

        for method in required_methods:
            self.assertIn(method, dir(RGB_ColorMatchingFunctions))


class TestXYZ_ColorMatchingFunctions(unittest.TestCase):
    """
    Defines :class:`color.computation.cmfs.XYZ_ColorMatchingFunctions` class units tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ("name",
                               "mapping",
                               "labels",
                               "data",
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
                            "normalize",
                            "clone")

        for method in required_methods:
            self.assertIn(method, dir(XYZ_ColorMatchingFunctions))


class TestRGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(unittest.TestCase):
    """
    Defines :func:`color.computation.cmfs.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition units tests methods.
    """

    def test_RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(self):
        """
        Tests :func:`color.computation.cmfs.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition.
        """

        # TODO: Change tests for tolerance testing.
        cmfs = color.dataset.cmfs.CMFS.get("CIE 1931 2 Degree Standard Observer")
        numpy.testing.assert_almost_equal(color.computation.cmfs.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(435),
                                          cmfs.get(435),
                                          decimal=2)

        numpy.testing.assert_almost_equal(color.computation.cmfs.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(545),
                                          cmfs.get(545),
                                          decimal=2)

        numpy.testing.assert_almost_equal(color.computation.cmfs.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700),
                                          cmfs.get(700),
                                          decimal=2)


class TestRGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(unittest.TestCase):
    """
    Defines :func:`color.computation.cmfs.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition units tests methods.
    """

    def test_RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(self):
        """
        Tests :func:`color.computation.cmfs.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition.
        """

        # TODO: Change tests for tolerance testing.
        cmfs = color.dataset.cmfs.CMFS.get("CIE 1964 10 Degree Standard Observer")
        numpy.testing.assert_almost_equal(color.computation.cmfs.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(435),
                                          cmfs.get(435),
                                          decimal=1)

        numpy.testing.assert_almost_equal(color.computation.cmfs.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(545),
                                          cmfs.get(545),
                                          decimal=1)

        numpy.testing.assert_almost_equal(color.computation.cmfs.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700),
                                          cmfs.get(700),
                                          decimal=1)


if __name__ == "__main__":
    unittest.main()
