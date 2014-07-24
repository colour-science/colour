# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_cie_xyy.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`colour.computation.colourspaces.cie_xyy` module.

**Others:**

"""

from __future__ import unicode_literals

import numpy
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import colour.computation.colourspaces.cie_xyy

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestXYZ_to_xyY",
           "TestxyY_to_XYZ",
           "Testxy_to_XYZ",
           "TestXYZ_to_xy",
           "TestIsWithinMacadamLimits"]


class TestXYZ_to_xyY(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_xyy.XYZ_to_xyY` definition units tests methods.
    """

    def test_XYZ_to_xyY(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_xyy.XYZ_to_xyY` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_xyy.XYZ_to_xyY(
                numpy.array([11.80583421, 10.34, 5.15089229])),
            numpy.array([0.4325, 0.3788, 10.34]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_xyy.XYZ_to_xyY(
                numpy.array([3.08690042, 3.2, 2.68925666])),
            numpy.array([0.3439, 0.3565, 3.20]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_xyy.XYZ_to_xyY(
                numpy.array([0, 0, 0]),
                (0.34567, 0.35850)),
            numpy.array([0.34567, 0.35850, 0]).reshape((3, 1)),
            decimal=7)


class TestxyY_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_xyy.xyY_to_XYZ` definition units tests methods.
    """

    def test_xyY_to_XYZ(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_xyy.xyY_to_XYZ` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_xyy.xyY_to_XYZ(
                numpy.array([0.4325, 0.3788, 10.34])),
            numpy.array([11.80583421, 10.34, 5.15089229]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_xyy.xyY_to_XYZ(
                numpy.array([0.3439, 0.3565, 3.20])),
            numpy.array([3.08690042, 3.2, 2.68925666]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_xyy.xyY_to_XYZ(
                numpy.array([0.4325, 0., 10.34])),
            numpy.array([0., 0., 0.]).reshape((3, 1)),
            decimal=7)


class Testxy_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_xyy.xy_to_XYZ` definition units tests methods.
    """

    def test_xy_to_XYZ(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_xyy.xy_to_XYZ` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_xyy.xy_to_XYZ(
                (0.32207410281368043, 0.3315655001362353)),
            numpy.array([0.97137399, 1., 1.04462134]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_xyy.xy_to_XYZ(
                (0.32174206617150575, 0.337609723160027)),
            numpy.array([0.953, 1.000, 1.009]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_xyy.xy_to_XYZ(
                (0.4474327628361859, 0.4074979625101875)),
            numpy.array([1.098, 1.000, 0.356]).reshape((3, 1)),
            decimal=7)


class TestXYZ_to_xy(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_xyy.XYZ_to_xy` definition units tests methods.
    """

    def test_XYZ_to_xy(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_xyy.XYZ_to_xy` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_xyy.XYZ_to_xy(
                (0.97137399, 1., 1.04462134)),
            (0.32207410281368043, 0.3315655001362353),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_xyy.XYZ_to_xy(
                (0.953, 1.000, 1.009)),
            (0.32174206617150575, 0.337609723160027),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_xyy.XYZ_to_xy(
                (1.098, 1.000, 0.356)),
            (0.4474327628361859, 0.4074979625101875),
            decimal=7)


class TestIsWithinMacadamLimits(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_xyy.is_within_macadam_limits` definition units tests methods.
    """

    def test_is_within_macadam_limits(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_xyy.is_within_macadam_limits` definition.
        """

        self.assertTrue(colour.computation.colourspaces.cie_xyy.is_within_macadam_limits((0.3205, 0.4131, 51), "A"))
        self.assertFalse(colour.computation.colourspaces.cie_xyy.is_within_macadam_limits((0.0005, 0.0031, 0.001), "A"))
        self.assertTrue(colour.computation.colourspaces.cie_xyy.is_within_macadam_limits((0.4325, 0.3788, 10.34), "C"))
        self.assertFalse(colour.computation.colourspaces.cie_xyy.is_within_macadam_limits((0.0025, 0.0088, 0.034), "C"))


if __name__ == "__main__":
    unittest.main()
