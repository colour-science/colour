# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_transformations.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`colour.colorimetry.transformations` module.

**Others:**

"""

from __future__ import unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.colorimetry import CMFS
from colour.colorimetry import RGB_10_degree_cmfs_to_LMS_10_degree_cmfs
from colour.colorimetry import RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs
from colour.colorimetry import RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs
from colour.colorimetry import LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs
from colour.colorimetry import LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestRGB_2_degree_cmfs_to_XYZ_2_degree_cmfs",
           "TestRGB_10_degree_cmfs_to_XYZ_10_degree_cmfs",
           "TestRGB_10_degree_cmfs_to_LMS_10_degree_cmfs",
           "TestLMS_2_degree_cmfs_to_XYZ_2_degree_cmfs",
           "TestLMS_10_degree_cmfs_to_XYZ_10_degree_cmfs"]


class TestRGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.transformations.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs`
    definition units tests methods.
    """

    def test_RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(self):
        """
        Tests :func:`colour.colorimetry.transformations.RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs`
        definition.
        """

        # TODO: Update test to tolerance matching.
        cmfs = CMFS.get("CIE 1931 2 Degree Standard Observer")
        np.testing.assert_almost_equal(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(435),
            cmfs.get(435),
            decimal=2)

        np.testing.assert_almost_equal(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(545),
            cmfs.get(545),
            decimal=2)

        np.testing.assert_almost_equal(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700),
            cmfs.get(700),
            decimal=2)


class TestRGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.transformations.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs`
    definition units tests methods.
    """

    def test_RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(self):
        """
        Tests :func:`colour.colorimetry.transformations.RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs`
        definition.
        """

        # TODO: Update test to tolerance matching.
        cmfs = CMFS.get("CIE 1964 10 Degree Standard Observer")
        np.testing.assert_almost_equal(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(435),
            cmfs.get(435),
            decimal=1)

        np.testing.assert_almost_equal(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(545),
            cmfs.get(545),
            decimal=1)

        np.testing.assert_almost_equal(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700),
            cmfs.get(700),
            decimal=1)


class TestRGB_10_degree_cmfs_to_LMS_10_degree_cmfs(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.transformations.RGB_10_degree_cmfs_to_LMS_10_degree_cmfs`
    definition units tests methods.
    """

    def test_RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(self):
        """
        Tests :func:`colour.colorimetry.transformations.RGB_10_degree_cmfs_to_LMS_10_degree_cmfs`
        definition.
        """

        # TODO: Update test to tolerance matching.
        cmfs = CMFS.get("Stockman & Sharpe 10 Degree Cone Fundamentals")
        np.testing.assert_almost_equal(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(435),
            cmfs.get(435),
            decimal=1)

        np.testing.assert_almost_equal(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(545),
            cmfs.get(545),
            decimal=1)

        np.testing.assert_almost_equal(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700),
            cmfs.get(700),
            decimal=1)


class TestLMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.transformations.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs`
    definition units tests methods.
    """

    def test_LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(self):
        """
        Tests :func:`colour.colorimetry.transformations.LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs`
        definition.
        """

        # TODO: Update test to tolerance matching.
        cmfs = CMFS.get("CIE 2012 2 Degree Standard Observer")
        np.testing.assert_almost_equal(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(435),
            cmfs.get(435),
            decimal=1)

        np.testing.assert_almost_equal(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(545),
            cmfs.get(545),
            decimal=1)

        np.testing.assert_almost_equal(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700),
            cmfs.get(700),
            decimal=1)


class TestLMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.transformations.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs`
    definition units tests methods.
    """

    def test_LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(self):
        """
        Tests :func:`colour.colorimetry.transformations.LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs`
        definition.
        """

        # TODO: Update test to tolerance matching.
        cmfs = CMFS.get("CIE 2012 10 Degree Standard Observer")
        np.testing.assert_almost_equal(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(435),
            cmfs.get(435),
            decimal=1)

        np.testing.assert_almost_equal(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(545),
            cmfs.get(545),
            decimal=1)

        np.testing.assert_almost_equal(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700),
            cmfs.get(700),
            decimal=1)


if __name__ == "__main__":
    unittest.main()
