#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_cie_xyy.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`colour.models.cie_xyy` module.

**Others:**

"""

from __future__ import unicode_literals

import sys
import numpy as np

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import XYZ_to_xyY, xyY_to_XYZ, xy_to_XYZ, XYZ_to_xy

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["TestXYZ_to_xyY",
           "TestxyY_to_XYZ",
           "Testxy_to_XYZ",
           "TestXYZ_to_xy"]


class TestXYZ_to_xyY(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.XYZ_to_xyY` definition units tests
    methods.
    """

    def test_XYZ_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xyY` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([11.80583421, 10.34, 5.15089229])),
            np.array([0.4325, 0.3788, 10.34]).reshape((3, 1)),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([3.08690042, 3.2, 2.68925666])),
            np.array([0.3439, 0.3565, 3.20]).reshape((3, 1)),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0, 0, 0]),
                       (0.34567, 0.35850)),
            np.array([0.34567, 0.35850, 0]).reshape((3, 1)),
            decimal=7)


class TestxyY_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.xyY_to_XYZ` definition units tests
    methods.
    """

    def test_xyY_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.4325, 0.3788, 10.34])),
            np.array([11.80583421, 10.34, 5.15089229]).reshape((3, 1)),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.3439, 0.3565, 3.20])),
            np.array([3.08690042, 3.2, 2.68925666]).reshape((3, 1)),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.4325, 0., 10.34])),
            np.array([0., 0., 0.]).reshape((3, 1)),
            decimal=7)


class Testxy_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.xy_to_XYZ` definition units tests
    methods.
    """

    def test_xy_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            xy_to_XYZ((0.32207410281368043, 0.3315655001362353)),
            np.array([0.97137399, 1., 1.04462134]).reshape((3, 1)),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_XYZ((0.32174206617150575, 0.337609723160027)),
            np.array([0.953, 1.000, 1.009]).reshape((3, 1)),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_XYZ((0.4474327628361859, 0.4074979625101875)),
            np.array([1.098, 1.000, 0.356]).reshape((3, 1)),
            decimal=7)


class TestXYZ_to_xy(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.XYZ_to_xy` definition units tests
    methods.
    """

    def test_XYZ_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xy` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_xy((0.97137399, 1., 1.04462134)),
            (0.32207410281368043, 0.3315655001362353),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xy((0.953, 1.000, 1.009)),
            (0.32174206617150575, 0.337609723160027),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xy((1.098, 1.000, 0.356)),
            (0.4474327628361859, 0.4074979625101875),
            decimal=7)


if __name__ == "__main__":
    unittest.main()
