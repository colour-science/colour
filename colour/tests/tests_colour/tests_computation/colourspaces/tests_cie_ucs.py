# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_cie_ucs.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`colour.computation.colourspaces.cie_ucs` module.

**Others:**

"""

from __future__ import unicode_literals

import numpy
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import colour.computation.colourspaces.cie_ucs

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestXYZ_to_UCS",
           "TestUCS_to_XYZ",
           "TestUCS_to_uv",
           "TestUCS_uv_to_xy",
           "TestXYZ_to_UVW"]


class TestXYZ_to_UCS(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_ucs.XYZ_to_UCS` definition units tests methods.
    """

    def test_XYZ_to_UCS(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_ucs.XYZ_to_UCS` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_ucs.XYZ_to_UCS(
                numpy.matrix([11.80583421, 10.34, 5.15089229]).reshape((3, 1))),
            numpy.matrix([7.87055614, 10.34, 12.18252904]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_ucs.XYZ_to_UCS(
                numpy.matrix([3.08690042, 3.2, 2.68925666]).reshape((3, 1))),
            numpy.matrix([2.05793361, 3.2, 4.60117812]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_ucs.XYZ_to_UCS(
                numpy.matrix([0.96907232, 1., 1.12179215]).reshape((3, 1))),
            numpy.matrix([0.64604821, 1., 1.57635992]).reshape((3, 1)),
            decimal=7)


class TestUCS_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_ucs.UCS_to_XYZ` definition units tests methods.
    """

    def test_UCS_to_XYZ(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_ucs.UCS_to_XYZ` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_ucs.UCS_to_XYZ(
                numpy.matrix([7.87055614, 10.34, 12.18252904]).reshape((3, 1))),
            numpy.matrix([11.80583421, 10.34, 5.15089229]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_ucs.UCS_to_XYZ(
                numpy.matrix([2.05793361, 3.2, 4.60117812]).reshape((3, 1))),
            numpy.matrix([3.08690042, 3.2, 2.68925666]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_ucs.UCS_to_XYZ(
                numpy.matrix([0.64604821, 1., 1.57635992]).reshape((3, 1))),
            numpy.matrix([0.96907232, 1., 1.12179215]).reshape((3, 1)),
            decimal=7)


class TestUCS_to_uv(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_ucs.UCS_to_uv` definition units tests methods.
    """

    def test_UCS_to_uv(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_ucs.UCS_to_uv` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_ucs.UCS_to_uv(
                numpy.matrix([7.87055614, 10.34, 12.18252904]).reshape((3, 1))),
            (0.25895877609618834, 0.34020896328103534),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_ucs.UCS_to_uv(
                numpy.matrix([2.05793361, 3.2, 4.60117812]).reshape((3, 1))),
            (0.20873418076173886, 0.32457285074301517),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_ucs.UCS_to_uv(
                numpy.matrix([0.64604821, 1., 1.57635992]).reshape((3, 1))),
            (0.20048615319251942, 0.31032692311386395),
            decimal=7)


class TestUCS_uv_to_xy(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_ucs.UCS_uv_to_xy` definition units tests methods.
    """

    def test_UCS_uv_to_xy(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_ucs.UCS_uv_to_xy` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_ucs.UCS_uv_to_xy((0.2033733344733139, 0.3140500001549052)),
            (0.32207410281368043, 0.33156550013623537),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_ucs.UCS_uv_to_xy((0.20873418102926322, 0.32457285063327812)),
            (0.3439000000209443, 0.35650000010917804),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_ucs.UCS_uv_to_xy((0.25585459629500179, 0.34952813701502972)),
            (0.4474327628361858, 0.40749796251018744),
            decimal=7)


if __name__ == "__main__":
    unittest.main()
