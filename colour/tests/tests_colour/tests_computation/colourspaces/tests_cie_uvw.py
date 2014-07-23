# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_cie_uvw.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`colour.computation.colourspaces.cie_uvw` module.

**Others:**

"""

from __future__ import unicode_literals

import numpy
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import colour.computation.colourspaces.cie_uvw

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestXYZ_to_UVW"]


class TestXYZ_to_UVW(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_uvw.XYZ_to_UVW` definition units tests methods.
    """

    def test_XYZ_to_UVW(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_uvw.XYZ_to_UVW` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_uvw.XYZ_to_UVW(
                numpy.array([0.96907232, 1., 1.12179215])),
            numpy.array([-0.90199113, -1.56588889, 8.]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_uvw.XYZ_to_UVW(
                numpy.array([1.92001986, 1., - 0.1241347])),
            numpy.array([26.5159289, 3.8694711, 8.]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_uvw.XYZ_to_UVW(
                numpy.array([1.0131677, 1., 2.11217686])),
            numpy.array([-2.89423113, -5.92004891, 8.]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_uvw.XYZ_to_UVW(
                numpy.array([1.0131677, 1., 2.11217686]),
                (0.44757, 0.40745)),
            numpy.array([-7.76195429, -8.43122502, 8.]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_uvw.XYZ_to_UVW(
                numpy.array([1.0131677, 1., 2.11217686]),
                (1. / 3., 1. / 3.)),
            numpy.array([-3.03641679, -4.92226526, 8.]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_uvw.XYZ_to_UVW(
                numpy.array([1.0131677, 1., 2.11217686]),
                (0.31271, 0.32902)),
            numpy.array([-1.7159427, -4.55119033, 8]).reshape((3, 1)),
            decimal=7)


if __name__ == "__main__":
    unittest.main()
