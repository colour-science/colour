# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_cie_lab.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`colour.computation.colourspaces.cie_lab` module.

**Others:**

"""

from __future__ import unicode_literals

import numpy
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import colour.computation.colourspaces.cie_lab

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestXYZ_to_Lab",
           "TestLab_to_XYZ",
           "TestLab_to_LCHab",
           "TestLCHab_to_Lab"]


class TestXYZ_to_Lab(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_lab.XY§Z_to_Lab` definition units tests methods.
    """

    def test_XYZ_to_Lab(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_lab.XYZ_to_Lab` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.XYZ_to_Lab(numpy.array([0.96907232, 1., 1.12179215])),
            numpy.array([100., 0.83871284, -21.55579303]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.XYZ_to_Lab(numpy.array([1.92001986, 1., -0.1241347])),
            numpy.array([100., 129.04406346, 406.69765889]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.XYZ_to_Lab(numpy.array([1.0131677, 1., 2.11217686])),
            numpy.array([100., 8.32281957, -73.58297716]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.XYZ_to_Lab(numpy.array([1.0131677, 1., 2.11217686]),
                                                               (0.44757, 0.40745)),
            numpy.array([100., -13.29228089, -162.12804888]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.XYZ_to_Lab(numpy.array([1.0131677, 1., 2.11217686]),
                                                               (1. / 3., 1. / 3.)),
            numpy.array([100., 2.18505384, -56.60990888]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.XYZ_to_Lab(numpy.array([1.0131677, 1., 2.11217686]),
                                                               (0.31271, 0.32902)),
            numpy.array([100., 10.76832763, -49.42733157]).reshape((3, 1)),
            decimal=7)


class TestLab_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_lab.Lab_to_XYZ` definition units tests methods.
    """

    def test_Lab_to_XYZ(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_lab.Lab_to_XYZ` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.Lab_to_XYZ(numpy.array([100., 0.83871284, -21.55579303])),
            numpy.array([0.96907232, 1., 1.12179215]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.Lab_to_XYZ(numpy.array([100., 129.04406346, 406.69765889])),
            numpy.array([1.92001986, 1., -0.1241347]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.Lab_to_XYZ(numpy.array([100., 8.32281957, -73.58297716])),
            numpy.array([1.0131677, 1., 2.11217686]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.Lab_to_XYZ(numpy.array([100., -13.29228089, -162.12804888]),
                                                               (0.44757, 0.40745)),
            numpy.array([1.0131677, 1., 2.11217686]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.Lab_to_XYZ(numpy.array([100., 2.18505384, -56.60990888]),
                                                               (1. / 3., 1. / 3.)),
            numpy.array([1.0131677, 1., 2.11217686]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.Lab_to_XYZ(numpy.array([100., 10.76832763, -49.42733157]),
                                                               (0.31271, 0.32902)),
            numpy.array([1.0131677, 1., 2.11217686]).reshape((3, 1)),
            decimal=7)


class TestLab_to_LCHab(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_lab.Lab_to_LCHab` definition units tests methods.
    """

    def test_Lab_to_LCHab(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_lab.Lab_to_LCHab` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.Lab_to_LCHab(numpy.array([100., 0.83871284, -21.55579303])),
            numpy.array([100., 21.57210357, 272.2281935]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.Lab_to_LCHab(numpy.array([100., 129.04406346, 406.69765889])),
            numpy.array([100., 426.67945353, 72.39590835]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.Lab_to_LCHab(numpy.array([100., 8.32281957, -73.58297716])),
            numpy.array([100., 74.05216981, 276.45318193]).reshape((3, 1)),
            decimal=7)


class TestLCHab_to_Lab(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.cie_lab.LCHab_to_Lab` definition units tests methods.
    """

    def test_LCHab_to_Lab(self):
        """
        Tests :func:`colour.computation.colourspaces.cie_lab.LCHab_to_Lab` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.LCHab_to_Lab(numpy.array([100., 21.57210357, 272.2281935])),
            numpy.array([100., 0.83871284, -21.55579303]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.LCHab_to_Lab(numpy.array([100., 426.67945353, 72.39590835])),
            numpy.array([100., 129.04406346, 406.69765889]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.cie_lab.LCHab_to_Lab(numpy.array([100., 74.05216981, 276.45318193])),
            numpy.array([100., 8.32281957, -73.58297716]).reshape((3, 1)),
            decimal=7)


if __name__ == "__main__":
    unittest.main()
