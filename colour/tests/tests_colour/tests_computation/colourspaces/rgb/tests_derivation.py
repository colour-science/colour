# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_derivation.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`colour.computation.colourspaces.rgb.derivation` module.

**Others:**

"""

from __future__ import unicode_literals

import sys

import numpy


if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import colour.computation.colourspaces.rgb.derivation

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["Testxy_to_z",
           "TestGetNormalisedPrimaryMatrix"]


class Testxy_to_z(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.rgb.derivation.xy_to_z` definition units tests methods.
    """

    def test_xy_to_z(self):
        """
        Tests :func:`colour.computation.colourspaces.rgb.derivation.xy_to_z` definition.
        """

        numpy.testing.assert_almost_equal(colour.computation.colourspaces.rgb.derivation.xy_to_z((0.25, 0.25)),
                                          0.5,
                                          decimal=7)

        numpy.testing.assert_almost_equal(colour.computation.colourspaces.rgb.derivation.xy_to_z((0.00010, -0.07700)),
                                          1.07690,
                                          decimal=7)

        numpy.testing.assert_almost_equal(colour.computation.colourspaces.rgb.derivation.xy_to_z((0.00000, 1.00000)),
                                          0.00000,
                                          decimal=7)


class TestGetNormalisedPrimaryMatrix(unittest.TestCase):
    """
    Defines :func:`colour.computation.colourspaces.rgb.derivation.get_normalised_primary_matrix` definition units tests methods.
    """

    def test_get_normalised_primary_matrix(self):
        """
        Tests :func:`colour.computation.colourspaces.rgb.derivation.get_normalised_primary_matrix` definition.
        """

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.rgb.derivation.get_normalised_primary_matrix(numpy.matrix([0.73470, 0.26530,
                                                                                      0.00000, 1.00000,
                                                                                      0.00010, -0.07700]).reshape(
                (3, 2)),
                                                                        (0.32168, 0.33767)),
            numpy.matrix([9.52552396e-01, 0.00000000e+00, 9.36786317e-05,
                          3.43966450e-01, 7.28166097e-01, -7.21325464e-02,
                          0.00000000e+00, 0.00000000e+00, 1.00882518e+00]).reshape((3, 3)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            colour.computation.colourspaces.rgb.derivation.get_normalised_primary_matrix(numpy.matrix([0.640, 0.330,
                                                                                      0.300, 0.600,
                                                                                      0.150, 0.060]).reshape((3, 2)),
                                                                        (0.3127, 0.3290)),
            numpy.matrix([0.4123908, 0.35758434, 0.18048079,
                          0.21263901, 0.71516868, 0.07219232,
                          0.01933082, 0.11919478, 0.95053215]).reshape((3, 3)),
            decimal=7)


if __name__ == "__main__":
    unittest.main()
