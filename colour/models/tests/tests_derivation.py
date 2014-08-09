#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_derivation.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`colour.models.rgb.derivation` module.

**Others:**

"""

from __future__ import unicode_literals

import sys
import re
import numpy as np

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import (
    get_normalised_primary_matrix,
    get_RGB_luminance_equation,
    get_RGB_luminance)
from colour.models.derivation import xy_to_z

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["Testxy_to_z",
           "TestGetNormalisedPrimaryMatrix",
           "TestGetRGBLuminanceEquation",
           "TestGetRGBLuminance"]


class Testxy_to_z(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.derivation.xy_to_z` definition units tests
    methods.
    """

    def test_xy_to_z(self):
        """
        Tests :func:`colour.models.rgb.derivation.xy_to_z` definition.
        """

        np.testing.assert_almost_equal(
            xy_to_z((0.25, 0.25)),
            0.5,
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_z((0.00010, -0.07700)),
            1.07690,
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_z((0.00000, 1.00000)),
            0.00000,
            decimal=7)


class TestGetNormalisedPrimaryMatrix(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.derivation.get_normalised_primary_matrix`
    definition units tests methods.
    """

    def test_get_normalised_primary_matrix(self):
        """
        Tests :func:`colour.models.rgb.derivation.get_normalised_primary_matrix`
        definition.
        """

        np.testing.assert_almost_equal(
            get_normalised_primary_matrix(
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                (0.32168, 0.33767)),
            np.array(
                [9.52552396e-01, 0.00000000e+00, 9.36786317e-05,
                 3.43966450e-01, 7.28166097e-01, -7.21325464e-02,
                 0.00000000e+00, 0.00000000e+00, 1.00882518e+00]
            ).reshape((3, 3)),
            decimal=7)

        np.testing.assert_almost_equal(
            get_normalised_primary_matrix(
                np.array([0.640, 0.330,
                          0.300, 0.600,
                          0.150, 0.060]),
                (0.3127, 0.3290)),
            np.array([0.4123908, 0.35758434, 0.18048079,
                      0.21263901, 0.71516868, 0.07219232,
                      0.01933082, 0.11919478, 0.95053215]).reshape((3, 3)),
            decimal=7)


class TestGetRGBLuminanceEquation(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.derivation.get_RGB_luminance_equation`
    definition units tests methods.
    """

    def test_get_RGB_luminance_equation(self):
        """
        Tests :func:`colour.models.rgb.derivation.get_RGB_luminance_equation`
        definition.
        """

        self.assertIsInstance(
            get_RGB_luminance_equation(
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                (0.32168, 0.33767)), unicode)

        self.assertTrue(re.match(
            r"Y\s?=\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.\(R\)\s?[\+-]\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.\(G\)\s?[\+-]\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.\(B\)",
            get_RGB_luminance_equation(
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                (0.32168, 0.33767))))


class TestGetRGBLuminance(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.derivation.get_RGB_luminance` definition
    units tests methods.
    """

    def test_get_RGB_luminance(self):
        """
        Tests :func:`colour.models.rgb.derivation.get_RGB_luminance` definition.
        """

        self.assertAlmostEqual(
            get_RGB_luminance(
                np.array([50., 50., 50.]),
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                (0.32168, 0.33767)),
            50.,
            places=7)

        self.assertAlmostEqual(
            get_RGB_luminance(
                np.array([74.6, 16.1, 100.]),
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                (0.32168, 0.33767)),
            30.1701166701,
            places=7)

        self.assertAlmostEqual(
            get_RGB_luminance(
                np.array([40.6, 4.2, 67.4]),
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                (0.32168, 0.33767)),
            12.1616018403,
            places=7)


if __name__ == "__main__":
    unittest.main()
