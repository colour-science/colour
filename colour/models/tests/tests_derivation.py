#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.derivation` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import re
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import (
    normalised_primary_matrix,
    RGB_luminance_equation,
    RGB_luminance)
from colour.models.derivation import xy_to_z

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Testxy_to_z',
           'TestNormalisedPrimaryMatrix',
           'TestRGBLuminanceEquation',
           'TestRGBLuminance']


class Testxy_to_z(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.derivation.xy_to_z` definition unit tests
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


class TestNormalisedPrimaryMatrix(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.derivation.normalised_primary_matrix`
    definition unit tests methods.
    """

    def test_normalised_primary_matrix(self):
        """
        Tests
        :func:`colour.models.rgb.derivation.normalised_primary_matrix`
        definition.
        """

        np.testing.assert_almost_equal(
            normalised_primary_matrix(
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
            normalised_primary_matrix(
                np.array([0.640, 0.330,
                          0.300, 0.600,
                          0.150, 0.060]),
                (0.3127, 0.3290)),
            np.array([0.4123908, 0.35758434, 0.18048079,
                      0.21263901, 0.71516868, 0.07219232,
                      0.01933082, 0.11919478, 0.95053215]).reshape((3, 3)),
            decimal=7)


class TestRGBLuminanceEquation(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.derivation.RGB_luminance_equation`
    definition unit tests methods.
    """

    def test_RGB_luminance_equation(self):
        """
        Tests :func:`colour.models.rgb.derivation.RGB_luminance_equation`
        definition.
        """

        self.assertIsInstance(
            RGB_luminance_equation(
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                (0.32168, 0.33767)), unicode)

        self.assertTrue(re.match(
            # TODO: Simplify that monster.
            'Y\s?=\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.\(R\)\s?[\+-]\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.\(G\)\s?[\+-]\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.\(B\)',  # noqa
            RGB_luminance_equation(
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                (0.32168, 0.33767))))


class TestRGBLuminance(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.derivation.RGB_luminance` definition
    unit tests methods.
    """

    def test_RGB_luminance(self):
        """
        Tests:func:`colour.models.rgb.derivation.RGB_luminance`
        definition.
        """

        self.assertAlmostEqual(
            RGB_luminance(
                np.array([50, 50, 50]),
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                (0.32168, 0.33767)),
            50.,
            places=7)

        self.assertAlmostEqual(
            RGB_luminance(
                np.array([74.6, 16.1, 100]),
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                (0.32168, 0.33767)),
            30.1701166701,
            places=7)

        self.assertAlmostEqual(
            RGB_luminance(
                np.array([40.6, 4.2, 67.4]),
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                (0.32168, 0.33767)),
            12.1616018403,
            places=7)


if __name__ == '__main__':
    unittest.main()
