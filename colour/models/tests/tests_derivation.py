#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.derivation` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import re
import unittest
from itertools import permutations

from colour.models import (
    normalised_primary_matrix,
    chromatically_adapted_primaries,
    primaries_whitepoint,
    RGB_luminance_equation,
    RGB_luminance)
from colour.models.derivation import xy_to_z
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Testxy_to_z',
           'TestNormalisedPrimaryMatrix',
           'TestChromaticallyAdaptedPrimaries',
           'TestPrimariesWhitepoint',
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
            xy_to_z(np.array([0.2500, 0.2500])),
            0.5,
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_z(np.array([0.0001, -0.0770])),
            1.0769,
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_z(np.array([0.0000, 1.0000])),
            0.0,
            decimal=7)

    def test_n_dimensional_xy_to_z(self):
        """
        Tests :func:`colour.models.rgb.derivation.xy_to_z` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.25, 0.25])
        z = 0.5
        np.testing.assert_almost_equal(
            xy_to_z(xy),
            z,
            decimal=7)

        xy = np.tile(xy, (6, 1))
        z = np.tile(z, 6, )
        np.testing.assert_almost_equal(
            xy_to_z(xy),
            z,
            decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        z = np.reshape(z, (2, 3))
        np.testing.assert_almost_equal(
            xy_to_z(xy),
            z,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_xy_to_z(self):
        """
        Tests :func:`colour.models.rgb.derivation.xy_to_z` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            xy_to_z(case)


class TestNormalisedPrimaryMatrix(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.derivation.normalised_primary_matrix`
    definition unit tests methods.
    """

    def test_normalised_primary_matrix(self):
        """
        Tests :func:`colour.models.rgb.derivation.normalised_primary_matrix`
        definition.
        """

        np.testing.assert_almost_equal(
            normalised_primary_matrix(
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                np.array([0.32168, 0.33767])),
            np.array(
                [[9.52552396e-01, 0.00000000e+00, 9.36786317e-05],
                 [3.43966450e-01, 7.28166097e-01, -7.21325464e-02],
                 [0.00000000e+00, 0.00000000e+00, 1.00882518e+00]]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalised_primary_matrix(
                np.array([0.640, 0.330,
                          0.300, 0.600,
                          0.150, 0.060]),
                np.array([0.3127, 0.3290])),
            np.array([[0.41239080, 0.35758434, 0.18048079],
                      [0.21263901, 0.71516868, 0.07219232],
                      [0.01933082, 0.11919478, 0.95053215]]),
            decimal=7)

    @ignore_numpy_errors
    def test_nan_normalised_primary_matrix(self):
        """
        Tests :func:`colour.models.rgb.derivation.normalised_primary_matrix`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            P = np.array(np.vstack((case, case, case)))
            W = np.array(case)
            try:
                normalised_primary_matrix(P, W)
            except np.linalg.linalg.LinAlgError:
                import traceback
                from colour.utilities import warning

                warning(traceback.format_exc())


class TestChromaticallyAdaptedPrimaries(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.derivation.\
chromatically_adapted_primaries` definition unit tests methods.
    """

    def test_chromatically_adapted_primaries(self):
        """
        Tests :func:`colour.models.rgb.derivation.\
chromatically_adapted_primaries` definition.
        """

        np.testing.assert_almost_equal(
            chromatically_adapted_primaries(
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                np.array([0.32168, 0.33767]),
                np.array([0.34567, 0.35850])),
            np.array([[0.7343147, 0.2669459],
                      [0.022058, 0.9804409],
                      [-0.0587459, -0.1256946]]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatically_adapted_primaries(
                np.array([0.640, 0.330,
                          0.300, 0.600,
                          0.150, 0.060]),
                np.array([0.31271, 0.32902]),
                np.array([0.34567, 0.35850])),
            np.array([[0.6492148, 0.3306242],
                      [0.3242141, 0.6023877],
                      [0.152359, 0.0611854]]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatically_adapted_primaries(
                np.array([0.640, 0.330,
                          0.300, 0.600,
                          0.150, 0.060]),
                np.array([0.31271, 0.32902]),
                np.array([0.34567, 0.35850]),
                'Bradford'),
            np.array([[0.6484318, 0.3308549],
                      [0.3211603, 0.5978621],
                      [0.155886, 0.0660431]]),
            decimal=7)

    @ignore_numpy_errors
    def test_nan_chromatically_adapted_primaries(self):
        """
        Tests :func:`colour.models.rgb.derivation.\
chromatically_adapted_primaries` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            P = np.array(np.vstack((case, case, case)))
            W = np.array(case)
            chromatically_adapted_primaries(P, W, W)


class TestPrimariesWhitepoint(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.derivation.primaries_whitepoint`
    definition unit tests methods.
    """

    def test_primaries_whitepoint(self):
        """
        Tests :func:`colour.models.rgb.derivation.primaries_whitepoint`
        definition.
        """

        P, W = primaries_whitepoint(
            np.array([[9.52552396e-01, 0.00000000e+00, 9.36786317e-05],
                      [3.43966450e-01, 7.28166097e-01, -7.21325464e-02],
                      [0.00000000e+00, 0.00000000e+00, 1.00882518e+00]]))
        np.testing.assert_almost_equal(
            P,
            np.array([[0.73470, 0.26530],
                      [0.00000, 1.00000],
                      [0.00010, -0.07700]]),
            decimal=7)
        np.testing.assert_almost_equal(
            W,
            np.array([0.32168, 0.33767]),
            decimal=7)

        P, W = primaries_whitepoint(
            np.array([[0.41239080, 0.35758434, 0.18048079],
                      [0.21263901, 0.71516868, 0.07219232],
                      [0.01933082, 0.11919478, 0.95053215]]))
        np.testing.assert_almost_equal(
            P,
            np.array([[0.640, 0.330],
                      [0.300, 0.600],
                      [0.150, 0.060]]),
            decimal=7)
        np.testing.assert_almost_equal(
            W,
            np.array([0.3127, 0.3290]),
            decimal=7)

    @ignore_numpy_errors
    def test_nan_primaries_whitepoint(self):
        """
        Tests :func:`colour.models.rgb.derivation.primaries_whitepoint`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            M = np.array(np.vstack((case, case, case)))
            primaries_whitepoint(M)


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
                np.array([0.32168, 0.33767])),
            unicode)  # noqa

        self.assertTrue(re.match(
            # TODO: Simplify that monster.
            ('Y\s?=\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.'
             '\(R\)\s?[\+-]\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.'
             '\(G\)\s?[\+-]\s?[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?.\(B\)'),
            RGB_luminance_equation(
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                np.array([0.32168, 0.33767]))))


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
                np.array([50.0, 50.0, 50.0]),
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                np.array([0.32168, 0.33767])),
            50.,
            places=7)

        self.assertAlmostEqual(
            RGB_luminance(
                np.array([74.6, 16.1, 100.0]),
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                np.array([0.32168, 0.33767])),
            30.1701166701,
            places=7)

        self.assertAlmostEqual(
            RGB_luminance(
                np.array([40.6, 4.2, 67.4]),
                np.array([0.73470, 0.26530,
                          0.00000, 1.00000,
                          0.00010, -0.07700]),
                np.array([0.32168, 0.33767])),
            12.1616018403,
            places=7)

    def test_n_dimensional_RGB_luminance(self):
        """
        Tests:func:`colour.models.rgb.derivation.RGB_luminance` definition
        n_dimensional arrays support.
        """

        RGB = np.array([50.0, 50.0, 50.0]),
        P = np.array([0.73470, 0.26530,
                      0.00000, 1.00000,
                      0.00010, -0.07700]),
        W = np.array([0.32168, 0.33767])
        Y = 50
        np.testing.assert_almost_equal(
            RGB_luminance(RGB, P, W),
            Y)

        RGB = np.tile(RGB, (6, 1))
        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(
            RGB_luminance(RGB, P, W),
            Y)

        RGB = np.reshape(RGB, (2, 3, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_almost_equal(
            RGB_luminance(RGB, P, W),
            Y)

    @ignore_numpy_errors
    def test_nan_RGB_luminance(self):
        """
        Tests :func:`colour.models.rgb.derivation.RGB_luminance`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            P = np.array(np.vstack((case[0:2], case[0:2], case[0:2])))
            W = np.array(case[0:2])
            try:
                RGB_luminance(RGB, P, W)
            except np.linalg.linalg.LinAlgError:
                import traceback
                from colour.utilities import warning

                warning(traceback.format_exc())


if __name__ == '__main__':
    unittest.main()
