#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.colorimetry.lightness` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.colorimetry import (
    lightness_Glasser1958,
    lightness_Wyszecki1963,
    lightness_1976)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLightnessGlasser1958',
           'TestLightnessWyszecki1963',
           'TestLightness1976']


class TestLightnessGlasser1958(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.lightness.lightness_Glasser1958`
    definition unit tests methods.
    """

    def test_lightness_Glasser1958(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Glasser1958`
        definition.
        """

        self.assertAlmostEqual(
            lightness_Glasser1958(10.08),
            36.2505626458,
            places=7)
        self.assertAlmostEqual(
            lightness_Glasser1958(56.76),
            78.8117999039,
            places=7)
        self.assertAlmostEqual(
            lightness_Glasser1958(98.32),
            98.3447052593,
            places=7)

    def test_n_dimensions_lightness_Glasser1958(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Glasser1958`
        definition n-dimensions support.
        """

        Y = 10.08
        L = 36.2505626458
        np.testing.assert_almost_equal(
            lightness_Glasser1958(Y),
            L,
            decimal=7)

        Y = [Y] * 6
        L = [L] * 6
        np.testing.assert_almost_equal(
            lightness_Glasser1958(Y),
            L,
            decimal=7)

        Y = np.reshape(Y, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            lightness_Glasser1958(Y),
            L,
            decimal=7)

        Y = np.reshape(Y, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            lightness_Glasser1958(Y),
            L,
            decimal=7)


class TestLightnessWyszecki1963(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.lightness.lightness_Wyszecki1963`
    definition unit tests methods.
    """

    def test_lightness_Wyszecki1963(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Wyszecki1963`
        definition.
        """

        self.assertAlmostEqual(
            lightness_Wyszecki1963(10.08),
            37.0041149128,
            places=7)
        self.assertAlmostEqual(
            lightness_Wyszecki1963(56.76),
            79.0773031869,
            places=7)
        self.assertAlmostEqual(
            lightness_Wyszecki1963(98.32),
            98.3862250488,
            places=7)

    def test_n_dimensions_lightness_Wyszecki1963(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_Wyszecki1963`
        definition n-dimensions support.
        """

        Y = 10.08
        W = 37.004114912764535
        np.testing.assert_almost_equal(
            lightness_Wyszecki1963(Y),
            W,
            decimal=7)

        Y = [Y] * 6
        W = [W] * 6
        np.testing.assert_almost_equal(
            lightness_Wyszecki1963(Y),
            W,
            decimal=7)

        Y = np.reshape(Y, (2, 3))
        W = np.reshape(W, (2, 3))
        np.testing.assert_almost_equal(
            lightness_Wyszecki1963(Y),
            W,
            decimal=7)

        Y = np.reshape(Y, (2, 3, 1))
        W = np.reshape(W, (2, 3, 1))
        np.testing.assert_almost_equal(
            lightness_Wyszecki1963(Y),
            W,
            decimal=7)


class TestLightness1976(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.lightness.lightness_1976` definition
    unit tests methods.
    """

    def test_lightness_1976(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_1976` definition.
        """

        self.assertAlmostEqual(
            lightness_1976(10.08),
            37.9856290977,
            places=7)
        self.assertAlmostEqual(
            lightness_1976(56.76),
            80.0444155585,
            places=7)
        self.assertAlmostEqual(
            lightness_1976(98.32),
            99.3467279026,
            places=7)
        self.assertAlmostEqual(
            lightness_1976(10.08, 50),
            52.01763049195023,
            places=7)
        self.assertAlmostEqual(
            lightness_1976(10.08, 75),
            43.41887325541973,
            places=7)
        self.assertAlmostEqual(
            lightness_1976(10.08, 95),
            38.91659875709282,
            places=7)

    def test_n_dimensions_lightness_1976(self):
        """
        Tests :func:`colour.colorimetry.lightness.lightness_1976`
        definition n-dimensions support.
        """

        Y = 10.08
        Lstar = 37.98562909765304
        np.testing.assert_almost_equal(
            lightness_1976(Y),
            Lstar,
            decimal=7)

        Y = [Y] * 6
        Lstar = [Lstar] * 6
        np.testing.assert_almost_equal(
            lightness_1976(Y),
            Lstar,
            decimal=7)

        Y = np.reshape(Y, (2, 3))
        Lstar = np.reshape(Lstar, (2, 3))
        np.testing.assert_almost_equal(
            lightness_1976(Y),
            Lstar,
            decimal=7)

        Y = np.reshape(Y, (2, 3, 1))
        Lstar = np.reshape(Lstar, (2, 3, 1))
        np.testing.assert_almost_equal(
            lightness_1976(Y),
            Lstar,
            decimal=7)


if __name__ == '__main__':
    unittest.main()
