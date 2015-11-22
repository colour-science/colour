#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.colorimetry.luminance` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.colorimetry.luminance import (
    luminance_Newhall1943,
    luminance_1976,
    luminance_ASTMD153508)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLuminanceNewhall1943',
           'TestLuminanceASTMD153508',
           'TestLuminance1976']


class TestLuminanceNewhall1943(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.luminance_Newhall1943`
    definition unit tests methods.
    """

    def test_luminance_Newhall1943(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Newhall1943`
        definition.
        """

        self.assertAlmostEqual(
            luminance_Newhall1943(3.74629715382),
            10.4089874577,
            places=7)

        self.assertAlmostEqual(
            luminance_Newhall1943(8.64728711385),
            71.3174801757,
            places=7)

        self.assertAlmostEqual(
            luminance_Newhall1943(1.52569021578),
            2.06998750444,
            places=7)

    def test_n_dimensional_luminance_Newhall1943(self):
        """
        Tests :func:`colour.colorimetry.lightness.luminance_Newhall1943`
        definition n-dimensional arrays support.
        """

        V = 3.74629715382
        Y = 10.408987457743208
        np.testing.assert_almost_equal(
            luminance_Newhall1943(V),
            Y,
            decimal=7)

        V = np.tile(V, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(
            luminance_Newhall1943(V),
            Y,
            decimal=7)

        V = np.reshape(V, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_almost_equal(
            luminance_Newhall1943(V),
            Y,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_almost_equal(
            luminance_Newhall1943(V),
            Y,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_luminance_Newhall1943(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_Newhall1943`
        definition nan support.
        """

        luminance_Newhall1943(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLuminanceASTMD153508(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.luminance_ASTMD153508`
    definition unit tests methods.
    """

    def test_luminance_ASTMD153508(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_ASTMD153508`
        definition.
        """

        self.assertAlmostEqual(
            luminance_ASTMD153508(3.74629715382),
            10.1488096782,
            places=7)

        self.assertAlmostEqual(
            luminance_ASTMD153508(8.64728711385),
            69.5324092373,
            places=7)

        self.assertAlmostEqual(
            luminance_ASTMD153508(1.52569021578),
            2.01830631474,
            places=7)

    def test_n_dimensional_luminance_ASTMD153508(self):
        """
        Tests :func:`colour.colorimetry.lightness.luminance_ASTMD153508`
        definition n-dimensional arrays support.
        """

        V = 3.74629715382
        Y = 10.148809678226682
        np.testing.assert_almost_equal(
            luminance_ASTMD153508(V),
            Y,
            decimal=7)

        V = np.tile(V, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(
            luminance_ASTMD153508(V),
            Y,
            decimal=7)

        V = np.reshape(V, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_almost_equal(
            luminance_ASTMD153508(V),
            Y,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_almost_equal(
            luminance_ASTMD153508(V),
            Y,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_luminance_ASTMD153508(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_ASTMD153508`
        definition nan support.
        """

        luminance_ASTMD153508(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLuminance1976(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.luminance_1976` definition
    unit tests methods.
    """

    def test_luminance_1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_1976` definition.
        """

        self.assertAlmostEqual(
            luminance_1976(37.9856290977),
            10.08,
            places=7)

        self.assertAlmostEqual(
            luminance_1976(80.0444155585),
            56.76,
            places=7)

        self.assertAlmostEqual(
            luminance_1976(99.3467279026),
            98.32,
            places=7)

        self.assertAlmostEqual(
            luminance_1976(37.9856290977, 50),
            5.040000000013152,
            places=7)

        self.assertAlmostEqual(
            luminance_1976(37.9856290977, 75),
            7.560000000019728,
            places=7)

        self.assertAlmostEqual(
            luminance_1976(37.9856290977, 95),
            9.576000000024989,
            places=7)

    def test_n_dimensional_luminance_1976(self):
        """
        Tests :func:`colour.colorimetry.lightness.luminance_1976`
        definition n-dimensional arrays support.
        """

        Lstar = 37.9856290977
        Y = 10.080000000026304
        np.testing.assert_almost_equal(
            luminance_1976(Lstar),
            Y,
            decimal=7)

        Lstar = np.tile(Lstar, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(
            luminance_1976(Lstar),
            Y,
            decimal=7)

        Lstar = np.reshape(Lstar, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_almost_equal(
            luminance_1976(Lstar),
            Y,
            decimal=7)

        Lstar = np.reshape(Lstar, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_almost_equal(
            luminance_1976(Lstar),
            Y,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_luminance_1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_1976`
        definition nan support.
        """

        luminance_1976(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
