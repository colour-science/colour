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
    luminance_CIE1976,
    luminance_ASTMD153508)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLuminanceNewhall1943',
           'TestLuminanceASTMD153508',
           'TestLuminanceCIE1976']


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
            luminance_Newhall1943(3.74629715),
            10.40898746,
            places=7)

        self.assertAlmostEqual(
            luminance_Newhall1943(8.64728711),
            71.31748010,
            places=7)

        self.assertAlmostEqual(
            luminance_Newhall1943(1.52569022),
            2.06998750,
            places=7)

    def test_n_dimensional_luminance_Newhall1943(self):
        """
        Tests :func:`colour.colorimetry.lightness.luminance_Newhall1943`
        definition n-dimensional arrays support.
        """

        V = 3.74629715
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
            luminance_ASTMD153508(3.74629715),
            10.14880968,
            places=7)

        self.assertAlmostEqual(
            luminance_ASTMD153508(8.64728711),
            69.53240916,
            places=7)

        self.assertAlmostEqual(
            luminance_ASTMD153508(1.52569022),
            2.01830631,
            places=7)

    def test_n_dimensional_luminance_ASTMD153508(self):
        """
        Tests :func:`colour.colorimetry.lightness.luminance_ASTMD153508`
        definition n-dimensional arrays support.
        """

        V = 3.74629715
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


class TestLuminanceCIE1976(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.luminance_CIE1976` definition
    unit tests methods.
    """

    def test_luminance_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_CIE1976`
        definition.
        """

        self.assertAlmostEqual(
            luminance_CIE1976(37.98562910),
            10.08000000,
            places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(80.04441556),
            56.76000000,
            places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(99.34672790),
            98.32000000,
            places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(37.98562910, 50),
            5.04000000,
            places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(37.98562910, 75),
            7.56000000,
            places=7)

        self.assertAlmostEqual(
            luminance_CIE1976(37.98562910, 95),
            9.57600000,
            places=7)

    def test_n_dimensional_luminance_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.lightness.luminance_CIE1976`
        definition n-dimensional arrays support.
        """

        Lstar = 37.98562910
        Y = 10.080000000026304
        np.testing.assert_almost_equal(
            luminance_CIE1976(Lstar),
            Y,
            decimal=7)

        Lstar = np.tile(Lstar, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(
            luminance_CIE1976(Lstar),
            Y,
            decimal=7)

        Lstar = np.reshape(Lstar, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_almost_equal(
            luminance_CIE1976(Lstar),
            Y,
            decimal=7)

        Lstar = np.reshape(Lstar, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_almost_equal(
            luminance_CIE1976(Lstar),
            Y,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_luminance_CIE1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_CIE1976`
        definition nan support.
        """

        luminance_CIE1976(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
