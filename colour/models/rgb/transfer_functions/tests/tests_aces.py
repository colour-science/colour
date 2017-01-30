#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.aces`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    log_encoding_ACESproxy,
    log_decoding_ACESproxy,
    log_encoding_ACEScc,
    log_decoding_ACEScc,
    log_encoding_ACEScct,
    log_decoding_ACEScct)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLogEncoding_ACESproxy',
           'TestLogDecoding_ACESproxy',
           'TestLogEncoding_ACEScc',
           'TestLogDecoding_ACEScc',
           'TestLogDecoding_ACEScct']


class TestLogEncoding_ACESproxy(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy`
    definition unit tests methods.
    """

    def test_log_encoding_ACESproxy(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition.
        """

        self.assertEqual(log_encoding_ACESproxy(0.0), 64)

        self.assertEqual(log_encoding_ACESproxy(0.18), 426)

        self.assertEqual(log_encoding_ACESproxy(1.0), 550)

    def test_n_dimensional_log_encoding_ACESproxy(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition n-dimensional arrays support.
        """

        linear = 0.18
        log = 426
        np.testing.assert_equal(
            log_encoding_ACESproxy(linear),
            log)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_equal(
            log_encoding_ACESproxy(linear),
            log)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_equal(
            log_encoding_ACESproxy(linear),
            log)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_ACESproxy(linear),
            log)

    @ignore_numpy_errors
    def test_nan_log_encoding_ACESproxy(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition nan support.
        """

        log_encoding_ACESproxy(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_ACESproxy(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy`
    definition unit tests methods.
    """

    def test_log_decoding_ACESproxy(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy` definition.
        """

        self.assertAlmostEqual(
            log_decoding_ACESproxy(64),
            0.001185737191792,
            places=7)

        self.assertAlmostEqual(
            log_decoding_ACESproxy(426),
            0.179244406001978,
            places=7)

        self.assertAlmostEqual(
            log_decoding_ACESproxy(550),
            1.0,
            places=7)

    def test_n_dimensional_log_decoding_ACESproxy(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy` definition n-dimensional arrays support.
        """

        log = 426.0
        linear = 0.179244406001978
        np.testing.assert_almost_equal(
            log_decoding_ACESproxy(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            log_decoding_ACESproxy(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_ACESproxy(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_ACESproxy(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_ACESproxy(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy` definition nan support.
        """

        log_decoding_ACESproxy(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_ACEScc(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition unit tests methods.
    """

    def test_log_encoding_ACEScc(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition.
        """

        self.assertAlmostEqual(
            log_encoding_ACEScc(0.0),
            -0.358447488584475,
            places=7)

        self.assertAlmostEqual(
            log_encoding_ACEScc(0.18),
            0.413588402492442,
            places=7)

        self.assertAlmostEqual(
            log_encoding_ACEScc(1.0),
            0.554794520547945,
            places=7)

    def test_n_dimensional_log_encoding_ACEScc(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.413588402492442
        np.testing.assert_almost_equal(
            log_encoding_ACEScc(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            log_encoding_ACEScc(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_ACEScc(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_ACEScc(linear),
            log,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_ACEScc(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition nan support.
        """

        log_encoding_ACEScc(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_ACEScc(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition unit tests methods.
    """

    def test_log_decoding_ACEScc(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition.
        """

        self.assertAlmostEqual(
            log_decoding_ACEScc(-0.358447488584475),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_decoding_ACEScc(0.413588402492442),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_ACEScc(0.554794520547945),
            1.0,
            places=7)

    def test_n_dimensional_log_decoding_ACEScc(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition n-dimensional arrays support.
        """

        log = 0.413588402492442
        linear = 0.18
        np.testing.assert_almost_equal(
            log_decoding_ACEScc(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            log_decoding_ACEScc(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_ACEScc(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_ACEScc(log),
            linear,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_ACEScc(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition nan support.
        """

        log_decoding_ACEScc(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_ACEScct(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition unit tests methods.
    """

    def test_log_encoding_ACEScct(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition.
        """

        self.assertAlmostEqual(
            log_encoding_ACEScct(0.0),
            0.072905534195835495,
            places=7)

        self.assertAlmostEqual(
            log_encoding_ACEScct(0.18),
            0.413588402492442,
            places=7)

        self.assertAlmostEqual(
            log_encoding_ACEScct(1.0),
            0.554794520547945,
            places=7)

    def test_n_dimensional_log_encoding_ACEScct(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition n-dimensional arrays support.
        """

        linear = 0.18
        log = 0.413588402492442
        np.testing.assert_almost_equal(
            log_encoding_ACEScct(linear),
            log,
            decimal=7)

        linear = np.tile(linear, 6)
        log = np.tile(log, 6)
        np.testing.assert_almost_equal(
            log_encoding_ACEScct(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3))
        log = np.reshape(log, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_ACEScct(linear),
            log,
            decimal=7)

        linear = np.reshape(linear, (2, 3, 1))
        log = np.reshape(log, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_ACEScct(linear),
            log,
            decimal=7)

    def test_ACEScc_equivalency_log_encoding_ACEScct(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition ACEScc equivalency, and explicit requirement
specified by AMPAS ACES specification S-2016-001 (https://github.com/ampas/\
aces-dev/blob/v1.0.3/documents/LaTeX/S-2016-001/introduction.tex#L14)
        """

        equiv = np.linspace(0.0078125, 222.86094420380761, 100)
        np.testing.assert_almost_equal(
            log_encoding_ACEScct(equiv),
            log_encoding_ACEScc(equiv),
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_ACEScct(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition nan support.
        """

        log_encoding_ACEScct(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_ACEScct(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition unit tests methods.
    """

    def test_log_decoding_ACEScct(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition.
        """

        self.assertAlmostEqual(
            log_decoding_ACEScct(0.072905534195835495),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_decoding_ACEScct(0.41358840249244228),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_ACEScct(0.554794520547945),
            1.0,
            places=7)

    def test_n_dimensional_log_decoding_ACEScct(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition n-dimensional arrays support.
        """

        log = 0.413588402492442
        linear = 0.18
        np.testing.assert_almost_equal(
            log_decoding_ACEScct(log),
            linear,
            decimal=7)

        log = np.tile(log, 6)
        linear = np.tile(linear, 6)
        np.testing.assert_almost_equal(
            log_decoding_ACEScct(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3))
        linear = np.reshape(linear, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_ACEScct(log),
            linear,
            decimal=7)

        log = np.reshape(log, (2, 3, 1))
        linear = np.reshape(linear, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_ACEScct(log),
            linear,
            decimal=7)

    def test_ACEScc_equivalency_log_decoding_ACEScct(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition ACEScc equivalency, and explicit requirement
specified by AMPAS ACES specification S-2016-001 (https://github.com/ampas/\
aces-dev/blob/v1.0.3/documents/LaTeX/S-2016-001/introduction.tex#L14)
        """

        equiv = np.linspace(0.15525114155251146, 1.0, 100)
        np.testing.assert_almost_equal(
            log_decoding_ACEScct(equiv),
            log_decoding_ACEScc(equiv),
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_ACEScct(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition nan support.
        """

        log_decoding_ACEScct(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
