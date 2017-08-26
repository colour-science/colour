#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.sony_slog`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    log_encoding_SLog, log_decoding_SLog, log_encoding_SLog2,
    log_decoding_SLog2, log_encoding_SLog3, log_decoding_SLog3)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestLogEncoding_SLog',
    'TestLogDecoding_SLog',
    'TestLogEncoding_SLog2',
    'TestLogDecoding_SLog2',
    'TestLogEncoding_SLog3',
    'TestLogDecoding_SLog3',
]


class TestLogEncoding_SLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog` definition unit tests methods.
    """

    def test_log_encoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog` definition.
        """

        self.assertAlmostEqual(
            log_encoding_SLog(0.0), 0.030001222851889307, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog(0.18), 0.37651272225459997, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog(0.18, 12), 0.37651272225459997, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog(0.18, 10, True), 0.38497081592867027, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog(0.18, 10, True, False),
            0.37082048237126808,
            places=7)

        self.assertAlmostEqual(
            log_encoding_SLog(1.0), 0.67264654494160947, places=7)

    def test_n_dimensional_log_encoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.37651272225459997
        np.testing.assert_almost_equal(log_encoding_SLog(L), V, decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(log_encoding_SLog(L), V, decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(log_encoding_SLog(L), V, decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_SLog(L), V, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog` definition nan support.
        """

        log_encoding_SLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_SLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog` definition unit tests methods.
    """

    def test_log_decoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog` definition.
        """

        self.assertAlmostEqual(
            log_decoding_SLog(0.030001222851889307), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog(0.37651272225459997), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog(0.37651272225459997, 12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog(0.38497081592867027, 10, True), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog(0.37082048237126808, 10, True, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_SLog(0.67264654494160947), 1.0, places=7)

    def test_n_dimensional_log_decoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog` definition n-dimensional arrays support.
        """

        V = 0.37651272225459997
        L = 0.18
        np.testing.assert_almost_equal(log_decoding_SLog(V), L, decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(log_decoding_SLog(V), L, decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(log_decoding_SLog(V), L, decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_SLog(V), L, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog` definition nan support.
        """

        log_decoding_SLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_SLog2(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog2` definition unit tests methods.
    """

    def test_log_encoding_SLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog2` definition.
        """

        self.assertAlmostEqual(
            log_encoding_SLog2(0.0), 0.030001222851889307, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog2(0.18), 0.32344951221501261, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog2(0.18, 12), 0.32344951221501261, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog2(0.18, 10, True), 0.33953252463377426, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog2(0.18, 10, True, False),
            0.32628653894679854,
            places=7)

        self.assertAlmostEqual(
            log_encoding_SLog2(1.0), 0.61021478759598913, places=7)

    def test_n_dimensional_log_encoding_SLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog2` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.32344951221501261
        np.testing.assert_almost_equal(log_encoding_SLog2(L), V, decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(log_encoding_SLog2(L), V, decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(log_encoding_SLog2(L), V, decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_SLog2(L), V, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_SLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog2` definition nan support.
        """

        log_encoding_SLog2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_SLog2(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog2` definition unit tests methods.
    """

    def test_log_decoding_SLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog2` definition.
        """

        self.assertAlmostEqual(
            log_decoding_SLog2(0.030001222851889307), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog2(0.32344951221501261), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog2(0.32344951221501261, 12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog2(0.33953252463377426, 10, True), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog2(0.32628653894679854, 10, True, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_SLog2(0.61021478759598913), 1.0, places=7)

    def test_n_dimensional_log_decoding_SLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog2` definition n-dimensional arrays support.
        """

        V = 0.32344951221501261
        L = 0.18
        np.testing.assert_almost_equal(log_decoding_SLog2(V), L, decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(log_decoding_SLog2(V), L, decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(log_decoding_SLog2(V), L, decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_SLog2(V), L, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_SLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog2` definition nan support.
        """

        log_decoding_SLog2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_SLog3(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog3` definition unit tests methods.
    """

    def test_log_encoding_SLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog3` definition.
        """

        self.assertAlmostEqual(
            log_encoding_SLog3(0.0), 0.092864125122189639, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog3(0.18), 0.41055718475073316, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog3(1.0), 0.59602734369012345, places=7)

    def test_n_dimensional_log_encoding_SLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog3` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.41055718475073316
        np.testing.assert_almost_equal(log_encoding_SLog3(L), V, decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(log_encoding_SLog3(L), V, decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(log_encoding_SLog3(L), V, decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_SLog3(L), V, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_SLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog3` definition nan support.
        """

        log_encoding_SLog3(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_SLog3(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog3` definition unit tests methods.
    """

    def test_log_decoding_SLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog3` definition.
        """

        self.assertAlmostEqual(
            log_decoding_SLog3(0.092864125122189639), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog3(0.41055718475073316), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog3(0.59602734369012345), 1.0, places=7)

    def test_n_dimensional_log_decoding_SLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog3` definition n-dimensional arrays support.
        """

        V = 0.41055718475073316
        L = 0.18
        np.testing.assert_almost_equal(log_decoding_SLog3(V), L, decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(log_decoding_SLog3(V), L, decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(log_decoding_SLog3(V), L, decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_SLog3(V), L, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_SLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog3` definition nan support.
        """

        log_decoding_SLog3(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
