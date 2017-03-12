#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.canon_log`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    log_encoding_CanonLog,
    log_decoding_CanonLog,
    log_encoding_CanonLog2,
    log_decoding_CanonLog2,
    log_encoding_CanonLog3,
    log_decoding_CanonLog3)

from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLogEncoding_CanonLog',
           'TestLogDecoding_CanonLog']


class TestLogEncoding_CanonLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog` definition unit tests methods.
    """

    def test_log_encoding_CanonLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog` definition.
        """

        self.assertAlmostEqual(
            log_encoding_CanonLog(-0.1),
            -0.088052640318143,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(0.0),
            0.073059700000000,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(0.18),
            0.312012855550395,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(1.0),
            0.627408304537653,
            places=7)

    def test_n_dimensional_log_encoding_CanonLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.312012855550395
        np.testing.assert_almost_equal(
            log_encoding_CanonLog(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            log_encoding_CanonLog(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_CanonLog(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_CanonLog(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog` definition nan support.
        """

        log_encoding_CanonLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_CanonLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog` definition unit tests methods.
    """

    def test_log_decoding_CanonLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog` definition.
        """

        self.assertAlmostEqual(
            log_decoding_CanonLog(-0.088052640318143),
            -0.1,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.073059700000000),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.312012855550395),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.627408304537653),
            1.0,
            places=7)

    def test_n_dimensional_log_decoding_CanonLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog` definition n-dimensional arrays support.
        """

        V = 0.312012855550395
        L = 0.18
        np.testing.assert_almost_equal(
            log_decoding_CanonLog(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            log_decoding_CanonLog(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_CanonLog(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_CanonLog(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog` definition nan support.
        """

        log_decoding_CanonLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_CanonLog2(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog2` definition unit tests methods.
    """

    def test_log_encoding_CanonLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog2` definition.
        """

        self.assertAlmostEqual(
            log_encoding_CanonLog2(-0.1),
            -0.242871750266172,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(0.0),
            0.035388127999999,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(0.18),
            0.379864582222983,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(1.0),
            0.583604185577946,
            places=7)

    def test_n_dimensional_log_encoding_CanonLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog2` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.379864582222983
        np.testing.assert_almost_equal(
            log_encoding_CanonLog2(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            log_encoding_CanonLog2(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_CanonLog2(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_CanonLog2(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog2` definition nan support.
        """

        log_encoding_CanonLog2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_CanonLog2(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog2` definition unit tests methods.
    """

    def test_log_decoding_CanonLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog2` definition.
        """

        self.assertAlmostEqual(
            log_decoding_CanonLog2(-0.242871750266172),
            -0.1,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.035388127999999),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.379864582222983),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.583604185577946),
            1.0,
            places=7)

    def test_n_dimensional_log_decoding_CanonLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog2` definition n-dimensional arrays support.
        """

        V = 0.379864582222983
        L = 0.18
        np.testing.assert_almost_equal(
            log_decoding_CanonLog2(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            log_decoding_CanonLog2(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_CanonLog2(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_CanonLog2(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog2` definition nan support.
        """

        log_decoding_CanonLog2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_CanonLog3(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog3` definition unit tests methods.
    """

    def test_log_encoding_CanonLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog3` definition.
        """

        self.assertAlmostEqual(
            log_encoding_CanonLog3(-0.1),
            -0.100664645796433,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(0.0),
            0.073059361000000,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(0.18),
            0.313436005886328,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(1.0),
            0.586137530935974,
            places=7)

    def test_n_dimensional_log_encoding_CanonLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog3` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.313436005886328
        np.testing.assert_almost_equal(
            log_encoding_CanonLog3(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            log_encoding_CanonLog3(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_CanonLog3(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_CanonLog3(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog3` definition nan support.
        """

        log_encoding_CanonLog3(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_CanonLog3(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog3` definition unit tests methods.
    """

    def test_log_decoding_CanonLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog3` definition.
        """

        self.assertAlmostEqual(
            log_decoding_CanonLog3(-0.100664645796433),
            -0.1,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.073059361000000),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.313436005886328),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.586137530935974),
            1.0,
            places=7)

    def test_n_dimensional_log_decoding_CanonLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog3` definition n-dimensional arrays support.
        """

        V = 0.313436005886328
        L = 0.18
        np.testing.assert_almost_equal(
            log_decoding_CanonLog3(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            log_decoding_CanonLog3(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_CanonLog3(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_CanonLog3(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog3` definition nan support.
        """

        log_decoding_CanonLog3(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
