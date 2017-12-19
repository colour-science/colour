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
    log_encoding_CanonLog, log_decoding_CanonLog, log_encoding_CanonLog2,
    log_decoding_CanonLog2, log_encoding_CanonLog3, log_decoding_CanonLog3)

from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLogEncoding_CanonLog', 'TestLogDecoding_CanonLog']


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
            log_encoding_CanonLog(-0.1), -0.100573065760254, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(0.0), 0.073059700000000, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(0.18), 0.327953896935809, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(0.18, 12), 0.327953896935809, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(0.18, 10, False),
            0.309927895622526,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(0.18, 10, False, False),
            0.291311816470381,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(1.0), 0.649551737177417, places=7)

    def test_n_dimensional_log_encoding_CanonLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.327953896935809
        np.testing.assert_almost_equal(log_encoding_CanonLog(L), V, decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(log_encoding_CanonLog(L), V, decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(log_encoding_CanonLog(L), V, decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_CanonLog(L), V, decimal=7)

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
            log_decoding_CanonLog(-0.100573065760254), -0.1, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.073059700000000), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.327953896935809), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.327953896935809, 12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.309927895622526, 10, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.291311816470381, 10, False, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.649551737177417), 1.0, places=7)

    def test_n_dimensional_log_decoding_CanonLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog` definition n-dimensional arrays support.
        """

        V = 0.327953896935809
        L = 0.18
        np.testing.assert_almost_equal(log_decoding_CanonLog(V), L, decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(log_decoding_CanonLog(V), L, decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(log_decoding_CanonLog(V), L, decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_CanonLog(V), L, decimal=7)

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
            log_encoding_CanonLog2(-0.1), -0.25450187789127, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(0.0), 0.035388127999999, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(0.18), 0.392025745397009, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(0.18, 12), 0.392025745397009, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(0.18, 10, False),
            0.384751526873448,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(0.18, 10, False, False),
            0.370549620564055,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(1.0), 0.596362507310829, places=7)

    def test_n_dimensional_log_encoding_CanonLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog2` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.392025745397009
        np.testing.assert_almost_equal(log_encoding_CanonLog2(L), V, decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(log_encoding_CanonLog2(L), V, decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(log_encoding_CanonLog2(L), V, decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_CanonLog2(L), V, decimal=7)

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
            log_decoding_CanonLog2(-0.25450187789127), -0.1, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.035388127999999), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.392025745397009), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.392025745397009, 12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.384751526873448, 10, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.370549620564055, 10, False, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.596362507310829), 1.0, places=7)

    def test_n_dimensional_log_decoding_CanonLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog2` definition n-dimensional arrays support.
        """

        V = 0.392025745397009
        L = 0.18
        np.testing.assert_almost_equal(log_decoding_CanonLog2(V), L, decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(log_decoding_CanonLog2(V), L, decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(log_decoding_CanonLog2(V), L, decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_CanonLog2(V), L, decimal=7)

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
            log_encoding_CanonLog3(-0.1), -0.112680937128071, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(0.0), 0.073059361000000, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(0.18), 0.327953567219893, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(0.18, 12), 0.327953567219893, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(0.18, 10, False),
            0.309927510577569,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(0.18, 10, False, False),
            0.292973783129810,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(1.0), 0.604593819123392, places=7)

    def test_n_dimensional_log_encoding_CanonLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog3` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.327953567219893
        np.testing.assert_almost_equal(log_encoding_CanonLog3(L), V, decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(log_encoding_CanonLog3(L), V, decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(log_encoding_CanonLog3(L), V, decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_CanonLog3(L), V, decimal=7)

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
            log_decoding_CanonLog3(-0.112680937128071), -0.1, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.073059361000000), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.327953567219893), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.327953567219893, 12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.309927510577569, 10, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.292973783129810, 10, False, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.604593819123392), 1.0, places=7)

    def test_n_dimensional_log_decoding_CanonLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog3` definition n-dimensional arrays support.
        """

        V = 0.327953567219893
        L = 0.18
        np.testing.assert_almost_equal(log_decoding_CanonLog3(V), L, decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(log_decoding_CanonLog3(V), L, decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(log_decoding_CanonLog3(V), L, decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_CanonLog3(V), L, decimal=7)

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
