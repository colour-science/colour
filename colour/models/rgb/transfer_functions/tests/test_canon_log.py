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

from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestLogEncoding_CanonLog', 'TestLogDecoding_CanonLog',
    'TestLogEncoding_CanonLog2', 'TestLogDecoding_CanonLog2',
    'TestLogEncoding_CanonLog3', 'TestLogDecoding_CanonLog3'
]


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
            log_encoding_CanonLog(-0.1), -0.023560122781997, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(0.0), 0.125122480156403, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(0.18), 0.343389651726069, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(0.18, 12), 0.343138084215647, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(0.18, 10, False),
            0.327953896935809,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(0.18, 10, False, False),
            0.312012855550395,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog(1.0), 0.618775485598649, places=7)

    def test_n_dimensional_log_encoding_CanonLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog` definition n-dimensional arrays support.
        """

        x = 0.18
        clog = log_encoding_CanonLog(x)

        x = np.tile(x, 6)
        clog = np.tile(clog, 6)
        np.testing.assert_almost_equal(
            log_encoding_CanonLog(x), clog, decimal=7)

        x = np.reshape(x, (2, 3))
        clog = np.reshape(clog, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_CanonLog(x), clog, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        clog = np.reshape(clog, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_CanonLog(x), clog, decimal=7)

    def test_domain_range_scale_log_encoding_CanonLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog` definition domain and range scale support.
        """

        x = 0.18
        clog = log_encoding_CanonLog(x)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_CanonLog(x * factor),
                    clog * factor,
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
            log_decoding_CanonLog(-0.023560122781997), -0.1, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.125122480156403), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.343389651726069), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.343138084215647, 12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.327953896935809, 10, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.312012855550395, 10, False, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog(0.618775485598649), 1.0, places=7)

    def test_n_dimensional_log_decoding_CanonLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog` definition n-dimensional arrays support.
        """

        clog = 0.343389651726069
        x = log_decoding_CanonLog(clog)

        clog = np.tile(clog, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(
            log_decoding_CanonLog(clog), x, decimal=7)

        clog = np.reshape(clog, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_CanonLog(clog), x, decimal=7)

        clog = np.reshape(clog, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_CanonLog(clog), x, decimal=7)

    def test_domain_range_scale_log_decoding_CanonLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog` definition domain and range scale support.
        """

        clog = 0.343389651726069
        x = log_decoding_CanonLog(clog)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_CanonLog(clog * factor),
                    x * factor,
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
            log_encoding_CanonLog2(-0.1), -0.155370131996824, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(0.0), 0.092864125247312, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(0.18), 0.398254694983167, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(0.18, 12), 0.397962933301861, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(0.18, 10, False),
            0.392025745397009,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(0.18, 10, False, False),
            0.379864582222983,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog2(1.0), 0.573229282897641, places=7)

    def test_n_dimensional_log_encoding_CanonLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog2` definition n-dimensional arrays support.
        """

        x = 0.18
        clog2 = log_encoding_CanonLog2(x)

        x = np.tile(x, 6)
        clog2 = np.tile(clog2, 6)
        np.testing.assert_almost_equal(
            log_encoding_CanonLog2(x), clog2, decimal=7)

        x = np.reshape(x, (2, 3))
        clog2 = np.reshape(clog2, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_CanonLog2(x), clog2, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        clog2 = np.reshape(clog2, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_CanonLog2(x), clog2, decimal=7)

    def test_domain_range_scale_log_encoding_CanonLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog2` definition domain and range scale support.
        """

        x = 0.18
        clog2 = log_encoding_CanonLog2(x)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_CanonLog2(x * factor),
                    clog2 * factor,
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
            log_decoding_CanonLog2(-0.155370131996824), -0.1, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.092864125247312), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.398254694983167), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.397962933301861, 12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.392025745397009, 10, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.379864582222983, 10, False, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog2(0.573229282897641), 1.0, places=7)

    def test_n_dimensional_log_decoding_CanonLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog2` definition n-dimensional arrays support.
        """

        clog2 = 0.398254694983167
        x = log_decoding_CanonLog2(clog2)

        clog2 = np.tile(clog2, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(
            log_decoding_CanonLog2(clog2), x, decimal=7)

        clog2 = np.reshape(clog2, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_CanonLog2(clog2), x, decimal=7)

        clog2 = np.reshape(clog2, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_CanonLog2(clog2), x, decimal=7)

    def test_domain_range_scale_log_decoding_CanonLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog2` definition domain and range scale support.
        """

        clog = 0.398254694983167
        x = log_decoding_CanonLog2(clog)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_CanonLog2(clog * factor),
                    x * factor,
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
            log_encoding_CanonLog3(-0.1), -0.028494506076432, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(0.0), 0.125122189869013, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(0.18), 0.343389369388687, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(0.18, 12), 0.343137802085105, places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(0.18, 10, False),
            0.327953567219893,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(0.18, 10, False, False),
            0.313436005886328,
            places=7)

        self.assertAlmostEqual(
            log_encoding_CanonLog3(1.0), 0.580277796238604, places=7)

    def test_n_dimensional_log_encoding_CanonLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog3` definition n-dimensional arrays support.
        """

        x = 0.18
        clog3 = log_encoding_CanonLog3(x)

        x = np.tile(x, 6)
        clog3 = np.tile(clog3, 6)
        np.testing.assert_almost_equal(
            log_encoding_CanonLog3(x), clog3, decimal=7)

        x = np.reshape(x, (2, 3))
        clog3 = np.reshape(clog3, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_CanonLog3(x), clog3, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        clog3 = np.reshape(clog3, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_CanonLog3(x), clog3, decimal=7)

    def test_domain_range_scale_log_encoding_CanonLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_encoding_CanonLog3` definition domain and range scale support.
        """

        x = 0.18
        clog3 = log_encoding_CanonLog3(x)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_CanonLog3(x * factor),
                    clog3 * factor,
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
            log_decoding_CanonLog3(-0.028494506076432), -0.1, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.125122189869013), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.343389369388687), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.343137802085105, 12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.327953567219893, 10, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.313436005886328, 10, False, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_CanonLog3(0.580277796238604), 1.0, places=7)

    def test_n_dimensional_log_decoding_CanonLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog3` definition n-dimensional arrays support.
        """

        clog3 = 0.343389369388687
        x = log_decoding_CanonLog3(clog3)

        clog3 = np.tile(clog3, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(
            log_decoding_CanonLog3(clog3), x, decimal=7)

        clog3 = np.reshape(clog3, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_CanonLog3(clog3), x, decimal=7)

        clog3 = np.reshape(clog3, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_CanonLog3(clog3), x, decimal=7)

    def test_domain_range_scale_log_decoding_CanonLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.canon_log.\
log_decoding_CanonLog3` definition domain and range scale support.
        """

        clog = 0.343389369388687
        x = log_decoding_CanonLog3(clog)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_CanonLog3(clog * factor),
                    x * factor,
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
