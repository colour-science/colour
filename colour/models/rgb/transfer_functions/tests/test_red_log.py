# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.red_log`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    log_encoding_REDLog, log_decoding_REDLog, log_encoding_REDLogFilm,
    log_decoding_REDLogFilm, log_encoding_Log3G10, log_decoding_Log3G10,
    log_encoding_Log3G12, log_decoding_Log3G12)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Development'

__all__ = [
    'TestLogEncoding_REDLog', 'TestLogDecoding_REDLog',
    'TestLogDecoding_REDLogFilm', 'TestLogDecoding_REDLogFilm',
    'TestLogDecoding_Log3G10', 'TestLogDecoding_Log3G10',
    'TestLogDecoding_Log3G12', 'TestLogDecoding_Log3G12'
]


class TestLogEncoding_REDLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_REDLog` definition unit tests methods.
    """

    def test_log_encoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_REDLog` definition.
        """

        self.assertAlmostEqual(log_encoding_REDLog(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            log_encoding_REDLog(0.18), 0.637621845988175, places=7)

        self.assertAlmostEqual(log_encoding_REDLog(1.0), 1.0, places=7)

    def test_n_dimensional_log_encoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_REDLog` definition n-dimensional arrays support.
        """

        x = 0.18
        y = 0.637621845988175
        np.testing.assert_almost_equal(log_encoding_REDLog(x), y, decimal=7)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_almost_equal(log_encoding_REDLog(x), y, decimal=7)

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_almost_equal(log_encoding_REDLog(x), y, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_REDLog(x), y, decimal=7)

    def test_domain_range_scale_log_encoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_REDLog` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_REDLog(x)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_REDLog(x * factor), y * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_REDLog` definition nan support.
        """

        log_encoding_REDLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_REDLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_REDLog` definition unit tests methods.
    """

    def test_log_decoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_REDLog` definition.
        """

        self.assertAlmostEqual(log_decoding_REDLog(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_REDLog(0.637621845988175), 0.18, places=7)

        self.assertAlmostEqual(log_decoding_REDLog(1.0), 1.0, places=7)

    def test_n_dimensional_log_decoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_REDLog` definition n-dimensional arrays support.
        """

        y = 0.637621845988175
        x = 0.18
        np.testing.assert_almost_equal(log_decoding_REDLog(y), x, decimal=7)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(log_decoding_REDLog(y), x, decimal=7)

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(log_decoding_REDLog(y), x, decimal=7)

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_REDLog(y), x, decimal=7)

    def test_domain_range_scale_log_decoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_REDLog` definition domain and range scale support.
        """

        y = 0.637621845988175
        x = log_decoding_REDLog(y)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_REDLog(y * factor), x * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_REDLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_REDLog` definition nan support.
        """

        log_decoding_REDLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_REDLogFilm(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_REDLogFilm` definition unit tests methods.
    """

    def test_log_encoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_REDLogFilm` definition.
        """

        self.assertAlmostEqual(
            log_encoding_REDLogFilm(0.0), 0.092864125122190, places=7)

        self.assertAlmostEqual(
            log_encoding_REDLogFilm(0.18), 0.457319613085418, places=7)

        self.assertAlmostEqual(
            log_encoding_REDLogFilm(1.0), 0.669599217986315, places=7)

    def test_n_dimensional_log_encoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_REDLogFilm` definition n-dimensional arrays support.
        """

        x = 0.18
        y = 0.457319613085418
        np.testing.assert_almost_equal(
            log_encoding_REDLogFilm(x), y, decimal=7)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_almost_equal(
            log_encoding_REDLogFilm(x), y, decimal=7)

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_REDLogFilm(x), y, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_REDLogFilm(x), y, decimal=7)

    def test_domain_range_scale_log_encoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_REDLogFilm` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_REDLogFilm(x)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_REDLogFilm(x * factor), y * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_REDLogFilm` definition nan support.
        """

        log_encoding_REDLogFilm(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_REDLogFilm(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_REDLogFilm` definition unit tests methods.
    """

    def test_log_decoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_REDLogFilm` definition.
        """

        self.assertAlmostEqual(
            log_decoding_REDLogFilm(0.092864125122190), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_REDLogFilm(0.457319613085418), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_REDLogFilm(0.669599217986315), 1.0, places=7)

    def test_n_dimensional_log_decoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_REDLogFilm` definition n-dimensional arrays support.
        """

        y = 0.457319613085418
        x = 0.18
        np.testing.assert_almost_equal(
            log_decoding_REDLogFilm(y), x, decimal=7)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(
            log_decoding_REDLogFilm(y), x, decimal=7)

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_REDLogFilm(y), x, decimal=7)

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_REDLogFilm(y), x, decimal=7)

    def test_domain_range_scale_log_decoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_REDLogFilm` definition domain and range scale support.
        """

        y = 0.457319613085418
        x = log_decoding_REDLogFilm(y)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_REDLogFilm(y * factor), x * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_REDLogFilm(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_REDLogFilm` definition nan support.
        """

        log_decoding_REDLogFilm(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_Log3G10(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_Log3G10` definition unit tests methods.
    """

    def test_log_encoding_Log3G10(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_Log3G10` definition.
        """

        self.assertAlmostEqual(
            log_encoding_Log3G10(-1.0, legacy_curve=True),
            -0.496483569056003,
            places=7)

        self.assertAlmostEqual(
            log_encoding_Log3G10(0.0, legacy_curve=True), 0.0, places=7)

        self.assertAlmostEqual(
            log_encoding_Log3G10(0.18, legacy_curve=True),
            0.333333644207707,
            places=7)

        self.assertAlmostEqual(
            log_encoding_Log3G10(-1.0, legacy_curve=False),
            -0.491512777522511,
            places=7)

        self.assertAlmostEqual(
            log_encoding_Log3G10(0.0, legacy_curve=False),
            0.091551487714745,
            places=7)

        self.assertAlmostEqual(
            log_encoding_Log3G10(0.18, legacy_curve=False),
            0.333332912025992,
            places=7)

    def test_n_dimensional_log_encoding_Log3G10(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_Log3G10` definition n-dimensional arrays support.
        """

        x = 0.18
        y1 = 0.333333644207707
        y2 = 0.333332912025992
        np.testing.assert_almost_equal(
            log_encoding_Log3G10(x, legacy_curve=True), y1, decimal=7)
        np.testing.assert_almost_equal(
            log_encoding_Log3G10(x, legacy_curve=False), y2, decimal=7)

        x = np.tile(x, 6)
        y1 = np.tile(y1, 6)
        y2 = np.tile(y2, 6)
        np.testing.assert_almost_equal(
            log_encoding_Log3G10(x, legacy_curve=True), y1, decimal=7)
        np.testing.assert_almost_equal(
            log_encoding_Log3G10(x, legacy_curve=False), y2, decimal=7)

        x = np.reshape(x, (2, 3))
        y1 = np.reshape(y1, (2, 3))
        y2 = np.reshape(y2, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_Log3G10(x, legacy_curve=True), y1, decimal=7)
        np.testing.assert_almost_equal(
            log_encoding_Log3G10(x, legacy_curve=False), y2, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        y1 = np.reshape(y1, (2, 3, 1))
        y2 = np.reshape(y2, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_Log3G10(x, legacy_curve=True), y1, decimal=7)
        np.testing.assert_almost_equal(
            log_encoding_Log3G10(x, legacy_curve=False), y2, decimal=7)

    def test_domain_range_scale_log_encoding_Log3G10(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_Log3G10` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_Log3G10(x)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_Log3G10(x * factor), y * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_Log3G10(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_Log3G10` definition nan support.
        """

        log_encoding_Log3G10(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]),
            legacy_curve=True)
        log_encoding_Log3G10(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]),
            legacy_curve=False)


class TestLogDecoding_Log3G10(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_Log3G10` definition unit tests methods.
    """

    def test_log_decoding_Log3G10(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_Log3G10` definition.
        """

        self.assertAlmostEqual(
            log_decoding_Log3G10(-0.496483569056003, legacy_curve=True),
            -1.0,
            places=7)

        self.assertAlmostEqual(
            log_decoding_Log3G10(0.0, legacy_curve=True), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_Log3G10(0.333333644207707, legacy_curve=True),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_Log3G10(-0.491512777522511, legacy_curve=False),
            -1.0,
            places=7)

        self.assertAlmostEqual(
            log_decoding_Log3G10(0.091551487714745, legacy_curve=False),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_decoding_Log3G10(0.333332912025992, legacy_curve=False),
            0.18,
            places=7)

    def test_n_dimensional_log_decoding_Log3G10(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_Log3G10` definition n-dimensional arrays support.
        """

        y1 = 0.333333644207707
        y2 = 0.333332912025992
        x = 0.18
        np.testing.assert_almost_equal(
            log_decoding_Log3G10(y1, legacy_curve=True), x, decimal=7)
        np.testing.assert_almost_equal(
            log_decoding_Log3G10(y2, legacy_curve=False), x, decimal=7)

        y1 = np.tile(y1, 6)
        y2 = np.tile(y2, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(
            log_decoding_Log3G10(y1, legacy_curve=True), x, decimal=7)
        np.testing.assert_almost_equal(
            log_decoding_Log3G10(y2, legacy_curve=False), x, decimal=7)

        y1 = np.reshape(y1, (2, 3))
        y2 = np.reshape(y2, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_Log3G10(y1, legacy_curve=True), x, decimal=7)
        np.testing.assert_almost_equal(
            log_decoding_Log3G10(y2, legacy_curve=False), x, decimal=7)

        y1 = np.reshape(y1, (2, 3, 1))
        y2 = np.reshape(y2, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_Log3G10(y1, legacy_curve=True), x, decimal=7)
        np.testing.assert_almost_equal(
            log_decoding_Log3G10(y2, legacy_curve=False), x, decimal=7)

    def test_domain_range_scale_log_decoding_Log3G10(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_Log3G10` definition domain and range scale support.
        """

        y = 0.333333644207707
        x = log_decoding_Log3G10(y)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_Log3G10(y * factor), x * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_Log3G10(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_Log3G10` definition nan support.
        """

        log_decoding_Log3G10(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]),
            legacy_curve=True)
        log_decoding_Log3G10(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]),
            legacy_curve=False)


class TestLogEncoding_Log3G12(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_Log3G12` definition unit tests methods.
    """

    def test_log_encoding_Log3G12(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_Log3G12` definition.
        """

        self.assertAlmostEqual(log_encoding_Log3G12(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            log_encoding_Log3G12(0.18), 0.333332662015923, places=7)

        self.assertAlmostEqual(
            log_encoding_Log3G12(1.0), 0.469991923234319, places=7)

        self.assertAlmostEqual(
            log_encoding_Log3G12(0.18 * 2 ** 12), 0.999997986792394, places=7)

    def test_n_dimensional_log_encoding_Log3G12(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_Log3G12` definition n-dimensional arrays support.
        """

        x = 0.18
        y = 0.333332662015923
        np.testing.assert_almost_equal(log_encoding_Log3G12(x), y, decimal=7)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_almost_equal(log_encoding_Log3G12(x), y, decimal=7)

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_almost_equal(log_encoding_Log3G12(x), y, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_Log3G12(x), y, decimal=7)

    def test_domain_range_scale_log_encoding_Log3G12(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_Log3G12` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_Log3G12(x)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_Log3G12(x * factor), y * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_Log3G12(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_encoding_Log3G12` definition nan support.
        """

        log_encoding_Log3G12(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Log3G12(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_Log3G12` definition unit tests methods.
    """

    def test_log_decoding_Log3G12(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_Log3G12` definition.
        """

        self.assertAlmostEqual(log_decoding_Log3G12(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_Log3G12(0.333332662015923), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_Log3G12(0.469991923234319), 1.0, places=7)

        self.assertAlmostEqual(
            log_decoding_Log3G12(1.0), 737.29848406719, places=7)

    def test_n_dimensional_log_decoding_Log3G12(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_Log3G12` definition n-dimensional arrays support.
        """

        y = 0.333332662015923
        x = 0.18
        np.testing.assert_almost_equal(log_decoding_Log3G12(y), x, decimal=7)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(log_decoding_Log3G12(y), x, decimal=7)

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(log_decoding_Log3G12(y), x, decimal=7)

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_Log3G12(y), x, decimal=7)

    def test_domain_range_scale_log_decoding_Log3G12(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_Log3G12` definition domain and range scale support.
        """

        y = 0.18
        x = log_decoding_Log3G12(y)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_Log3G12(y * factor), x * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_Log3G12(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.red_log.\
log_decoding_Log3G12` definition nan support.
        """

        log_decoding_Log3G12(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
