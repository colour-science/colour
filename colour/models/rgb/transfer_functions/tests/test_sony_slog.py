# -*- coding: utf-8 -*-
"""
Defines the unit tests for the
:mod:`colour.models.rgb.transfer_functions.sony_slog` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    log_encoding_SLog,
    log_decoding_SLog,
    log_encoding_SLog2,
    log_decoding_SLog2,
    log_encoding_SLog3,
    log_decoding_SLog3,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
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
            log_encoding_SLog(0.0), 0.088251291513446, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog(0.18), 0.384970815928670, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog(0.18, 12), 0.384688786026891, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog(0.18, 10, False), 0.376512722254600, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog(0.18, 10, False, False),
            0.359987846422154,
            places=7)

        self.assertAlmostEqual(
            log_encoding_SLog(1.0), 0.638551684622532, places=7)

    def test_n_dimensional_log_encoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_SLog(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_almost_equal(log_encoding_SLog(x), y, decimal=7)

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_almost_equal(log_encoding_SLog(x), y, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_SLog(x), y, decimal=7)

    def test_domain_range_scale_log_encoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_SLog(x)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_SLog(x * factor), y * factor, decimal=7)

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
            log_decoding_SLog(0.088251291513446), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog(0.384970815928670), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog(0.384688786026891, 12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog(0.376512722254600, 10, False), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog(0.359987846422154, 10, False, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_SLog(0.638551684622532), 1.0, places=7)

    def test_n_dimensional_log_decoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog` definition n-dimensional arrays support.
        """

        y = 0.384970815928670
        x = log_decoding_SLog(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(log_decoding_SLog(y), x, decimal=7)

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(log_decoding_SLog(y), x, decimal=7)

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_SLog(y), x, decimal=7)

    def test_domain_range_scale_log_decoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog` definition domain and range scale support.
        """

        y = 0.384970815928670
        x = log_decoding_SLog(y)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_SLog(y * factor), x * factor, decimal=7)

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
            log_encoding_SLog2(0.0), 0.088251291513446, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog2(0.18), 0.339532524633774, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog2(0.18, 12), 0.339283782857486, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog2(0.18, 10, False), 0.323449512215013, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog2(0.18, 10, False, False),
            0.307980741258647,
            places=7)

        self.assertAlmostEqual(
            log_encoding_SLog2(1.0), 0.585091059564112, places=7)

    def test_n_dimensional_log_encoding_SLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog2` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_SLog2(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_almost_equal(log_encoding_SLog2(x), y, decimal=7)

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_almost_equal(log_encoding_SLog2(x), y, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_SLog2(x), y, decimal=7)

    def test_domain_range_scale_log_encoding_SLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog2` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_SLog2(x)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_SLog2(x * factor), y * factor, decimal=7)

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
            log_decoding_SLog2(0.088251291513446), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog2(0.339532524633774), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog2(0.339283782857486, 12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog2(0.323449512215013, 10, False), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog2(0.307980741258647, 10, False, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_SLog2(0.585091059564112), 1.0, places=7)

    def test_n_dimensional_log_decoding_SLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog2` definition n-dimensional arrays support.
        """

        y = 0.339532524633774
        x = log_decoding_SLog2(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(log_decoding_SLog2(y), x, decimal=7)

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(log_decoding_SLog2(y), x, decimal=7)

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_SLog2(y), x, decimal=7)

    def test_domain_range_scale_log_decoding_SLog2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog2` definition domain and range scale support.
        """

        y = 0.339532524633774
        x = log_decoding_SLog2(y)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_SLog2(y * factor), x * factor, decimal=7)

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
            log_encoding_SLog3(0.0), 0.092864125122190, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog3(0.18), 0.41055718475073, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog3(0.18, 12), 0.410557184750733, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog3(0.18, 10, False), 0.406392694063927, places=7)

        self.assertAlmostEqual(
            log_encoding_SLog3(0.18, 10, False, False),
            0.393489294768447,
            places=7)

        self.assertAlmostEqual(
            log_encoding_SLog3(1.0), 0.596027343690123, places=7)

    def test_n_dimensional_log_encoding_SLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog3` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_SLog3(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_almost_equal(log_encoding_SLog3(x), y, decimal=7)

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_almost_equal(log_encoding_SLog3(x), y, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_SLog3(x), y, decimal=7)

    def test_domain_range_scale_log_encoding_SLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog3` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_SLog3(x)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_SLog3(x * factor), y * factor, decimal=7)

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
            log_decoding_SLog3(0.092864125122190), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog3(0.41055718475073), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog3(0.410557184750733, 12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog3(0.406392694063927, 10, False), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_SLog3(0.393489294768447, 10, False, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_SLog3(0.596027343690123), 1.0, places=7)

    def test_n_dimensional_log_decoding_SLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog3` definition n-dimensional arrays support.
        """

        y = 0.41055718475073
        x = log_decoding_SLog3(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(log_decoding_SLog3(y), x, decimal=7)

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(log_decoding_SLog3(y), x, decimal=7)

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_SLog3(y), x, decimal=7)

    def test_domain_range_scale_log_decoding_SLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog3` definition domain and range scale support.
        """

        y = 0.41055718475073
        x = log_decoding_SLog3(y)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_SLog3(y * factor), x * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_SLog3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog3` definition nan support.
        """

        log_decoding_SLog3(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
