# -*- coding: utf-8 -*-
"""
Defines the unit tests for the
:mod:`colour.models.rgb.transfer_functions.viper_log` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    log_encoding_ViperLog,
    log_decoding_ViperLog,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestLogEncoding_ViperLog',
    'TestLogDecoding_ViperLog',
]


class TestLogEncoding_ViperLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.viper_log.\
log_encoding_ViperLog` definition unit tests methods.
    """

    def test_log_encoding_ViperLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.viper_log.\
log_encoding_ViperLog` definition.
        """

        self.assertAlmostEqual(log_encoding_ViperLog(0.0), -np.inf, places=7)

        self.assertAlmostEqual(
            log_encoding_ViperLog(0.18), 0.636008067010413, places=7)

        self.assertAlmostEqual(log_encoding_ViperLog(1.0), 1.0, places=7)

    def test_n_dimensional_log_encoding_ViperLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.viper_log.\
log_encoding_ViperLog` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_ViperLog(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_almost_equal(log_encoding_ViperLog(x), y, decimal=7)

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_almost_equal(log_encoding_ViperLog(x), y, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_ViperLog(x), y, decimal=7)

    def test_domain_range_scale_log_encoding_ViperLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.viper_log.\
log_encoding_ViperLog` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_ViperLog(x)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_ViperLog(x * factor), y * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_ViperLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.viper_log.\
log_encoding_ViperLog` definition nan support.
        """

        log_encoding_ViperLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_ViperLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.viper_log.\
log_decoding_ViperLog` definition unit tests methods.
    """

    def test_log_decoding_ViperLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.viper_log.\
log_decoding_ViperLog` definition.
        """

        self.assertAlmostEqual(log_decoding_ViperLog(-np.inf), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_ViperLog(0.636008067010413), 0.18, places=7)

        self.assertAlmostEqual(log_decoding_ViperLog(1.0), 1.0, places=7)

    def test_n_dimensional_log_decoding_ViperLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.viper_log.\
log_decoding_ViperLog` definition n-dimensional arrays support.
        """

        y = 0.636008067010413
        x = log_decoding_ViperLog(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(log_decoding_ViperLog(y), x, decimal=7)

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(log_decoding_ViperLog(y), x, decimal=7)

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_ViperLog(y), x, decimal=7)

    def test_domain_range_scale_log_decoding_ViperLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.viper_log.\
log_decoding_ViperLog` definition domain and range scale support.
        """

        y = 0.636008067010413
        x = log_decoding_ViperLog(y)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_ViperLog(y * factor), x * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_ViperLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.viper_log.\
log_decoding_ViperLog` definition nan support.
        """

        log_decoding_ViperLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
