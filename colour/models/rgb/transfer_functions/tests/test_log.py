# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.log`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (log_encoding_Log2,
                                                  log_decoding_Log2)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestLogEncoding_Log2', 'TestLogDecoding_Log2']


class TestLogEncoding_Log2(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.log.log_encoding_Log2`
    definition unit tests methods.
    """

    def test_log_encoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition.
        """

        self.assertAlmostEqual(
            log_encoding_Log2(18), 0.407732889704347, places=7)

        self.assertAlmostEqual(
            log_encoding_Log2(18, 0.12), 0.443642737727224, places=7)

        self.assertAlmostEqual(
            log_encoding_Log2(18, 0.12, 0.0045), 0.443556955303088, places=7)

        self.assertAlmostEqual(
            log_encoding_Log2(18, 0.12, 0.0045, 15.0),
            0.481765775765788,
            places=7)

    def test_n_dimensional_log_encoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition n-dimensional arrays support.
        """

        x = 18
        y = log_encoding_Log2(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_almost_equal(log_encoding_Log2(x), y, decimal=7)

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_almost_equal(log_encoding_Log2(x), y, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_Log2(x), y, decimal=7)

    def test_domain_range_scale_log_encoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition domain and range scale support.
        """

        x = 18
        y = log_encoding_Log2(x)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_Log2(x * factor), y * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition nan support.
        """

        log_encoding_Log2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Log2(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition unit tests methods.
    """

    def test_log_decoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition.
        """

        self.assertAlmostEqual(
            log_decoding_Log2(0.407732889704347), 18.000000000000075, places=7)

        self.assertAlmostEqual(
            log_decoding_Log2(0.443642737727224, 0.12),
            17.999999999999904,
            places=7)

        self.assertAlmostEqual(
            log_decoding_Log2(0.443556955303088, 0.12, 0.0045),
            17.999999999999982,
            places=7)

        self.assertAlmostEqual(
            log_decoding_Log2(0.481765775765788, 0.12, 0.0045, 15.0),
            17.999999999999918,
            places=7)

    def test_n_dimensional_log_decoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition n-dimensional arrays support.
        """

        y = 0.407732889704347
        x = log_decoding_Log2(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(log_decoding_Log2(y), x, decimal=7)

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(log_decoding_Log2(y), x, decimal=7)

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_Log2(y), x, decimal=7)

    def test_domain_range_scale_log_decoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition domain and range scale support.
        """

        y = 0.407732889704347
        x = log_decoding_Log2(y)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_Log2(y * factor), x * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_Log2(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition nan support.
        """

        log_decoding_Log2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
