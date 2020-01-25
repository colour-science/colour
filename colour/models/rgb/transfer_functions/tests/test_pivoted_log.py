# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.\
pivoted_log` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (log_encoding_PivotedLog,
                                                  log_decoding_PivotedLog)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestLogEncoding_PivotedLog', 'TestLogDecoding_PivotedLog']


class TestLogEncoding_PivotedLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_encoding_PivotedLog` definition unit tests methods.
    """

    def test_log_encoding_PivotedLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_encoding_PivotedLog` definition.
        """

        self.assertAlmostEqual(log_encoding_PivotedLog(0.0), -np.inf, places=7)

        self.assertAlmostEqual(
            log_encoding_PivotedLog(0.18), 0.434995112414467, places=7)

        self.assertAlmostEqual(
            log_encoding_PivotedLog(1.0), 0.653390272208219, places=7)

    def test_n_dimensional_log_encoding_PivotedLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_encoding_PivotedLog` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_PivotedLog(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_almost_equal(
            log_encoding_PivotedLog(x), y, decimal=7)

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_PivotedLog(x), y, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_PivotedLog(x), y, decimal=7)

    def test_domain_range_scale_log_encoding_PivotedLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_encoding_PivotedLog` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_PivotedLog(x)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_PivotedLog(x * factor), y * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_PivotedLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_encoding_PivotedLog` definition nan support.
        """

        log_encoding_PivotedLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_PivotedLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_decoding_PivotedLog` definition unit tests methods.
    """

    def test_log_decoding_PivotedLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_decoding_PivotedLog` definition.
        """

        self.assertAlmostEqual(log_decoding_PivotedLog(-np.inf), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_PivotedLog(0.434995112414467), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_PivotedLog(0.653390272208219), 1.0, places=7)

    def test_n_dimensional_log_decoding_PivotedLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_decoding_PivotedLog` definition n-dimensional arrays support.
        """

        y = 0.434995112414467
        x = log_decoding_PivotedLog(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(
            log_decoding_PivotedLog(y), x, decimal=7)

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_PivotedLog(y), x, decimal=7)

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_PivotedLog(y), x, decimal=7)

    def test_domain_range_scale_log_decoding_PivotedLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_decoding_PivotedLog` definition domain and range scale support.
        """

        y = 0.434995112414467
        x = log_decoding_PivotedLog(y)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_PivotedLog(y * factor), x * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_PivotedLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_decoding_PivotedLog` definition nan support.
        """

        log_decoding_PivotedLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
