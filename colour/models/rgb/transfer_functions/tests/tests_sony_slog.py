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
    log_encoding_SLog,
    log_decoding_SLog)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLogEncoding_SLog',
           'TestLogDecoding_SLog']


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
            log_encoding_SLog(0.0),
            0.030001222851889,
            places=7)

        self.assertAlmostEqual(
            log_encoding_SLog(0.18),
            0.359987846422154,
            places=7)

        self.assertAlmostEqual(
            log_encoding_SLog(1.0),
            0.653529251225308,
            places=7)

    def test_n_dimensional_log_encoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.359987846422154
        np.testing.assert_almost_equal(
            log_encoding_SLog(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            log_encoding_SLog(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_SLog(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_SLog(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_encoding_SLog` definition nan support.
        """

        log_encoding_SLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


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
            log_decoding_SLog(0.0),
            -0.005545814446077,
            places=7)

        self.assertAlmostEqual(
            log_decoding_SLog(0.359987846422154),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_SLog(0.653529251225308),
            1.0,
            places=7)

    def test_n_dimensional_log_decoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog` definition n-dimensional arrays support.
        """

        V = 0.359987846422154
        L = 0.18
        np.testing.assert_almost_equal(
            log_decoding_SLog(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            log_decoding_SLog(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_SLog(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_SLog(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_SLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sony_slog.\
log_decoding_SLog` definition nan support.
        """

        log_decoding_SLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
