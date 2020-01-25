# -*- coding: utf-8 -*-
"""
Defines unit tests for
:mod:`colour.models.rgb.transfer_functions.arri_alexa_log_c` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (log_encoding_ALEXALogC,
                                                  log_decoding_ALEXALogC)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestLogEncoding_ALEXALogC', 'TestLogDecoding_ALEXALogC']


class TestLogEncoding_ALEXALogC(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.arri_alexa_log_c.\
log_encoding_ALEXALogC` definition unit tests methods.
    """

    def test_log_encoding_ALEXALogC(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arri_alexa_log_c.\
log_encoding_ALEXALogC` definition.
        """

        self.assertAlmostEqual(
            log_encoding_ALEXALogC(0.0), 0.092809000000000, places=7)

        self.assertAlmostEqual(
            log_encoding_ALEXALogC(0.18), 0.391006832034084, places=7)

        self.assertAlmostEqual(
            log_encoding_ALEXALogC(1.0), 0.570631558120417, places=7)

    def test_n_dimensional_log_encoding_ALEXALogC(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arri_alexa_log_c.\
log_encoding_ALEXALogC` definition n-dimensional arrays support.
        """

        x = 0.18
        t = log_encoding_ALEXALogC(x)

        x = np.tile(x, 6)
        t = np.tile(t, 6)
        np.testing.assert_almost_equal(log_encoding_ALEXALogC(x), t, decimal=7)

        x = np.reshape(x, (2, 3))
        t = np.reshape(t, (2, 3))
        np.testing.assert_almost_equal(log_encoding_ALEXALogC(x), t, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        t = np.reshape(t, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_ALEXALogC(x), t, decimal=7)

    def test_domain_range_scale_log_encoding_ALEXALogC(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arri_alexa_log_c.\
log_encoding_ALEXALogC` definition domain and range scale support.
        """

        x = 0.18
        t = log_encoding_ALEXALogC(x)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_ALEXALogC(x * factor), t * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_ALEXALogC(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arri_alexa_log_c.\
log_encoding_ALEXALogC` definition nan support.
        """

        log_encoding_ALEXALogC(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_ALEXALogC(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.arri_alexa_log_c.\
log_decoding_ALEXALogC` definition unit tests methods.
    """

    def test_log_decoding_ALEXALogC(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arri_alexa_log_c.\
log_decoding_ALEXALogC` definition.
        """

        self.assertAlmostEqual(log_decoding_ALEXALogC(0.092809), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_ALEXALogC(0.391006832034084), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_ALEXALogC(0.570631558120417), 1.0, places=7)

    def test_n_dimensional_log_decoding_ALEXALogC(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arri_alexa_log_c.\
log_decoding_ALEXALogC` definition n-dimensional arrays support.
        """

        t = 0.391006832034084
        x = log_decoding_ALEXALogC(t)

        t = np.tile(t, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(log_decoding_ALEXALogC(t), x, decimal=7)

        t = np.reshape(t, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(log_decoding_ALEXALogC(t), x, decimal=7)

        t = np.reshape(t, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_ALEXALogC(t), x, decimal=7)

    def test_domain_range_scale_log_decoding_ALEXALogC(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arri_alexa_log_c.\
log_decoding_ALEXALogC` definition domain and range scale support.
        """

        t = 0.391006832034084
        x = log_decoding_ALEXALogC(t)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_ALEXALogC(t * factor), x * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_ALEXALogC(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.arri_alexa_log_c.\
log_decoding_ALEXALogC` definition nan support.
        """

        log_decoding_ALEXALogC(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
