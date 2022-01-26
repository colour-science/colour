# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
nikon_nlog` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    log_encoding_NLog,
    log_decoding_NLog,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestLogEncoding_VLog',
    'TestLogDecoding_VLog',
]


class TestLogEncoding_VLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.nikon_nlog.\
log_encoding_NLog` definition unit tests methods.
    """

    def test_log_encoding_NLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.nikon_nlog.\
log_encoding_NLog` definition.
        """

        self.assertAlmostEqual(
            log_encoding_NLog(0.0), 0.124372627896372, places=7)

        self.assertAlmostEqual(
            log_encoding_NLog(0.18), 0.363667770117139, places=7)

        self.assertAlmostEqual(
            log_encoding_NLog(0.18, 12), 0.363667770117139, places=7)

        self.assertAlmostEqual(
            log_encoding_NLog(0.18, 10, False), 0.351634850262366, places=7)

        self.assertAlmostEqual(
            log_encoding_NLog(0.18, 10, False, False),
            0.337584957293328,
            places=7)

        self.assertAlmostEqual(
            log_encoding_NLog(1.0), 0.605083088954056, places=7)

    def test_n_dimensional_log_encoding_NLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.nikon_nlog.\
log_encoding_NLog` definition n-dimensional arrays support.
        """

        L_in = 0.18
        V_out = log_encoding_NLog(L_in)

        L_in = np.tile(L_in, 6)
        V_out = np.tile(V_out, 6)
        np.testing.assert_almost_equal(
            log_encoding_NLog(L_in), V_out, decimal=7)

        L_in = np.reshape(L_in, (2, 3))
        V_out = np.reshape(V_out, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_NLog(L_in), V_out, decimal=7)

        L_in = np.reshape(L_in, (2, 3, 1))
        V_out = np.reshape(V_out, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_NLog(L_in), V_out, decimal=7)

    def test_domain_range_scale_log_encoding_NLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.nikon_nlog.\
log_encoding_NLog` definition domain and range scale support.
        """

        L_in = 0.18
        V_out = log_encoding_NLog(L_in)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_NLog(L_in * factor),
                    V_out * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_NLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.nikon_nlog.\
log_encoding_NLog` definition nan support.
        """

        log_encoding_NLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_VLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.nikon_nlog.\
log_decoding_NLog` definition unit tests methods.
    """

    def test_log_decoding_NLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.nikon_nlog.\
log_decoding_NLog` definition.
        """

        self.assertAlmostEqual(
            log_decoding_NLog(0.124372627896372), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_NLog(0.363667770117139), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_NLog(0.363667770117139, 12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_NLog(0.351634850262366, 10, False), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_NLog(0.337584957293328, 10, False, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_NLog(0.605083088954056), 1.0, places=7)

    def test_n_dimensional_log_decoding_NLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.nikon_nlog.\
log_decoding_NLog` definition n-dimensional arrays support.
        """

        V_out = 0.363667770117139
        L_in = log_decoding_NLog(V_out)

        V_out = np.tile(V_out, 6)
        L_in = np.tile(L_in, 6)
        np.testing.assert_almost_equal(
            log_decoding_NLog(V_out), L_in, decimal=7)

        V_out = np.reshape(V_out, (2, 3))
        L_in = np.reshape(L_in, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_NLog(V_out), L_in, decimal=7)

        V_out = np.reshape(V_out, (2, 3, 1))
        L_in = np.reshape(L_in, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_NLog(V_out), L_in, decimal=7)

    def test_domain_range_scale_log_decoding_NLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.nikon_nlog.\
log_decoding_NLog` definition domain and range scale support.
        """

        V_out = 0.363667770117139
        L_in = log_decoding_NLog(V_out)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_NLog(V_out * factor),
                    L_in * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_NLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.nikon_nlog.\
log_decoding_NLog` definition nan support.
        """

        log_decoding_NLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
