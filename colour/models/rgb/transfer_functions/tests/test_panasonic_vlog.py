# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
panasonic_vlog` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    log_encoding_VLog,
    log_decoding_VLog,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
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
    Defines :func:`colour.models.rgb.transfer_functions.panasonic_vlog.\
log_encoding_VLog` definition unit tests methods.
    """

    def test_log_encoding_VLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panasonic_vlog.\
log_encoding_VLog` definition.
        """

        self.assertAlmostEqual(log_encoding_VLog(0.0), 0.125, places=7)

        self.assertAlmostEqual(
            log_encoding_VLog(0.18), 0.423311448760136, places=7)

        self.assertAlmostEqual(
            log_encoding_VLog(0.18, 12), 0.423311448760136, places=7)

        self.assertAlmostEqual(
            log_encoding_VLog(0.18, 10, False), 0.421287228403675, places=7)

        self.assertAlmostEqual(
            log_encoding_VLog(0.18, 10, False, False),
            0.409009628526078,
            places=7)

        self.assertAlmostEqual(
            log_encoding_VLog(1.0), 0.599117700158146, places=7)

    def test_n_dimensional_log_encoding_VLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panasonic_vlog.\
log_encoding_VLog` definition n-dimensional arrays support.
        """

        L_in = 0.18
        V_out = log_encoding_VLog(L_in)

        L_in = np.tile(L_in, 6)
        V_out = np.tile(V_out, 6)
        np.testing.assert_almost_equal(
            log_encoding_VLog(L_in), V_out, decimal=7)

        L_in = np.reshape(L_in, (2, 3))
        V_out = np.reshape(V_out, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_VLog(L_in), V_out, decimal=7)

        L_in = np.reshape(L_in, (2, 3, 1))
        V_out = np.reshape(V_out, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_VLog(L_in), V_out, decimal=7)

    def test_domain_range_scale_log_encoding_VLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panasonic_vlog.\
log_encoding_VLog` definition domain and range scale support.
        """

        L_in = 0.18
        V_out = log_encoding_VLog(L_in)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_VLog(L_in * factor),
                    V_out * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_VLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panasonic_vlog.\
log_encoding_VLog` definition nan support.
        """

        log_encoding_VLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_VLog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.panasonic_vlog.\
log_decoding_VLog` definition unit tests methods.
    """

    def test_log_decoding_VLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panasonic_vlog.\
log_decoding_VLog` definition.
        """

        self.assertAlmostEqual(log_decoding_VLog(0.125), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_VLog(0.423311448760136), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_VLog(0.423311448760136, 12), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_VLog(0.421287228403675, 10, False), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_VLog(0.409009628526078, 10, False, False),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_VLog(0.599117700158146), 1.0, places=7)

    def test_n_dimensional_log_decoding_VLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panasonic_vlog.\
log_decoding_VLog` definition n-dimensional arrays support.
        """

        V_out = 0.423311448760136
        L_in = log_decoding_VLog(V_out)

        V_out = np.tile(V_out, 6)
        L_in = np.tile(L_in, 6)
        np.testing.assert_almost_equal(
            log_decoding_VLog(V_out), L_in, decimal=7)

        V_out = np.reshape(V_out, (2, 3))
        L_in = np.reshape(L_in, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_VLog(V_out), L_in, decimal=7)

        V_out = np.reshape(V_out, (2, 3, 1))
        L_in = np.reshape(L_in, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_VLog(V_out), L_in, decimal=7)

    def test_domain_range_scale_log_decoding_VLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panasonic_vlog.\
log_decoding_VLog` definition domain and range scale support.
        """

        V_out = 0.423311448760136
        L_in = log_decoding_VLog(V_out)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_VLog(V_out * factor),
                    L_in * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_VLog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panasonic_vlog.\
log_decoding_VLog` definition nan support.
        """

        log_decoding_VLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
