# -*- coding: utf-8 -*-
"""
Defines unit tests for
:mod:`colour.models.rgb.transfer_functions.rimm_romm_rgb` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    oetf_ROMMRGB, eotf_ROMMRGB, oetf_RIMMRGB, eotf_RIMMRGB,
    log_encoding_ERIMMRGB, log_decoding_ERIMMRGB)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestOetf_ROMMRGB', 'TestEotf_ROMMRGB', 'TestOetf_RIMMRGB',
    'TestEotf_RIMMRGB', 'TestLog_encoding_ERIMMRGB',
    'TestLog_decoding_ERIMMRGB'
]


class TestOetf_ROMMRGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_ROMMRGB` definition unit tests methods.
    """

    def test_oetf_ROMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_ROMMRGB` definition.
        """

        self.assertAlmostEqual(oetf_ROMMRGB(0.0), 0.0, places=7)

        self.assertAlmostEqual(oetf_ROMMRGB(0.18), 0.385711424751138, places=7)

        self.assertAlmostEqual(oetf_ROMMRGB(1.0), 1.0, places=7)

        self.assertEqual(oetf_ROMMRGB(0.18, out_int=True), 98)

        self.assertEqual(oetf_ROMMRGB(0.18, bit_depth=12, out_int=True), 1579)

    def test_n_dimensional_oetf_ROMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_ROMMRGB` definition n-dimensional arrays support.
        """

        X = 0.18
        X_ROMM = 0.385711424751138
        np.testing.assert_almost_equal(oetf_ROMMRGB(X), X_ROMM, decimal=7)

        X = np.tile(X, 6)
        X_ROMM = np.tile(X_ROMM, 6)
        np.testing.assert_almost_equal(oetf_ROMMRGB(X), X_ROMM, decimal=7)

        X = np.reshape(X, (2, 3))
        X_ROMM = np.reshape(X_ROMM, (2, 3))
        np.testing.assert_almost_equal(oetf_ROMMRGB(X), X_ROMM, decimal=7)

        X = np.reshape(X, (2, 3, 1))
        X_ROMM = np.reshape(X_ROMM, (2, 3, 1))
        np.testing.assert_almost_equal(oetf_ROMMRGB(X), X_ROMM, decimal=7)

    def test_domain_range_scale_oetf_ROMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_ROMMRGB` definition domain and range scale support.
        """

        X = 0.18
        X_p = oetf_ROMMRGB(X)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    oetf_ROMMRGB(X * factor), X_p * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_ROMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_ROMMRGB` definition nan support.
        """

        oetf_ROMMRGB(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_ROMMRGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.
eotf_ROMMRGB` definition unit tests methods.
    """

    def test_eotf_ROMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
eotf_ROMMRGB` definition.
        """

        self.assertAlmostEqual(eotf_ROMMRGB(0.0), 0.0, places=7)

        self.assertAlmostEqual(eotf_ROMMRGB(0.385711424751138), 0.18, places=7)

        self.assertAlmostEqual(eotf_ROMMRGB(1.0), 1.0, places=7)

        np.testing.assert_allclose(
            eotf_ROMMRGB(98, in_int=True), 0.18, atol=0.001, rtol=0.001)

        np.testing.assert_allclose(
            eotf_ROMMRGB(1579, bit_depth=12, in_int=True),
            0.18,
            atol=0.001,
            rtol=0.001)

    def test_n_dimensional_eotf_ROMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
eotf_ROMMRGB` definition n-dimensional arrays support.
        """

        X_p = 0.385711424751138
        X = 0.18
        np.testing.assert_almost_equal(eotf_ROMMRGB(X_p), X, decimal=7)

        X_p = np.tile(X_p, 6)
        X = np.tile(X, 6)
        np.testing.assert_almost_equal(eotf_ROMMRGB(X_p), X, decimal=7)

        X_p = np.reshape(X_p, (2, 3))
        X = np.reshape(X, (2, 3))
        np.testing.assert_almost_equal(eotf_ROMMRGB(X_p), X, decimal=7)

        X_p = np.reshape(X_p, (2, 3, 1))
        X = np.reshape(X, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_ROMMRGB(X_p), X, decimal=7)

    def test_domain_range_scale_eotf_ROMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
eotf_ROMMRGB` definition domain and range scale support.
        """

        X_p = 0.385711424751138
        X = eotf_ROMMRGB(X_p)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_ROMMRGB(X_p * factor), X * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_ROMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
eotf_ROMMRGB` definition nan support.
        """

        eotf_ROMMRGB(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_RIMMRGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_RIMMRGB` definition unit tests methods.
    """

    def test_oetf_RIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_RIMMRGB` definition.
        """

        self.assertAlmostEqual(oetf_RIMMRGB(0.0), 0.0, places=7)

        self.assertAlmostEqual(oetf_RIMMRGB(0.18), 0.291673732475746, places=7)

        self.assertAlmostEqual(oetf_RIMMRGB(1.0), 0.713125234297525, places=7)

        self.assertEqual(oetf_RIMMRGB(0.18, out_int=True), 74)

        self.assertEqual(oetf_RIMMRGB(0.18, bit_depth=12, out_int=True), 1194)

    def test_n_dimensional_oetf_RIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_RIMMRGB` definition n-dimensional arrays support.
        """

        X = 0.18
        X_p = 0.291673732475746
        np.testing.assert_almost_equal(oetf_RIMMRGB(X), X_p, decimal=7)

        X = np.tile(X, 6)
        X_p = np.tile(X_p, 6)
        np.testing.assert_almost_equal(oetf_RIMMRGB(X), X_p, decimal=7)

        X = np.reshape(X, (2, 3))
        X_p = np.reshape(X_p, (2, 3))
        np.testing.assert_almost_equal(oetf_RIMMRGB(X), X_p, decimal=7)

        X = np.reshape(X, (2, 3, 1))
        X_p = np.reshape(X_p, (2, 3, 1))
        np.testing.assert_almost_equal(oetf_RIMMRGB(X), X_p, decimal=7)

    def test_domain_range_scale_oetf_RIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_RIMMRGB` definition domain and range scale support.
        """

        X = 0.18
        X_p = oetf_RIMMRGB(X)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    oetf_RIMMRGB(X * factor), X_p * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_RIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_RIMMRGB` definition nan support.
        """

        oetf_RIMMRGB(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_RIMMRGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.
eotf_RIMMRGB` definition unit tests methods.
    """

    def test_eotf_RIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
eotf_RIMMRGB` definition.
        """

        self.assertAlmostEqual(eotf_RIMMRGB(0.0), 0.0, places=7)

        self.assertAlmostEqual(eotf_RIMMRGB(0.291673732475746), 0.18, places=7)

        self.assertAlmostEqual(eotf_RIMMRGB(0.713125234297525), 1.0, places=7)

        np.testing.assert_allclose(
            eotf_RIMMRGB(74, in_int=True), 0.18, atol=0.005, rtol=0.005)

        np.testing.assert_allclose(
            eotf_RIMMRGB(1194, bit_depth=12, in_int=True),
            0.18,
            atol=0.005,
            rtol=0.005)

    def test_n_dimensional_eotf_RIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
eotf_RIMMRGB` definition n-dimensional arrays support.
        """

        X_p = 0.291673732475746
        X = 0.18
        np.testing.assert_almost_equal(eotf_RIMMRGB(X_p), X, decimal=7)

        X_p = np.tile(X_p, 6)
        X = np.tile(X, 6)
        np.testing.assert_almost_equal(eotf_RIMMRGB(X_p), X, decimal=7)

        X_p = np.reshape(X_p, (2, 3))
        X = np.reshape(X, (2, 3))
        np.testing.assert_almost_equal(eotf_RIMMRGB(X_p), X, decimal=7)

        X_p = np.reshape(X_p, (2, 3, 1))
        X = np.reshape(X, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_RIMMRGB(X_p), X, decimal=7)

    def test_domain_range_scale_eotf_RIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
eotf_RIMMRGB` definition domain and range scale support.
        """

        X_p = 0.291673732475746
        X = eotf_RIMMRGB(X_p)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_RIMMRGB(X_p * factor), X * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_RIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
eotf_RIMMRGB` definition nan support.
        """

        eotf_RIMMRGB(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLog_encoding_ERIMMRGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition unit tests methods.
    """

    def test_log_encoding_ERIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition.
        """

        self.assertAlmostEqual(log_encoding_ERIMMRGB(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            log_encoding_ERIMMRGB(0.18), 0.410052389492129, places=7)

        self.assertAlmostEqual(
            log_encoding_ERIMMRGB(1.0), 0.545458327405113, places=7)

        self.assertEqual(log_encoding_ERIMMRGB(0.18, out_int=True), 105)

        self.assertEqual(
            log_encoding_ERIMMRGB(0.18, bit_depth=12, out_int=True), 1679)

    def test_n_dimensional_log_encoding_ERIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition n-dimensional arrays support.
        """

        X = 0.18
        X_p = 0.410052389492129
        np.testing.assert_almost_equal(
            log_encoding_ERIMMRGB(X), X_p, decimal=7)

        X = np.tile(X, 6)
        X_p = np.tile(X_p, 6)
        np.testing.assert_almost_equal(
            log_encoding_ERIMMRGB(X), X_p, decimal=7)

        X = np.reshape(X, (2, 3))
        X_p = np.reshape(X_p, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_ERIMMRGB(X), X_p, decimal=7)

        X = np.reshape(X, (2, 3, 1))
        X_p = np.reshape(X_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_ERIMMRGB(X), X_p, decimal=7)

    def test_domain_range_scale_log_encoding_ERIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition domain and range scale support.
        """

        X = 0.18
        X_p = log_encoding_ERIMMRGB(X)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_ERIMMRGB(X * factor), X_p * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_ERIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition nan support.
        """

        log_encoding_ERIMMRGB(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLog_decoding_ERIMMRGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.
log_decoding_ERIMMRGB` definition unit tests methods.
    """

    def test_log_decoding_ERIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_decoding_ERIMMRGB` definition.
        """

        self.assertAlmostEqual(log_decoding_ERIMMRGB(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_ERIMMRGB(0.410052389492129), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_ERIMMRGB(0.545458327405113), 1.0, places=7)

        np.testing.assert_allclose(
            log_decoding_ERIMMRGB(105, in_int=True),
            0.18,
            atol=0.005,
            rtol=0.005)

        np.testing.assert_allclose(
            log_decoding_ERIMMRGB(1679, bit_depth=12, in_int=True),
            0.18,
            atol=0.005,
            rtol=0.005)

    def test_n_dimensional_log_decoding_ERIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_decoding_ERIMMRGB` definition n-dimensional arrays support.
        """

        X_p = 0.410052389492129
        X = 0.18
        np.testing.assert_almost_equal(
            log_decoding_ERIMMRGB(X_p), X, decimal=7)

        X_p = np.tile(X_p, 6)
        X = np.tile(X, 6)
        np.testing.assert_almost_equal(
            log_decoding_ERIMMRGB(X_p), X, decimal=7)

        X_p = np.reshape(X_p, (2, 3))
        X = np.reshape(X, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_ERIMMRGB(X_p), X, decimal=7)

        X_p = np.reshape(X_p, (2, 3, 1))
        X = np.reshape(X, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_ERIMMRGB(X_p), X, decimal=7)

    def test_domain_range_scale_log_decoding_ERIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_decoding_ERIMMRGB` definition domain and range scale support.
        """

        X_p = 0.410052389492129
        X = log_decoding_ERIMMRGB(X_p)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_ERIMMRGB(X_p * factor), X * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_ERIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_decoding_ERIMMRGB` definition nan support.
        """

        log_decoding_ERIMMRGB(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
