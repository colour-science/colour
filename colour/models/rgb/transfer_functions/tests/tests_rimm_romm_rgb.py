#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for
:mod:`colour.models.rgb.transfer_functions.rimm_romm_rgb` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    oetf_ROMMRGB,
    eotf_ROMMRGB,
    oetf_RIMMRGB,
    eotf_RIMMRGB,
    log_encoding_ERIMMRGB,
    log_decoding_ERIMMRGB)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestOetf_ROMMRGB',
           'TestEotf_ROMMRGB',
           'TestOetf_RIMMRGB',
           'TestEotf_RIMMRGB',
           'TestLog_encoding_ERIMMRGB',
           'TestLog_decoding_ERIMMRGB']


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

        self.assertAlmostEqual(
            oetf_ROMMRGB(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            oetf_ROMMRGB(0.18),
            98.356413311540095,
            places=7)

        self.assertAlmostEqual(
            oetf_ROMMRGB(1.0),
            255.0,
            places=7)

    def test_n_dimensional_oetf_ROMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_ROMMRGB` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 98.356413311540095
        np.testing.assert_almost_equal(
            oetf_ROMMRGB(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            oetf_ROMMRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            oetf_ROMMRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            oetf_ROMMRGB(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_ROMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_ROMMRGB` definition nan support.
        """

        oetf_ROMMRGB(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


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

        self.assertAlmostEqual(
            eotf_ROMMRGB(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            eotf_ROMMRGB(98.356413311540095),
            0.18,
            places=7)

        self.assertAlmostEqual(
            eotf_ROMMRGB(255.0),
            1.0,
            places=7)

    def test_n_dimensional_eotf_ROMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
eotf_ROMMRGB` definition n-dimensional arrays support.
        """

        L = 98.356413311540095
        V = 0.18
        np.testing.assert_almost_equal(
            eotf_ROMMRGB(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            eotf_ROMMRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            eotf_ROMMRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            eotf_ROMMRGB(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_ROMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
eotf_ROMMRGB` definition nan support.
        """

        eotf_ROMMRGB(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


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

        self.assertAlmostEqual(
            oetf_RIMMRGB(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            oetf_RIMMRGB(0.18),
            74.376801781315210,
            places=7)

        self.assertAlmostEqual(
            oetf_RIMMRGB(1.0),
            181.846934745868940,
            places=7)

    def test_n_dimensional_oetf_RIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_RIMMRGB` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 74.376801781315210
        np.testing.assert_almost_equal(
            oetf_RIMMRGB(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            oetf_RIMMRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            oetf_RIMMRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            oetf_RIMMRGB(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_RIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
oetf_RIMMRGB` definition nan support.
        """

        oetf_RIMMRGB(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


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

        self.assertAlmostEqual(
            eotf_RIMMRGB(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            eotf_RIMMRGB(74.376801781315210),
            0.18,
            places=7)

        self.assertAlmostEqual(
            eotf_RIMMRGB(181.846934745868940),
            1.0,
            places=7)

    def test_n_dimensional_eotf_RIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
eotf_RIMMRGB` definition n-dimensional arrays support.
        """

        L = 74.376801781315210
        V = 0.18
        np.testing.assert_almost_equal(
            eotf_RIMMRGB(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            eotf_RIMMRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            eotf_RIMMRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            eotf_RIMMRGB(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_RIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
eotf_RIMMRGB` definition nan support.
        """

        eotf_RIMMRGB(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


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

        self.assertAlmostEqual(
            log_encoding_ERIMMRGB(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_encoding_ERIMMRGB(0.18),
            104.563359320492940,
            places=7)

        self.assertAlmostEqual(
            log_encoding_ERIMMRGB(1.0),
            139.09187348830370,
            places=7)

    def test_n_dimensional_log_encoding_ERIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 104.563359320492940
        np.testing.assert_almost_equal(
            log_encoding_ERIMMRGB(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            log_encoding_ERIMMRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_ERIMMRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_ERIMMRGB(L),
            V,
            decimal=7)

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

        self.assertAlmostEqual(
            log_decoding_ERIMMRGB(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_decoding_ERIMMRGB(104.563359320492940),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_ERIMMRGB(139.09187348830370),
            1.0,
            places=7)

    def test_n_dimensional_log_decoding_ERIMMRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_decoding_ERIMMRGB` definition n-dimensional arrays support.
        """

        L = 104.563359320492940
        V = 0.18
        np.testing.assert_almost_equal(
            log_decoding_ERIMMRGB(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            log_decoding_ERIMMRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_ERIMMRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_ERIMMRGB(L),
            V,
            decimal=7)

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
