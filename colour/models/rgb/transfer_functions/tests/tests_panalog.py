#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.panalog`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    log_encoding_Panalog,
    log_decoding_Panalog)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLogEncoding_Panalog',
           'TestLogDecoding_Panalog']


class TestLogEncoding_Panalog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.panalog.\
log_encoding_Panalog` definition unit tests methods.
    """

    def test_log_encoding_Panalog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panalog.\
log_encoding_Panalog` definition.
        """

        self.assertAlmostEqual(
            log_encoding_Panalog(0.0),
            0.062561094819159,
            places=7)

        self.assertAlmostEqual(
            log_encoding_Panalog(0.18),
            0.374576791382298,
            places=7)

        self.assertAlmostEqual(
            log_encoding_Panalog(1.0),
            0.665689149560117,
            places=7)

    def test_n_dimensional_log_encoding_Panalog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panalog.\
log_encoding_Panalog` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.374576791382298
        np.testing.assert_almost_equal(
            log_encoding_Panalog(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            log_encoding_Panalog(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_Panalog(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_Panalog(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_Panalog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panalog.\
log_encoding_Panalog` definition nan support.
        """

        log_encoding_Panalog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Panalog(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.panalog.\
log_decoding_Panalog` definition unit tests methods.
    """

    def test_log_decoding_Panalog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panalog.\
log_decoding_Panalog` definition.
        """

        self.assertAlmostEqual(
            log_decoding_Panalog(0.062561094819159),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_decoding_Panalog(0.374576791382298),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_Panalog(0.665689149560117),
            1.0,
            places=7)

    def test_n_dimensional_log_decoding_Panalog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panalog.\
log_decoding_Panalog` definition n-dimensional arrays support.
        """

        V = 0.374576791382298
        L = 0.18
        np.testing.assert_almost_equal(
            log_decoding_Panalog(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            log_decoding_Panalog(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_Panalog(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_Panalog(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_Panalog(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.panalog.\
log_decoding_Panalog` definition nan support.
        """

        log_decoding_Panalog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
