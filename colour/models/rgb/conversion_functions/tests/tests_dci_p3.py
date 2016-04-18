#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.conversion_functions.dci_p3`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.conversion_functions import (
    log_encoding_DCIP3,
    log_decoding_DCIP3)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLogEncoding_DCIP3',
           'TestLogDecoding_DCIP3']


class TestLogEncoding_DCIP3(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.conversion_functions.dci_p3.\
log_encoding_DCIP3` definition unit tests methods.
    """

    def test_log_encoding_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.dci_p3.\
log_encoding_DCIP3` definition.
        """

        self.assertAlmostEqual(
            log_encoding_DCIP3(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_encoding_DCIP3(0.18),
            461.99220597484737,
            places=7)

        self.assertAlmostEqual(
            log_encoding_DCIP3(1.0),
            893.44598340527841,
            places=7)

    def test_n_dimensional_log_encoding_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.dci_p3.\
log_encoding_DCIP3` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 461.99220597484737
        np.testing.assert_almost_equal(
            log_encoding_DCIP3(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            log_encoding_DCIP3(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            log_encoding_DCIP3(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_encoding_DCIP3(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.dci_p3.\
log_encoding_DCIP3` definition nan support.
        """

        log_encoding_DCIP3(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_DCIP3(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.conversion_functions.dci_p3.
log_decoding_DCIP3` definition unit tests methods.
    """

    def test_log_decoding_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.dci_p3.\
log_decoding_DCIP3` definition.
        """

        self.assertAlmostEqual(
            log_decoding_DCIP3(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            log_decoding_DCIP3(461.99220597484737),
            0.18,
            places=7)

        self.assertAlmostEqual(
            log_decoding_DCIP3(893.44598340527841),
            1.0,
            places=7)

    def test_n_dimensional_log_decoding_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.dci_p3.\
log_decoding_DCIP3` definition n-dimensional arrays support.
        """

        V = 461.99220597484737
        L = 0.18
        np.testing.assert_almost_equal(
            log_decoding_DCIP3(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            log_decoding_DCIP3(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            log_decoding_DCIP3(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            log_decoding_DCIP3(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.dci_p3.\
log_decoding_DCIP3` definition nan support.
        """

        log_decoding_DCIP3(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
