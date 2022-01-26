# -*- coding: utf-8 -*-
"""
Defines the unit tests for the
:mod:`colour.models.rgb.transfer_functions.cineon` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    log_encoding_Cineon,
    log_decoding_Cineon,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestLogEncoding_Cineon',
    'TestLogDecoding_Cineon',
]


class TestLogEncoding_Cineon(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition unit tests methods.
    """

    def test_log_encoding_Cineon(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition.
        """

        self.assertAlmostEqual(
            log_encoding_Cineon(0.0), 0.092864125122190, places=7)

        self.assertAlmostEqual(
            log_encoding_Cineon(0.18), 0.457319613085418, places=7)

        self.assertAlmostEqual(
            log_encoding_Cineon(1.0), 0.669599217986315, places=7)

    def test_n_dimensional_log_encoding_Cineon(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_Cineon(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_almost_equal(log_encoding_Cineon(x), y, decimal=7)

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_almost_equal(log_encoding_Cineon(x), y, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_Cineon(x), y, decimal=7)

    def test_domain_range_scale_log_encoding_Cineon(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_Cineon(x)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_encoding_Cineon(x * factor), y * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_Cineon(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition nan support.
        """

        log_encoding_Cineon(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Cineon(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition unit tests methods.
    """

    def test_log_decoding_Cineon(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition.
        """

        self.assertAlmostEqual(
            log_decoding_Cineon(0.092864125122190), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_Cineon(0.457319613085418), 0.18, places=7)

        self.assertAlmostEqual(
            log_decoding_Cineon(0.669599217986315), 1.0, places=7)

    def test_n_dimensional_log_decoding_Cineon(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition n-dimensional arrays support.
        """

        y = 0.457319613085418
        x = log_decoding_Cineon(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(log_decoding_Cineon(y), x, decimal=7)

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(log_decoding_Cineon(y), x, decimal=7)

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_Cineon(y), x, decimal=7)

    def test_domain_range_scale_log_decoding_Cineon(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition domain and range scale support.
        """

        y = 0.457319613085418
        x = log_decoding_Cineon(y)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    log_decoding_Cineon(y * factor), x * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_Cineon(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition nan support.
        """

        log_decoding_Cineon(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
