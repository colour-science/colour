# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.gopro`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (log_encoding_Protune,
                                                  log_decoding_Protune)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLogEncoding_Protune', 'TestLogDecoding_Protune']


class TestLogEncoding_Protune(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.gopro.\
log_encoding_Protune` definition unit tests methods.
    """

    def test_log_encoding_Protune(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.gopro.\
log_encoding_Protune` definition.
        """

        self.assertAlmostEqual(log_encoding_Protune(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            log_encoding_Protune(0.18), 0.645623486803636, places=7)

        self.assertAlmostEqual(log_encoding_Protune(1.0), 1.0, places=7)

    def test_n_dimensional_log_encoding_Protune(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.gopro.\
log_encoding_Protune` definition n-dimensional arrays support.
        """

        x = 0.18
        y = 0.645623486803636
        np.testing.assert_almost_equal(log_encoding_Protune(x), y, decimal=7)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_almost_equal(log_encoding_Protune(x), y, decimal=7)

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_almost_equal(log_encoding_Protune(x), y, decimal=7)

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_almost_equal(log_encoding_Protune(x), y, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_encoding_Protune(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.gopro.\
log_encoding_Protune` definition nan support.
        """

        log_encoding_Protune(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Protune(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.gopro.\
log_decoding_Protune` definition unit tests methods.
    """

    def test_log_decoding_Protune(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.gopro.\
log_decoding_Protune` definition.
        """

        self.assertAlmostEqual(log_decoding_Protune(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            log_decoding_Protune(0.645623486803636), 0.18, places=7)

        self.assertAlmostEqual(log_decoding_Protune(1.0), 1.0, places=7)

    def test_n_dimensional_log_decoding_Protune(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.gopro.\
log_decoding_Protune` definition n-dimensional arrays support.
        """

        y = 0.645623486803636
        x = 0.18
        np.testing.assert_almost_equal(log_decoding_Protune(y), x, decimal=7)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_almost_equal(log_decoding_Protune(y), x, decimal=7)

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_almost_equal(log_decoding_Protune(y), x, decimal=7)

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_almost_equal(log_decoding_Protune(y), x, decimal=7)

    @ignore_numpy_errors
    def test_nan_log_decoding_Protune(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.gopro.\
log_decoding_Protune` definition nan support.
        """

        log_decoding_Protune(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
