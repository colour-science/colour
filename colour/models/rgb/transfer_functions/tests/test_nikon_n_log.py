"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
nikon_n_log` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    log_encoding_NLog,
    log_decoding_NLog,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLogEncoding_NLog",
    "TestLogDecoding_NLog",
]


class TestLogEncoding_NLog(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_encoding_NLog` definition unit tests methods.
    """

    def test_log_encoding_NLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_encoding_NLog` definition.
        """

        self.assertAlmostEqual(
            log_encoding_NLog(0.0), 0.124372627896372, places=7
        )

        self.assertAlmostEqual(
            log_encoding_NLog(0.18), 0.363667770117139, places=7
        )

        self.assertAlmostEqual(
            log_encoding_NLog(0.18, 12), 0.363667770117139, places=7
        )

        self.assertAlmostEqual(
            log_encoding_NLog(0.18, 10, False), 0.351634850262366, places=7
        )

        self.assertAlmostEqual(
            log_encoding_NLog(0.18, 10, False, False),
            0.337584957293328,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_NLog(1.0), 0.605083088954056, places=7
        )

    def test_n_dimensional_log_encoding_NLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_encoding_NLog` definition n-dimensional arrays support.
        """

        y = 0.18
        x = log_encoding_NLog(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_array_almost_equal(
            log_encoding_NLog(y), x, decimal=7
        )

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_array_almost_equal(
            log_encoding_NLog(y), x, decimal=7
        )

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            log_encoding_NLog(y), x, decimal=7
        )

    def test_domain_range_scale_log_encoding_NLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_encoding_NLog` definition domain and range scale support.
        """

        y = 0.18
        x = log_encoding_NLog(y)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    log_encoding_NLog(y * factor), x * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_NLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_encoding_NLog` definition nan support.
        """

        log_encoding_NLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_NLog(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_decoding_NLog` definition unit tests methods.
    """

    def test_log_decoding_NLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_decoding_NLog` definition.
        """

        self.assertAlmostEqual(
            log_decoding_NLog(0.124372627896372), 0.0, places=7
        )

        self.assertAlmostEqual(
            log_decoding_NLog(0.363667770117139), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_NLog(0.363667770117139, 12), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_NLog(0.351634850262366, 10, False), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_NLog(0.337584957293328, 10, False, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_NLog(0.605083088954056), 1.0, places=7
        )

    def test_n_dimensional_log_decoding_NLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_decoding_NLog` definition n-dimensional arrays support.
        """

        x = 0.363667770117139
        y = log_decoding_NLog(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_array_almost_equal(
            log_decoding_NLog(x), y, decimal=7
        )

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_array_almost_equal(
            log_decoding_NLog(x), y, decimal=7
        )

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_array_almost_equal(
            log_decoding_NLog(x), y, decimal=7
        )

    def test_domain_range_scale_log_decoding_NLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_decoding_NLog` definition domain and range scale support.
        """

        x = 0.363667770117139
        y = log_decoding_NLog(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    log_decoding_NLog(x * factor), y * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_NLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_decoding_NLog` definition nan support.
        """

        log_decoding_NLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == "__main__":
    unittest.main()
