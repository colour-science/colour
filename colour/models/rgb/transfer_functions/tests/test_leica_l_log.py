"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
leica_l_log` module.
"""

import unittest

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_LLog,
    log_encoding_LLog,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLogEncoding_LLog",
    "TestLogDecoding_LLog",
]


class TestLogEncoding_LLog(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.leica_l_log.\
log_encoding_LLog` definition unit tests methods.
    """

    def test_log_encoding_LLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.leica_l_log.\
log_encoding_LLog` definition.
        """

        self.assertAlmostEqual(
            log_encoding_LLog(0.0), 0.089999999999999, places=7
        )

        self.assertAlmostEqual(
            log_encoding_LLog(0.18), 0.435313904043927, places=7
        )

        self.assertAlmostEqual(
            log_encoding_LLog(0.18, 12), 0.435313904043927, places=7
        )

        self.assertAlmostEqual(
            log_encoding_LLog(0.18, 10, False), 0.4353037943344028, places=7
        )

        self.assertAlmostEqual(
            log_encoding_LLog(0.18, 10, False, False),
            0.421586960452824,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_LLog(1.0), 0.631797439630121, places=7
        )

    def test_n_dimensional_log_encoding_LLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.leica_l_log.\
log_encoding_LLog` definition n-dimensional arrays support.
        """

        y = 0.18
        x = log_encoding_LLog(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_encoding_LLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_encoding_LLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_LLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_LLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.leica_l_log.\
log_encoding_LLog` definition domain and range scale support.
        """

        y = 0.18
        x = log_encoding_LLog(y)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_LLog(y * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_LLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.leica_l_log.\
log_encoding_LLog` definition nan support.
        """

        log_encoding_LLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_LLog(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.leica_l_log.\
log_decoding_LLog` definition unit tests methods.
    """

    def test_log_decoding_LLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.leica_l_log.\
log_decoding_LLog` definition.
        """

        self.assertAlmostEqual(
            log_decoding_LLog(0.089999999999999), 0.0, places=7
        )

        self.assertAlmostEqual(
            log_decoding_LLog(0.435313904043927), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_LLog(0.435313904043927, 12), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_LLog(0.4353037943344028, 10, False), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_LLog(0.421586960452824, 10, False, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_LLog(0.631797439630121), 1.0, places=7
        )

    def test_n_dimensional_log_decoding_LLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.leica_l_log.\
log_decoding_LLog` definition n-dimensional arrays support.
        """

        x = 0.435313904043927
        y = log_decoding_LLog(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_allclose(
            log_decoding_LLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_allclose(
            log_decoding_LLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_LLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_LLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.leica_l_log.\
log_decoding_LLog` definition domain and range scale support.
        """

        x = 0.435313904043927
        y = log_decoding_LLog(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_LLog(x * factor),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_LLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.leica_l_log.\
log_decoding_LLog` definition nan support.
        """

        log_decoding_LLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == "__main__":
    unittest.main()
