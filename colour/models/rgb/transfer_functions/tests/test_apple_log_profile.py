"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
apple_log_profile` module.
"""

import unittest

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_AppleLogProfile,
    log_encoding_AppleLogProfile,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLogEncoding_AppleLogProfile",
    "TestLogDecoding_AppleLogProfile",
]


class TestLogEncoding_AppleLogProfile(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_encoding_AppleLogProfile` definition unit tests methods.
    """

    def test_log_encoding_AppleLogProfile(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_encoding_AppleLogProfile` definition.
        """

        np.testing.assert_allclose(
            log_encoding_AppleLogProfile(0.0),
            0.150476452300913,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_AppleLogProfile(0.18),
            0.488272458526868,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_AppleLogProfile(1.0),
            0.694552983055191,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_DLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_encoding_AppleLogProfile` definition n-dimensional arrays support.
        """

        R = 0.18
        P = log_encoding_AppleLogProfile(R)

        R = np.tile(R, 6)
        P = np.tile(P, 6)
        np.testing.assert_allclose(
            log_encoding_AppleLogProfile(R), P, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        R = np.reshape(R, (2, 3))
        P = np.reshape(P, (2, 3))
        np.testing.assert_allclose(
            log_encoding_AppleLogProfile(R), P, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        R = np.reshape(R, (2, 3, 1))
        P = np.reshape(P, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_AppleLogProfile(R), P, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_DLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_encoding_AppleLogProfile` definition domain and range scale support.
        """

        R = 0.18
        P = log_encoding_AppleLogProfile(R)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_AppleLogProfile(R * factor),
                    P * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_DLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_encoding_AppleLogProfile` definition nan support.
        """

        log_encoding_AppleLogProfile(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_AppleLogProfile(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_decoding_AppleLogProfile` definition unit tests methods.
    """

    def test_log_decoding_AppleLogProfile(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_decoding_AppleLogProfile` definition.
        """

        np.testing.assert_allclose(
            log_decoding_AppleLogProfile(0.150476452300913),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_AppleLogProfile(0.488272458526868),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_AppleLogProfile(0.694552983055191),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_DLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_decoding_AppleLogProfile` definition n-dimensional arrays support.
        """

        P = 0.398764556189331
        R = log_decoding_AppleLogProfile(P)

        P = np.tile(P, 6)
        R = np.tile(R, 6)
        np.testing.assert_allclose(
            log_decoding_AppleLogProfile(P), R, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        P = np.reshape(P, (2, 3))
        R = np.reshape(R, (2, 3))
        np.testing.assert_allclose(
            log_decoding_AppleLogProfile(P), R, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        P = np.reshape(P, (2, 3, 1))
        R = np.reshape(R, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_AppleLogProfile(P), R, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_DLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_decoding_AppleLogProfile` definition domain and range scale support.
        """

        P = 0.398764556189331
        R = log_decoding_AppleLogProfile(P)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_AppleLogProfile(P * factor),
                    R * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_DLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_decoding_AppleLogProfile` definition nan support.
        """

        log_decoding_AppleLogProfile(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


if __name__ == "__main__":
    unittest.main()
