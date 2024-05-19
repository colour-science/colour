"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.gopro` module.
"""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_Protune,
    log_encoding_Protune,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLogEncoding_Protune",
    "TestLogDecoding_Protune",
]


class TestLogEncoding_Protune:
    """
    Define :func:`colour.models.rgb.transfer_functions.gopro.\
log_encoding_Protune` definition unit tests methods.
    """

    def test_log_encoding_Protune(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.gopro.\
log_encoding_Protune` definition.
        """

        np.testing.assert_allclose(
            log_encoding_Protune(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_encoding_Protune(0.18),
            0.645623486803636,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_Protune(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_log_encoding_Protune(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.gopro.\
log_encoding_Protune` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_Protune(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_allclose(
            log_encoding_Protune(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_allclose(
            log_encoding_Protune(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_Protune(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_Protune(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.gopro.\
log_encoding_Protune` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_Protune(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_Protune(x * factor),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_Protune(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.gopro.\
log_encoding_Protune` definition nan support.
        """

        log_encoding_Protune(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Protune:
    """
    Define :func:`colour.models.rgb.transfer_functions.gopro.\
log_decoding_Protune` definition unit tests methods.
    """

    def test_log_decoding_Protune(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.gopro.\
log_decoding_Protune` definition.
        """

        np.testing.assert_allclose(
            log_decoding_Protune(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_decoding_Protune(0.645623486803636),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Protune(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_log_decoding_Protune(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.gopro.\
log_decoding_Protune` definition n-dimensional arrays support.
        """

        y = 0.645623486803636
        x = log_decoding_Protune(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_Protune(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_Protune(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_Protune(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_Protune(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.gopro.\
log_decoding_Protune` definition domain and range scale support.
        """

        y = 0.645623486803636
        x = log_decoding_Protune(y)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_Protune(y * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_Protune(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.gopro.\
log_decoding_Protune` definition nan support.
        """

        log_decoding_Protune(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
