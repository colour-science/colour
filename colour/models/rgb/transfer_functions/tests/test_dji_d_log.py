"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
dji_d_log` module.
"""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_DJIDLog,
    log_encoding_DJIDLog,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLogEncoding_DJIDLog",
    "TestLogDecoding_DJIDLog",
]


class TestLogEncoding_DJIDLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.dji_d_log.\
log_encoding_DJIDLog` definition unit tests methods.
    """

    def test_log_encoding_DJIDLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dji_d_log.\
log_encoding_DJIDLog` definition.
        """

        np.testing.assert_allclose(
            log_encoding_DJIDLog(0.0), 0.0929, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_encoding_DJIDLog(0.18),
            0.398764556189331,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_DJIDLog(1.0), 0.584555, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_log_encoding_DLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dji_d_log.\
log_encoding_DJIDLog` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_DJIDLog(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_allclose(
            log_encoding_DJIDLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_allclose(
            log_encoding_DJIDLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_DJIDLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_DLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dji_d_log.\
log_encoding_DJIDLog` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_DJIDLog(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_DJIDLog(x * factor),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_DLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dji_d_log.\
log_encoding_DJIDLog` definition nan support.
        """

        log_encoding_DJIDLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_DJIDLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.dji_d_log.\
log_decoding_DJIDLog` definition unit tests methods.
    """

    def test_log_decoding_DJIDLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dji_d_log.\
log_decoding_DJIDLog` definition.
        """

        np.testing.assert_allclose(
            log_decoding_DJIDLog(0.0929), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_decoding_DJIDLog(0.398764556189331), 0.18, atol=1e-6
        )

        np.testing.assert_allclose(log_decoding_DJIDLog(0.584555), 1.0, atol=1e-6)

    def test_n_dimensional_log_decoding_DLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dji_d_log.\
log_decoding_DJIDLog` definition n-dimensional arrays support.
        """

        y = 0.398764556189331
        x = log_decoding_DJIDLog(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_DJIDLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_DJIDLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_DJIDLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_DLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dji_d_log.\
log_decoding_DJIDLog` definition domain and range scale support.
        """

        y = 0.398764556189331
        x = log_decoding_DJIDLog(y)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_DJIDLog(y * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_DLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.dji_d_log.\
log_decoding_DJIDLog` definition nan support.
        """

        log_decoding_DJIDLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
