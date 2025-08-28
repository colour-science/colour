"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
xiaomi_mi_log` module.
"""

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_MiLog,
    log_encoding_MiLog,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLogEncoding_MiLog",
    "TestLogDecoding_MiLog",
]


class TestLogEncoding_MiLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.xiaomi_mi_log.\
log_encoding_MiLog` definition unit tests methods.
    """

    def test_log_encoding_MiLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.xiaomi_mi_log.\
log_encoding_MiLog` definition.
        """

        # Test values from the whitepaper
        np.testing.assert_allclose(
            log_encoding_MiLog(0.0),
            0.14742742,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_MiLog(0.18),
            0.45345968,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_MiLog(0.90),
            0.66086763,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_MiLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.xiaomi_mi_log.\
log_encoding_MiLog` definition n-dimensional arrays support.
        """

        R = 0.18
        P = log_encoding_MiLog(R)

        R = np.tile(R, 6)
        P = np.tile(P, 6)
        np.testing.assert_allclose(
            log_encoding_MiLog(R), P, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        R = np.reshape(R, (2, 3))
        P = np.reshape(P, (2, 3))
        np.testing.assert_allclose(
            log_encoding_MiLog(R), P, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        R = np.reshape(R, (2, 3, 1))
        P = np.reshape(P, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_MiLog(R), P, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_MiLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.xiaomi_mi_log.\
log_encoding_MiLog` definition domain and range scale support.
        """

        R = 0.18
        P = log_encoding_MiLog(R)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_MiLog(R * factor),
                    P * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_MiLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.xiaomi_mi_log.\
log_encoding_MiLog` definition nan support.
        """

        log_encoding_MiLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_MiLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.xiaomi_mi_log.\
log_decoding_MiLog` definition unit tests methods.
    """

    def test_log_decoding_MiLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.xiaomi_mi_log.\
log_decoding_MiLog` definition.
        """

        # Test inverse of values from the whitepaper
        np.testing.assert_allclose(
            log_decoding_MiLog(0.14742742),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_MiLog(0.45345968),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_MiLog(0.66086763),
            0.90,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_MiLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.xiaomi_mi_log.\
log_decoding_MiLog` definition n-dimensional arrays support.
        """

        P = 0.45345968
        R = log_decoding_MiLog(P)

        P = np.tile(P, 6)
        R = np.tile(R, 6)
        np.testing.assert_allclose(
            log_decoding_MiLog(P), R, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        P = np.reshape(P, (2, 3))
        R = np.reshape(R, (2, 3))
        np.testing.assert_allclose(
            log_decoding_MiLog(P), R, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        P = np.reshape(P, (2, 3, 1))
        R = np.reshape(R, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_MiLog(P), R, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_MiLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.xiaomi_mi_log.\
log_decoding_MiLog` definition domain and range scale support.
        """

        P = 0.45345968
        R = log_decoding_MiLog(P)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_MiLog(P * factor),
                    R * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_MiLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.xiaomi_mi_log.\
log_decoding_MiLog` definition nan support.
        """

        log_decoding_MiLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
