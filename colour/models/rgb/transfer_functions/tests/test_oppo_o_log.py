"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
oppo_o_log` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_OPPOOLog,
    log_encoding_OPPOOLog,
)
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLogEncoding_OPPOOLog",
    "TestLogDecoding_OPPOOLog",
]


class TestLogEncoding_OPPOOLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.oppo_o_log.\
log_encoding_OPPOOLog` definition unit tests methods.
    """

    def test_log_encoding_OPPOOLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.oppo_o_log.\
log_encoding_OPPOOLog` definition.
        """

        xp_assert_close(
            log_encoding_OPPOOLog(xp_as_array(0.0, xp=xp)),
            0.06309903,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_OPPOOLog(xp_as_array(0.18, xp=xp)),
            0.38959139,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_OPPOOLog(xp_as_array(0.90, xp=xp)),
            0.60225879,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_OPPOOLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.oppo_o_log.\
log_encoding_OPPOOLog` definition n-dimensional arrays support.
        """

        R = 0.18
        P = as_ndarray(log_encoding_OPPOOLog(xp_as_array(R, xp=xp)))

        R = xp.tile(xp_as_array(R, xp=xp), (6,))
        P = xp.tile(xp_as_array(P, xp=xp), (6,))
        xp_assert_close(log_encoding_OPPOOLog(R), P, atol=TOLERANCE_ABSOLUTE_TESTS)

        R = xp_reshape(xp_as_array(R, xp=xp), (2, 3), xp=xp)
        P = xp_reshape(xp_as_array(P, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_OPPOOLog(R), P, atol=TOLERANCE_ABSOLUTE_TESTS)

        R = xp_reshape(xp_as_array(R, xp=xp), (2, 3, 1), xp=xp)
        P = xp_reshape(xp_as_array(P, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_OPPOOLog(R), P, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_OPPOOLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.oppo_o_log.\
log_encoding_OPPOOLog` definition domain and range scale support.
        """

        R = 0.18
        P = as_ndarray(log_encoding_OPPOOLog(xp_as_array(R, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_OPPOOLog(xp_as_array(R * factor, xp=xp)),
                    P * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_OPPOOLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.oppo_o_log.\
log_encoding_OPPOOLog` definition nan support.
        """

        log_encoding_OPPOOLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_OPPOOLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.oppo_o_log.\
log_decoding_OPPOOLog` definition unit tests methods.
    """

    def test_log_decoding_OPPOOLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.oppo_o_log.\
log_decoding_OPPOOLog` definition.
        """

        xp_assert_close(
            log_decoding_OPPOOLog(xp_as_array(0.06309903, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_OPPOOLog(xp_as_array(0.38959139, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_OPPOOLog(xp_as_array(0.60225879, xp=xp)),
            0.90,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_OPPOOLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.oppo_o_log.\
log_decoding_OPPOOLog` definition n-dimensional arrays support.
        """

        P = 0.38959139
        R = as_ndarray(log_decoding_OPPOOLog(xp_as_array(P, xp=xp)))

        P = xp.tile(xp_as_array(P, xp=xp), (6,))
        R = xp.tile(xp_as_array(R, xp=xp), (6,))
        xp_assert_close(log_decoding_OPPOOLog(P), R, atol=TOLERANCE_ABSOLUTE_TESTS)

        P = xp_reshape(xp_as_array(P, xp=xp), (2, 3), xp=xp)
        R = xp_reshape(xp_as_array(R, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_OPPOOLog(P), R, atol=TOLERANCE_ABSOLUTE_TESTS)

        P = xp_reshape(xp_as_array(P, xp=xp), (2, 3, 1), xp=xp)
        R = xp_reshape(xp_as_array(R, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_OPPOOLog(P), R, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_OPPOOLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.oppo_o_log.\
log_decoding_OPPOOLog` definition domain and range scale support.
        """

        P = 0.38959139
        R = as_ndarray(log_decoding_OPPOOLog(xp_as_array(P, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_OPPOOLog(xp_as_array(P * factor, xp=xp)),
                    R * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_OPPOOLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.oppo_o_log.\
log_decoding_OPPOOLog` definition nan support.
        """

        log_decoding_OPPOOLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
