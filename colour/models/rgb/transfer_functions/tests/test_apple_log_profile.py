"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
apple_log_profile` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_AppleLogProfile,
    log_encoding_AppleLogProfile,
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
    "TestLogEncoding_AppleLogProfile",
    "TestLogDecoding_AppleLogProfile",
]


class TestLogEncoding_AppleLogProfile:
    """
    Define :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_encoding_AppleLogProfile` definition unit tests methods.
    """

    def test_log_encoding_AppleLogProfile(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_encoding_AppleLogProfile` definition.
        """

        xp_assert_close(
            log_encoding_AppleLogProfile(xp_as_array(0.0, xp=xp)),
            0.150476452300913,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_AppleLogProfile(xp_as_array(0.18, xp=xp)),
            0.488272458526868,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_AppleLogProfile(xp_as_array(1.0, xp=xp)),
            0.694552983055191,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_DLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_encoding_AppleLogProfile` definition n-dimensional arrays support.
        """

        R = 0.18
        P = as_ndarray(log_encoding_AppleLogProfile(xp_as_array(R, xp=xp)))

        R = xp.tile(xp_as_array(R, xp=xp), (6,))
        P = xp.tile(xp_as_array(P, xp=xp), (6,))
        xp_assert_close(
            log_encoding_AppleLogProfile(R),
            P,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        R = xp_reshape(xp_as_array(R, xp=xp), (2, 3), xp=xp)
        P = xp_reshape(xp_as_array(P, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_encoding_AppleLogProfile(R),
            P,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        R = xp_reshape(xp_as_array(R, xp=xp), (2, 3, 1), xp=xp)
        P = xp_reshape(xp_as_array(P, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_encoding_AppleLogProfile(R),
            P,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_DLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_encoding_AppleLogProfile` definition domain and range scale support.
        """

        R = 0.18
        P = as_ndarray(log_encoding_AppleLogProfile(xp_as_array(R, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_AppleLogProfile(xp_as_array(R * factor, xp=xp)),
                    P * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_DLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_encoding_AppleLogProfile` definition nan support.
        """

        log_encoding_AppleLogProfile(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_AppleLogProfile:
    """
    Define :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_decoding_AppleLogProfile` definition unit tests methods.
    """

    def test_log_decoding_AppleLogProfile(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_decoding_AppleLogProfile` definition.
        """

        xp_assert_close(
            log_decoding_AppleLogProfile(xp_as_array(0.150476452300913, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_AppleLogProfile(xp_as_array(0.488272458526868, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_AppleLogProfile(xp_as_array(0.694552983055191, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_DLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_decoding_AppleLogProfile` definition n-dimensional arrays support.
        """

        P = 0.398764556189331
        R = as_ndarray(log_decoding_AppleLogProfile(xp_as_array(P, xp=xp)))

        P = xp.tile(xp_as_array(P, xp=xp), (6,))
        R = xp.tile(xp_as_array(R, xp=xp), (6,))
        xp_assert_close(
            log_decoding_AppleLogProfile(P),
            R,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        P = xp_reshape(xp_as_array(P, xp=xp), (2, 3), xp=xp)
        R = xp_reshape(xp_as_array(R, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_decoding_AppleLogProfile(P),
            R,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        P = xp_reshape(xp_as_array(P, xp=xp), (2, 3, 1), xp=xp)
        R = xp_reshape(xp_as_array(R, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_decoding_AppleLogProfile(P),
            R,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_DLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_decoding_AppleLogProfile` definition domain and range scale support.
        """

        P = 0.398764556189331
        R = as_ndarray(log_decoding_AppleLogProfile(xp_as_array(P, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_AppleLogProfile(xp_as_array(P * factor, xp=xp)),
                    R * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_DLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.apple_log_profile.\
log_decoding_AppleLogProfile` definition nan support.
        """

        log_decoding_AppleLogProfile(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )
