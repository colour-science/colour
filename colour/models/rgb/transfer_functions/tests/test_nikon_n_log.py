"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
nikon_n_log` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import log_decoding_NLog, log_encoding_NLog
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
    "TestLogEncoding_NLog",
    "TestLogDecoding_NLog",
]


class TestLogEncoding_NLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_encoding_NLog` definition unit tests methods.
    """

    def test_log_encoding_NLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_encoding_NLog` definition.
        """

        xp_assert_close(
            log_encoding_NLog(xp_as_array(0.0, xp=xp)),
            0.124372627896372,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_NLog(xp_as_array(0.18, xp=xp)),
            0.363667770117139,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_NLog(xp_as_array(0.18, xp=xp), 12),
            0.363667770117139,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_NLog(xp_as_array(0.18, xp=xp), 10, False),
            0.351634850262366,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_NLog(xp_as_array(0.18, xp=xp), 10, False, False),
            0.337584957293328,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_NLog(xp_as_array(1.0, xp=xp)),
            0.605083088954056,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_NLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_encoding_NLog` definition n-dimensional arrays support.
        """

        y = 0.18
        x = as_ndarray(log_encoding_NLog(xp_as_array(y, xp=xp)))

        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_encoding_NLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_NLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_NLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_NLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_encoding_NLog` definition domain and range scale support.
        """

        y = 0.18
        x = as_ndarray(log_encoding_NLog(xp_as_array(y, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_NLog(xp_as_array(y * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_NLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_encoding_NLog` definition nan support.
        """

        log_encoding_NLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_NLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_decoding_NLog` definition unit tests methods.
    """

    def test_log_decoding_NLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_decoding_NLog` definition.
        """

        xp_assert_close(
            log_decoding_NLog(xp_as_array(0.124372627896372, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_NLog(xp_as_array(0.363667770117139, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_NLog(xp_as_array(0.363667770117139, xp=xp), 12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_NLog(xp_as_array(0.351634850262366, xp=xp), 10, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_NLog(xp_as_array(0.337584957293328, xp=xp), 10, False, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_NLog(xp_as_array(0.605083088954056, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_NLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_decoding_NLog` definition n-dimensional arrays support.
        """

        x = 0.363667770117139
        y = as_ndarray(log_decoding_NLog(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        xp_assert_close(log_decoding_NLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_NLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_NLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_NLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_decoding_NLog` definition domain and range scale support.
        """

        x = 0.363667770117139
        y = as_ndarray(log_decoding_NLog(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_NLog(xp_as_array(x * factor, xp=xp)),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_NLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.nikon_n_log.\
log_decoding_NLog` definition nan support.
        """

        log_decoding_NLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
