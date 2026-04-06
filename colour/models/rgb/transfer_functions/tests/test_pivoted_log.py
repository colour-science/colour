"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
pivoted_log` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_PivotedLog,
    log_encoding_PivotedLog,
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
    "TestLogEncoding_PivotedLog",
    "TestLogDecoding_PivotedLog",
]


class TestLogEncoding_PivotedLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_encoding_PivotedLog` definition unit tests methods.
    """

    def test_log_encoding_PivotedLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_encoding_PivotedLog` definition.
        """

        xp_assert_close(
            log_encoding_PivotedLog(xp_as_array(0.0, xp=xp)),
            -np.inf,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_PivotedLog(xp_as_array(0.18, xp=xp)),
            0.434995112414467,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_PivotedLog(xp_as_array(1.0, xp=xp)),
            0.653390272208219,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_PivotedLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_encoding_PivotedLog` definition n-dimensional arrays support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_PivotedLog(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        xp_assert_close(log_encoding_PivotedLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_PivotedLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_PivotedLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_PivotedLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_encoding_PivotedLog` definition domain and range scale support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_PivotedLog(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_PivotedLog(xp_as_array(x * factor, xp=xp)),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_PivotedLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_encoding_PivotedLog` definition nan support.
        """

        log_encoding_PivotedLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_PivotedLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_decoding_PivotedLog` definition unit tests methods.
    """

    def test_log_decoding_PivotedLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_decoding_PivotedLog` definition.
        """

        xp_assert_close(
            log_decoding_PivotedLog(xp_as_array(-np.inf, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_PivotedLog(xp_as_array(0.434995112414467, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_PivotedLog(xp_as_array(0.653390272208219, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_PivotedLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_decoding_PivotedLog` definition n-dimensional arrays support.
        """

        y = 0.434995112414467
        x = as_ndarray(log_decoding_PivotedLog(xp_as_array(y, xp=xp)))

        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_decoding_PivotedLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_PivotedLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_PivotedLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_PivotedLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_decoding_PivotedLog` definition domain and range scale support.
        """

        y = 0.434995112414467
        x = as_ndarray(log_decoding_PivotedLog(xp_as_array(y, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_PivotedLog(xp_as_array(y * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_PivotedLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.pivoted_log.\
log_decoding_PivotedLog` definition nan support.
        """

        log_decoding_PivotedLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
