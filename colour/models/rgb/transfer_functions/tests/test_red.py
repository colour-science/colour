"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.red` module.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_Log3G12,
    log_decoding_REDLog,
    log_decoding_REDLogFilm,
    log_encoding_Log3G12,
    log_encoding_REDLog,
    log_encoding_REDLogFilm,
)
from colour.models.rgb.transfer_functions.red import (
    log_decoding_Log3G10_v1,
    log_decoding_Log3G10_v2,
    log_decoding_Log3G10_v3,
    log_encoding_Log3G10_v1,
    log_encoding_Log3G10_v2,
    log_encoding_Log3G10_v3,
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
__status__ = "Development"

__all__ = [
    "TestLogEncoding_REDLog",
    "TestLogDecoding_REDLog",
    "TestLogEncoding_REDLogFilm",
    "TestLogDecoding_REDLogFilm",
    "TestLogEncoding_Log3G10_v1",
    "TestLogDecoding_Log3G10_v1",
    "TestLogEncoding_Log3G10_v2",
    "TestLogDecoding_Log3G10_v2",
    "TestLogEncoding_Log3G10_v3",
    "TestLogDecoding_Log3G10_v3",
    "TestLogEncoding_Log3G12",
    "TestLogDecoding_Log3G12",
]


class TestLogEncoding_REDLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLog` definition unit tests methods.
    """

    def test_log_encoding_REDLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLog` definition.
        """

        xp_assert_close(
            log_encoding_REDLog(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_REDLog(xp_as_array(0.18, xp=xp)),
            0.637621845988175,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_REDLog(xp_as_array(1.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_REDLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLog` definition n-dimensional arrays support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_REDLog(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        xp_assert_close(log_encoding_REDLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_REDLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_REDLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_REDLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLog` definition domain and range scale support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_REDLog(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_REDLog(xp_as_array(x * factor, xp=xp)),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_REDLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLog` definition nan support.
        """

        log_encoding_REDLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_REDLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLog` definition unit tests methods.
    """

    def test_log_decoding_REDLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLog` definition.
        """

        xp_assert_close(
            log_decoding_REDLog(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_REDLog(xp_as_array(0.637621845988175, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_REDLog(xp_as_array(1.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_REDLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLog` definition n-dimensional arrays support.
        """

        y = 0.637621845988175
        x = as_ndarray(log_decoding_REDLog(xp_as_array(y, xp=xp)))

        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_decoding_REDLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_REDLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_REDLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_REDLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLog` definition domain and range scale support.
        """

        y = 0.637621845988175
        x = as_ndarray(log_decoding_REDLog(xp_as_array(y, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_REDLog(xp_as_array(y * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_REDLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLog` definition nan support.
        """

        log_decoding_REDLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_REDLogFilm:
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLogFilm` definition unit tests methods.
    """

    def test_log_encoding_REDLogFilm(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLogFilm` definition.
        """

        xp_assert_close(
            log_encoding_REDLogFilm(xp_as_array(0.0, xp=xp)),
            0.092864125122190,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_REDLogFilm(xp_as_array(0.18, xp=xp)),
            0.457319613085418,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_REDLogFilm(xp_as_array(1.0, xp=xp)),
            0.669599217986315,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_REDLogFilm(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLogFilm` definition n-dimensional arrays support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_REDLogFilm(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        xp_assert_close(log_encoding_REDLogFilm(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_REDLogFilm(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_REDLogFilm(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_REDLogFilm(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLogFilm` definition domain and range scale support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_REDLogFilm(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_REDLogFilm(xp_as_array(x * factor, xp=xp)),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_REDLogFilm(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLogFilm` definition nan support.
        """

        log_encoding_REDLogFilm(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_REDLogFilm:
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLogFilm` definition unit tests methods.
    """

    def test_log_decoding_REDLogFilm(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLogFilm` definition.
        """

        xp_assert_close(
            log_decoding_REDLogFilm(xp_as_array(0.092864125122190, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_REDLogFilm(xp_as_array(0.457319613085418, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_REDLogFilm(xp_as_array(0.669599217986315, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_REDLogFilm(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLogFilm` definition n-dimensional arrays support.
        """

        y = 0.457319613085418
        x = as_ndarray(log_decoding_REDLogFilm(xp_as_array(y, xp=xp)))

        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_decoding_REDLogFilm(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_REDLogFilm(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_REDLogFilm(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_REDLogFilm(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLogFilm` definition domain and range scale support.
        """

        y = 0.457319613085418
        x = as_ndarray(log_decoding_REDLogFilm(xp_as_array(y, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_REDLogFilm(xp_as_array(y * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_REDLogFilm(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLogFilm` definition nan support.
        """

        log_decoding_REDLogFilm(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_Log3G10_v1:
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v1` definition unit tests methods.
    """

    def test_log_encoding_Log3G10_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v1` definition.
        """

        xp_assert_close(
            log_encoding_Log3G10_v1(xp_as_array(-1.0, xp=xp)),
            -0.496483569056003,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log3G10_v1(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log3G10_v1(xp_as_array(0.18, xp=xp)),
            0.333333644207707,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_Log3G10_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v1` definition n-dimensional arrays support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_Log3G10_v1(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        xp_assert_close(log_encoding_Log3G10_v1(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_Log3G10_v1(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_Log3G10_v1(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_Log3G10_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v1` definition domain and range scale support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_Log3G10_v1(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_Log3G10_v1(xp_as_array(x * factor, xp=xp)),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_Log3G10_v1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v1` definition nan support.
        """

        log_encoding_Log3G10_v1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Log3G10_v1:
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v1` definition unit tests methods.
    """

    def test_log_decoding_Log3G10_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v1` definition.
        """

        xp_assert_close(
            log_decoding_Log3G10_v1(xp_as_array(-0.496483569056003, xp=xp)),
            -1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log3G10_v1(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log3G10_v1(xp_as_array(0.333333644207707, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_Log3G10_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v1` definition n-dimensional arrays support.
        """

        y = 0.333333644207707
        x = as_ndarray(log_decoding_Log3G10_v1(xp_as_array(y, xp=xp)))

        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_decoding_Log3G10_v1(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_Log3G10_v1(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_Log3G10_v1(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_Log3G10_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v1` definition domain and range scale support.
        """

        y = 0.333333644207707
        x = as_ndarray(log_decoding_Log3G10_v1(xp_as_array(y, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_Log3G10_v1(xp_as_array(y * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_Log3G10_v1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v1` definition nan support.
        """

        log_decoding_Log3G10_v1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_Log3G10_v2:
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v2` definition unit tests methods.
    """

    def test_log_encoding_Log3G10_v2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v2` definition.
        """

        xp_assert_close(
            log_encoding_Log3G10_v2(xp_as_array(-1.0, xp=xp)),
            -0.491512777522511,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log3G10_v2(xp_as_array(0.0, xp=xp)),
            0.091551487714745,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log3G10_v2(xp_as_array(0.18, xp=xp)),
            0.333332912025992,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_Log3G10_v2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v2` definition n-dimensional arrays support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_Log3G10_v2(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        xp_assert_close(log_encoding_Log3G10_v2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_Log3G10_v2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_Log3G10_v2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_Log3G10_v2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v2` definition domain and range scale support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_Log3G10_v2(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_Log3G10_v2(xp_as_array(x * factor, xp=xp)),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_Log3G10_v2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v2` definition nan support.
        """

        log_encoding_Log3G10_v2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Log3G10_v2:
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v2` definition unit tests methods.
    """

    def test_log_decoding_Log3G10_v2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v2` definition.
        """

        xp_assert_close(
            log_decoding_Log3G10_v2(xp_as_array(-0.491512777522511, xp=xp)),
            -1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log3G10_v2(xp_as_array(0.091551487714745, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log3G10_v2(xp_as_array(0.333332912025992, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_Log3G10_v2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v2` definition n-dimensional arrays support.
        """

        y = 0.333332912025992
        x = as_ndarray(log_decoding_Log3G10_v2(xp_as_array(y, xp=xp)))

        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_decoding_Log3G10_v2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_Log3G10_v2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_Log3G10_v2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_Log3G10_v2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v2` definition domain and range scale support.
        """

        y = 0.333333644207707
        x = as_ndarray(log_decoding_Log3G10_v2(xp_as_array(y, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_Log3G10_v2(xp_as_array(y * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_Log3G10_v2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v2` definition nan support.
        """

        log_decoding_Log3G10_v2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_Log3G10_v3:
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v3` definition unit tests methods.
    """

    def test_log_encoding_Log3G10_v3(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v3` definition.
        """

        xp_assert_close(
            log_encoding_Log3G10_v3(xp_as_array(-1.0, xp=xp)),
            -15.040773,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log3G10_v3(xp_as_array(0.0, xp=xp)),
            0.091551487714745,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log3G10_v3(xp_as_array(0.18, xp=xp)),
            0.333332912025992,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_Log3G10_v3(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v3` definition n-dimensional arrays support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_Log3G10_v3(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        xp_assert_close(log_encoding_Log3G10_v3(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_Log3G10_v3(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_Log3G10_v3(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_Log3G10_v3(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v3` definition domain and range scale support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_Log3G10_v3(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_Log3G10_v3(xp_as_array(x * factor, xp=xp)),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_Log3G10_v3(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v3` definition nan support.
        """

        log_encoding_Log3G10_v3(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Log3G10_v3:
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v3` definition unit tests methods.
    """

    def test_log_decoding_Log3G10_v3(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v3` definition.
        """

        xp_assert_close(
            log_decoding_Log3G10_v3(xp_as_array(-15.040773, xp=xp)),
            -1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log3G10_v3(xp_as_array(0.091551487714745, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log3G10_v3(xp_as_array(0.333332912025992, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_Log3G10_v3(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v3` definition n-dimensional arrays support.
        """

        y = 0.333332912025992
        x = as_ndarray(log_decoding_Log3G10_v3(xp_as_array(y, xp=xp)))

        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_decoding_Log3G10_v3(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_Log3G10_v3(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_Log3G10_v3(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_Log3G10_v3(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v3` definition domain and range scale support.
        """

        y = 0.333333644207707
        x = as_ndarray(log_decoding_Log3G10_v3(xp_as_array(y, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_Log3G10_v3(xp_as_array(y * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_Log3G10_v3(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v3` definition nan support.
        """

        log_decoding_Log3G10_v3(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_Log3G12:
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G12` definition unit tests methods.
    """

    def test_log_encoding_Log3G12(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G12` definition.
        """

        xp_assert_close(
            log_encoding_Log3G12(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log3G12(xp_as_array(0.18, xp=xp)),
            0.333332662015923,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log3G12(xp_as_array(1.0, xp=xp)),
            0.469991923234319,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log3G12(xp_as_array(0.18 * 2**12, xp=xp)),
            0.999997986792394,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_Log3G12(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G12` definition n-dimensional arrays support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_Log3G12(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        xp_assert_close(log_encoding_Log3G12(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_Log3G12(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_Log3G12(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_Log3G12(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G12` definition domain and range scale support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_Log3G12(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_Log3G12(xp_as_array(x * factor, xp=xp)),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_Log3G12(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G12` definition nan support.
        """

        log_encoding_Log3G12(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Log3G12:
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G12` definition unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1e-2)
    def test_log_decoding_Log3G12(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G12` definition.
        """

        xp_assert_close(
            log_decoding_Log3G12(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log3G12(xp_as_array(0.333332662015923, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log3G12(xp_as_array(0.469991923234319, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log3G12(xp_as_array(1.0, xp=xp)),
            737.29848406719,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_Log3G12(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G12` definition n-dimensional arrays support.
        """

        y = 0.333332662015923
        x = as_ndarray(log_decoding_Log3G12(xp_as_array(y, xp=xp)))

        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_decoding_Log3G12(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_Log3G12(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_Log3G12(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_Log3G12(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G12` definition domain and range scale support.
        """

        y = 0.18
        x = as_ndarray(log_decoding_Log3G12(xp_as_array(y, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_Log3G12(xp_as_array(y * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_Log3G12(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G12` definition nan support.
        """

        log_decoding_Log3G12(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
