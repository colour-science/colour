"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
filmlight_t_log` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_FilmLightTLog,
    log_encoding_FilmLightTLog,
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
    "TestLogEncoding_FilmLightTLog",
    "TestLogDecoding_FilmLightTLog",
]


class TestLogEncoding_FilmLightTLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_encoding_FilmLightTLog` definition unit tests methods.
    """

    def test_log_encoding_FilmLightTLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_encoding_FilmLightTLog` definition.
        """

        xp_assert_close(
            log_encoding_FilmLightTLog(xp_as_array(0.0, xp=xp)),
            0.075,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_FilmLightTLog(xp_as_array(0.18, xp=xp)),
            0.396567801298332,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_FilmLightTLog(xp_as_array(1.0, xp=xp)),
            0.552537881005859,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_TLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_encoding_FilmLightTLog` definition n-dimensional arrays support.
        """

        x = 0.18
        t = as_ndarray(log_encoding_FilmLightTLog(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        t = xp.tile(xp_as_array(t, xp=xp), (6,))
        xp_assert_close(log_encoding_FilmLightTLog(x), t, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        t = xp_reshape(xp_as_array(t, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_FilmLightTLog(x), t, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        t = xp_reshape(xp_as_array(t, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_FilmLightTLog(x), t, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_TLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_encoding_FilmLightTLog` definition domain and range scale support.
        """

        x = 0.18
        t = as_ndarray(log_encoding_FilmLightTLog(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_FilmLightTLog(xp_as_array(x * factor, xp=xp)),
                    t * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_TLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_encoding_FilmLightTLog` definition nan support.
        """

        log_encoding_FilmLightTLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_FilmLightTLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_decoding_FilmLightTLog` definition unit tests methods.
    """

    def test_log_decoding_FilmLightTLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_decoding_FilmLightTLog` definition.
        """

        xp_assert_close(
            log_decoding_FilmLightTLog(xp_as_array(0.075, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_FilmLightTLog(xp_as_array(0.396567801298332, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_FilmLightTLog(xp_as_array(0.552537881005859, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_TLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_decoding_FilmLightTLog` definition n-dimensional arrays support.
        """

        t = 0.396567801298332
        x = as_ndarray(log_decoding_FilmLightTLog(xp_as_array(t, xp=xp)))

        t = xp.tile(xp_as_array(t, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_decoding_FilmLightTLog(t), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        t = xp_reshape(xp_as_array(t, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_FilmLightTLog(t), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        t = xp_reshape(xp_as_array(t, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_FilmLightTLog(t), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_TLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_decoding_FilmLightTLog` definition domain and range scale support.
        """

        t = 0.396567801298332
        x = as_ndarray(log_decoding_FilmLightTLog(xp_as_array(t, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_FilmLightTLog(xp_as_array(t * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_TLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_decoding_FilmLightTLog` definition nan support.
        """

        log_decoding_FilmLightTLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
