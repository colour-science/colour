"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.panalog` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_Panalog,
    log_encoding_Panalog,
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
    "TestLogEncoding_Panalog",
    "TestLogDecoding_Panalog",
]


class TestLogEncoding_Panalog:
    """
    Define :func:`colour.models.rgb.transfer_functions.panalog.\
log_encoding_Panalog` definition unit tests methods.
    """

    def test_log_encoding_Panalog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.panalog.\
log_encoding_Panalog` definition.
        """

        xp_assert_close(
            log_encoding_Panalog(xp_as_array(0.0, xp=xp)),
            0.062561094819159,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Panalog(xp_as_array(0.18, xp=xp)),
            0.374576791382298,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Panalog(xp_as_array(1.0, xp=xp)),
            0.665689149560117,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_Panalog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.panalog.\
log_encoding_Panalog` definition n-dimensional arrays support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_Panalog(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        xp_assert_close(log_encoding_Panalog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_Panalog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_Panalog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_Panalog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.panalog.\
log_encoding_Panalog` definition domain and range scale support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_Panalog(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_Panalog(xp_as_array(x * factor, xp=xp)),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_Panalog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.panalog.\
log_encoding_Panalog` definition nan support.
        """

        log_encoding_Panalog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Panalog:
    """
    Define :func:`colour.models.rgb.transfer_functions.panalog.\
log_decoding_Panalog` definition unit tests methods.
    """

    def test_log_decoding_Panalog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.panalog.\
log_decoding_Panalog` definition.
        """

        xp_assert_close(
            log_decoding_Panalog(xp_as_array(0.062561094819159, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Panalog(xp_as_array(0.374576791382298, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Panalog(xp_as_array(0.665689149560117, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_Panalog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.panalog.\
log_decoding_Panalog` definition n-dimensional arrays support.
        """

        y = 0.374576791382298
        x = as_ndarray(log_decoding_Panalog(xp_as_array(y, xp=xp)))

        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_decoding_Panalog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_Panalog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_Panalog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_Panalog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.panalog.\
log_decoding_Panalog` definition domain and range scale support.
        """

        y = 0.374576791382298
        x = as_ndarray(log_decoding_Panalog(xp_as_array(y, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_Panalog(xp_as_array(y * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_Panalog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.panalog.\
log_decoding_Panalog` definition nan support.
        """

        log_decoding_Panalog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
