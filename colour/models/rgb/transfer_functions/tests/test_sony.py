"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.sony` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_SLog,
    log_decoding_SLog2,
    log_decoding_SLog3,
    log_encoding_SLog,
    log_encoding_SLog2,
    log_encoding_SLog3,
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
    "TestLogEncoding_SLog",
    "TestLogDecoding_SLog",
    "TestLogEncoding_SLog2",
    "TestLogDecoding_SLog2",
    "TestLogEncoding_SLog3",
    "TestLogDecoding_SLog3",
]


class TestLogEncoding_SLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog` definition unit tests methods.
    """

    def test_log_encoding_SLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog` definition.
        """

        xp_assert_close(
            log_encoding_SLog(xp_as_array(0.0, xp=xp)),
            0.088251291513446,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog(xp_as_array(0.18, xp=xp)),
            0.384970815928670,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog(xp_as_array(0.18, xp=xp), 12),
            0.384688786026891,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog(xp_as_array(0.18, xp=xp), 10, False),
            0.376512722254600,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog(xp_as_array(0.18, xp=xp), 10, False, False),
            0.359987846422154,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog(xp_as_array(1.0, xp=xp)),
            0.638551684622532,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_SLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog` definition n-dimensional arrays support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_SLog(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        xp_assert_close(log_encoding_SLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_SLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_SLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_SLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog` definition domain and range scale support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_SLog(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_SLog(xp_as_array(x * factor, xp=xp)),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_SLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog` definition nan support.
        """

        log_encoding_SLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_SLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog` definition unit tests methods.
    """

    def test_log_decoding_SLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog` definition.
        """

        xp_assert_close(
            log_decoding_SLog(xp_as_array(0.088251291513446, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog(xp_as_array(0.384970815928670, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog(xp_as_array(0.384688786026891, xp=xp), 12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog(xp_as_array(0.376512722254600, xp=xp), 10, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog(xp_as_array(0.359987846422154, xp=xp), 10, False, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog(xp_as_array(0.638551684622532, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_SLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog` definition n-dimensional arrays support.
        """

        y = 0.384970815928670
        x = as_ndarray(log_decoding_SLog(xp_as_array(y, xp=xp)))

        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_decoding_SLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_SLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_SLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_SLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog` definition domain and range scale support.
        """

        y = 0.384970815928670
        x = as_ndarray(log_decoding_SLog(xp_as_array(y, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_SLog(xp_as_array(y * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_SLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog` definition nan support.
        """

        log_decoding_SLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_SLog2:
    """
    Define :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog2` definition unit tests methods.
    """

    def test_log_encoding_SLog2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog2` definition.
        """

        xp_assert_close(
            log_encoding_SLog2(xp_as_array(0.0, xp=xp)),
            0.088251291513446,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog2(xp_as_array(0.18, xp=xp)),
            0.339532524633774,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog2(xp_as_array(0.18, xp=xp), 12),
            0.339283782857486,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog2(xp_as_array(0.18, xp=xp), 10, False),
            0.323449512215013,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog2(xp_as_array(0.18, xp=xp), 10, False, False),
            0.307980741258647,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog2(xp_as_array(1.0, xp=xp)),
            0.585091059564112,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_SLog2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog2` definition n-dimensional arrays support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_SLog2(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        xp_assert_close(log_encoding_SLog2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_SLog2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_SLog2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_SLog2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog2` definition domain and range scale support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_SLog2(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_SLog2(xp_as_array(x * factor, xp=xp)),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_SLog2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog2` definition nan support.
        """

        log_encoding_SLog2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_SLog2:
    """
    Define :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog2` definition unit tests methods.
    """

    def test_log_decoding_SLog2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog2` definition.
        """

        xp_assert_close(
            log_decoding_SLog2(xp_as_array(0.088251291513446, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog2(xp_as_array(0.339532524633774, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog2(xp_as_array(0.339283782857486, xp=xp), 12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog2(xp_as_array(0.323449512215013, xp=xp), 10, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog2(xp_as_array(0.307980741258647, xp=xp), 10, False, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog2(xp_as_array(0.585091059564112, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_SLog2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog2` definition n-dimensional arrays support.
        """

        y = 0.339532524633774
        x = as_ndarray(log_decoding_SLog2(xp_as_array(y, xp=xp)))

        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_decoding_SLog2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_SLog2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_SLog2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_SLog2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog2` definition domain and range scale support.
        """

        y = 0.339532524633774
        x = as_ndarray(log_decoding_SLog2(xp_as_array(y, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_SLog2(xp_as_array(y * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_SLog2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog2` definition nan support.
        """

        log_decoding_SLog2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_SLog3:
    """
    Define :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog3` definition unit tests methods.
    """

    def test_log_encoding_SLog3(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog3` definition.
        """

        xp_assert_close(
            log_encoding_SLog3(xp_as_array(0.0, xp=xp)),
            0.092864125122190,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog3(xp_as_array(0.18, xp=xp)),
            0.41055718475073,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog3(xp_as_array(0.18, xp=xp), 12),
            0.410557184750733,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog3(xp_as_array(0.18, xp=xp), 10, False),
            0.406392694063927,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog3(xp_as_array(0.18, xp=xp), 10, False, False),
            0.393489294768447,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_SLog3(xp_as_array(1.0, xp=xp)),
            0.596027343690123,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_SLog3(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog3` definition n-dimensional arrays support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_SLog3(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        xp_assert_close(log_encoding_SLog3(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_SLog3(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_SLog3(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_SLog3(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog3` definition domain and range scale support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_SLog3(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_SLog3(xp_as_array(x * factor, xp=xp)),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_SLog3(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_encoding_SLog3` definition nan support.
        """

        log_encoding_SLog3(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_SLog3:
    """
    Define :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog3` definition unit tests methods.
    """

    def test_log_decoding_SLog3(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog3` definition.
        """

        xp_assert_close(
            log_decoding_SLog3(xp_as_array(0.092864125122190, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog3(xp_as_array(0.41055718475073, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog3(xp_as_array(0.410557184750733, xp=xp), 12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog3(xp_as_array(0.406392694063927, xp=xp), 10, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog3(xp_as_array(0.393489294768447, xp=xp), 10, False, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_SLog3(xp_as_array(0.596027343690123, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_SLog3(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog3` definition n-dimensional arrays support.
        """

        y = 0.41055718475073
        x = as_ndarray(log_decoding_SLog3(xp_as_array(y, xp=xp)))

        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_decoding_SLog3(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_SLog3(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_SLog3(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_SLog3(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog3` definition domain and range scale support.
        """

        y = 0.41055718475073
        x = as_ndarray(log_decoding_SLog3(xp_as_array(y, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_SLog3(xp_as_array(y * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_SLog3(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.sony.\
log_decoding_SLog3` definition nan support.
        """

        log_decoding_SLog3(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
