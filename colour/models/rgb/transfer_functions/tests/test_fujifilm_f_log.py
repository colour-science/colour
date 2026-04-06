"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
fujifilm_f_log` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_FLog,
    log_decoding_FLog2,
    log_encoding_FLog,
    log_encoding_FLog2,
)
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_assert_equal,
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
    "TestLogEncoding_FLog",
    "TestLogDecoding_FLog",
    "TestLogEncoding_FLog2",
    "TestLogDecoding_FLog2",
]


class TestLogEncoding_FLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.fujifilm_f_log.\
log_encoding_FLog` definition unit tests methods.
    """

    def test_log_encoding_FLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_f_log.\
log_encoding_FLog` definition.
        """

        xp_assert_close(
            log_encoding_FLog(xp_as_array(0.0, xp=xp)),
            0.092864000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_FLog(xp_as_array(0.18, xp=xp)),
            0.459318458661621,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_FLog(xp_as_array(0.18, xp=xp), 12),
            0.459318458661621,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_FLog(xp_as_array(0.18, xp=xp), 10, False),
            0.463336510514656,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_FLog(xp_as_array(0.18, xp=xp), 10, False, False),
            0.446590337236003,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_FLog(xp_as_array(1.0, xp=xp)),
            0.704996409216428,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_FLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_f_log.\
log_encoding_FLog` definition n-dimensional arrays support.
        """

        L_in = 0.18
        V_out = as_ndarray(log_encoding_FLog(xp_as_array(L_in, xp=xp)))

        L_in = xp.tile(xp_as_array(L_in, xp=xp), (6,))
        V_out = xp.tile(xp_as_array(V_out, xp=xp), (6,))
        xp_assert_close(log_encoding_FLog(L_in), V_out, atol=TOLERANCE_ABSOLUTE_TESTS)

        L_in = xp_reshape(xp_as_array(L_in, xp=xp), (2, 3), xp=xp)
        V_out = xp_reshape(xp_as_array(V_out, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_FLog(L_in), V_out, atol=TOLERANCE_ABSOLUTE_TESTS)

        L_in = xp_reshape(xp_as_array(L_in, xp=xp), (2, 3, 1), xp=xp)
        V_out = xp_reshape(xp_as_array(V_out, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_FLog(L_in), V_out, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_FLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_f_log.\
log_encoding_FLog` definition domain and range scale support.
        """

        L_in = 0.18
        V_out = as_ndarray(log_encoding_FLog(xp_as_array(L_in, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_FLog(xp_as_array(L_in * factor, xp=xp)),
                    V_out * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_FLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_f_log.\
log_encoding_FLog` definition nan support.
        """

        log_encoding_FLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_FLog:
    """
    Define :func:`colour.models.rgb.transfer_functions.fujifilm_f_log.\
log_decoding_FLog` definition unit tests methods.
    """

    def test_log_decoding_FLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_f_log.\
log_decoding_FLog` definition.
        """

        xp_assert_close(
            log_decoding_FLog(xp_as_array(0.092864000000000, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_FLog(xp_as_array(0.459318458661621, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_FLog(xp_as_array(0.459318458661621, xp=xp), 12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_FLog(xp_as_array(0.463336510514656, xp=xp), 10, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_FLog(xp_as_array(0.446590337236003, xp=xp), 10, False, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_FLog(xp_as_array(0.704996409216428, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_FLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_f_log.\
log_decoding_FLog` definition n-dimensional arrays support.
        """

        V_out = 0.459318458661621
        L_in = as_ndarray(log_decoding_FLog(xp_as_array(V_out, xp=xp)))

        V_out = xp.tile(xp_as_array(V_out, xp=xp), (6,))
        L_in = xp.tile(xp_as_array(L_in, xp=xp), (6,))
        xp_assert_close(log_decoding_FLog(V_out), L_in, atol=TOLERANCE_ABSOLUTE_TESTS)

        V_out = xp_reshape(xp_as_array(V_out, xp=xp), (2, 3), xp=xp)
        L_in = xp_reshape(xp_as_array(L_in, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_FLog(V_out), L_in, atol=TOLERANCE_ABSOLUTE_TESTS)

        V_out = xp_reshape(xp_as_array(V_out, xp=xp), (2, 3, 1), xp=xp)
        L_in = xp_reshape(xp_as_array(L_in, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_FLog(V_out), L_in, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_FLog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_f_log.\
log_decoding_FLog` definition domain and range scale support.
        """

        V_out = 0.459318458661621
        L_in = as_ndarray(log_decoding_FLog(xp_as_array(V_out, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_FLog(xp_as_array(V_out * factor, xp=xp)),
                    L_in * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_FLog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_f_log.\
log_decoding_FLog` definition nan support.
        """

        log_decoding_FLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_FLog2:
    """
    Define :func:`colour.models.rgb.transfer_functions.fujifilm_flog.\
log_encoding_FLog2` definition unit tests methods.
    """

    def test_log_encoding_FLog2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_flog.\
log_encoding_FLog2` definition.
        """

        xp_assert_close(
            log_encoding_FLog2(xp_as_array(0.0, xp=xp)),
            0.092864000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_FLog2(xp_as_array(0.18, xp=xp)),
            0.39100724189123,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_FLog2(xp_as_array(0.18, xp=xp), 12),
            0.39100724189123,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_FLog2(xp_as_array(0.18, xp=xp), 10, False),
            0.383562110108137,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_FLog2(xp_as_array(0.18, xp=xp), 10, False, False),
            0.371293971820387,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_FLog2(xp_as_array(1.0, xp=xp)),
            0.568219370444443,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_FLog2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_flog.\
log_encoding_FLog2` definition n-dimensional arrays support.
        """

        L_in = 0.18
        V_out = as_ndarray(log_encoding_FLog2(xp_as_array(L_in, xp=xp)))

        L_in = xp.tile(xp_as_array(L_in, xp=xp), (6,))
        V_out = xp.tile(xp_as_array(V_out, xp=xp), (6,))
        xp_assert_close(log_encoding_FLog2(L_in), V_out, atol=TOLERANCE_ABSOLUTE_TESTS)

        L_in = xp_reshape(xp_as_array(L_in, xp=xp), (2, 3), xp=xp)
        V_out = xp_reshape(xp_as_array(V_out, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_FLog2(L_in), V_out, atol=TOLERANCE_ABSOLUTE_TESTS)

        L_in = xp_reshape(xp_as_array(L_in, xp=xp), (2, 3, 1), xp=xp)
        V_out = xp_reshape(xp_as_array(V_out, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_FLog2(L_in), V_out, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_FLog2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_flog.\
log_encoding_FLog2` definition domain and range scale support.
        """

        L_in = 0.18
        V_out = as_ndarray(log_encoding_FLog2(xp_as_array(L_in, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_equal(
                    log_encoding_FLog2(xp_as_array(L_in * factor, xp=xp)),
                    V_out * factor,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_FLog2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_flog.\
log_encoding_FLog2` definition nan support.
        """

        log_encoding_FLog2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_FLog2:
    """
    Define :func:`colour.models.rgb.transfer_functions.fujifilm_flog.\
log_decoding_FLog2` definition unit tests methods.
    """

    def test_log_decoding_FLog2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_flog.\
log_decoding_FLog2` definition.
        """

        xp_assert_close(
            log_decoding_FLog2(xp_as_array(0.092864000000000, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_FLog2(xp_as_array(0.391007241891230, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_FLog2(xp_as_array(0.391007241891230, xp=xp), 12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_FLog2(xp_as_array(0.383562110108137, xp=xp), 10, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_FLog2(xp_as_array(0.371293971820387, xp=xp), 10, False, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_FLog2(xp_as_array(0.568219370444443, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_FLog2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_flog.\
log_decoding_FLog2` definition n-dimensional arrays support.
        """

        V_out = 0.39100724189123
        L_in = as_ndarray(log_decoding_FLog2(xp_as_array(V_out, xp=xp)))

        V_out = xp.tile(xp_as_array(V_out, xp=xp), (6,))
        L_in = xp.tile(xp_as_array(L_in, xp=xp), (6,))
        xp_assert_close(log_decoding_FLog2(V_out), L_in, atol=TOLERANCE_ABSOLUTE_TESTS)

        V_out = xp_reshape(xp_as_array(V_out, xp=xp), (2, 3), xp=xp)
        L_in = xp_reshape(xp_as_array(L_in, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_FLog2(V_out), L_in, atol=TOLERANCE_ABSOLUTE_TESTS)

        V_out = xp_reshape(xp_as_array(V_out, xp=xp), (2, 3, 1), xp=xp)
        L_in = xp_reshape(xp_as_array(L_in, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_FLog2(V_out), L_in, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_FLog2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_flog.\
log_decoding_FLog2` definition domain and range scale support.
        """

        V_out = 0.39100724189123
        L_in = as_ndarray(log_decoding_FLog2(xp_as_array(V_out, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_equal(
                    log_decoding_FLog2(xp_as_array(V_out * factor, xp=xp)),
                    L_in * factor,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_FLog2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.fujifilm_flog.\
log_decoding_FLog2` definition nan support.
        """

        log_decoding_FLog2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
