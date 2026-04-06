"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.rimm_romm_rgb` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    cctf_decoding_RIMMRGB,
    cctf_decoding_ROMMRGB,
    cctf_encoding_RIMMRGB,
    cctf_encoding_ROMMRGB,
    log_decoding_ERIMMRGB,
    log_encoding_ERIMMRGB,
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
    "TestCctfEncoding_ROMMRGB",
    "TestCctfDecoding_ROMMRGB",
    "TestCctfEncoding_RIMMRGB",
    "TestCctfDecoding_RIMMRGB",
    "TestLog_encoding_ERIMMRGB",
    "TestLog_decoding_ERIMMRGB",
]


class TestCctfEncoding_ROMMRGB:
    """
    Define :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_ROMMRGB` definition unit tests methods.
    """

    def test_cctf_encoding_ROMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_ROMMRGB` definition.
        """

        xp_assert_close(
            cctf_encoding_ROMMRGB(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            cctf_encoding_ROMMRGB(xp_as_array(0.18, xp=xp)),
            0.385711424751138,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            cctf_encoding_ROMMRGB(xp_as_array(1.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert (
            as_ndarray(cctf_encoding_ROMMRGB(xp_as_array(0.18, xp=xp), out_int=True))
            == 98
        )

        assert (
            as_ndarray(
                cctf_encoding_ROMMRGB(
                    xp_as_array(0.18, xp=xp), bit_depth=12, out_int=True
                )
            )
            == 1579
        )

    def test_n_dimensional_cctf_encoding_ROMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_ROMMRGB` definition n-dimensional arrays support.
        """

        X = 0.18
        X_ROMM = as_ndarray(cctf_encoding_ROMMRGB(xp_as_array(X, xp=xp)))

        X = xp.tile(xp_as_array(X, xp=xp), (6,))
        X_ROMM = xp.tile(xp_as_array(X_ROMM, xp=xp), (6,))
        xp_assert_close(cctf_encoding_ROMMRGB(X), X_ROMM, atol=TOLERANCE_ABSOLUTE_TESTS)

        X = xp_reshape(xp_as_array(X, xp=xp), (2, 3), xp=xp)
        X_ROMM = xp_reshape(xp_as_array(X_ROMM, xp=xp), (2, 3), xp=xp)
        xp_assert_close(cctf_encoding_ROMMRGB(X), X_ROMM, atol=TOLERANCE_ABSOLUTE_TESTS)

        X = xp_reshape(xp_as_array(X, xp=xp), (2, 3, 1), xp=xp)
        X_ROMM = xp_reshape(xp_as_array(X_ROMM, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(cctf_encoding_ROMMRGB(X), X_ROMM, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_cctf_encoding_ROMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_ROMMRGB` definition domain and range scale support.
        """

        X = 0.18
        X_p = as_ndarray(cctf_encoding_ROMMRGB(xp_as_array(X, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    cctf_encoding_ROMMRGB(xp_as_array(X * factor, xp=xp)),
                    X_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_cctf_encoding_ROMMRGB(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_ROMMRGB` definition nan support.
        """

        cctf_encoding_ROMMRGB(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestCctfDecoding_ROMMRGB:
    """
        Define :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.
    cctf_decoding_ROMMRGB` definition unit tests methods.
    """

    def test_cctf_decoding_ROMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_ROMMRGB` definition.
        """

        xp_assert_close(
            cctf_decoding_ROMMRGB(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            cctf_decoding_ROMMRGB(xp_as_array(0.385711424751138, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            cctf_decoding_ROMMRGB(xp_as_array(1.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            cctf_decoding_ROMMRGB(xp_as_array(98, xp=xp), in_int=True),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

        xp_assert_close(
            cctf_decoding_ROMMRGB(xp_as_array(1579, xp=xp), bit_depth=12, in_int=True),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS * 10000,
        )

    def test_n_dimensional_cctf_decoding_ROMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_ROMMRGB` definition n-dimensional arrays support.
        """

        X_p = 0.385711424751138
        X = as_ndarray(cctf_decoding_ROMMRGB(xp_as_array(X_p, xp=xp)))

        X_p = xp.tile(xp_as_array(X_p, xp=xp), (6,))
        X = xp.tile(xp_as_array(X, xp=xp), (6,))
        xp_assert_close(cctf_decoding_ROMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS)

        X_p = xp_reshape(xp_as_array(X_p, xp=xp), (2, 3), xp=xp)
        X = xp_reshape(xp_as_array(X, xp=xp), (2, 3), xp=xp)
        xp_assert_close(cctf_decoding_ROMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS)

        X_p = xp_reshape(xp_as_array(X_p, xp=xp), (2, 3, 1), xp=xp)
        X = xp_reshape(xp_as_array(X, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(cctf_decoding_ROMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_cctf_decoding_ROMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_ROMMRGB` definition domain and range scale support.
        """

        X_p = 0.385711424751138
        X = as_ndarray(cctf_decoding_ROMMRGB(xp_as_array(X_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    cctf_decoding_ROMMRGB(xp_as_array(X_p * factor, xp=xp)),
                    X * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_cctf_decoding_ROMMRGB(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_ROMMRGB` definition nan support.
        """

        cctf_decoding_ROMMRGB(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestCctfEncoding_RIMMRGB:
    """
    Define :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_RIMMRGB` definition unit tests methods.
    """

    def test_cctf_encoding_RIMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_RIMMRGB` definition.
        """

        xp_assert_close(
            cctf_encoding_RIMMRGB(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            cctf_encoding_RIMMRGB(xp_as_array(0.18, xp=xp)),
            0.291673732475746,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            cctf_encoding_RIMMRGB(xp_as_array(1.0, xp=xp)),
            0.713125234297525,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert (
            as_ndarray(cctf_encoding_RIMMRGB(xp_as_array(0.18, xp=xp), out_int=True))
            == 74
        )

        assert (
            as_ndarray(
                cctf_encoding_RIMMRGB(
                    xp_as_array(0.18, xp=xp), bit_depth=12, out_int=True
                )
            )
            == 1194
        )

    def test_n_dimensional_cctf_encoding_RIMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_RIMMRGB` definition n-dimensional arrays support.
        """

        X = 0.18
        X_p = as_ndarray(cctf_encoding_RIMMRGB(xp_as_array(X, xp=xp)))

        X = xp.tile(xp_as_array(X, xp=xp), (6,))
        X_p = xp.tile(xp_as_array(X_p, xp=xp), (6,))
        xp_assert_close(cctf_encoding_RIMMRGB(X), X_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        X = xp_reshape(xp_as_array(X, xp=xp), (2, 3), xp=xp)
        X_p = xp_reshape(xp_as_array(X_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(cctf_encoding_RIMMRGB(X), X_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        X = xp_reshape(xp_as_array(X, xp=xp), (2, 3, 1), xp=xp)
        X_p = xp_reshape(xp_as_array(X_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(cctf_encoding_RIMMRGB(X), X_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_cctf_encoding_RIMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_RIMMRGB` definition domain and range scale support.
        """

        X = 0.18
        X_p = as_ndarray(cctf_encoding_RIMMRGB(xp_as_array(X, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    cctf_encoding_RIMMRGB(xp_as_array(X * factor, xp=xp)),
                    X_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_cctf_encoding_RIMMRGB(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_RIMMRGB` definition nan support.
        """

        cctf_encoding_RIMMRGB(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestCctfDecoding_RIMMRGB:
    """
        Define :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.
    cctf_decoding_RIMMRGB` definition unit tests methods.
    """

    def test_cctf_decoding_RIMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_RIMMRGB` definition.
        """

        xp_assert_close(
            cctf_decoding_RIMMRGB(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            cctf_decoding_RIMMRGB(xp_as_array(0.291673732475746, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            cctf_decoding_RIMMRGB(xp_as_array(0.713125234297525, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            cctf_decoding_RIMMRGB(xp_as_array(74, xp=xp), in_int=True),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS * 50000,
        )

        xp_assert_close(
            cctf_decoding_RIMMRGB(xp_as_array(1194, xp=xp), bit_depth=12, in_int=True),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS * 50000,
        )

    def test_n_dimensional_cctf_decoding_RIMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_RIMMRGB` definition n-dimensional arrays support.
        """

        X_p = 0.291673732475746
        X = as_ndarray(cctf_decoding_RIMMRGB(xp_as_array(X_p, xp=xp)))

        X_p = xp.tile(xp_as_array(X_p, xp=xp), (6,))
        X = xp.tile(xp_as_array(X, xp=xp), (6,))
        xp_assert_close(cctf_decoding_RIMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS)

        X_p = xp_reshape(xp_as_array(X_p, xp=xp), (2, 3), xp=xp)
        X = xp_reshape(xp_as_array(X, xp=xp), (2, 3), xp=xp)
        xp_assert_close(cctf_decoding_RIMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS)

        X_p = xp_reshape(xp_as_array(X_p, xp=xp), (2, 3, 1), xp=xp)
        X = xp_reshape(xp_as_array(X, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(cctf_decoding_RIMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_cctf_decoding_RIMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_RIMMRGB` definition domain and range scale support.
        """

        X_p = 0.291673732475746
        X = as_ndarray(cctf_decoding_RIMMRGB(xp_as_array(X_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    cctf_decoding_RIMMRGB(xp_as_array(X_p * factor, xp=xp)),
                    X * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_cctf_decoding_RIMMRGB(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_RIMMRGB` definition nan support.
        """

        cctf_decoding_RIMMRGB(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLog_encoding_ERIMMRGB:
    """
    Define :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition unit tests methods.
    """

    def test_log_encoding_ERIMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition.
        """

        xp_assert_close(
            log_encoding_ERIMMRGB(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ERIMMRGB(xp_as_array(0.18, xp=xp)),
            0.410052389492129,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ERIMMRGB(xp_as_array(1.0, xp=xp)),
            0.545458327405113,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert (
            as_ndarray(log_encoding_ERIMMRGB(xp_as_array(0.18, xp=xp), out_int=True))
            == 105
        )

        assert (
            as_ndarray(
                log_encoding_ERIMMRGB(
                    xp_as_array(0.18, xp=xp), bit_depth=12, out_int=True
                )
            )
            == 1679
        )

    def test_n_dimensional_log_encoding_ERIMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition n-dimensional arrays support.
        """

        X = 0.18
        X_p = as_ndarray(log_encoding_ERIMMRGB(xp_as_array(X, xp=xp)))

        X = xp.tile(xp_as_array(X, xp=xp), (6,))
        X_p = xp.tile(xp_as_array(X_p, xp=xp), (6,))
        xp_assert_close(log_encoding_ERIMMRGB(X), X_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        X = xp_reshape(xp_as_array(X, xp=xp), (2, 3), xp=xp)
        X_p = xp_reshape(xp_as_array(X_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_ERIMMRGB(X), X_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        X = xp_reshape(xp_as_array(X, xp=xp), (2, 3, 1), xp=xp)
        X_p = xp_reshape(xp_as_array(X_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_ERIMMRGB(X), X_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_encoding_ERIMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition domain and range scale support.
        """

        X = 0.18
        X_p = as_ndarray(log_encoding_ERIMMRGB(xp_as_array(X, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_ERIMMRGB(xp_as_array(X * factor, xp=xp)),
                    X_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_ERIMMRGB(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition nan support.
        """

        log_encoding_ERIMMRGB(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLog_decoding_ERIMMRGB:
    """
        Define :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.
    log_decoding_ERIMMRGB` definition unit tests methods.
    """

    def test_log_decoding_ERIMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_decoding_ERIMMRGB` definition.
        """

        xp_assert_close(
            log_decoding_ERIMMRGB(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_ERIMMRGB(xp_as_array(0.410052389492129, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_ERIMMRGB(xp_as_array(0.545458327405113, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_ERIMMRGB(xp_as_array(105, xp=xp), in_int=True),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS * 50000,
        )

        xp_assert_close(
            log_decoding_ERIMMRGB(xp_as_array(1679, xp=xp), bit_depth=12, in_int=True),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS * 50000,
        )

    def test_n_dimensional_log_decoding_ERIMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_decoding_ERIMMRGB` definition n-dimensional arrays support.
        """

        X_p = 0.410052389492129
        X = as_ndarray(log_decoding_ERIMMRGB(xp_as_array(X_p, xp=xp)))

        X_p = xp.tile(xp_as_array(X_p, xp=xp), (6,))
        X = xp.tile(xp_as_array(X, xp=xp), (6,))
        xp_assert_close(log_decoding_ERIMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS)

        X_p = xp_reshape(xp_as_array(X_p, xp=xp), (2, 3), xp=xp)
        X = xp_reshape(xp_as_array(X, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_ERIMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS)

        X_p = xp_reshape(xp_as_array(X_p, xp=xp), (2, 3, 1), xp=xp)
        X = xp_reshape(xp_as_array(X, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_ERIMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_log_decoding_ERIMMRGB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_decoding_ERIMMRGB` definition domain and range scale support.
        """

        X_p = 0.410052389492129
        X = as_ndarray(log_decoding_ERIMMRGB(xp_as_array(X_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_ERIMMRGB(xp_as_array(X_p * factor, xp=xp)),
                    X * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_ERIMMRGB(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_decoding_ERIMMRGB` definition nan support.
        """

        log_decoding_ERIMMRGB(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
