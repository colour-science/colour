"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.aces`
module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_ACEScc,
    log_decoding_ACEScct,
    log_decoding_ACESproxy,
    log_encoding_ACEScc,
    log_encoding_ACEScct,
    log_encoding_ACESproxy,
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
    "TestLogEncoding_ACESproxy",
    "TestLogDecoding_ACESproxy",
    "TestLogEncoding_ACEScc",
    "TestLogDecoding_ACEScc",
    "TestLogDecoding_ACEScct",
]


class TestLogEncoding_ACESproxy:
    """
    Define :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy`
    definition unit tests methods.
    """

    def test_log_encoding_ACESproxy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition.
        """

        xp_assert_close(
            log_encoding_ACESproxy(xp_as_array(0.0, xp=xp)),
            0.062561094819159,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACESproxy(xp_as_array(0.18, xp=xp)),
            0.416422287390029,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACESproxy(xp_as_array(0.18, xp=xp), 12),
            0.416361416361416,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACESproxy(xp_as_array(1.0, xp=xp)),
            0.537634408602151,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert (
            as_ndarray(log_encoding_ACESproxy(xp_as_array(0.18, xp=xp), out_int=True))
            == 426
        )

    def test_n_dimensional_log_encoding_ACESproxy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition n-dimensional arrays support.
        """

        lin_AP1 = 0.18
        ACESproxy = as_ndarray(log_encoding_ACESproxy(xp_as_array(lin_AP1, xp=xp)))

        lin_AP1 = xp.tile(xp_as_array(lin_AP1, xp=xp), (6,))
        ACESproxy = xp.tile(xp_as_array(ACESproxy, xp=xp), (6,))
        xp_assert_close(
            log_encoding_ACESproxy(lin_AP1),
            ACESproxy,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3), xp=xp)
        ACESproxy = xp_reshape(xp_as_array(ACESproxy, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_encoding_ACESproxy(lin_AP1),
            ACESproxy,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3, 1), xp=xp)
        ACESproxy = xp_reshape(xp_as_array(ACESproxy, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_encoding_ACESproxy(lin_AP1),
            ACESproxy,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_ACESproxy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition domain and range scale support.
        """

        lin_AP1 = 0.18
        ACESproxy = as_ndarray(log_encoding_ACESproxy(xp_as_array(lin_AP1, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_ACESproxy(xp_as_array(lin_AP1 * factor, xp=xp)),
                    ACESproxy * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_ACESproxy(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACESproxy` definition nan support.
        """

        log_encoding_ACESproxy(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_ACESproxy:
    """
    Define :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy`
    definition unit tests methods.
    """

    def test_log_decoding_ACESproxy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy` definition.
        """

        xp_assert_close(
            log_decoding_ACESproxy(xp_as_array(0.062561094819159, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

        xp_assert_close(
            log_decoding_ACESproxy(xp_as_array(0.416422287390029, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

        xp_assert_close(
            log_decoding_ACESproxy(xp_as_array(0.416361416361416, xp=xp), 12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

        xp_assert_close(
            log_decoding_ACESproxy(xp_as_array(0.537634408602151, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

        xp_assert_close(
            log_decoding_ACESproxy(xp_as_array(426, xp=xp), in_int=True),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

    def test_n_dimensional_log_decoding_ACESproxy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy` definition n-dimensional arrays support.
        """

        ACESproxy = 0.416422287390029
        lin_AP1 = as_ndarray(log_decoding_ACESproxy(xp_as_array(ACESproxy, xp=xp)))

        ACESproxy = xp.tile(xp_as_array(ACESproxy, xp=xp), (6,))
        lin_AP1 = xp.tile(xp_as_array(lin_AP1, xp=xp), (6,))
        xp_assert_close(
            log_decoding_ACESproxy(ACESproxy),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        ACESproxy = xp_reshape(xp_as_array(ACESproxy, xp=xp), (2, 3), xp=xp)
        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_decoding_ACESproxy(ACESproxy),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        ACESproxy = xp_reshape(xp_as_array(ACESproxy, xp=xp), (2, 3, 1), xp=xp)
        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_decoding_ACESproxy(ACESproxy),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_ACESproxy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy` definition domain and range scale support.
        """

        ACESproxy = 426.0
        lin_AP1 = as_ndarray(log_decoding_ACESproxy(xp_as_array(ACESproxy, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_ACESproxy(xp_as_array(ACESproxy * factor, xp=xp)),
                    lin_AP1 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_ACESproxy(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACESproxy` definition nan support.
        """

        log_decoding_ACESproxy(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_ACEScc:
    """
    Define :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition unit tests methods.
    """

    def test_log_encoding_ACEScc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition.
        """

        xp_assert_close(
            log_encoding_ACEScc(xp_as_array(0.0, xp=xp)),
            -0.358447488584475,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACEScc(xp_as_array(0.18, xp=xp)),
            0.413588402492442,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACEScc(xp_as_array(1.0, xp=xp)),
            0.554794520547945,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_ACEScc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition n-dimensional arrays support.
        """

        lin_AP1 = 0.18
        ACEScc = as_ndarray(log_encoding_ACEScc(xp_as_array(lin_AP1, xp=xp)))

        lin_AP1 = xp.tile(xp_as_array(lin_AP1, xp=xp), (6,))
        ACEScc = xp.tile(xp_as_array(ACEScc, xp=xp), (6,))
        xp_assert_close(
            log_encoding_ACEScc(lin_AP1),
            ACEScc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3), xp=xp)
        ACEScc = xp_reshape(xp_as_array(ACEScc, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_encoding_ACEScc(lin_AP1),
            ACEScc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3, 1), xp=xp)
        ACEScc = xp_reshape(xp_as_array(ACEScc, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_encoding_ACEScc(lin_AP1),
            ACEScc,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_ACEScc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition domain and range scale support.
        """

        lin_AP1 = 0.18
        ACEScc = as_ndarray(log_encoding_ACEScc(xp_as_array(lin_AP1, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_ACEScc(xp_as_array(lin_AP1 * factor, xp=xp)),
                    ACEScc * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_ACEScc(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScc` definition nan support.
        """

        log_encoding_ACEScc(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_ACEScc:
    """
    Define :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition unit tests methods.
    """

    def test_log_decoding_ACEScc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition.
        """

        xp_assert_close(
            log_decoding_ACEScc(xp_as_array(-0.358447488584475, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_ACEScc(xp_as_array(0.413588402492442, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_ACEScc(xp_as_array(0.554794520547945, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_ACEScc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition n-dimensional arrays support.
        """

        ACEScc = 0.413588402492442
        lin_AP1 = as_ndarray(log_decoding_ACEScc(xp_as_array(ACEScc, xp=xp)))

        ACEScc = xp.tile(xp_as_array(ACEScc, xp=xp), (6,))
        lin_AP1 = xp.tile(xp_as_array(lin_AP1, xp=xp), (6,))
        xp_assert_close(
            log_decoding_ACEScc(ACEScc),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        ACEScc = xp_reshape(xp_as_array(ACEScc, xp=xp), (2, 3), xp=xp)
        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_decoding_ACEScc(ACEScc),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        ACEScc = xp_reshape(xp_as_array(ACEScc, xp=xp), (2, 3, 1), xp=xp)
        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_decoding_ACEScc(ACEScc),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_ACEScc(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition domain and range scale support.
        """

        ACEScc = 0.413588402492442
        lin_AP1 = as_ndarray(log_decoding_ACEScc(xp_as_array(ACEScc, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_ACEScc(xp_as_array(ACEScc * factor, xp=xp)),
                    lin_AP1 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_ACEScc(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScc` definition nan support.
        """

        log_decoding_ACEScc(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_ACEScct:
    """
    Define :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition unit tests methods.
    """

    def test_log_encoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition.
        """

        xp_assert_close(
            log_encoding_ACEScct(xp_as_array(0.0, xp=xp)),
            0.072905534195835495,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACEScct(xp_as_array(0.18, xp=xp)),
            0.413588402492442,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_ACEScct(xp_as_array(1.0, xp=xp)),
            0.554794520547945,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition n-dimensional arrays support.
        """

        lin_AP1 = 0.18
        ACEScct = as_ndarray(log_encoding_ACEScct(xp_as_array(lin_AP1, xp=xp)))

        lin_AP1 = xp.tile(xp_as_array(lin_AP1, xp=xp), (6,))
        ACEScct = xp.tile(xp_as_array(ACEScct, xp=xp), (6,))
        xp_assert_close(
            log_encoding_ACEScct(lin_AP1),
            ACEScct,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3), xp=xp)
        ACEScct = xp_reshape(xp_as_array(ACEScct, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_encoding_ACEScct(lin_AP1),
            ACEScct,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3, 1), xp=xp)
        ACEScct = xp_reshape(xp_as_array(ACEScct, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_encoding_ACEScct(lin_AP1),
            ACEScct,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition domain and range scale support.
        """

        lin_AP1 = 0.18
        ACEScct = as_ndarray(log_encoding_ACEScct(xp_as_array(lin_AP1, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_ACEScct(xp_as_array(lin_AP1 * factor, xp=xp)),
                    ACEScct * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    def test_ACEScc_equivalency_log_encoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition ACEScc equivalency, and explicit requirement
        specified by AMPAS ACES specification S-2016-001
        (https://github.com/ampas/aces-dev/blob/v1.0.3/documents/LaTeX/\
S-2016-001/introduction.tex#L14)
        """

        equiv = np.linspace(0.0078125, 222.86094420380761, 100)
        xp_assert_close(
            log_encoding_ACEScct(xp_as_array(equiv, xp=xp)),
            as_ndarray(log_encoding_ACEScc(xp_as_array(equiv, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_log_encoding_ACEScct(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_encoding_ACEScct` definition nan support.
        """

        log_encoding_ACEScct(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_ACEScct:
    """
    Define :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition unit tests methods.
    """

    def test_log_decoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition.
        """

        xp_assert_close(
            log_decoding_ACEScct(xp_as_array(0.072905534195835495, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_ACEScct(xp_as_array(0.41358840249244228, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_ACEScct(xp_as_array(0.554794520547945, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition n-dimensional arrays support.
        """

        ACEScct = 0.413588402492442
        lin_AP1 = as_ndarray(log_decoding_ACEScct(xp_as_array(ACEScct, xp=xp)))

        ACEScct = xp.tile(xp_as_array(ACEScct, xp=xp), (6,))
        lin_AP1 = xp.tile(xp_as_array(lin_AP1, xp=xp), (6,))
        xp_assert_close(
            log_decoding_ACEScct(ACEScct),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        ACEScct = xp_reshape(xp_as_array(ACEScct, xp=xp), (2, 3), xp=xp)
        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_decoding_ACEScct(ACEScct),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        ACEScct = xp_reshape(xp_as_array(ACEScct, xp=xp), (2, 3, 1), xp=xp)
        lin_AP1 = xp_reshape(xp_as_array(lin_AP1, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_decoding_ACEScct(ACEScct),
            lin_AP1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition domain and range scale support.
        """

        ACEScc = 0.413588402492442
        lin_AP1 = as_ndarray(log_decoding_ACEScct(xp_as_array(ACEScc, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_ACEScct(xp_as_array(ACEScc * factor, xp=xp)),
                    lin_AP1 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    def test_ACEScc_equivalency_log_decoding_ACEScct(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition ACEScc equivalency, and explicit requirement
        specified by AMPAS ACES specification S-2016-001
        (https://github.com/ampas/aces-dev/blob/v1.0.3/documents/LaTeX/\
S-2016-001/introduction.tex#L14)
        """

        equiv = np.linspace(0.15525114155251146, 1.0, 100)
        xp_assert_close(
            log_decoding_ACEScct(xp_as_array(equiv, xp=xp)),
            as_ndarray(log_decoding_ACEScc(xp_as_array(equiv, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_log_decoding_ACEScct(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.aces.\
log_decoding_ACEScct` definition nan support.
        """

        log_decoding_ACEScct(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
