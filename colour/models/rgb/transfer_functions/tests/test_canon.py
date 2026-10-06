"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.canon` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions.canon import (
    log_decoding_CanonLog2_v1,
    log_decoding_CanonLog2_v1_2,
    log_decoding_CanonLog3_v1,
    log_decoding_CanonLog3_v1_2,
    log_decoding_CanonLog_v1,
    log_decoding_CanonLog_v1_2,
    log_encoding_CanonLog2_v1,
    log_encoding_CanonLog2_v1_2,
    log_encoding_CanonLog3_v1,
    log_encoding_CanonLog3_v1_2,
    log_encoding_CanonLog_v1,
    log_encoding_CanonLog_v1_2,
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
    "TestLogEncoding_CanonLog_v1",
    "TestLogDecoding_CanonLog_v1",
    "TestLogEncoding_CanonLog_v1_2",
    "TestLogDecoding_CanonLog_v1_2",
    "TestLogEncoding_CanonLog2_v1",
    "TestLogDecoding_CanonLog2_v1",
    "TestLogEncoding_CanonLog2_v1_2",
    "TestLogDecoding_CanonLog2_v1_2",
    "TestLogEncoding_CanonLog3_v1",
    "TestLogDecoding_CanonLog3_v1",
    "TestLogEncoding_CanonLog3_v1_2",
    "TestLogDecoding_CanonLog3_v1_2",
]


class TestLogEncoding_CanonLog_v1:
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1` definition unit tests methods.
    """

    def test_log_encoding_CanonLog_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1` definition.
        """

        xp_assert_close(
            log_encoding_CanonLog_v1(xp_as_array(-0.1, xp=xp)),
            -0.023560122781997,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1(xp_as_array(0.0, xp=xp)),
            0.125122480156403,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1(xp_as_array(0.18, xp=xp)),
            0.343389651726069,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1(xp_as_array(0.18, xp=xp), 12),
            0.343138084215647,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1(xp_as_array(0.18, xp=xp), 10, False),
            0.327953896935809,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1(xp_as_array(0.18, xp=xp), 10, False, False),
            0.312012855550395,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1(xp_as_array(1.0, xp=xp)),
            0.618775485598649,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_CanonLog_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1` definition n-dimensional arrays support.
        """

        x = 0.18
        clog = as_ndarray(log_encoding_CanonLog_v1(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        clog = xp.tile(xp_as_array(clog, xp=xp), (6,))
        xp_assert_close(
            log_encoding_CanonLog_v1(x), clog, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        clog = xp_reshape(xp_as_array(clog, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_encoding_CanonLog_v1(x), clog, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        clog = xp_reshape(xp_as_array(clog, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_encoding_CanonLog_v1(x), clog, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_CanonLog_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1` definition domain and range scale support.
        """

        x = 0.18
        clog = as_ndarray(log_encoding_CanonLog_v1(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_CanonLog_v1(xp_as_array(x * factor, xp=xp)),
                    clog * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog_v1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1` definition nan support.
        """

        log_encoding_CanonLog_v1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_CanonLog_v1:
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1` definition unit tests methods.
    """

    def test_log_decoding_CanonLog_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1` definition.
        """

        xp_assert_close(
            log_decoding_CanonLog_v1(xp_as_array(-0.023560122781997, xp=xp)),
            -0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1(xp_as_array(0.125122480156403, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1(xp_as_array(0.343389651726069, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1(xp_as_array(0.343138084215647, xp=xp), 12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1(xp_as_array(0.327953896935809, xp=xp), 10, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1(
                xp_as_array(0.312012855550395, xp=xp), 10, False, False
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1(xp_as_array(0.618775485598649, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_CanonLog_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1` definition n-dimensional arrays support.
        """

        clog = 0.343389651726069
        x = as_ndarray(log_decoding_CanonLog_v1(xp_as_array(clog, xp=xp)))

        clog = xp.tile(xp_as_array(clog, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(
            log_decoding_CanonLog_v1(clog), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        clog = xp_reshape(xp_as_array(clog, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_decoding_CanonLog_v1(clog), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        clog = xp_reshape(xp_as_array(clog, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_decoding_CanonLog_v1(clog), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_CanonLog_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1` definition domain and range scale support.
        """

        clog = 0.343389651726069
        x = as_ndarray(log_decoding_CanonLog_v1(xp_as_array(clog, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_CanonLog_v1(xp_as_array(clog * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog_v1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1` definition nan support.
        """

        log_decoding_CanonLog_v1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_CanonLog_v1_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1_2` definition unit tests methods.
    """

    def test_log_encoding_CanonLog_v1_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1_2` definition.
        """

        xp_assert_close(
            log_encoding_CanonLog_v1_2(xp_as_array(-0.1, xp=xp)),
            -0.023560121389098,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1_2(xp_as_array(0.0, xp=xp)),
            0.125122480000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1_2(xp_as_array(0.18, xp=xp)),
            0.343389649295280,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1_2(xp_as_array(0.18, xp=xp), 12),
            0.343389649295281,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1_2(xp_as_array(0.18, xp=xp), 10, False),
            0.327953894097114,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1_2(xp_as_array(0.18, xp=xp), 10, False, False),
            0.312012852877809,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1_2(xp_as_array(1.0, xp=xp)),
            0.618775480298287,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        samples = np.linspace(0, 1, 10000)

        xp_assert_close(
            log_encoding_CanonLog_v1(xp_as_array(samples, xp=xp)),
            as_ndarray(log_encoding_CanonLog_v1_2(xp_as_array(samples, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1(xp_as_array(samples, xp=xp)),
            as_ndarray(log_encoding_CanonLog_v1_2(xp_as_array(samples, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog_v1(
                xp_as_array(samples, xp=xp), out_normalised_code_value=False
            ),
            as_ndarray(
                log_encoding_CanonLog_v1_2(
                    xp_as_array(samples, xp=xp), out_normalised_code_value=False
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            log_encoding_CanonLog_v1(xp_as_array(samples, xp=xp), in_reflection=False),
            as_ndarray(
                log_encoding_CanonLog_v1_2(
                    xp_as_array(samples, xp=xp), in_reflection=False
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_CanonLog_v1_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1_2` definition n-dimensional arrays support.
        """

        x = 0.18
        clog = as_ndarray(log_encoding_CanonLog_v1_2(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        clog = xp.tile(xp_as_array(clog, xp=xp), (6,))
        xp_assert_close(
            log_encoding_CanonLog_v1_2(x),
            clog,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        clog = xp_reshape(xp_as_array(clog, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_encoding_CanonLog_v1_2(x),
            clog,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        clog = xp_reshape(xp_as_array(clog, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_encoding_CanonLog_v1_2(x),
            clog,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_CanonLog_v1_2(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1_2` definition domain and range scale support.
        """

        x = 0.18
        clog = as_ndarray(log_encoding_CanonLog_v1_2(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_CanonLog_v1_2(xp_as_array(x * factor, xp=xp)),
                    clog * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog_v1_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1_2` definition nan support.
        """

        log_encoding_CanonLog_v1_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_CanonLog_v1_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1_2` definition unit tests methods.
    """

    def test_log_decoding_CanonLog_v1_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1_2` definition.
        """

        xp_assert_close(
            log_decoding_CanonLog_v1_2(xp_as_array(-0.023560121389098, xp=xp)),
            -0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1_2(xp_as_array(0.125122480000000, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1_2(xp_as_array(0.343389649295280, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1_2(xp_as_array(0.343389649295281, xp=xp), 12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1_2(
                xp_as_array(0.327953894097114, xp=xp), 10, False
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1_2(
                xp_as_array(0.312012852877809, xp=xp), 10, False, False
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1_2(xp_as_array(0.618775480298287, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        samples = np.linspace(0, 1, 10000)

        xp_assert_close(
            log_decoding_CanonLog_v1(xp_as_array(samples, xp=xp)),
            as_ndarray(log_decoding_CanonLog_v1_2(xp_as_array(samples, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1(xp_as_array(samples, xp=xp)),
            as_ndarray(log_decoding_CanonLog_v1_2(xp_as_array(samples, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1(
                xp_as_array(samples, xp=xp), in_normalised_code_value=False
            ),
            as_ndarray(
                log_decoding_CanonLog_v1_2(
                    xp_as_array(samples, xp=xp), in_normalised_code_value=False
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            log_decoding_CanonLog_v1(xp_as_array(samples, xp=xp), out_reflection=False),
            as_ndarray(
                log_decoding_CanonLog_v1_2(
                    xp_as_array(samples, xp=xp), out_reflection=False
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_CanonLog_v1_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1_2` definition n-dimensional arrays support.
        """

        clog = 0.343389649295280
        x = as_ndarray(log_decoding_CanonLog_v1_2(xp_as_array(clog, xp=xp)))

        clog = xp.tile(xp_as_array(clog, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(
            log_decoding_CanonLog_v1_2(clog),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog = xp_reshape(xp_as_array(clog, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_decoding_CanonLog_v1_2(clog),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog = xp_reshape(xp_as_array(clog, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_decoding_CanonLog_v1_2(clog),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_CanonLog_v1_2(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1_2` definition domain and range scale support.
        """

        clog = 0.343389649295280
        x = as_ndarray(log_decoding_CanonLog_v1_2(xp_as_array(clog, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_CanonLog_v1_2(xp_as_array(clog * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog_v1_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1_2` definition nan support.
        """

        log_decoding_CanonLog_v1_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_CanonLog2_v1:
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1` definition unit tests methods.
    """

    def test_log_encoding_CanonLog2_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1` definition.
        """

        xp_assert_close(
            log_encoding_CanonLog2_v1(xp_as_array(-0.1, xp=xp)),
            -0.155370131996824,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1(xp_as_array(0.0, xp=xp)),
            0.092864125247312,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1(xp_as_array(0.18, xp=xp)),
            0.398254694983167,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1(xp_as_array(0.18, xp=xp), 12),
            0.397962933301861,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1(xp_as_array(0.18, xp=xp), 10, False),
            0.392025745397009,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1(xp_as_array(0.18, xp=xp), 10, False, False),
            0.379864582222983,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1(xp_as_array(1.0, xp=xp)),
            0.573229282897641,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_CanonLog2_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1` definition n-dimensional arrays support.
        """

        x = 0.18
        clog2 = as_ndarray(log_encoding_CanonLog2_v1(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        clog2 = xp.tile(xp_as_array(clog2, xp=xp), (6,))
        xp_assert_close(
            log_encoding_CanonLog2_v1(x),
            clog2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        clog2 = xp_reshape(xp_as_array(clog2, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_encoding_CanonLog2_v1(x),
            clog2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        clog2 = xp_reshape(xp_as_array(clog2, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_encoding_CanonLog2_v1(x),
            clog2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_CanonLog2_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1` definition domain and range scale support.
        """

        x = 0.18
        clog2 = as_ndarray(log_encoding_CanonLog2_v1(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_CanonLog2_v1(xp_as_array(x * factor, xp=xp)),
                    clog2 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog2_v1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1` definition nan support.
        """

        log_encoding_CanonLog2_v1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_CanonLog2_v1:
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1` definition unit tests methods.
    """

    def test_log_decoding_CanonLog2_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1` definition.
        """

        xp_assert_close(
            log_decoding_CanonLog2_v1(xp_as_array(-0.155370131996824, xp=xp)),
            -0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog2_v1(xp_as_array(0.092864125247312, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog2_v1(xp_as_array(0.398254694983167, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog2_v1(xp_as_array(0.397962933301861, xp=xp), 12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog2_v1(xp_as_array(0.392025745397009, xp=xp), 10, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog2_v1(
                xp_as_array(0.379864582222983, xp=xp), 10, False, False
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog2_v1(xp_as_array(0.573229282897641, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_CanonLog2_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1` definition n-dimensional arrays support.
        """

        clog2 = 0.398254694983167
        x = as_ndarray(log_decoding_CanonLog2_v1(xp_as_array(clog2, xp=xp)))

        clog2 = xp.tile(xp_as_array(clog2, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(
            log_decoding_CanonLog2_v1(clog2),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog2 = xp_reshape(xp_as_array(clog2, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_decoding_CanonLog2_v1(clog2),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog2 = xp_reshape(xp_as_array(clog2, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_decoding_CanonLog2_v1(clog2),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_CanonLog2_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1` definition domain and range scale support.
        """

        clog = 0.398254694983167
        x = as_ndarray(log_decoding_CanonLog2_v1(xp_as_array(clog, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_CanonLog2_v1(xp_as_array(clog * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog2_v1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1` definition nan support.
        """

        log_decoding_CanonLog2_v1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_CanonLog2_v1_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1_2` definition unit tests methods.
    """

    def test_log_encoding_CanonLog2_v1_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1_2` definition.
        """

        xp_assert_close(
            log_encoding_CanonLog2_v1_2(xp_as_array(-0.1, xp=xp)),
            -0.155370130476722,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1_2(xp_as_array(0.0, xp=xp)),
            0.092864125000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1_2(xp_as_array(0.18, xp=xp)),
            0.398254692561492,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1_2(xp_as_array(0.18, xp=xp), 12),
            0.398254692561492,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1_2(xp_as_array(0.18, xp=xp), 10, False),
            0.392025742568957,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1_2(xp_as_array(0.18, xp=xp), 10, False, False),
            0.379864579481518,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1_2(xp_as_array(1.0, xp=xp)),
            0.573229279230156,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        samples = np.linspace(0, 1, 10000)

        xp_assert_close(
            log_encoding_CanonLog2_v1(xp_as_array(samples, xp=xp)),
            as_ndarray(log_encoding_CanonLog2_v1_2(xp_as_array(samples, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1(xp_as_array(samples, xp=xp)),
            as_ndarray(log_encoding_CanonLog2_v1_2(xp_as_array(samples, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog2_v1(
                xp_as_array(samples, xp=xp), out_normalised_code_value=False
            ),
            as_ndarray(
                log_encoding_CanonLog2_v1_2(
                    xp_as_array(samples, xp=xp), out_normalised_code_value=False
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            log_encoding_CanonLog2_v1(xp_as_array(samples, xp=xp), in_reflection=False),
            as_ndarray(
                log_encoding_CanonLog2_v1_2(
                    xp_as_array(samples, xp=xp), in_reflection=False
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_CanonLog2_v1_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1_2` definition n-dimensional arrays support.
        """

        x = 0.18
        clog2 = as_ndarray(log_encoding_CanonLog2_v1_2(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        clog2 = xp.tile(xp_as_array(clog2, xp=xp), (6,))
        xp_assert_close(
            log_encoding_CanonLog2_v1_2(x),
            clog2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        clog2 = xp_reshape(xp_as_array(clog2, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_encoding_CanonLog2_v1_2(x),
            clog2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        clog2 = xp_reshape(xp_as_array(clog2, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_encoding_CanonLog2_v1_2(x),
            clog2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_CanonLog2_v1_2(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1_2` definition domain and range scale support.
        """

        x = 0.18
        clog2 = as_ndarray(log_encoding_CanonLog2_v1_2(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_CanonLog2_v1_2(xp_as_array(x * factor, xp=xp)),
                    clog2 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog2_v1_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1_2` definition nan support.
        """

        log_encoding_CanonLog2_v1_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_CanonLog2_v1_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1_2` definition unit tests methods.
    """

    def test_log_decoding_CanonLog2_v1_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1_2` definition.
        """

        xp_assert_close(
            log_decoding_CanonLog2_v1_2(xp_as_array(-0.155370130476722, xp=xp)),
            -0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog2_v1_2(xp_as_array(0.092864125000000, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog2_v1_2(xp_as_array(0.398254692561492, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog2_v1_2(xp_as_array(0.398254692561492, xp=xp), 12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog2_v1_2(
                xp_as_array(0.392025742568957, xp=xp), 10, False
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog2_v1_2(
                xp_as_array(0.379864579481518, xp=xp), 10, False, False
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog2_v1_2(xp_as_array(0.573229279230156, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        samples = np.linspace(0, 1, 10000)

        xp_assert_close(
            log_decoding_CanonLog_v1(xp_as_array(samples, xp=xp)),
            as_ndarray(log_decoding_CanonLog_v1_2(xp_as_array(samples, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1(xp_as_array(samples, xp=xp)),
            as_ndarray(log_decoding_CanonLog_v1_2(xp_as_array(samples, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog_v1(
                xp_as_array(samples, xp=xp), in_normalised_code_value=False
            ),
            as_ndarray(
                log_decoding_CanonLog_v1_2(
                    xp_as_array(samples, xp=xp), in_normalised_code_value=False
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            log_decoding_CanonLog_v1(xp_as_array(samples, xp=xp), out_reflection=False),
            as_ndarray(
                log_decoding_CanonLog_v1_2(
                    xp_as_array(samples, xp=xp), out_reflection=False
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_CanonLog2_v1_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1_2` definition n-dimensional arrays support.
        """

        clog2 = 0.398254692561492
        x = as_ndarray(log_decoding_CanonLog2_v1_2(xp_as_array(clog2, xp=xp)))

        clog2 = xp.tile(xp_as_array(clog2, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(
            log_decoding_CanonLog2_v1_2(clog2),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog2 = xp_reshape(xp_as_array(clog2, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_decoding_CanonLog2_v1_2(clog2),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog2 = xp_reshape(xp_as_array(clog2, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_decoding_CanonLog2_v1_2(clog2),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_CanonLog2_v1_2(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1_2` definition domain and range scale support.
        """

        clog = 0.398254692561492
        x = as_ndarray(log_decoding_CanonLog2_v1_2(xp_as_array(clog, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_CanonLog2_v1_2(xp_as_array(clog * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog2_v1_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1_2` definition nan support.
        """

        log_decoding_CanonLog2_v1_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_CanonLog3_v1:
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1` definition unit tests methods.
    """

    def test_log_encoding_CanonLog3_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1` definition.
        """

        xp_assert_close(
            log_encoding_CanonLog3_v1(xp_as_array(-0.1, xp=xp)),
            -0.028494506076432,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1(xp_as_array(0.0, xp=xp)),
            0.125122189869013,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1(xp_as_array(0.18, xp=xp)),
            0.343389369388687,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1(xp_as_array(0.18, xp=xp), 12),
            0.343137802085105,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1(xp_as_array(0.18, xp=xp), 10, False),
            0.327953567219893,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1(xp_as_array(0.18, xp=xp), 10, False, False),
            0.313436005886328,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1(xp_as_array(1.0, xp=xp)),
            0.580277796238604,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_CanonLog3_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1` definition n-dimensional arrays support.
        """

        x = 0.18
        clog3 = as_ndarray(log_encoding_CanonLog3_v1(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        clog3 = xp.tile(xp_as_array(clog3, xp=xp), (6,))
        xp_assert_close(
            log_encoding_CanonLog3_v1(x),
            clog3,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        clog3 = xp_reshape(xp_as_array(clog3, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_encoding_CanonLog3_v1(x),
            clog3,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        clog3 = xp_reshape(xp_as_array(clog3, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_encoding_CanonLog3_v1(x),
            clog3,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_CanonLog3_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1` definition domain and range scale support.
        """

        x = 0.18
        clog3 = as_ndarray(log_encoding_CanonLog3_v1(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_CanonLog3_v1(xp_as_array(x * factor, xp=xp)),
                    clog3 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog3_v1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1` definition nan support.
        """

        log_encoding_CanonLog3_v1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_CanonLog3_v1:
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1` definition unit tests methods.
    """

    def test_log_decoding_CanonLog3_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1` definition.
        """

        xp_assert_close(
            log_decoding_CanonLog3_v1(xp_as_array(-0.028494506076432, xp=xp)),
            -0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1(xp_as_array(0.125122189869013, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1(xp_as_array(0.343389369388687, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1(xp_as_array(0.343137802085105, xp=xp), 12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1(xp_as_array(0.327953567219893, xp=xp), 10, False),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1(
                xp_as_array(0.313436005886328, xp=xp), 10, False, False
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1(xp_as_array(0.580277796238604, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_CanonLog3_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1` definition n-dimensional arrays support.
        """

        clog3 = 0.343389369388687
        x = as_ndarray(log_decoding_CanonLog3_v1(xp_as_array(clog3, xp=xp)))

        clog3 = xp.tile(xp_as_array(clog3, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(
            log_decoding_CanonLog3_v1(clog3),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog3 = xp_reshape(xp_as_array(clog3, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_decoding_CanonLog3_v1(clog3),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog3 = xp_reshape(xp_as_array(clog3, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_decoding_CanonLog3_v1(clog3),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_CanonLog3_v1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1` definition domain and range scale support.
        """

        clog = 0.343389369388687
        x = as_ndarray(log_decoding_CanonLog3_v1(xp_as_array(clog, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_CanonLog3_v1(xp_as_array(clog * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog3_v1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1` definition nan support.
        """

        log_decoding_CanonLog3_v1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogEncoding_CanonLog3_v1_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1_2` definition unit tests methods.
    """

    def test_log_encoding_CanonLog3_v1_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1_2` definition.
        """

        xp_assert_close(
            log_encoding_CanonLog3_v1_2(xp_as_array(-0.1, xp=xp)),
            -0.028494507620494,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1_2(xp_as_array(0.0, xp=xp)),
            0.125122189999999,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1_2(xp_as_array(0.18, xp=xp)),
            0.343389370373936,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1_2(xp_as_array(0.18, xp=xp), 12),
            0.343389370373936,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1_2(xp_as_array(0.18, xp=xp), 10, False),
            0.327953568370475,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1_2(xp_as_array(0.18, xp=xp), 10, False, False),
            0.313436007221221,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1_2(xp_as_array(1.0, xp=xp)),
            0.580277794216371,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        samples = np.linspace(0, 1, 10000)

        xp_assert_close(
            log_encoding_CanonLog3_v1(xp_as_array(samples, xp=xp)),
            as_ndarray(log_encoding_CanonLog3_v1_2(xp_as_array(samples, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1(xp_as_array(samples, xp=xp)),
            as_ndarray(log_encoding_CanonLog3_v1_2(xp_as_array(samples, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_CanonLog3_v1(
                xp_as_array(samples, xp=xp), out_normalised_code_value=False
            ),
            as_ndarray(
                log_encoding_CanonLog3_v1_2(
                    xp_as_array(samples, xp=xp), out_normalised_code_value=False
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            log_encoding_CanonLog3_v1(xp_as_array(samples, xp=xp), in_reflection=False),
            as_ndarray(
                log_encoding_CanonLog3_v1_2(
                    xp_as_array(samples, xp=xp), in_reflection=False
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_CanonLog3_v1_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1_2` definition n-dimensional arrays support.
        """

        x = 0.18
        clog3 = as_ndarray(log_encoding_CanonLog3_v1_2(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        clog3 = xp.tile(xp_as_array(clog3, xp=xp), (6,))
        xp_assert_close(
            log_encoding_CanonLog3_v1_2(x),
            clog3,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        clog3 = xp_reshape(xp_as_array(clog3, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_encoding_CanonLog3_v1_2(x),
            clog3,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        clog3 = xp_reshape(xp_as_array(clog3, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_encoding_CanonLog3_v1_2(x),
            clog3,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_CanonLog3_v1_2(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1_2` definition domain and range scale support.
        """

        x = 0.18
        clog3 = as_ndarray(log_encoding_CanonLog3_v1_2(xp_as_array(x, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_encoding_CanonLog3_v1_2(xp_as_array(x * factor, xp=xp)),
                    clog3 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog3_v1_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1_2` definition nan support.
        """

        log_encoding_CanonLog3_v1_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_CanonLog3_v1_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1_2` definition unit tests methods.
    """

    def test_log_decoding_CanonLog3_v1_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1_2` definition.
        """

        xp_assert_close(
            log_decoding_CanonLog3_v1_2(xp_as_array(-0.028494507620494, xp=xp)),
            -0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1_2(xp_as_array(0.125122189999999, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1_2(xp_as_array(0.343389370373936, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1_2(xp_as_array(0.343389370373936, xp=xp), 12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1_2(
                xp_as_array(0.327953568370475, xp=xp), 10, False
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1_2(
                xp_as_array(0.313436007221221, xp=xp), 10, False, False
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1_2(xp_as_array(0.580277794216371, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        samples = np.linspace(0, 1, 10000)

        xp_assert_close(
            log_decoding_CanonLog3_v1(xp_as_array(samples, xp=xp)),
            as_ndarray(log_decoding_CanonLog3_v1_2(xp_as_array(samples, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1(xp_as_array(samples, xp=xp)),
            as_ndarray(log_decoding_CanonLog3_v1_2(xp_as_array(samples, xp=xp))),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_CanonLog3_v1(
                xp_as_array(samples, xp=xp), in_normalised_code_value=False
            ),
            as_ndarray(
                log_decoding_CanonLog3_v1_2(
                    xp_as_array(samples, xp=xp), in_normalised_code_value=False
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            log_decoding_CanonLog3_v1(
                xp_as_array(samples, xp=xp), out_reflection=False
            ),
            as_ndarray(
                log_decoding_CanonLog3_v1_2(
                    xp_as_array(samples, xp=xp), out_reflection=False
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_CanonLog3_v1_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1_2` definition n-dimensional arrays support.
        """

        clog3 = 0.343389370373936
        x = as_ndarray(log_decoding_CanonLog3_v1_2(xp_as_array(clog3, xp=xp)))

        clog3 = xp.tile(xp_as_array(clog3, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(
            log_decoding_CanonLog3_v1_2(clog3),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog3 = xp_reshape(xp_as_array(clog3, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            log_decoding_CanonLog3_v1_2(clog3),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog3 = xp_reshape(xp_as_array(clog3, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            log_decoding_CanonLog3_v1_2(clog3),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_CanonLog3_v1_2(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1_2` definition domain and range scale support.
        """

        clog = 0.343389370373936
        x = as_ndarray(log_decoding_CanonLog3_v1_2(xp_as_array(clog, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    log_decoding_CanonLog3_v1_2(xp_as_array(clog * factor, xp=xp)),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog3_v1_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1_2` definition nan support.
        """

        log_decoding_CanonLog3_v1_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
