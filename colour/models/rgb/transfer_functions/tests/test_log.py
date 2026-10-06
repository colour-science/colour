"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.log` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_Log2,
    log_encoding_Log2,
    logarithmic_function_basic,
    logarithmic_function_camera,
    logarithmic_function_quasilog,
)
from colour.utilities import (
    as_ndarray,
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
    "TestLogarithmFunction_Basic",
    "TestLogarithmFunction_Quasilog",
    "TestLogarithmFunction_Camera",
    "TestLogEncoding_Log2",
    "TestLogDecoding_Log2",
]


class TestLogarithmFunction_Basic:
    """
    Define :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_basic` definition unit tests methods.
    """

    def test_logarithmic_function_basic(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_basic` definition.
        """

        xp_assert_close(
            logarithmic_function_basic(xp_as_array(0.18, xp=xp)),
            -2.473931188332412,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_basic(
                xp_as_array(-2.473931188332412, xp=xp), "antiLog2"
            ),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_basic(xp_as_array(0.18, xp=xp), "log10"),
            -0.744727494896694,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_basic(
                xp_as_array(-0.744727494896694, xp=xp), "antiLog10"
            ),
            0.179999999999999,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_basic(xp_as_array(0.18, xp=xp), "logB", 3),
            -1.560876795007312,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_basic(
                xp_as_array(-1.560876795007312, xp=xp), "antiLogB", 3
            ),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_logarithmic_function_basic(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_basic` definition n-dimensional arrays support.
        """

        styles = ["log10", "antiLog10", "log2", "antiLog2", "logB", "antiLogB"]

        for style in styles:
            a = 0.18
            a_p = as_ndarray(logarithmic_function_basic(xp_as_array(a, xp=xp), style))

            a = xp.tile(xp_as_array(a, xp=xp), (6,))
            a_p = xp.tile(xp_as_array(a_p, xp=xp), (6,))
            xp_assert_close(
                logarithmic_function_basic(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            a = xp_reshape(xp_as_array(a, xp=xp), (2, 3), xp=xp)
            a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3), xp=xp)
            xp_assert_close(
                logarithmic_function_basic(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            a = xp_reshape(xp_as_array(a, xp=xp), (2, 3, 1), xp=xp)
            a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3, 1), xp=xp)
            xp_assert_close(
                logarithmic_function_basic(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    @ignore_numpy_errors
    def test_nan_logarithmic_function_basic(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_basic` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        styles = ["log10", "antiLog10", "log2", "antiLog2", "logB", "antiLogB"]
        for style in styles:
            logarithmic_function_basic(cases, style)


class TestLogarithmFunction_Quasilog:
    """
    Define :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_quasilog` definition unit tests methods.
    """

    def test_logarithmic_function_quasilog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_quasilog` definition.
        """

        xp_assert_close(
            logarithmic_function_quasilog(xp_as_array(0.18, xp=xp)),
            -2.473931188332412,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_quasilog(
                xp_as_array(-2.473931188332412, xp=xp), "logToLin"
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_quasilog(xp_as_array(0.18, xp=xp), "linToLog", 10),
            -0.744727494896694,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_quasilog(
                xp_as_array(-0.744727494896694, xp=xp), "logToLin", 10
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_quasilog(
                xp_as_array(0.18, xp=xp), "linToLog", 10, 0.75
            ),
            -0.558545621172520,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_quasilog(
                xp_as_array(-0.558545621172520, xp=xp), "logToLin", 10, 0.75
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_quasilog(
                xp_as_array(0.18, xp=xp), "linToLog", 10, 0.75, 0.75
            ),
            -0.652249673628745,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_quasilog(
                -0.652249673628745, "logToLin", 10, 0.75, 0.75
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_quasilog(
                xp_as_array(0.18, xp=xp), "linToLog", 10, 0.75, 0.75, 0.001
            ),
            -0.651249673628745,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_quasilog(
                -0.651249673628745, "logToLin", 10, 0.75, 0.75, 0.001
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_quasilog(
                0.18, "linToLog", 10, 0.75, 0.75, 0.001, 0.01
            ),
            -0.627973998323769,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_quasilog(
                -0.627973998323769, "logToLin", 10, 0.75, 0.75, 0.001, 0.01
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_logarithmic_function_quasilog(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_quasilog` definition n-dimensional arrays support.
        """

        styles = ["lintolog", "logtolin"]

        for style in styles:
            a = 0.18
            a_p = as_ndarray(
                logarithmic_function_quasilog(xp_as_array(a, xp=xp), style)
            )

            a = xp.tile(xp_as_array(a, xp=xp), (6,))
            a_p = xp.tile(xp_as_array(a_p, xp=xp), (6,))
            xp_assert_close(
                logarithmic_function_quasilog(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            a = xp_reshape(xp_as_array(a, xp=xp), (2, 3), xp=xp)
            a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3), xp=xp)
            xp_assert_close(
                logarithmic_function_quasilog(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            a = xp_reshape(xp_as_array(a, xp=xp), (2, 3, 1), xp=xp)
            a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3, 1), xp=xp)
            xp_assert_close(
                logarithmic_function_quasilog(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    @ignore_numpy_errors
    def test_nan_logarithmic_function_quasilog(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_quasilog` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        styles = ["lintolog", "logtolin"]
        for style in styles:
            logarithmic_function_quasilog(cases, style)


class TestLogarithmFunction_Camera:
    """
    Define :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_camera` definition unit tests methods.
    """

    def test_logarithmic_function_camera(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_camera` definition.
        """

        xp_assert_close(
            logarithmic_function_camera(xp_as_array(0, xp=xp), "cameraLinToLog"),
            -9.08655123066369,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                xp_as_array(-9.08655123066369, xp=xp), "cameraLogToLin"
            ),
            0.000000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(xp_as_array(0.18, xp=xp), "cameraLinToLog"),
            -2.473931188332412,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                xp_as_array(-2.473931188332412, xp=xp), "cameraLogToLin"
            ),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(xp_as_array(1, xp=xp), "cameraLinToLog"),
            0.000000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(xp_as_array(0, xp=xp), "cameraLogToLin"),
            1.000000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(xp_as_array(0.18, xp=xp), "cameraLinToLog", 10),
            -0.744727494896693,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                xp_as_array(-0.744727494896693, xp=xp), "cameraLogToLin", 10
            ),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                xp_as_array(0.18, xp=xp), "cameraLinToLog", 10, 0.25
            ),
            -0.186181873724173,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                xp_as_array(-0.186181873724173, xp=xp), "cameraLogToLin", 10, 0.25
            ),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                xp_as_array(0.18, xp=xp), "cameraLinToLog", 10, 0.25, 0.95
            ),
            -0.191750972401961,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                -0.191750972401961, "cameraLogToLin", 10, 0.25, 0.95
            ),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                xp_as_array(0.18, xp=xp), "cameraLinToLog", 10, 0.25, 0.95, 0.6
            ),
            0.408249027598038,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                0.408249027598038, "cameraLogToLin", 10, 0.25, 0.95, 0.6
            ),
            0.179999999999999,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                0.18, "cameraLinToLog", 10, 0.25, 0.95, 0.6, 0.01
            ),
            0.414419643717296,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                0.414419643717296, "cameraLogToLin", 10, 0.25, 0.95, 0.6, 0.01
            ),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                0.005, "cameraLinToLog", 10, 0.25, 0.95, 0.6, 0.01, 0.01
            ),
            0.146061232468316,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                0.146061232468316,
                "cameraLogToLin",
                10,
                0.25,
                0.95,
                0.6,
                0.01,
                0.01,
            ),
            0.005000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                0.005, "cameraLinToLog", 10, 0.25, 0.95, 0.6, 0.01, 0.01, 6
            ),
            0.142508652840630,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            logarithmic_function_camera(
                0.142508652840630,
                "cameraLogToLin",
                10,
                0.25,
                0.95,
                0.6,
                0.01,
                0.01,
                6,
            ),
            0.005000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_logarithmic_function_camera(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_camera` definition n-dimensional arrays support.
        """

        styles = ["cameraLinToLog", "cameraLogToLin"]

        for style in styles:
            a = 0.18
            a_p = as_ndarray(logarithmic_function_camera(xp_as_array(a, xp=xp), style))

            a = xp.tile(xp_as_array(a, xp=xp), (6,))
            a_p = xp.tile(xp_as_array(a_p, xp=xp), (6,))
            xp_assert_close(
                logarithmic_function_camera(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            a = xp_reshape(xp_as_array(a, xp=xp), (2, 3), xp=xp)
            a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3), xp=xp)
            xp_assert_close(
                logarithmic_function_camera(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            a = xp_reshape(xp_as_array(a, xp=xp), (2, 3, 1), xp=xp)
            a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3, 1), xp=xp)
            xp_assert_close(
                logarithmic_function_camera(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    @ignore_numpy_errors
    def test_nan_logarithmic_function_camera(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_camera` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        styles = ["cameraLinToLog", "cameraLogToLin"]
        for style in styles:
            logarithmic_function_camera(cases, style)


class TestLogEncoding_Log2:
    """
    Define :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition unit tests methods.
    """

    def test_log_encoding_Log2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition.
        """

        xp_assert_close(
            log_encoding_Log2(xp_as_array(0.0, xp=xp)),
            -np.inf,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log2(xp_as_array(0.18, xp=xp)),
            0.5,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log2(xp_as_array(1.0, xp=xp)),
            0.690302399102493,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log2(xp_as_array(0.18, xp=xp), 0.12),
            0.544997115440089,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log2(xp_as_array(0.18, xp=xp), 0.12, 2**-10),
            0.089857490719529,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_encoding_Log2(xp_as_array(0.18, xp=xp), 0.12, 2**-10, 2**10),
            0.000570299311674,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_Log2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition n-dimensional arrays support.
        """

        x = 0.18
        y = as_ndarray(log_encoding_Log2(xp_as_array(x, xp=xp)))

        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        xp_assert_close(log_encoding_Log2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_encoding_Log2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_encoding_Log2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_log_encoding_Log2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition nan support.
        """

        log_encoding_Log2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Log2:
    """
    Define :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition unit tests methods.
    """

    def test_log_decoding_Log2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition.
        """

        xp_assert_close(
            log_decoding_Log2(xp_as_array(0.0, xp=xp)),
            0.001988737822087,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log2(xp_as_array(0.5, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log2(xp_as_array(0.690302399102493, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log2(xp_as_array(0.544997115440089, xp=xp), 0.12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log2(xp_as_array(0.089857490719529, xp=xp), 0.12, 2**-10),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            log_decoding_Log2(
                xp_as_array(0.000570299311674, xp=xp), 0.12, 2**-10, 2**10
            ),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_Log2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition n-dimensional arrays support.
        """

        y = 0.5
        x = as_ndarray(log_decoding_Log2(xp_as_array(y, xp=xp)))

        y = xp.tile(xp_as_array(y, xp=xp), (6,))
        x = xp.tile(xp_as_array(x, xp=xp), (6,))
        xp_assert_close(log_decoding_Log2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3), xp=xp)
        xp_assert_close(log_decoding_Log2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

        y = xp_reshape(xp_as_array(y, xp=xp), (2, 3, 1), xp=xp)
        x = xp_reshape(xp_as_array(x, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(log_decoding_Log2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_log_decoding_Log2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition nan support.
        """

        log_decoding_Log2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
