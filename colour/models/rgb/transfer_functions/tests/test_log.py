"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.log` module.
"""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_Log2,
    log_encoding_Log2,
    logarithmic_function_basic,
    logarithmic_function_camera,
    logarithmic_function_quasilog,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

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

    def test_logarithmic_function_basic(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_basic` definition.
        """

        np.testing.assert_allclose(
            logarithmic_function_basic(0.18),
            -2.473931188332412,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_basic(-2.473931188332412, "antiLog2"),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_basic(0.18, "log10"),
            -0.744727494896694,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_basic(-0.744727494896694, "antiLog10"),
            0.179999999999999,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_basic(0.18, "logB", 3),
            -1.560876795007312,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_basic(-1.560876795007312, "antiLogB", 3),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_logarithmic_function_basic(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_basic` definition n-dimensional arrays support.
        """

        styles = ["log10", "antiLog10", "log2", "antiLog2", "logB", "antiLogB"]

        for style in styles:
            a = 0.18
            a_p = logarithmic_function_basic(a, style)

            a = np.tile(a, 6)
            a_p = np.tile(a_p, 6)
            np.testing.assert_allclose(
                logarithmic_function_basic(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            a = np.reshape(a, (2, 3))
            a_p = np.reshape(a_p, (2, 3))
            np.testing.assert_allclose(
                logarithmic_function_basic(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            a = np.reshape(a, (2, 3, 1))
            a_p = np.reshape(a_p, (2, 3, 1))
            np.testing.assert_allclose(
                logarithmic_function_basic(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    @ignore_numpy_errors
    def test_nan_logarithmic_function_basic(self):
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

    def test_logarithmic_function_quasilog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_quasilog` definition.
        """

        np.testing.assert_allclose(
            logarithmic_function_quasilog(0.18),
            -2.473931188332412,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_quasilog(-2.473931188332412, "logToLin"),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_quasilog(0.18, "linToLog", 10),
            -0.744727494896694,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_quasilog(-0.744727494896694, "logToLin", 10),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_quasilog(0.18, "linToLog", 10, 0.75),
            -0.558545621172520,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_quasilog(-0.558545621172520, "logToLin", 10, 0.75),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_quasilog(0.18, "linToLog", 10, 0.75, 0.75),
            -0.652249673628745,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_quasilog(
                -0.652249673628745, "logToLin", 10, 0.75, 0.75
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_quasilog(0.18, "linToLog", 10, 0.75, 0.75, 0.001),
            -0.651249673628745,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_quasilog(
                -0.651249673628745, "logToLin", 10, 0.75, 0.75, 0.001
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_quasilog(
                0.18, "linToLog", 10, 0.75, 0.75, 0.001, 0.01
            ),
            -0.627973998323769,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_quasilog(
                -0.627973998323769, "logToLin", 10, 0.75, 0.75, 0.001, 0.01
            ),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_logarithmic_function_quasilog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_quasilog` definition n-dimensional arrays support.
        """

        styles = ["lintolog", "logtolin"]

        for style in styles:
            a = 0.18
            a_p = logarithmic_function_quasilog(a, style)

            a = np.tile(a, 6)
            a_p = np.tile(a_p, 6)
            np.testing.assert_allclose(
                logarithmic_function_quasilog(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            a = np.reshape(a, (2, 3))
            a_p = np.reshape(a_p, (2, 3))
            np.testing.assert_allclose(
                logarithmic_function_quasilog(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            a = np.reshape(a, (2, 3, 1))
            a_p = np.reshape(a_p, (2, 3, 1))
            np.testing.assert_allclose(
                logarithmic_function_quasilog(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    @ignore_numpy_errors
    def test_nan_logarithmic_function_quasilog(self):
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

    def test_logarithmic_function_camera(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_camera` definition.
        """

        np.testing.assert_allclose(
            logarithmic_function_camera(0, "cameraLinToLog"),
            -9.08655123066369,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(-9.08655123066369, "cameraLogToLin"),
            0.000000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(0.18, "cameraLinToLog"),
            -2.473931188332412,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(-2.473931188332412, "cameraLogToLin"),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(1, "cameraLinToLog"),
            0.000000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(0, "cameraLogToLin"),
            1.000000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(0.18, "cameraLinToLog", 10),
            -0.744727494896693,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(-0.744727494896693, "cameraLogToLin", 10),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(0.18, "cameraLinToLog", 10, 0.25),
            -0.186181873724173,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(-0.186181873724173, "cameraLogToLin", 10, 0.25),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(0.18, "cameraLinToLog", 10, 0.25, 0.95),
            -0.191750972401961,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(
                -0.191750972401961, "cameraLogToLin", 10, 0.25, 0.95
            ),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(0.18, "cameraLinToLog", 10, 0.25, 0.95, 0.6),
            0.408249027598038,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(
                0.408249027598038, "cameraLogToLin", 10, 0.25, 0.95, 0.6
            ),
            0.179999999999999,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(
                0.18, "cameraLinToLog", 10, 0.25, 0.95, 0.6, 0.01
            ),
            0.414419643717296,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(
                0.414419643717296, "cameraLogToLin", 10, 0.25, 0.95, 0.6, 0.01
            ),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            logarithmic_function_camera(
                0.005, "cameraLinToLog", 10, 0.25, 0.95, 0.6, 0.01, 0.01
            ),
            0.146061232468316,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
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

        np.testing.assert_allclose(
            logarithmic_function_camera(
                0.005, "cameraLinToLog", 10, 0.25, 0.95, 0.6, 0.01, 0.01, 6
            ),
            0.142508652840630,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
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

    def test_n_dimensional_logarithmic_function_camera(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
logarithmic_function_camera` definition n-dimensional arrays support.
        """

        styles = ["cameraLinToLog", "cameraLogToLin"]

        for style in styles:
            a = 0.18
            a_p = logarithmic_function_camera(a, style)

            a = np.tile(a, 6)
            a_p = np.tile(a_p, 6)
            np.testing.assert_allclose(
                logarithmic_function_camera(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            a = np.reshape(a, (2, 3))
            a_p = np.reshape(a_p, (2, 3))
            np.testing.assert_allclose(
                logarithmic_function_camera(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

            a = np.reshape(a, (2, 3, 1))
            a_p = np.reshape(a_p, (2, 3, 1))
            np.testing.assert_allclose(
                logarithmic_function_camera(a, style),
                a_p,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    @ignore_numpy_errors
    def test_nan_logarithmic_function_camera(self):
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

    def test_log_encoding_Log2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition.
        """

        np.testing.assert_allclose(
            log_encoding_Log2(0.0), -np.inf, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_encoding_Log2(0.18), 0.5, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_encoding_Log2(1.0),
            0.690302399102493,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_Log2(0.18, 0.12),
            0.544997115440089,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_Log2(0.18, 0.12, 2**-10),
            0.089857490719529,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_Log2(0.18, 0.12, 2**-10, 2**10),
            0.000570299311674,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_Log2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_Log2(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_allclose(
            log_encoding_Log2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_allclose(
            log_encoding_Log2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_Log2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_Log2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
log_encoding_Log2` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_Log2(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_Log2(x * factor),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_Log2(self):
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

    def test_log_decoding_Log2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition.
        """

        np.testing.assert_allclose(
            log_decoding_Log2(0.0),
            0.001988737822087,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Log2(0.5), 0.18, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_decoding_Log2(0.690302399102493),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Log2(0.544997115440089, 0.12),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Log2(0.089857490719529, 0.12, 2**-10),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Log2(0.000570299311674, 0.12, 2**-10, 2**10),
            0.180000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_Log2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition n-dimensional arrays support.
        """

        y = 0.5
        x = log_decoding_Log2(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_Log2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_Log2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_Log2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_Log2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition domain and range scale support.
        """

        y = 0.5
        x = log_decoding_Log2(y)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_Log2(y * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_Log2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.log.\
log_decoding_Log2` definition nan support.
        """

        log_decoding_Log2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
