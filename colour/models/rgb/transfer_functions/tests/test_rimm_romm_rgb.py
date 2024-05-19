"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.rimm_romm_rgb` module.
"""


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
from colour.utilities import domain_range_scale, ignore_numpy_errors

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

    def test_cctf_encoding_ROMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_ROMMRGB` definition.
        """

        np.testing.assert_allclose(
            cctf_encoding_ROMMRGB(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            cctf_encoding_ROMMRGB(0.18),
            0.385711424751138,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            cctf_encoding_ROMMRGB(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        assert cctf_encoding_ROMMRGB(0.18, out_int=True) == 98

        assert cctf_encoding_ROMMRGB(0.18, bit_depth=12, out_int=True) == 1579

    def test_n_dimensional_cctf_encoding_ROMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_ROMMRGB` definition n-dimensional arrays support.
        """

        X = 0.18
        X_ROMM = cctf_encoding_ROMMRGB(X)

        X = np.tile(X, 6)
        X_ROMM = np.tile(X_ROMM, 6)
        np.testing.assert_allclose(
            cctf_encoding_ROMMRGB(X), X_ROMM, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        X = np.reshape(X, (2, 3))
        X_ROMM = np.reshape(X_ROMM, (2, 3))
        np.testing.assert_allclose(
            cctf_encoding_ROMMRGB(X), X_ROMM, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        X = np.reshape(X, (2, 3, 1))
        X_ROMM = np.reshape(X_ROMM, (2, 3, 1))
        np.testing.assert_allclose(
            cctf_encoding_ROMMRGB(X), X_ROMM, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_cctf_encoding_ROMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_ROMMRGB` definition domain and range scale support.
        """

        X = 0.18
        X_p = cctf_encoding_ROMMRGB(X)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    cctf_encoding_ROMMRGB(X * factor),
                    X_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_cctf_encoding_ROMMRGB(self):
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

    def test_cctf_decoding_ROMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_ROMMRGB` definition.
        """

        np.testing.assert_allclose(
            cctf_decoding_ROMMRGB(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            cctf_decoding_ROMMRGB(0.385711424751138),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            cctf_decoding_ROMMRGB(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            cctf_decoding_ROMMRGB(98, in_int=True),
            0.18,
            atol=0.01,
        )

        np.testing.assert_allclose(
            cctf_decoding_ROMMRGB(1579, bit_depth=12, in_int=True),
            0.18,
            atol=0.001,
        )

    def test_n_dimensional_cctf_decoding_ROMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_ROMMRGB` definition n-dimensional arrays support.
        """

        X_p = 0.385711424751138
        X = cctf_decoding_ROMMRGB(X_p)

        X_p = np.tile(X_p, 6)
        X = np.tile(X, 6)
        np.testing.assert_allclose(
            cctf_decoding_ROMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        X_p = np.reshape(X_p, (2, 3))
        X = np.reshape(X, (2, 3))
        np.testing.assert_allclose(
            cctf_decoding_ROMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        X_p = np.reshape(X_p, (2, 3, 1))
        X = np.reshape(X, (2, 3, 1))
        np.testing.assert_allclose(
            cctf_decoding_ROMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_cctf_decoding_ROMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_ROMMRGB` definition domain and range scale support.
        """

        X_p = 0.385711424751138
        X = cctf_decoding_ROMMRGB(X_p)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    cctf_decoding_ROMMRGB(X_p * factor),
                    X * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_cctf_decoding_ROMMRGB(self):
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

    def test_cctf_encoding_RIMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_RIMMRGB` definition.
        """

        np.testing.assert_allclose(
            cctf_encoding_RIMMRGB(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            cctf_encoding_RIMMRGB(0.18),
            0.291673732475746,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            cctf_encoding_RIMMRGB(1.0),
            0.713125234297525,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert cctf_encoding_RIMMRGB(0.18, out_int=True) == 74

        assert cctf_encoding_RIMMRGB(0.18, bit_depth=12, out_int=True) == 1194

    def test_n_dimensional_cctf_encoding_RIMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_RIMMRGB` definition n-dimensional arrays support.
        """

        X = 0.18
        X_p = cctf_encoding_RIMMRGB(X)

        X = np.tile(X, 6)
        X_p = np.tile(X_p, 6)
        np.testing.assert_allclose(
            cctf_encoding_RIMMRGB(X), X_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        X = np.reshape(X, (2, 3))
        X_p = np.reshape(X_p, (2, 3))
        np.testing.assert_allclose(
            cctf_encoding_RIMMRGB(X), X_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        X = np.reshape(X, (2, 3, 1))
        X_p = np.reshape(X_p, (2, 3, 1))
        np.testing.assert_allclose(
            cctf_encoding_RIMMRGB(X), X_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_cctf_encoding_RIMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_encoding_RIMMRGB` definition domain and range scale support.
        """

        X = 0.18
        X_p = cctf_encoding_RIMMRGB(X)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    cctf_encoding_RIMMRGB(X * factor),
                    X_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_cctf_encoding_RIMMRGB(self):
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

    def test_cctf_decoding_RIMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_RIMMRGB` definition.
        """

        np.testing.assert_allclose(
            cctf_decoding_RIMMRGB(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            cctf_decoding_RIMMRGB(0.291673732475746),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            cctf_decoding_RIMMRGB(0.713125234297525),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            cctf_decoding_RIMMRGB(74, in_int=True),
            0.18,
            atol=0.005,
        )

        np.testing.assert_allclose(
            cctf_decoding_RIMMRGB(1194, bit_depth=12, in_int=True),
            0.18,
            atol=0.005,
        )

    def test_n_dimensional_cctf_decoding_RIMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_RIMMRGB` definition n-dimensional arrays support.
        """

        X_p = 0.291673732475746
        X = cctf_decoding_RIMMRGB(X_p)

        X_p = np.tile(X_p, 6)
        X = np.tile(X, 6)
        np.testing.assert_allclose(
            cctf_decoding_RIMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        X_p = np.reshape(X_p, (2, 3))
        X = np.reshape(X, (2, 3))
        np.testing.assert_allclose(
            cctf_decoding_RIMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        X_p = np.reshape(X_p, (2, 3, 1))
        X = np.reshape(X, (2, 3, 1))
        np.testing.assert_allclose(
            cctf_decoding_RIMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_cctf_decoding_RIMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
cctf_decoding_RIMMRGB` definition domain and range scale support.
        """

        X_p = 0.291673732475746
        X = cctf_decoding_RIMMRGB(X_p)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    cctf_decoding_RIMMRGB(X_p * factor),
                    X * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_cctf_decoding_RIMMRGB(self):
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

    def test_log_encoding_ERIMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition.
        """

        np.testing.assert_allclose(
            log_encoding_ERIMMRGB(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_encoding_ERIMMRGB(0.18),
            0.410052389492129,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_ERIMMRGB(1.0),
            0.545458327405113,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert log_encoding_ERIMMRGB(0.18, out_int=True) == 105

        assert log_encoding_ERIMMRGB(0.18, bit_depth=12, out_int=True) == 1679

    def test_n_dimensional_log_encoding_ERIMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition n-dimensional arrays support.
        """

        X = 0.18
        X_p = log_encoding_ERIMMRGB(X)

        X = np.tile(X, 6)
        X_p = np.tile(X_p, 6)
        np.testing.assert_allclose(
            log_encoding_ERIMMRGB(X), X_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        X = np.reshape(X, (2, 3))
        X_p = np.reshape(X_p, (2, 3))
        np.testing.assert_allclose(
            log_encoding_ERIMMRGB(X), X_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        X = np.reshape(X, (2, 3, 1))
        X_p = np.reshape(X_p, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_ERIMMRGB(X), X_p, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_ERIMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_encoding_ERIMMRGB` definition domain and range scale support.
        """

        X = 0.18
        X_p = log_encoding_ERIMMRGB(X)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_ERIMMRGB(X * factor),
                    X_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_ERIMMRGB(self):
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

    def test_log_decoding_ERIMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_decoding_ERIMMRGB` definition.
        """

        np.testing.assert_allclose(
            log_decoding_ERIMMRGB(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_decoding_ERIMMRGB(0.410052389492129),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_ERIMMRGB(0.545458327405113),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_ERIMMRGB(105, in_int=True),
            0.18,
            atol=0.005,
        )

        np.testing.assert_allclose(
            log_decoding_ERIMMRGB(1679, bit_depth=12, in_int=True),
            0.18,
            atol=0.005,
        )

    def test_n_dimensional_log_decoding_ERIMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_decoding_ERIMMRGB` definition n-dimensional arrays support.
        """

        X_p = 0.410052389492129
        X = log_decoding_ERIMMRGB(X_p)

        X_p = np.tile(X_p, 6)
        X = np.tile(X, 6)
        np.testing.assert_allclose(
            log_decoding_ERIMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        X_p = np.reshape(X_p, (2, 3))
        X = np.reshape(X, (2, 3))
        np.testing.assert_allclose(
            log_decoding_ERIMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        X_p = np.reshape(X_p, (2, 3, 1))
        X = np.reshape(X, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_ERIMMRGB(X_p), X, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_ERIMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_decoding_ERIMMRGB` definition domain and range scale support.
        """

        X_p = 0.410052389492129
        X = log_decoding_ERIMMRGB(X_p)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_ERIMMRGB(X_p * factor),
                    X * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_ERIMMRGB(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.rimm_romm_rgb.\
log_decoding_ERIMMRGB` definition nan support.
        """

        log_decoding_ERIMMRGB(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
