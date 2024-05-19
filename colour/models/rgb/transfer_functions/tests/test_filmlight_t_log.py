"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
filmlight_t_log` module.
"""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_FilmLightTLog,
    log_encoding_FilmLightTLog,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

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

    def test_log_encoding_FilmLightTLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_encoding_FilmLightTLog` definition.
        """

        np.testing.assert_allclose(
            log_encoding_FilmLightTLog(0.0),
            0.075,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_FilmLightTLog(0.18),
            0.396567801298332,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_FilmLightTLog(1.0),
            0.552537881005859,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_TLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_encoding_FilmLightTLog` definition n-dimensional arrays support.
        """

        x = 0.18
        t = log_encoding_FilmLightTLog(x)

        x = np.tile(x, 6)
        t = np.tile(t, 6)
        np.testing.assert_allclose(
            log_encoding_FilmLightTLog(x), t, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        t = np.reshape(t, (2, 3))
        np.testing.assert_allclose(
            log_encoding_FilmLightTLog(x), t, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        t = np.reshape(t, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_FilmLightTLog(x), t, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_TLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_encoding_FilmLightTLog` definition domain and range scale support.
        """

        x = 0.18
        t = log_encoding_FilmLightTLog(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_FilmLightTLog(x * factor),
                    t * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_TLog(self):
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

    def test_log_decoding_FilmLightTLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_decoding_FilmLightTLog` definition.
        """

        np.testing.assert_allclose(
            log_decoding_FilmLightTLog(0.075),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_FilmLightTLog(0.396567801298332),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_FilmLightTLog(0.552537881005859),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_TLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_decoding_FilmLightTLog` definition n-dimensional arrays support.
        """

        t = 0.396567801298332
        x = log_decoding_FilmLightTLog(t)

        t = np.tile(t, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_FilmLightTLog(t), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        t = np.reshape(t, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_FilmLightTLog(t), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        t = np.reshape(t, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_FilmLightTLog(t), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_TLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_decoding_FilmLightTLog` definition domain and range scale support.
        """

        t = 0.396567801298332
        x = log_decoding_FilmLightTLog(t)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_FilmLightTLog(t * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_TLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.filmlight_t_log.\
log_decoding_FilmLightTLog` definition nan support.
        """

        log_decoding_FilmLightTLog(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
