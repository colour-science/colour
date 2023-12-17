"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.red` module.
"""

import unittest

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_Log3G12,
    log_decoding_REDLog,
    log_decoding_REDLogFilm,
    log_encoding_Log3G12,
    log_encoding_REDLog,
    log_encoding_REDLogFilm,
)
from colour.models.rgb.transfer_functions.red import (
    log_decoding_Log3G10_v1,
    log_decoding_Log3G10_v2,
    log_decoding_Log3G10_v3,
    log_encoding_Log3G10_v1,
    log_encoding_Log3G10_v2,
    log_encoding_Log3G10_v3,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Development"

__all__ = [
    "TestLogEncoding_REDLog",
    "TestLogDecoding_REDLog",
    "TestLogEncoding_REDLogFilm",
    "TestLogDecoding_REDLogFilm",
    "TestLogEncoding_Log3G10_v1",
    "TestLogDecoding_Log3G10_v1",
    "TestLogEncoding_Log3G10_v2",
    "TestLogDecoding_Log3G10_v2",
    "TestLogEncoding_Log3G10_v3",
    "TestLogDecoding_Log3G10_v3",
    "TestLogEncoding_Log3G12",
    "TestLogDecoding_Log3G12",
]


class TestLogEncoding_REDLog(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLog` definition unit tests methods.
    """

    def test_log_encoding_REDLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLog` definition.
        """

        np.testing.assert_allclose(
            log_encoding_REDLog(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_encoding_REDLog(0.18),
            0.637621845988175,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_REDLog(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_log_encoding_REDLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLog` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_REDLog(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_allclose(
            log_encoding_REDLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_allclose(
            log_encoding_REDLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_REDLog(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_REDLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLog` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_REDLog(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_REDLog(x * factor),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_REDLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLog` definition nan support.
        """

        log_encoding_REDLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_REDLog(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLog` definition unit tests methods.
    """

    def test_log_decoding_REDLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLog` definition.
        """

        np.testing.assert_allclose(
            log_decoding_REDLog(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_decoding_REDLog(0.637621845988175),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_REDLog(1.0), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_log_decoding_REDLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLog` definition n-dimensional arrays support.
        """

        y = 0.637621845988175
        x = log_decoding_REDLog(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_REDLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_REDLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_REDLog(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_REDLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLog` definition domain and range scale support.
        """

        y = 0.637621845988175
        x = log_decoding_REDLog(y)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_REDLog(y * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_REDLog(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLog` definition nan support.
        """

        log_decoding_REDLog(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogEncoding_REDLogFilm(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLogFilm` definition unit tests methods.
    """

    def test_log_encoding_REDLogFilm(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLogFilm` definition.
        """

        np.testing.assert_allclose(
            log_encoding_REDLogFilm(0.0),
            0.092864125122190,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_REDLogFilm(0.18),
            0.457319613085418,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_REDLogFilm(1.0),
            0.669599217986315,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_REDLogFilm(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLogFilm` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_REDLogFilm(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_allclose(
            log_encoding_REDLogFilm(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_allclose(
            log_encoding_REDLogFilm(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_REDLogFilm(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_REDLogFilm(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLogFilm` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_REDLogFilm(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_REDLogFilm(x * factor),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_REDLogFilm(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_REDLogFilm` definition nan support.
        """

        log_encoding_REDLogFilm(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_REDLogFilm(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLogFilm` definition unit tests methods.
    """

    def test_log_decoding_REDLogFilm(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLogFilm` definition.
        """

        np.testing.assert_allclose(
            log_decoding_REDLogFilm(0.092864125122190),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_REDLogFilm(0.457319613085418),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_REDLogFilm(0.669599217986315),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_REDLogFilm(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLogFilm` definition n-dimensional arrays support.
        """

        y = 0.457319613085418
        x = log_decoding_REDLogFilm(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_REDLogFilm(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_REDLogFilm(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_REDLogFilm(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_REDLogFilm(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLogFilm` definition domain and range scale support.
        """

        y = 0.457319613085418
        x = log_decoding_REDLogFilm(y)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_REDLogFilm(y * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_REDLogFilm(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_REDLogFilm` definition nan support.
        """

        log_decoding_REDLogFilm(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogEncoding_Log3G10_v1(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v1` definition unit tests methods.
    """

    def test_log_encoding_Log3G10_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v1` definition.
        """

        np.testing.assert_allclose(
            log_encoding_Log3G10_v1(-1.0),
            -0.496483569056003,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_Log3G10_v1(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_encoding_Log3G10_v1(0.18),
            0.333333644207707,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_Log3G10_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v1` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_Log3G10_v1(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_allclose(
            log_encoding_Log3G10_v1(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_allclose(
            log_encoding_Log3G10_v1(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_Log3G10_v1(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_Log3G10_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v1` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_Log3G10_v1(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_Log3G10_v1(x * factor),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_Log3G10_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v1` definition nan support.
        """

        log_encoding_Log3G10_v1(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_Log3G10_v1(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v1` definition unit tests methods.
    """

    def test_log_decoding_Log3G10_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v1` definition.
        """

        np.testing.assert_allclose(
            log_decoding_Log3G10_v1(-0.496483569056003),
            -1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Log3G10_v1(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_decoding_Log3G10_v1(0.333333644207707),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_Log3G10_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v1` definition n-dimensional arrays support.
        """

        y = 0.333333644207707
        x = log_decoding_Log3G10_v1(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_Log3G10_v1(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_Log3G10_v1(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_Log3G10_v1(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_Log3G10_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v1` definition domain and range scale support.
        """

        y = 0.333333644207707
        x = log_decoding_Log3G10_v1(y)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_Log3G10_v1(y * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_Log3G10_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v1` definition nan support.
        """

        log_decoding_Log3G10_v1(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogEncoding_Log3G10_v2(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v2` definition unit tests methods.
    """

    def test_log_encoding_Log3G10_v2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v2` definition.
        """

        np.testing.assert_allclose(
            log_encoding_Log3G10_v2(-1.0),
            -0.491512777522511,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_Log3G10_v2(0.0),
            0.091551487714745,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_Log3G10_v2(0.18),
            0.333332912025992,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_Log3G10_v2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v2` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_Log3G10_v2(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_allclose(
            log_encoding_Log3G10_v2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_allclose(
            log_encoding_Log3G10_v2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_Log3G10_v2(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_Log3G10_v2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v2` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_Log3G10_v2(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_Log3G10_v2(x * factor),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_Log3G10_v2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v2` definition nan support.
        """

        log_encoding_Log3G10_v2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_Log3G10_v2(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v2` definition unit tests methods.
    """

    def test_log_decoding_Log3G10_v2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v2` definition.
        """

        np.testing.assert_allclose(
            log_decoding_Log3G10_v2(-0.491512777522511),
            -1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Log3G10_v2(0.091551487714745),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Log3G10_v2(0.333332912025992),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_Log3G10_v2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v2` definition n-dimensional arrays support.
        """

        y = 0.333332912025992
        x = log_decoding_Log3G10_v2(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_Log3G10_v2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_Log3G10_v2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_Log3G10_v2(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_Log3G10_v2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v2` definition domain and range scale support.
        """

        y = 0.333333644207707
        x = log_decoding_Log3G10_v2(y)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_Log3G10_v2(y * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_Log3G10_v2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v2` definition nan support.
        """

        log_decoding_Log3G10_v2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogEncoding_Log3G10_v3(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v3` definition unit tests methods.
    """

    def test_log_encoding_Log3G10_v3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v3` definition.
        """

        np.testing.assert_allclose(
            log_encoding_Log3G10_v3(-1.0),
            -15.040773,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_Log3G10_v3(0.0),
            0.091551487714745,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_Log3G10_v3(0.18),
            0.333332912025992,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_Log3G10_v3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v3` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_Log3G10_v3(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_allclose(
            log_encoding_Log3G10_v3(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_allclose(
            log_encoding_Log3G10_v3(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_Log3G10_v3(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_Log3G10_v3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v3` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_Log3G10_v3(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_Log3G10_v3(x * factor),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_Log3G10_v3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G10_v3` definition nan support.
        """

        log_encoding_Log3G10_v3(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_Log3G10_v3(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v3` definition unit tests methods.
    """

    def test_log_decoding_Log3G10_v3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v3` definition.
        """

        np.testing.assert_allclose(
            log_decoding_Log3G10_v3(-15.040773),
            -1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Log3G10_v3(0.091551487714745),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Log3G10_v3(0.333332912025992),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_Log3G10_v3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v3` definition n-dimensional arrays support.
        """

        y = 0.333332912025992
        x = log_decoding_Log3G10_v3(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_Log3G10_v3(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_Log3G10_v3(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_Log3G10_v3(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_Log3G10_v3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v3` definition domain and range scale support.
        """

        y = 0.333333644207707
        x = log_decoding_Log3G10_v3(y)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_Log3G10_v3(y * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_Log3G10_v3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G10_v3` definition nan support.
        """

        log_decoding_Log3G10_v3(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogEncoding_Log3G12(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G12` definition unit tests methods.
    """

    def test_log_encoding_Log3G12(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G12` definition.
        """

        np.testing.assert_allclose(
            log_encoding_Log3G12(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_encoding_Log3G12(0.18),
            0.333332662015923,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_Log3G12(1.0),
            0.469991923234319,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_Log3G12(0.18 * 2**12),
            0.999997986792394,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_Log3G12(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G12` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_Log3G12(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_allclose(
            log_encoding_Log3G12(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_allclose(
            log_encoding_Log3G12(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_Log3G12(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_Log3G12(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G12` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_Log3G12(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_Log3G12(x * factor),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_Log3G12(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_encoding_Log3G12` definition nan support.
        """

        log_encoding_Log3G12(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_Log3G12(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G12` definition unit tests methods.
    """

    def test_log_decoding_Log3G12(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G12` definition.
        """

        np.testing.assert_allclose(
            log_decoding_Log3G12(0.0), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            log_decoding_Log3G12(0.333332662015923),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Log3G12(0.469991923234319),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Log3G12(1.0),
            737.29848406719,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_Log3G12(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G12` definition n-dimensional arrays support.
        """

        y = 0.333332662015923
        x = log_decoding_Log3G12(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_Log3G12(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_Log3G12(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_Log3G12(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_Log3G12(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G12` definition domain and range scale support.
        """

        y = 0.18
        x = log_decoding_Log3G12(y)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_Log3G12(y * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_Log3G12(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.red.\
log_decoding_Log3G12` definition nan support.
        """

        log_decoding_Log3G12(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


if __name__ == "__main__":
    unittest.main()
