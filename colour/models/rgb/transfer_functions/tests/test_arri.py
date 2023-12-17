"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.arri` module.
"""

import unittest

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_ARRILogC3,
    log_decoding_ARRILogC4,
    log_encoding_ARRILogC3,
    log_encoding_ARRILogC4,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLogEncoding_ARRILogC3",
    "TestLogDecoding_ARRILogC3",
    "TestLogEncoding_ARRILogC4",
    "TestLogDecoding_ARRILogC4",
]


class TestLogEncoding_ARRILogC3(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.arri.\
log_encoding_ARRILogC3` definition unit tests methods.
    """

    def test_log_encoding_ARRILogC3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_encoding_ARRILogC3` definition.
        """

        np.testing.assert_allclose(
            log_encoding_ARRILogC3(0.0),
            0.092809000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_ARRILogC3(0.18),
            0.391006832034084,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_ARRILogC3(1.0),
            0.570631558120417,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_ARRILogC3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_encoding_ARRILogC3` definition n-dimensional arrays support.
        """

        x = 0.18
        t = log_encoding_ARRILogC3(x)

        x = np.tile(x, 6)
        t = np.tile(t, 6)
        np.testing.assert_allclose(
            log_encoding_ARRILogC3(x), t, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        t = np.reshape(t, (2, 3))
        np.testing.assert_allclose(
            log_encoding_ARRILogC3(x), t, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        t = np.reshape(t, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_ARRILogC3(x), t, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_ARRILogC3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_encoding_ARRILogC3` definition domain and range scale support.
        """

        x = 0.18
        t = log_encoding_ARRILogC3(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_ARRILogC3(x * factor),
                    t * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_ARRILogC3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_encoding_ARRILogC3` definition nan support.
        """

        log_encoding_ARRILogC3(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_ARRILogC3(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.arri.\
log_decoding_ARRILogC3` definition unit tests methods.
    """

    def test_log_decoding_ARRILogC3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_decoding_ARRILogC3` definition.
        """

        np.testing.assert_allclose(
            log_decoding_ARRILogC3(0.092809),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_ARRILogC3(0.391006832034084),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_ARRILogC3(0.570631558120417),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_ARRILogC3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_decoding_ARRILogC3` definition n-dimensional arrays support.
        """

        t = 0.391006832034084
        x = log_decoding_ARRILogC3(t)

        t = np.tile(t, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_ARRILogC3(t), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        t = np.reshape(t, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_ARRILogC3(t), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        t = np.reshape(t, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_ARRILogC3(t), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_ARRILogC3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_decoding_ARRILogC3` definition domain and range scale support.
        """

        t = 0.391006832034084
        x = log_decoding_ARRILogC3(t)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_ARRILogC3(t * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_ARRILogC3(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_decoding_ARRILogC3` definition nan support.
        """

        log_decoding_ARRILogC3(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogEncoding_ARRILogC4(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.arri.\
log_encoding_ARRILogC4` definition unit tests methods.
    """

    def test_log_encoding_ARRILogC4(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_encoding_ARRILogC4` definition.
        """

        np.testing.assert_allclose(
            log_encoding_ARRILogC4(0.0),
            0.092864125122190,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_ARRILogC4(0.18),
            0.278395836548265,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_ARRILogC4(1.0),
            0.427519364835306,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_ARRILogC4(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_encoding_ARRILogC4` definition n-dimensional arrays support.
        """

        x = 0.18
        t = log_encoding_ARRILogC4(x)

        x = np.tile(x, 6)
        t = np.tile(t, 6)
        np.testing.assert_allclose(
            log_encoding_ARRILogC4(x), t, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        t = np.reshape(t, (2, 3))
        np.testing.assert_allclose(
            log_encoding_ARRILogC4(x), t, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        t = np.reshape(t, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_ARRILogC4(x), t, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_ARRILogC4(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_encoding_ARRILogC4` definition domain and range scale support.
        """

        x = 0.18
        t = log_encoding_ARRILogC4(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_ARRILogC4(x * factor),
                    t * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_ARRILogC4(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_encoding_ARRILogC4` definition nan support.
        """

        log_encoding_ARRILogC4(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_ARRILogC4(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.arri.\
log_decoding_ARRILogC4` definition unit tests methods.
    """

    def test_log_decoding_ARRILogC4(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_decoding_ARRILogC4` definition.
        """

        np.testing.assert_allclose(
            log_decoding_ARRILogC4(0.092864125122190),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_ARRILogC4(0.278395836548265),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_ARRILogC4(0.427519364835306),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_ARRILogC4(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_decoding_ARRILogC4` definition n-dimensional arrays support.
        """

        t = 0.278395836548265
        x = log_decoding_ARRILogC4(t)

        t = np.tile(t, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_ARRILogC4(t), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        t = np.reshape(t, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_ARRILogC4(t), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        t = np.reshape(t, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_ARRILogC4(t), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_ARRILogC4(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_decoding_ARRILogC4` definition domain and range scale support.
        """

        t = 0.278395836548265
        x = log_decoding_ARRILogC4(t)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_ARRILogC4(t * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_ARRILogC4(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.arri.\
log_decoding_ARRILogC4` definition nan support.
        """

        log_decoding_ARRILogC4(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


if __name__ == "__main__":
    unittest.main()
