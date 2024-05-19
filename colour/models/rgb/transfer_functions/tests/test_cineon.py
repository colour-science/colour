"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.cineon` module.
"""


import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    log_decoding_Cineon,
    log_encoding_Cineon,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLogEncoding_Cineon",
    "TestLogDecoding_Cineon",
]


class TestLogEncoding_Cineon:
    """
    Define :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition unit tests methods.
    """

    def test_log_encoding_Cineon(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition.
        """

        np.testing.assert_allclose(
            log_encoding_Cineon(0.0),
            0.092864125122190,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_Cineon(0.18),
            0.457319613085418,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_Cineon(1.0),
            0.669599217986315,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_Cineon(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition n-dimensional arrays support.
        """

        x = 0.18
        y = log_encoding_Cineon(x)

        x = np.tile(x, 6)
        y = np.tile(y, 6)
        np.testing.assert_allclose(
            log_encoding_Cineon(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        y = np.reshape(y, (2, 3))
        np.testing.assert_allclose(
            log_encoding_Cineon(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        y = np.reshape(y, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_Cineon(x), y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_Cineon(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition domain and range scale support.
        """

        x = 0.18
        y = log_encoding_Cineon(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_Cineon(x * factor),
                    y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_Cineon(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_encoding_Cineon` definition nan support.
        """

        log_encoding_Cineon(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLogDecoding_Cineon:
    """
    Define :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition unit tests methods.
    """

    def test_log_decoding_Cineon(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition.
        """

        np.testing.assert_allclose(
            log_decoding_Cineon(0.092864125122190),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Cineon(0.457319613085418),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_Cineon(0.669599217986315),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_Cineon(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition n-dimensional arrays support.
        """

        y = 0.457319613085418
        x = log_decoding_Cineon(y)

        y = np.tile(y, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_Cineon(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_Cineon(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        y = np.reshape(y, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_Cineon(y), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_Cineon(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition domain and range scale support.
        """

        y = 0.457319613085418
        x = log_decoding_Cineon(y)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_Cineon(y * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_Cineon(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.cineon.\
log_decoding_Cineon` definition nan support.
        """

        log_decoding_Cineon(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
