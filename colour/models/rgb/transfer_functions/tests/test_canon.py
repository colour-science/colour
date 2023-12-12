"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.canon` module.
"""

import unittest

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
from colour.utilities import domain_range_scale, ignore_numpy_errors

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


class TestLogEncoding_CanonLog_v1(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1` definition unit tests methods.
    """

    def test_log_encoding_CanonLog_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1` definition.
        """

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1(-0.1), -0.023560122781997, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1(0.0), 0.125122480156403, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1(0.18), 0.343389651726069, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1(0.18, 12), 0.343138084215647, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1(0.18, 10, False),
            0.327953896935809,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1(0.18, 10, False, False),
            0.312012855550395,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1(1.0), 0.618775485598649, places=7
        )

    def test_n_dimensional_log_encoding_CanonLog_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1` definition n-dimensional arrays support.
        """

        x = 0.18
        clog = log_encoding_CanonLog_v1(x)

        x = np.tile(x, 6)
        clog = np.tile(clog, 6)
        np.testing.assert_allclose(
            log_encoding_CanonLog_v1(x), clog, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        clog = np.reshape(clog, (2, 3))
        np.testing.assert_allclose(
            log_encoding_CanonLog_v1(x), clog, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        clog = np.reshape(clog, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_CanonLog_v1(x), clog, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_CanonLog_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1` definition domain and range scale support.
        """

        x = 0.18
        clog = log_encoding_CanonLog_v1(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_CanonLog_v1(x * factor),
                    clog * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1` definition nan support.
        """

        log_encoding_CanonLog_v1(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_CanonLog_v1(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1` definition unit tests methods.
    """

    def test_log_decoding_CanonLog_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1` definition.
        """

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1(-0.023560122781997), -0.1, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1(0.125122480156403), 0.0, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1(0.343389651726069), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1(0.343138084215647, 12), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1(0.327953896935809, 10, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1(0.312012855550395, 10, False, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1(0.618775485598649), 1.0, places=7
        )

    def test_n_dimensional_log_decoding_CanonLog_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1` definition n-dimensional arrays support.
        """

        clog = 0.343389651726069
        x = log_decoding_CanonLog_v1(clog)

        clog = np.tile(clog, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_CanonLog_v1(clog), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        clog = np.reshape(clog, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_CanonLog_v1(clog), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        clog = np.reshape(clog, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_CanonLog_v1(clog), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_CanonLog_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1` definition domain and range scale support.
        """

        clog = 0.343389651726069
        x = log_decoding_CanonLog_v1(clog)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_CanonLog_v1(clog * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1` definition nan support.
        """

        log_decoding_CanonLog_v1(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogEncoding_CanonLog_v1_2(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1_2` definition unit tests methods.
    """

    def test_log_encoding_CanonLog_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1_2` definition.
        """

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1_2(-0.1), -0.023560121389098, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1_2(0.0), 0.125122480000000, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1_2(0.18), 0.343389649295280, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1_2(0.18, 12), 0.343389649295281, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1_2(0.18, 10, False),
            0.327953894097114,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1_2(0.18, 10, False, False),
            0.312012852877809,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog_v1_2(1.0), 0.618775480298287, places=7
        )

        samples = np.linspace(0, 1, 10000)

        np.testing.assert_allclose(
            log_encoding_CanonLog_v1(samples),
            log_encoding_CanonLog_v1_2(samples),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_CanonLog_v1(samples),
            log_encoding_CanonLog_v1_2(samples),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_CanonLog_v1(samples, out_normalised_code_value=False),
            log_encoding_CanonLog_v1_2(
                samples, out_normalised_code_value=False
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            log_encoding_CanonLog_v1(samples, in_reflection=False),
            log_encoding_CanonLog_v1_2(samples, in_reflection=False),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_CanonLog_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1_2` definition n-dimensional arrays support.
        """

        x = 0.18
        clog = log_encoding_CanonLog_v1_2(x)

        x = np.tile(x, 6)
        clog = np.tile(clog, 6)
        np.testing.assert_allclose(
            log_encoding_CanonLog_v1_2(x), clog, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        clog = np.reshape(clog, (2, 3))
        np.testing.assert_allclose(
            log_encoding_CanonLog_v1_2(x), clog, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        clog = np.reshape(clog, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_CanonLog_v1_2(x), clog, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_CanonLog_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1_2` definition domain and range scale support.
        """

        x = 0.18
        clog = log_encoding_CanonLog_v1_2(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_CanonLog_v1_2(x * factor),
                    clog * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog_v1_2` definition nan support.
        """

        log_encoding_CanonLog_v1_2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_CanonLog_v1_2(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1_2` definition unit tests methods.
    """

    def test_log_decoding_CanonLog_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1_2` definition.
        """

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1_2(-0.023560121389098), -0.1, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1_2(0.125122480000000), 0.0, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1_2(0.343389649295280), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1_2(0.343389649295281, 12), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1_2(0.327953894097114, 10, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1_2(0.312012852877809, 10, False, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog_v1_2(0.618775480298287), 1.0, places=7
        )

        samples = np.linspace(0, 1, 10000)

        np.testing.assert_allclose(
            log_decoding_CanonLog_v1(samples),
            log_decoding_CanonLog_v1_2(samples),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_CanonLog_v1(samples),
            log_decoding_CanonLog_v1_2(samples),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_CanonLog_v1(samples, in_normalised_code_value=False),
            log_decoding_CanonLog_v1_2(
                samples, in_normalised_code_value=False
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            log_decoding_CanonLog_v1(samples, out_reflection=False),
            log_decoding_CanonLog_v1_2(samples, out_reflection=False),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_CanonLog_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1_2` definition n-dimensional arrays support.
        """

        clog = 0.343389649295280
        x = log_decoding_CanonLog_v1_2(clog)

        clog = np.tile(clog, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_CanonLog_v1_2(clog), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        clog = np.reshape(clog, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_CanonLog_v1_2(clog), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        clog = np.reshape(clog, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_CanonLog_v1_2(clog), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_CanonLog_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1_2` definition domain and range scale support.
        """

        clog = 0.343389649295280
        x = log_decoding_CanonLog_v1_2(clog)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_CanonLog_v1_2(clog * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog_v1_2` definition nan support.
        """

        log_decoding_CanonLog_v1_2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogEncoding_CanonLog2_v1(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1` definition unit tests methods.
    """

    def test_log_encoding_CanonLog2_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1` definition.
        """

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1(-0.1), -0.155370131996824, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1(0.0), 0.092864125247312, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1(0.18), 0.398254694983167, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1(0.18, 12), 0.397962933301861, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1(0.18, 10, False),
            0.392025745397009,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1(0.18, 10, False, False),
            0.379864582222983,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1(1.0), 0.573229282897641, places=7
        )

    def test_n_dimensional_log_encoding_CanonLog2_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1` definition n-dimensional arrays support.
        """

        x = 0.18
        clog2 = log_encoding_CanonLog2_v1(x)

        x = np.tile(x, 6)
        clog2 = np.tile(clog2, 6)
        np.testing.assert_allclose(
            log_encoding_CanonLog2_v1(x), clog2, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        clog2 = np.reshape(clog2, (2, 3))
        np.testing.assert_allclose(
            log_encoding_CanonLog2_v1(x), clog2, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        clog2 = np.reshape(clog2, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_CanonLog2_v1(x), clog2, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_CanonLog2_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1` definition domain and range scale support.
        """

        x = 0.18
        clog2 = log_encoding_CanonLog2_v1(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_CanonLog2_v1(x * factor),
                    clog2 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog2_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1` definition nan support.
        """

        log_encoding_CanonLog2_v1(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_CanonLog2_v1(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1` definition unit tests methods.
    """

    def test_log_decoding_CanonLog2_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1` definition.
        """

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1(-0.155370131996824), -0.1, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1(0.092864125247312), 0.0, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1(0.398254694983167), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1(0.397962933301861, 12), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1(0.392025745397009, 10, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1(0.379864582222983, 10, False, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1(0.573229282897641), 1.0, places=7
        )

    def test_n_dimensional_log_decoding_CanonLog2_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1` definition n-dimensional arrays support.
        """

        clog2 = 0.398254694983167
        x = log_decoding_CanonLog2_v1(clog2)

        clog2 = np.tile(clog2, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_CanonLog2_v1(clog2), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        clog2 = np.reshape(clog2, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_CanonLog2_v1(clog2), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        clog2 = np.reshape(clog2, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_CanonLog2_v1(clog2), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_CanonLog2_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1` definition domain and range scale support.
        """

        clog = 0.398254694983167
        x = log_decoding_CanonLog2_v1(clog)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_CanonLog2_v1(clog * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog2_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1` definition nan support.
        """

        log_decoding_CanonLog2_v1(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogEncoding_CanonLog2_v1_2(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1_2` definition unit tests methods.
    """

    def test_log_encoding_CanonLog2_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1_2` definition.
        """

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1_2(-0.1), -0.155370130476722, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1_2(0.0), 0.092864125000000, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1_2(0.18), 0.398254692561492, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1_2(0.18, 12), 0.398254692561492, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1_2(0.18, 10, False),
            0.392025742568957,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1_2(0.18, 10, False, False),
            0.379864579481518,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog2_v1_2(1.0), 0.573229279230156, places=7
        )

        samples = np.linspace(0, 1, 10000)

        np.testing.assert_allclose(
            log_encoding_CanonLog2_v1(samples),
            log_encoding_CanonLog2_v1_2(samples),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_CanonLog2_v1(samples),
            log_encoding_CanonLog2_v1_2(samples),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_CanonLog2_v1(
                samples, out_normalised_code_value=False
            ),
            log_encoding_CanonLog2_v1_2(
                samples, out_normalised_code_value=False
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            log_encoding_CanonLog2_v1(samples, in_reflection=False),
            log_encoding_CanonLog2_v1_2(samples, in_reflection=False),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_CanonLog2_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1_2` definition n-dimensional arrays support.
        """

        x = 0.18
        clog2 = log_encoding_CanonLog2_v1_2(x)

        x = np.tile(x, 6)
        clog2 = np.tile(clog2, 6)
        np.testing.assert_allclose(
            log_encoding_CanonLog2_v1_2(x),
            clog2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = np.reshape(x, (2, 3))
        clog2 = np.reshape(clog2, (2, 3))
        np.testing.assert_allclose(
            log_encoding_CanonLog2_v1_2(x),
            clog2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = np.reshape(x, (2, 3, 1))
        clog2 = np.reshape(clog2, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_CanonLog2_v1_2(x),
            clog2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_CanonLog2_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1_2` definition domain and range scale support.
        """

        x = 0.18
        clog2 = log_encoding_CanonLog2_v1_2(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_CanonLog2_v1_2(x * factor),
                    clog2 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog2_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog2_v1_2` definition nan support.
        """

        log_encoding_CanonLog2_v1_2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_CanonLog2_v1_2(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1_2` definition unit tests methods.
    """

    def test_log_decoding_CanonLog2_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1_2` definition.
        """

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1_2(-0.155370130476722), -0.1, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1_2(0.092864125000000), 0.0, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1_2(0.398254692561492), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1_2(0.398254692561492, 12), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1_2(0.392025742568957, 10, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1_2(0.379864579481518, 10, False, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog2_v1_2(0.573229279230156), 1.0, places=7
        )

        samples = np.linspace(0, 1, 10000)

        np.testing.assert_allclose(
            log_decoding_CanonLog_v1(samples),
            log_decoding_CanonLog_v1_2(samples),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_CanonLog_v1(samples),
            log_decoding_CanonLog_v1_2(samples),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_CanonLog_v1(samples, in_normalised_code_value=False),
            log_decoding_CanonLog_v1_2(
                samples, in_normalised_code_value=False
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            log_decoding_CanonLog_v1(samples, out_reflection=False),
            log_decoding_CanonLog_v1_2(samples, out_reflection=False),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_CanonLog2_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1_2` definition n-dimensional arrays support.
        """

        clog2 = 0.398254692561492
        x = log_decoding_CanonLog2_v1_2(clog2)

        clog2 = np.tile(clog2, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_CanonLog2_v1_2(clog2),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog2 = np.reshape(clog2, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_CanonLog2_v1_2(clog2),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog2 = np.reshape(clog2, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_CanonLog2_v1_2(clog2),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_CanonLog2_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1_2` definition domain and range scale support.
        """

        clog = 0.398254692561492
        x = log_decoding_CanonLog2_v1_2(clog)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_CanonLog2_v1_2(clog * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog2_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog2_v1_2` definition nan support.
        """

        log_decoding_CanonLog2_v1_2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogEncoding_CanonLog3_v1(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1` definition unit tests methods.
    """

    def test_log_encoding_CanonLog3_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1` definition.
        """

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1(-0.1), -0.028494506076432, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1(0.0), 0.125122189869013, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1(0.18), 0.343389369388687, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1(0.18, 12), 0.343137802085105, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1(0.18, 10, False),
            0.327953567219893,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1(0.18, 10, False, False),
            0.313436005886328,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1(1.0), 0.580277796238604, places=7
        )

    def test_n_dimensional_log_encoding_CanonLog3_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1` definition n-dimensional arrays support.
        """

        x = 0.18
        clog3 = log_encoding_CanonLog3_v1(x)

        x = np.tile(x, 6)
        clog3 = np.tile(clog3, 6)
        np.testing.assert_allclose(
            log_encoding_CanonLog3_v1(x), clog3, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3))
        clog3 = np.reshape(clog3, (2, 3))
        np.testing.assert_allclose(
            log_encoding_CanonLog3_v1(x), clog3, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        x = np.reshape(x, (2, 3, 1))
        clog3 = np.reshape(clog3, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_CanonLog3_v1(x), clog3, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_encoding_CanonLog3_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1` definition domain and range scale support.
        """

        x = 0.18
        clog3 = log_encoding_CanonLog3_v1(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_CanonLog3_v1(x * factor),
                    clog3 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog3_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1` definition nan support.
        """

        log_encoding_CanonLog3_v1(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_CanonLog3_v1(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1` definition unit tests methods.
    """

    def test_log_decoding_CanonLog3_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1` definition.
        """

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1(-0.028494506076432), -0.1, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1(0.125122189869013), 0.0, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1(0.343389369388687), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1(0.343137802085105, 12), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1(0.327953567219893, 10, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1(0.313436005886328, 10, False, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1(0.580277796238604), 1.0, places=7
        )

    def test_n_dimensional_log_decoding_CanonLog3_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1` definition n-dimensional arrays support.
        """

        clog3 = 0.343389369388687
        x = log_decoding_CanonLog3_v1(clog3)

        clog3 = np.tile(clog3, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_CanonLog3_v1(clog3), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        clog3 = np.reshape(clog3, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_CanonLog3_v1(clog3), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        clog3 = np.reshape(clog3, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_CanonLog3_v1(clog3), x, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_log_decoding_CanonLog3_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1` definition domain and range scale support.
        """

        clog = 0.343389369388687
        x = log_decoding_CanonLog3_v1(clog)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_CanonLog3_v1(clog * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog3_v1(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1` definition nan support.
        """

        log_decoding_CanonLog3_v1(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogEncoding_CanonLog3_v1_2(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1_2` definition unit tests methods.
    """

    def test_log_encoding_CanonLog3_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1_2` definition.
        """

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1_2(-0.1), -0.028494507620494, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1_2(0.0), 0.125122189999999, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1_2(0.18), 0.343389370373936, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1_2(0.18, 12), 0.343389370373936, places=7
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1_2(0.18, 10, False),
            0.327953568370475,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1_2(0.18, 10, False, False),
            0.313436007221221,
            places=7,
        )

        self.assertAlmostEqual(
            log_encoding_CanonLog3_v1_2(1.0), 0.580277794216371, places=7
        )

        samples = np.linspace(0, 1, 10000)

        np.testing.assert_allclose(
            log_encoding_CanonLog3_v1(samples),
            log_encoding_CanonLog3_v1_2(samples),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_CanonLog3_v1(samples),
            log_encoding_CanonLog3_v1_2(samples),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_encoding_CanonLog3_v1(
                samples, out_normalised_code_value=False
            ),
            log_encoding_CanonLog3_v1_2(
                samples, out_normalised_code_value=False
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            log_encoding_CanonLog3_v1(samples, in_reflection=False),
            log_encoding_CanonLog3_v1_2(samples, in_reflection=False),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_encoding_CanonLog3_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1_2` definition n-dimensional arrays support.
        """

        x = 0.18
        clog3 = log_encoding_CanonLog3_v1_2(x)

        x = np.tile(x, 6)
        clog3 = np.tile(clog3, 6)
        np.testing.assert_allclose(
            log_encoding_CanonLog3_v1_2(x),
            clog3,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = np.reshape(x, (2, 3))
        clog3 = np.reshape(clog3, (2, 3))
        np.testing.assert_allclose(
            log_encoding_CanonLog3_v1_2(x),
            clog3,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = np.reshape(x, (2, 3, 1))
        clog3 = np.reshape(clog3, (2, 3, 1))
        np.testing.assert_allclose(
            log_encoding_CanonLog3_v1_2(x),
            clog3,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_encoding_CanonLog3_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1_2` definition domain and range scale support.
        """

        x = 0.18
        clog3 = log_encoding_CanonLog3_v1_2(x)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_encoding_CanonLog3_v1_2(x * factor),
                    clog3 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_encoding_CanonLog3_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_encoding_CanonLog3_v1_2` definition nan support.
        """

        log_encoding_CanonLog3_v1_2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLogDecoding_CanonLog3_v1_2(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1_2` definition unit tests methods.
    """

    def test_log_decoding_CanonLog3_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1_2` definition.
        """

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1_2(-0.028494507620494), -0.1, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1_2(0.125122189999999), 0.0, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1_2(0.343389370373936), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1_2(0.343389370373936, 12), 0.18, places=7
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1_2(0.327953568370475, 10, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1_2(0.313436007221221, 10, False, False),
            0.18,
            places=7,
        )

        self.assertAlmostEqual(
            log_decoding_CanonLog3_v1_2(0.580277794216371), 1.0, places=7
        )

        samples = np.linspace(0, 1, 10000)

        np.testing.assert_allclose(
            log_decoding_CanonLog3_v1(samples),
            log_decoding_CanonLog3_v1_2(samples),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_CanonLog3_v1(samples),
            log_decoding_CanonLog3_v1_2(samples),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            log_decoding_CanonLog3_v1(samples, in_normalised_code_value=False),
            log_decoding_CanonLog3_v1_2(
                samples, in_normalised_code_value=False
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            log_decoding_CanonLog3_v1(samples, out_reflection=False),
            log_decoding_CanonLog3_v1_2(samples, out_reflection=False),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_log_decoding_CanonLog3_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1_2` definition n-dimensional arrays support.
        """

        clog3 = 0.343389370373936
        x = log_decoding_CanonLog3_v1_2(clog3)

        clog3 = np.tile(clog3, 6)
        x = np.tile(x, 6)
        np.testing.assert_allclose(
            log_decoding_CanonLog3_v1_2(clog3),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog3 = np.reshape(clog3, (2, 3))
        x = np.reshape(x, (2, 3))
        np.testing.assert_allclose(
            log_decoding_CanonLog3_v1_2(clog3),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        clog3 = np.reshape(clog3, (2, 3, 1))
        x = np.reshape(x, (2, 3, 1))
        np.testing.assert_allclose(
            log_decoding_CanonLog3_v1_2(clog3),
            x,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_log_decoding_CanonLog3_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1_2` definition domain and range scale support.
        """

        clog = 0.343389370373936
        x = log_decoding_CanonLog3_v1_2(clog)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    log_decoding_CanonLog3_v1_2(clog * factor),
                    x * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_log_decoding_CanonLog3_v1_2(self):
        """
        Test :func:`colour.models.rgb.transfer_functions.canon.\
log_decoding_CanonLog3_v1_2` definition nan support.
        """

        log_decoding_CanonLog3_v1_2(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


if __name__ == "__main__":
    unittest.main()
