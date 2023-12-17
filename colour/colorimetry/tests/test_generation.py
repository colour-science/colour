# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.colorimetry.generation` module."""

import unittest

import numpy as np

from colour.colorimetry.generation import (
    msds_constant,
    msds_ones,
    msds_zeros,
    sd_constant,
    sd_gaussian_fwhm,
    sd_gaussian_normal,
    sd_multi_leds_Ohno2005,
    sd_ones,
    sd_single_led_Ohno2005,
    sd_zeros,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestSdConstant",
    "TestSdZeros",
    "TestSdOnes",
    "TestMsdsConstant",
    "TestMsdsZeros",
    "TestMsdsOnes",
    "TestSdGaussianNormal",
    "TestSdGaussianFwhm",
    "TestSdSingleLedOhno2005",
    "TestSdMultiLedsOhno2005",
]


class TestSdConstant(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.generation.sd_constant` definition unit
    tests methods.
    """

    def test_sd_constant(self):
        """Test :func:`colour.colorimetry.generation.sd_constant` definition."""

        sd = sd_constant(np.pi)

        np.testing.assert_allclose(
            sd[360], np.pi, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            sd[555], np.pi, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            sd[780], np.pi, atol=TOLERANCE_ABSOLUTE_TESTS
        )


class TestSdZeros(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.generation.sd_zeros` definition unit
    tests methods.
    """

    def test_sd_zeros(self):
        """
        Test :func:`colour.colorimetry.generation.sd_zeros`
        definition.
        """

        sd = sd_zeros()

        self.assertEqual(sd[360], 0)

        self.assertEqual(sd[555], 0)

        self.assertEqual(sd[780], 0)


class TestSdOnes(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.generation.sd_ones` definition unit
    tests methods.
    """

    def test_sd_ones(self):
        """Test :func:`colour.colorimetry.generation.sd_ones` definition."""

        sd = sd_ones()

        self.assertEqual(sd[360], 1)

        self.assertEqual(sd[555], 1)

        self.assertEqual(sd[780], 1)


class TestMsdsConstant(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.generation.msds_constant` definition unit
    tests methods.
    """

    def test_msds_constant(self):
        """Test :func:`colour.colorimetry.generation.msds_constant` definition."""

        msds = msds_constant(np.pi, labels=["a", "b", "c"])

        np.testing.assert_allclose(
            msds[360],
            np.array([np.pi, np.pi, np.pi]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            msds[555],
            np.array([np.pi, np.pi, np.pi]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            msds[780],
            np.array([np.pi, np.pi, np.pi]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestMsdsZeros(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.generation.msds_zeros` definition unit
    tests methods.
    """

    def test_msds_zeros(self):
        """
        Test :func:`colour.colorimetry.generation.msds_zeros`
        definition.
        """

        msds = msds_zeros(labels=["a", "b", "c"])

        np.testing.assert_equal(msds[360], np.array([0, 0, 0]))

        np.testing.assert_equal(msds[555], np.array([0, 0, 0]))

        np.testing.assert_equal(msds[780], np.array([0, 0, 0]))


class TestMsdsOnes(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.generation.msds_ones` definition unit
    tests methods.
    """

    def test_msds_ones(self):
        """Test :func:`colour.colorimetry.generation.msds_ones` definition."""

        msds = msds_ones(labels=["a", "b", "c"])

        np.testing.assert_equal(msds[360], np.array([1, 1, 1]))

        np.testing.assert_equal(msds[555], np.array([1, 1, 1]))

        np.testing.assert_equal(msds[780], np.array([1, 1, 1]))


class TestSdGaussianNormal(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.generation.sd_gaussian_normal`
    definition unit tests methods.
    """

    def test_sd_gaussian_normal(self):
        """
        Test :func:`colour.colorimetry.generation.sd_gaussian_normal`
        definition.
        """

        sd = sd_gaussian_normal(555, 25)

        np.testing.assert_allclose(
            sd[530], 0.606530659712633, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(sd[555], 1, atol=TOLERANCE_ABSOLUTE_TESTS)

        np.testing.assert_allclose(
            sd[580], 0.606530659712633, atol=TOLERANCE_ABSOLUTE_TESTS
        )


class TestSdGaussianFwhm(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.generation.sd_gaussian_fwhm` definition
    unit tests methods.
    """

    def test_sd_gaussian_fwhm(self):
        """
        Test :func:`colour.colorimetry.generation.sd_gaussian_fwhm` definition.
        """

        sd = sd_gaussian_fwhm(555, 25)

        np.testing.assert_allclose(
            sd[530], 0.0625, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(sd[555], 1, atol=TOLERANCE_ABSOLUTE_TESTS)

        np.testing.assert_allclose(
            sd[580], 0.062499999999999, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            sd[555 - 25 / 2], 0.5, atol=TOLERANCE_ABSOLUTE_TESTS
        )


class TestSdSingleLedOhno2005(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.generation.sd_single_led_Ohno2005`
    definition unit tests methods.
    """

    def test_sd_single_led_Ohno2005(self):
        """
        Test :func:`colour.colorimetry.generation.sd_single_led_Ohno2005`
        definition.
        """

        sd = sd_single_led_Ohno2005(555, 25)

        np.testing.assert_allclose(
            sd[530], 0.127118445056538, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(sd[555], 1, atol=TOLERANCE_ABSOLUTE_TESTS)

        np.testing.assert_allclose(
            sd[580], 0.127118445056538, atol=TOLERANCE_ABSOLUTE_TESTS
        )


class TestSdMultiLedsOhno2005(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.generation.sd_multi_leds_Ohno2005`
    definition unit tests methods.
    """

    def test_sd_multi_leds_Ohno2005(self):
        """
        Test :func:`colour.colorimetry.generation.sd_multi_leds_Ohno2005`
        definition.
        """

        sd = sd_multi_leds_Ohno2005(
            np.array([457, 530, 615]),
            np.array([20, 30, 20]),
            np.array([0.731, 1.000, 1.660]),
        )

        np.testing.assert_allclose(
            sd[500], 0.129513248576116, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            sd[570], 0.059932156222703, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            sd[640], 0.116433257970624, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        sd = sd_multi_leds_Ohno2005(
            np.array([457, 530, 615]),
            np.array([20, 30, 20]),
        )

        np.testing.assert_allclose(
            sd[500], 0.130394510062799, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            sd[570], 0.058539618824187, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        np.testing.assert_allclose(
            sd[640], 0.070140708922879, atol=TOLERANCE_ABSOLUTE_TESTS
        )


if __name__ == "__main__":
    unittest.main()
