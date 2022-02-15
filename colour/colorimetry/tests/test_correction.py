"""Defines the unit tests for the :mod:`colour.colorimetry.correction` module."""

import numpy as np
import unittest

from colour.colorimetry import (
    SpectralDistribution,
    bandpass_correction_Stearns1988,
)
from colour.hints import Tuple

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DATA_NON_BANDPASS_CORRECTED",
    "DATA_BANDPASS_CORRECTED",
    "TestBandpassCorrectionStearns1988",
]

DATA_NON_BANDPASS_CORRECTED: Tuple = (
    9.3700,
    12.3200,
    12.4600,
    9.5100,
    5.9200,
    4.3300,
    4.2900,
    3.8800,
    4.5100,
    10.9200,
    27.5000,
    49.6700,
    69.5900,
    81.7300,
    88.1900,
    86.0500,
)

DATA_BANDPASS_CORRECTED: Tuple = (
    9.12515000,
    12.57355255,
    12.69542514,
    9.54357971,
    5.75121288,
    4.21535933,
    4.33022518,
    3.79034131,
    4.03770167,
    10.11509076,
    27.10283747,
    49.88971449,
    70.21750370,
    82.14935719,
    88.88373581,
    85.87238000,
)


class TestBandpassCorrectionStearns1988(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.correction.\
bandpass_correction_Stearns1988` definition unit tests methods.
    """

    def test_bandpass_correction_Stearns1988(self):
        """
        Test :func:`colour.colorimetry.correction.\
bandpass_correction_Stearns1988` definition.
        """

        sd = SpectralDistribution(
            dict(
                zip(
                    range(len(DATA_NON_BANDPASS_CORRECTED)),
                    DATA_NON_BANDPASS_CORRECTED,
                )
            )
        )

        np.testing.assert_almost_equal(
            bandpass_correction_Stearns1988(sd).values, DATA_BANDPASS_CORRECTED
        )


if __name__ == "__main__":
    unittest.main()
