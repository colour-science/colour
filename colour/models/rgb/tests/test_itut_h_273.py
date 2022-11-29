# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.rgb.itut_h_273` module."""

import unittest

from colour.models import (
    describe_video_signal_colour_primaries,
    describe_video_signal_transfer_characteristics,
    describe_video_signal_matrix_coefficients,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestDescribeVideoSignalColourPrimaries",
    "TestDescribeVideoSignalTransferCharacteristics",
    "TestDescribeVideoSignalMatrixCoefficients",
]


class TestDescribeVideoSignalColourPrimaries(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.itut_h_273.\
describe_video_signal_colour_primaries` definition unit tests methods.
    """

    def test_describe_video_signal_colour_primaries(self):
        """
        Test
        :func:`colour.models.rgb.itut_h_273.\
describe_video_signal_colour_primaries` definition.
        """

        description = describe_video_signal_colour_primaries(1)
        self.assertIsInstance(description, str)


class TestDescribeVideoSignalTransferCharacteristics(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.itut_h_273.\
describe_video_signal_transfer_characteristics` definition unit tests methods.
    """

    def test_describe_video_signal_transfer_characteristics(self):
        """
        Test :func:`colour.models.rgb.itut_h_273.\
describe_video_signal_transfer_characteristics` definition.
        """

        description = describe_video_signal_transfer_characteristics(1)
        self.assertIsInstance(description, str)


class TestDescribeVideoSignalMatrixCoefficients(unittest.TestCase):
    """
    Define :func:`colour.models.rgb.itut_h_273.\
describe_video_signal_matrix_coefficients` definition unit tests methods.
    """

    def test_describe_video_signal_matrix_coefficients(self):
        """
        Test :func:`colour.models.rgb.itut_h_273.\
describe_video_signal_matrix_coefficients` definition.
        """

        description = describe_video_signal_matrix_coefficients(1)
        self.assertIsInstance(description, str)


if __name__ == "__main__":
    unittest.main()
