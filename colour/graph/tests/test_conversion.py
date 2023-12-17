# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.graph.conversion` module."""

import unittest

import numpy as np

from colour.characterisation import SDS_COLOURCHECKERS
from colour.colorimetry import CCS_ILLUMINANTS, SDS_ILLUMINANTS
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.graph import convert, describe_conversion_path
from colour.models import COLOURSPACE_MODELS, RGB_COLOURSPACE_ACES2065_1

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestDescribeConversionPath",
    "TestConvert",
]


class TestDescribeConversionPath(unittest.TestCase):
    """
    Define :func:`colour.graph.conversion.describe_conversion_path` definition
    unit tests methods.
    """

    def test_describe_conversion_path(self):
        """
        Test :func:`colour.graph.conversion.describe_conversion_path`
        definition.
        """

        describe_conversion_path("Spectral Distribution", "sRGB")

        describe_conversion_path("Spectral Distribution", "sRGB", mode="Long")

        describe_conversion_path(
            "Spectral Distribution",
            "sRGB",
            mode="Extended",
            sd_to_XYZ={
                "illuminant": SDS_ILLUMINANTS["FL2"],
                "return": np.array([0.47924575, 0.31676968, 0.17362725]),
            },
        )


class TestConvert(unittest.TestCase):
    """
    Define :func:`colour.graph.conversion.convert` definition unit tests
    methods.
    """

    def test_convert(self):
        """Test :func:`colour.graph.conversion.convert` definition."""

        # NOTE: Reduced precision for random unit tests failure.
        RGB_a = convert(
            SDS_COLOURCHECKERS["ColorChecker N Ohta"]["dark skin"],
            "Spectral Distribution",
            "sRGB",
        )
        np.testing.assert_allclose(
            RGB_a,
            np.array([0.49034776, 0.30185875, 0.23587685]),
            atol=5e-5,
        )

        # NOTE: Reduced precision for random unit tests failure.
        Jpapbp = convert(RGB_a, "Output-Referred RGB", "CAM16UCS")
        np.testing.assert_allclose(
            Jpapbp,
            np.array([0.40738741, 0.12046560, 0.09284385]),
            atol=5e-4,
        )

        RGB_b = convert(
            Jpapbp, "CAM16UCS", "sRGB", verbose={"mode": "Extended"}
        )
        # NOTE: The "CIE XYZ" tristimulus values to "sRGB" matrix is given
        # rounded at 4 decimals as per "IEC 61966-2-1:1999" and thus preventing
        # exact roundtrip.
        np.testing.assert_allclose(RGB_a, RGB_b, atol=1e-4)

        np.testing.assert_allclose(
            convert("#808080", "Hexadecimal", "Scene-Referred RGB"),
            np.array([0.21586050, 0.21586050, 0.21586050]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            convert("#808080", "Hexadecimal", "RGB Luminance"),
            0.21586050,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            convert(
                convert(
                    np.array([0.5, 0.5, 0.5]),
                    "Output-Referred RGB",
                    "Scene-Referred RGB",
                ),
                "RGB",
                "YCbCr",
            ),
            np.array([0.49215686, 0.50196078, 0.50196078]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # NOTE: Reduced precision for random unit tests failure.
        np.testing.assert_allclose(
            convert(
                RGB_a,
                "RGB",
                "Scene-Referred RGB",
                RGB_to_RGB={"output_colourspace": RGB_COLOURSPACE_ACES2065_1},
            ),
            np.array([0.37308227, 0.31241444, 0.24746366]),
            atol=5e-5,
        )

        # Consistency check to verify that all the colour models are properly
        # named in the graph:
        for model in COLOURSPACE_MODELS:
            convert(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                "CIE XYZ",
                model,
            )

    def test_convert_direct_keyword_argument_passing(self):
        """
        Test :func:`colour.graph.conversion.convert` definition behaviour when
        direct keyword arguments are passed.
        """

        a = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"][
            "D50"
        ]
        np.testing.assert_allclose(
            convert(
                a, "CIE XYZ", "CIE UVW", XYZ_to_UVW={"illuminant": illuminant}
            ),
            convert(a, "CIE XYZ", "CIE UVW", illuminant=illuminant),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Illuminant "ndarray" is converted to tuple here so that it can
        # be hashed by the "sd_to_XYZ" definition, this should never occur
        # in practical application.
        self.assertRaises(
            AttributeError,
            lambda: convert(
                SDS_COLOURCHECKERS["ColorChecker N Ohta"]["dark skin"],
                "Spectral Distribution",
                "sRGB",
                illuminant=tuple(illuminant),
            ),
        )


if __name__ == "__main__":
    unittest.main()
