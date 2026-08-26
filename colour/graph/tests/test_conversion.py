"""Define the unit tests for the :mod:`colour.graph.conversion` module."""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    sys.version_info >= (3, 14, 1),
    reason="networkx 3.6 is incompatible with Python 3.14.1+",
)

from colour.characterisation import SDS_COLOURCHECKERS  # noqa: E402
from colour.colorimetry import CCS_ILLUMINANTS, SDS_ILLUMINANTS  # noqa: E402
from colour.constants import TOLERANCE_ABSOLUTE_TESTS  # noqa: E402
from colour.graph import convert, describe_conversion_path  # noqa: E402
from colour.models import (  # noqa: E402
    COLOURSPACE_MODELS,
    RGB_COLOURSPACE_ACES2065_1,
    XYZ_to_Lab,
)
from colour.utilities import (  # noqa: E402
    get_domain_range_scale_metadata,
    xp_assert_close,
)

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


class TestDescribeConversionPath:
    """
    Define :func:`colour.graph.conversion.describe_conversion_path` definition
    unit tests methods.
    """

    def test_describe_conversion_path(self) -> None:
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


class TestConvert:
    """
    Define :func:`colour.graph.conversion.convert` definition unit tests
    methods.
    """

    def test_convert(self) -> None:
        """Test :func:`colour.graph.conversion.convert` definition."""

        RGB_a = convert(
            SDS_COLOURCHECKERS["ColorChecker N Ohta"]["dark skin"],
            "Spectral Distribution",
            "sRGB",
        )
        xp_assert_close(
            RGB_a,
            [0.49034776, 0.30185875, 0.23587685],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Jpapbp = convert(RGB_a, "Output-Referred RGB", "CAM16UCS")
        xp_assert_close(
            Jpapbp,
            [0.40738741, 0.12046560, 0.09284385],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB_b = convert(Jpapbp, "CAM16UCS", "sRGB", verbose={"mode": "Extended"})
        # NOTE: The "CIE XYZ" tristimulus values to "sRGB" matrix is given
        # rounded at 4 decimals as per "IEC 61966-2-1:1999" and thus preventing
        # exact roundtrip.
        xp_assert_close(RGB_a, RGB_b, atol=TOLERANCE_ABSOLUTE_TESTS * 1000)

        xp_assert_close(
            convert("#808080", "Hexadecimal", "Scene-Referred RGB"),
            [0.21586050, 0.21586050, 0.21586050],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            convert("#808080", "Hexadecimal", "RGB Luminance"),
            0.21586050,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            convert(
                convert(
                    [0.5, 0.5, 0.5],
                    "Output-Referred RGB",
                    "Scene-Referred RGB",
                ),
                "RGB",
                "YCbCr",
            ),
            [0.49215686, 0.50196078, 0.50196078],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            convert(
                RGB_a,
                "RGB",
                "Scene-Referred RGB",
                RGB_to_RGB={"output_colourspace": RGB_COLOURSPACE_ACES2065_1},
            ),
            [0.37308227, 0.31241444, 0.24746366],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Consistency check to verify that all the colour models are properly
        # named in the graph:
        for model in COLOURSPACE_MODELS:
            convert(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                "CIE XYZ",
                model,
            )

    def test_convert_direct_keyword_argument_passing(self) -> None:
        """
        Test :func:`colour.graph.conversion.convert` definition behaviour when
        direct keyword arguments are passed.
        """

        a = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D50"]
        xp_assert_close(
            convert(a, "CIE XYZ", "CIE UVW", XYZ_to_UVW={"illuminant": illuminant}),
            convert(a, "CIE XYZ", "CIE UVW", illuminant=illuminant),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Illuminant "ndarray" is converted to tuple here so that it can
        # be hashed by the "sd_to_XYZ" definition, this should never occur
        # in practical application.
        with pytest.raises(AttributeError):
            convert(
                SDS_COLOURCHECKERS["ColorChecker N Ohta"]["dark skin"],
                "Spectral Distribution",
                "sRGB",
                illuminant=tuple(illuminant),
            )

    def test_convert_reference_scale(self) -> None:
        """
        Test :func:`colour.graph.conversion.convert` definition behaviour with
        `from_reference_scale` and `to_reference_scale` parameters.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])

        Lab_auto = convert(XYZ, "CIE XYZ", "CIE Lab", to_reference_scale=True)

        Lab_manual = convert(XYZ, "CIE XYZ", "CIE Lab")
        metadata = get_domain_range_scale_metadata(XYZ_to_Lab)
        range_scale = metadata["range"]
        Lab_manual_scaled = Lab_manual * range_scale

        xp_assert_close(
            Lab_auto,
            Lab_manual_scaled,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Lab_native = Lab_auto

        XYZ_auto = convert(
            Lab_native,
            "CIE Lab",
            "CIE XYZ",
            from_reference_scale=True,
        )

        Lab_manual_normalized = Lab_native / range_scale
        XYZ_manual = convert(Lab_manual_normalized, "CIE Lab", "CIE XYZ")

        xp_assert_close(
            XYZ_auto,
            XYZ_manual,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_roundtrip = convert(
            convert(
                XYZ,
                "CIE XYZ",
                "CIE Lab",
                to_reference_scale=True,
            ),
            "CIE Lab",
            "CIE XYZ",
            from_reference_scale=True,
        )

        xp_assert_close(
            XYZ_roundtrip,
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])

        # Test CIE Lab and CIE LCHab consistency
        Lab = convert(XYZ, "CIE XYZ", "CIE Lab", to_reference_scale=True)
        LCHab = convert(XYZ, "CIE XYZ", "CIE LCHab", to_reference_scale=True)

        # L component should be identical
        xp_assert_close(
            Lab[0],
            LCHab[0],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # C should equal sqrt(a^2 + b^2)
        expected_C = np.sqrt(Lab[1] ** 2 + Lab[2] ** 2)
        xp_assert_close(
            LCHab[1],
            expected_C,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test CIE Lab <-> CIE LCHab roundtrip conversion
        Lab_roundtrip = convert(
            LCHab,
            "CIE LCHab",
            "CIE Lab",
            from_reference_scale=True,
            to_reference_scale=True,
        )
        xp_assert_close(
            Lab_roundtrip,
            Lab,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        LCHab_roundtrip = convert(
            Lab,
            "CIE Lab",
            "CIE LCHab",
            from_reference_scale=True,
            to_reference_scale=True,
        )
        xp_assert_close(
            LCHab_roundtrip,
            LCHab,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test CIE Luv and CIE LCHuv consistency
        Luv = convert(XYZ, "CIE XYZ", "CIE Luv", to_reference_scale=True)
        LCHuv = convert(XYZ, "CIE XYZ", "CIE LCHuv", to_reference_scale=True)

        # L component should be identical
        xp_assert_close(
            Luv[0],
            LCHuv[0],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # C should equal sqrt(u^2 + v^2)
        expected_C = np.sqrt(Luv[1] ** 2 + Luv[2] ** 2)
        xp_assert_close(
            LCHuv[1],
            expected_C,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test CIE Luv <-> CIE LCHuv roundtrip conversion
        Luv_roundtrip = convert(
            LCHuv,
            "CIE LCHuv",
            "CIE Luv",
            from_reference_scale=True,
            to_reference_scale=True,
        )
        xp_assert_close(
            Luv_roundtrip,
            Luv,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        LCHuv_roundtrip = convert(
            Luv,
            "CIE Luv",
            "CIE LCHuv",
            from_reference_scale=True,
            to_reference_scale=True,
        )
        xp_assert_close(
            LCHuv_roundtrip,
            LCHuv,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
