"""Define the unit tests for the :mod:`colour.recovery.smits1999` module."""

from __future__ import annotations

import numpy as np

from colour.colorimetry import sd_to_XYZ_integration
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.recovery import (
    MSDS_SMITS1999,
    SDS_SMITS1999,
    RGB_to_msds_Smits1999,
    RGB_to_sd_Smits1999,
)
from colour.recovery.smits1999 import XYZ_to_RGB_Smits1999
from colour.utilities import domain_range_scale

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestMsds_from_RGB_Smits1999",
    "TestRGB_to_msds_Smits1999",
    "TestSd_from_RGB_Smits1999",
    "TestRGB_to_sd_Smits1999",
]


class TestMsds_from_RGB_Smits1999:
    """
    Define :func:`colour.recovery.smits1999.RGB_to_msds_Smits1999`
    definition unit tests methods.
    """

    def test_RGB_to_msds_Smits1999(self) -> None:
        """
        Test :func:`colour.recovery.smits1999.RGB_to_msds_Smits1999`
        definition.
        """

        RGB = np.array(
            [
                [0.45623196, 0.03080455, 0.04093343],
                [0.05438271, 0.29877169, 0.07188444],
                [0.01863137, 0.05139773, 0.28887675],
            ]
        )

        msds = RGB_to_msds_Smits1999(RGB, MSDS_SMITS1999)

        assert msds.shape == (3, 10)

        np.testing.assert_allclose(
            msds,
            np.array(
                [
                    [
                        0.08296164,
                        0.06232130,
                        0.04061129,
                        0.03304071,
                        0.03077991,
                        0.03126229,
                        0.38501744,
                        0.46241991,
                        0.46241991,
                        0.46237838,
                    ],
                    [
                        0.07137689,
                        0.07087984,
                        0.07808527,
                        0.25193903,
                        0.29874044,
                        0.28556823,
                        0.09612190,
                        0.05438271,
                        0.05438271,
                        0.05494993,
                    ],
                    [
                        0.28792653,
                        0.28699596,
                        0.26315510,
                        0.13032190,
                        0.05140576,
                        0.05141694,
                        0.02382727,
                        0.02739435,
                        0.03010161,
                        0.03041033,
                    ],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestRGB_to_msds_Smits1999:
    """
    Define :func:`colour.recovery.smits1999.RGB_to_msds_Smits1999`
    definition unit tests methods.
    """

    def test_RGB_to_msds_Smits1999(self) -> None:
        """
        Test :func:`colour.recovery.smits1999.RGB_to_msds_Smits1999`
        definition.
        """

        RGB = np.array(
            [
                [0.45623196, 0.03080455, 0.04093343],
                [0.05438271, 0.29877169, 0.07188444],
                [0.01863137, 0.05139773, 0.28887675],
            ]
        )

        msds = RGB_to_msds_Smits1999(RGB)

        assert msds.shape == (3, 10)

        np.testing.assert_allclose(
            msds[0, 0],
            0.08296164,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestSd_from_RGB_Smits1999:
    """
    Define :func:`colour.recovery.smits1999.RGB_to_sd_Smits1999`
    definition unit tests methods.
    """

    def test_RGB_to_sd_Smits1999(self) -> None:
        """
        Test :func:`colour.recovery.smits1999.RGB_to_sd_Smits1999`
        definition.
        """

        RGB = np.array([0.4, 0.03, 0.04])
        sd = RGB_to_sd_Smits1999(RGB, MSDS_SMITS1999, "test")

        np.testing.assert_equal(sd.name, "test")
        np.testing.assert_equal(sd.shape, SDS_SMITS1999["white"].shape)

        np.testing.assert_allclose(
            sd.values,
            np.array(
                [
                    0.076432,
                    0.05854,
                    0.039682,
                    0.032208,
                    0.029976,
                    0.030452,
                    0.338069,
                    0.405364,
                    0.405364,
                    0.405323,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB_white = np.array([1.0, 1.0, 1.0])
        sd_white = RGB_to_sd_Smits1999(RGB_white, MSDS_SMITS1999, "white")
        np.testing.assert_allclose(
            sd_white.values,
            SDS_SMITS1999["white"].values,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB_red = np.array([1.0, 0.0, 0.0])
        sd_red = RGB_to_sd_Smits1999(RGB_red, MSDS_SMITS1999, "red")
        np.testing.assert_allclose(
            sd_red.values,
            SDS_SMITS1999["red"].values,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB_green = np.array([0.0, 1.0, 0.0])
        sd_green = RGB_to_sd_Smits1999(RGB_green, MSDS_SMITS1999, "green")
        np.testing.assert_allclose(
            sd_green.values,
            SDS_SMITS1999["green"].values,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB_blue = np.array([0.0, 0.0, 1.0])
        sd_blue = RGB_to_sd_Smits1999(RGB_blue, MSDS_SMITS1999, "blue")
        np.testing.assert_allclose(
            sd_blue.values,
            SDS_SMITS1999["blue"].values,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestRGB_to_sd_Smits1999:
    """
    Define :func:`colour.recovery.smits1999.RGB_to_sd_Smits1999`
    definition unit tests methods.
    """

    def test_RGB_to_sd_Smits1999(self) -> None:
        """
        Test :func:`colour.recovery.smits1999.RGB_to_sd_Smits1999`
        definition.
        """

        np.testing.assert_allclose(
            RGB_to_sd_Smits1999(
                XYZ_to_RGB_Smits1999(np.array([0.21781186, 0.12541048, 0.04697113]))
            ).values,
            np.array(
                [
                    0.07691923,
                    0.05870050,
                    0.03943195,
                    0.03024978,
                    0.02750692,
                    0.02808645,
                    0.34298985,
                    0.41185795,
                    0.41185795,
                    0.41180754,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            RGB_to_sd_Smits1999(
                XYZ_to_RGB_Smits1999(np.array([0.15434689, 0.22960951, 0.09620221]))
            ).values,
            np.array(
                [
                    0.06981477,
                    0.06981351,
                    0.07713379,
                    0.25139495,
                    0.30063408,
                    0.28797045,
                    0.11990414,
                    0.08186170,
                    0.08198613,
                    0.08272671,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            RGB_to_sd_Smits1999(
                XYZ_to_RGB_Smits1999(np.array([0.07683480, 0.06006092, 0.25833845]))
            ).values,
            np.array(
                [
                    0.29091152,
                    0.29010285,
                    0.26572455,
                    0.13140471,
                    0.05160646,
                    0.05162034,
                    0.02765638,
                    0.03199188,
                    0.03472939,
                    0.03504156,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            RGB_to_sd_Smits1999(XYZ_to_RGB_Smits1999(np.array([0.0, 1.0, 0.0]))).values,
            np.array(
                [
                    -0.2549796,
                    -0.2848386,
                    -0.1634905,
                    1.5254829,
                    1.9800433,
                    1.8510762,
                    -0.7327702,
                    -1.2758621,
                    -1.2758621,
                    -1.2703551,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            RGB_to_sd_Smits1999(XYZ_to_RGB_Smits1999(np.array([1.0, 1.0, 0.0]))).values,
            np.array(
                [
                    -0.1168428,
                    -0.1396982,
                    -0.0414535,
                    0.581391,
                    0.9563091,
                    0.9562111,
                    1.3366949,
                    1.3742666,
                    1.3853491,
                    1.4027005,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            RGB_to_sd_Smits1999(XYZ_to_RGB_Smits1999(np.array([0.5, 0.0, 1.0]))).values,
            np.array(
                [
                    1.1938776,
                    1.1938776,
                    1.1213867,
                    -0.067889,
                    -0.4668587,
                    -0.4030985,
                    0.703056,
                    0.9407334,
                    0.9437298,
                    0.9383386,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_RGB_to_sd_Smits1999(self) -> None:
        """
        Test :func:`colour.recovery.smits1999.RGB_to_sd_Smits1999`
        definition domain and range scale support.
        """

        XYZ_i = np.array([0.20654008, 0.12197225, 0.05136952])
        RGB_i = XYZ_to_RGB_Smits1999(XYZ_i)
        XYZ_o = sd_to_XYZ_integration(RGB_to_sd_Smits1999(RGB_i))

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    sd_to_XYZ_integration(RGB_to_sd_Smits1999(RGB_i * factor_a)),
                    XYZ_o * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
