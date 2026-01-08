"""Define the unit tests for the :mod:`colour.recovery.gaussian` module."""

from __future__ import annotations

import numpy as np

from colour.colorimetry import SpectralShape, sd_to_XYZ_integration
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.recovery import (
    MSDS_GAUSSIAN_BASIS,
    RGB_to_msds_Gaussian,
    RGB_to_sd_Gaussian,
)
from colour.recovery.gaussian import (
    XYZ_to_RGB_Gaussian,
    optimise_gaussian_basis_parameters,
)
from colour.recovery.smits1999 import RGB_to_msds_Smits1999, RGB_to_sd_Smits1999
from colour.utilities import domain_range_scale

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestOptimiseGaussianBasisParameters",
    "TestRGB_to_msds_Gaussian",
    "TestRGB_to_sd_Gaussian",
]


class TestOptimiseGaussianBasisParameters:
    """
    Define :func:`colour.recovery.gaussian.optimise_gaussian_basis_parameters`
    definition unit tests methods.
    """

    def test_optimise_gaussian_basis_parameters(self) -> None:
        """
        Test :func:`colour.recovery.gaussian.optimise_gaussian_basis_parameters`
        definition.
        """

        peak_wavelengths, fwhm, exponent = optimise_gaussian_basis_parameters()

        assert set(peak_wavelengths.keys()) == {
            "red",
            "green",
            "blue",
            "cyan",
            "magenta",
            "yellow",
        }
        assert set(fwhm.keys()) == {
            "red",
            "green",
            "blue",
            "cyan",
            "magenta",
            "yellow",
        }
        assert set(exponent.keys()) == {
            "red",
            "green",
            "blue",
            "cyan",
            "magenta",
            "yellow",
        }

        # Primary colours
        assert 560 < peak_wavelengths["red"] < 650
        assert 510 < peak_wavelengths["green"] < 570
        assert 400 < peak_wavelengths["blue"] <= 480

        # Secondary colours
        assert 460 < peak_wavelengths["cyan"] < 540  # below green
        assert 510 < peak_wavelengths["magenta"] < 570  # valley at green
        assert 550 < peak_wavelengths["yellow"] < 620  # above green

        for colour in peak_wavelengths:
            assert 15 <= fwhm[colour] <= 200
            assert 2.0 <= exponent[colour] <= 20.0


class TestRGB_to_msds_Gaussian:
    """
    Define :func:`colour.recovery.gaussian.RGB_to_msds_Gaussian`
    definition unit tests methods.
    """

    def test_RGB_to_msds_Gaussian(self) -> None:
        """
        Test :func:`colour.recovery.gaussian.RGB_to_msds_Gaussian`
        definition.
        """

        RGB = np.array(
            [
                [0.45623196, 0.03080455, 0.04093343],
                [0.05438271, 0.29877169, 0.07188444],
                [0.01863137, 0.05139773, 0.28887675],
            ]
        )

        msds = RGB_to_msds_Gaussian(RGB)

        assert msds.shape == (3, 421)

        np.testing.assert_allclose(
            msds[0, 0],
            0.04093343,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test with interpolated basis at 10nm for full array comparison
        basis_10nm = MSDS_GAUSSIAN_BASIS.copy().align(SpectralShape(360, 780, 10))
        msds_10nm = RGB_to_msds_Smits1999(RGB, basis_10nm)

        assert msds_10nm.shape == (3, 43)

        np.testing.assert_allclose(
            msds_10nm,
            np.array(
                [
                    [
                        0.04093343,
                        0.04093343,
                        0.04093343,
                        0.04093343,
                        0.04093343,
                        0.04093343,
                        0.04093343,
                        0.04093342,
                        0.04093283,
                        0.04092024,
                        0.04079976,
                        0.04022487,
                        0.03870567,
                        0.03627245,
                        0.03373241,
                        0.03192536,
                        0.03105929,
                        0.03082168,
                        0.03080456,
                        0.03084335,
                        0.03118516,
                        0.03224929,
                        0.03426685,
                        0.27468947,
                        0.45414631,
                        0.45572176,
                        0.45614793,
                        0.45622493,
                        0.4562317,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                    ],
                    [
                        0.07188444,
                        0.07188444,
                        0.07188444,
                        0.07188444,
                        0.07188444,
                        0.07188444,
                        0.07188445,
                        0.07188522,
                        0.0719184,
                        0.07248562,
                        0.07690389,
                        0.09455081,
                        0.13465147,
                        0.19131626,
                        0.2446856,
                        0.27949054,
                        0.29482228,
                        0.29856816,
                        0.29877046,
                        0.29787769,
                        0.29062145,
                        0.2681007,
                        0.22384602,
                        0.15968467,
                        0.0986633,
                        0.06765821,
                        0.05672074,
                        0.05459388,
                        0.0543912,
                        0.05438284,
                        0.05438271,
                        0.05438271,
                        0.05438271,
                        0.05438271,
                        0.05438271,
                        0.05438271,
                        0.05438271,
                        0.05438271,
                        0.05438271,
                        0.05438271,
                        0.05438271,
                        0.05438271,
                        0.05438271,
                    ],
                    [
                        0.28887675,
                        0.28887675,
                        0.28887675,
                        0.28887675,
                        0.28887675,
                        0.28887675,
                        0.28887675,
                        0.28887675,
                        0.28887675,
                        0.28390382,
                        0.26481179,
                        0.23200821,
                        0.19080431,
                        0.14871237,
                        0.11240365,
                        0.08553414,
                        0.06835537,
                        0.05883975,
                        0.05427016,
                        0.05236839,
                        0.05167181,
                        0.05125795,
                        0.04904247,
                        0.03597392,
                        0.01909977,
                        0.01863144,
                        0.01863138,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                        0.01863137,
                    ],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestRGB_to_sd_Gaussian:
    """
    Define :func:`colour.recovery.gaussian.RGB_to_sd_Gaussian`
    definition unit tests methods.
    """

    def test_RGB_to_sd_Gaussian(self) -> None:
        """
        Test :func:`colour.recovery.gaussian.RGB_to_sd_Gaussian`
        definition.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        RGB = XYZ_to_RGB_Gaussian(XYZ)

        sd = RGB_to_sd_Gaussian(RGB)

        assert sd.values.shape == (421,)

        # Test with interpolated basis at 10nm for full array comparison
        basis_10nm = MSDS_GAUSSIAN_BASIS.copy().align(SpectralShape(360, 780, 10))
        sd_10nm = RGB_to_sd_Smits1999(RGB, basis_10nm, "test")

        assert sd_10nm.values.shape == (43,)

        np.testing.assert_allclose(
            sd_10nm.values,
            np.array(
                [
                    0.04502017,
                    0.04502017,
                    0.04502017,
                    0.04502017,
                    0.04502017,
                    0.04502017,
                    0.04502017,
                    0.04502015,
                    0.04501943,
                    0.04500378,
                    0.04485416,
                    0.04414019,
                    0.04225346,
                    0.03923158,
                    0.03607704,
                    0.03383282,
                    0.03275723,
                    0.03246214,
                    0.03244087,
                    0.03248904,
                    0.03291355,
                    0.03423511,
                    0.03674078,
                    0.23102363,
                    0.37618614,
                    0.37801185,
                    0.37854111,
                    0.37863674,
                    0.37864515,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                    0.37864547,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_RGB_to_sd_Gaussian(self) -> None:
        """
        Test :func:`colour.recovery.gaussian.RGB_to_sd_Gaussian`
        definition domain and range scale support.
        """

        XYZ_i = np.array([0.20654008, 0.12197225, 0.05136952])
        RGB_i = XYZ_to_RGB_Gaussian(XYZ_i)
        XYZ_o = sd_to_XYZ_integration(RGB_to_sd_Gaussian(RGB_i))

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    sd_to_XYZ_integration(RGB_to_sd_Gaussian(RGB_i * factor_a)),
                    XYZ_o * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
