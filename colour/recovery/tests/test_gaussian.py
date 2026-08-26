"""Define the unit tests for the :mod:`colour.recovery.gaussian` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

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
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    xp_as_array,
    xp_assert_close,
)

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
        assert 475 < peak_wavelengths["red"] < 790
        assert 400 < peak_wavelengths["green"] < 670
        assert 360 <= peak_wavelengths["blue"] < 510

        # Secondary colours
        assert 420 < peak_wavelengths["cyan"] < 700
        assert 400 < peak_wavelengths["magenta"] < 670  # valley at green
        assert 400 < peak_wavelengths["yellow"] < 670

        for colour in peak_wavelengths:
            assert 55 <= fwhm[colour] <= 240
            assert 2.0 <= exponent[colour] <= 5.0


class TestRGB_to_msds_Gaussian:
    """
    Define :func:`colour.recovery.gaussian.RGB_to_msds_Gaussian`
    definition unit tests methods.
    """

    def test_RGB_to_msds_Gaussian(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.recovery.gaussian.RGB_to_msds_Gaussian`
        definition.
        """

        RGB = xp_as_array(
            [
                [0.45623196, 0.03080455, 0.04093343],
                [0.05438271, 0.29877169, 0.07188444],
                [0.01863137, 0.05139773, 0.28887675],
            ],
            xp=xp,
        )

        msds = RGB_to_msds_Gaussian(RGB, as_array=True)

        assert msds.shape == (3, 421)

        xp_assert_close(
            msds[0, 0],
            0.04093343,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test with interpolated basis at 10nm for full array comparison
        basis_10nm = MSDS_GAUSSIAN_BASIS.copy().align(SpectralShape(360, 780, 10))
        msds_10nm = RGB_to_msds_Smits1999(RGB, basis_10nm, as_array=True)

        assert msds_10nm.shape == (3, 43)

        xp_assert_close(
            msds_10nm,
            [
                [
                    0.04093337,
                    0.04093320,
                    0.04093260,
                    0.04093072,
                    0.04092523,
                    0.04091063,
                    0.04087508,
                    0.04079585,
                    0.04063436,
                    0.04033363,
                    0.03982269,
                    0.03903235,
                    0.03792300,
                    0.03651767,
                    0.03492546,
                    0.03333915,
                    0.03200184,
                    0.03117438,
                    0.03124197,
                    0.03337885,
                    0.04158892,
                    0.06509127,
                    0.11770222,
                    0.20781604,
                    0.32145436,
                    0.41792699,
                    0.45505576,
                    0.45559176,
                    0.45591017,
                    0.45608272,
                    0.45616815,
                    0.45620682,
                    0.45622284,
                    0.45622892,
                    0.45623103,
                    0.45623170,
                    0.45623189,
                    0.45623194,
                    0.45623196,
                    0.45623196,
                    0.45623196,
                    0.45623196,
                    0.45623196,
                ],
                [
                    0.07188456,
                    0.07188506,
                    0.07188742,
                    0.07189721,
                    0.07193369,
                    0.07205532,
                    0.07241839,
                    0.07338861,
                    0.07570918,
                    0.08067403,
                    0.09016571,
                    0.10635006,
                    0.13088528,
                    0.16377607,
                    0.20239089,
                    0.24136455,
                    0.27383472,
                    0.29375824,
                    0.29866593,
                    0.28952972,
                    0.26550968,
                    0.23037931,
                    0.18714401,
                    0.14347913,
                    0.10818026,
                    0.08407715,
                    0.06945102,
                    0.06137903,
                    0.05733465,
                    0.05550949,
                    0.05477091,
                    0.05450325,
                    0.05441641,
                    0.05439118,
                    0.05438462,
                    0.05438310,
                    0.05438278,
                    0.05438272,
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
                    0.28514242,
                    0.26366776,
                    0.22802585,
                    0.18606289,
                    0.14547288,
                    0.11161469,
                    0.08671499,
                    0.07037699,
                    0.06074309,
                    0.05561408,
                    0.05314075,
                    0.05205795,
                    0.05149186,
                    0.04442883,
                    0.03265094,
                    0.02386852,
                    0.01997155,
                    0.01886579,
                    0.01865936,
                    0.01863365,
                    0.01863150,
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
                ],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestRGB_to_sd_Gaussian:
    """
    Define :func:`colour.recovery.gaussian.RGB_to_sd_Gaussian`
    definition unit tests methods.
    """

    def test_RGB_to_sd_Gaussian(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.recovery.gaussian.RGB_to_sd_Gaussian`
        definition.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        RGB = as_ndarray(XYZ_to_RGB_Gaussian(XYZ))

        sd = RGB_to_sd_Gaussian(RGB)

        assert sd.values.shape == (421,)

        # Test with interpolated basis at 10nm for full array comparison
        basis_10nm = MSDS_GAUSSIAN_BASIS.copy().align(SpectralShape(360, 780, 10))
        sd_10nm = RGB_to_sd_Smits1999(RGB, basis_10nm, "test")

        assert sd_10nm.values.shape == (43,)

        xp_assert_close(
            sd_10nm.values,
            [
                0.04502009,
                0.04501988,
                0.04501914,
                0.04501680,
                0.04500998,
                0.04499185,
                0.04494770,
                0.04484930,
                0.04464874,
                0.04427526,
                0.04364072,
                0.04265916,
                0.04128144,
                0.03953613,
                0.03755869,
                0.03558836,
                0.03392457,
                0.03287194,
                0.03279225,
                0.03461775,
                0.04157706,
                0.06103056,
                0.10398511,
                0.17707942,
                0.26899683,
                0.34699756,
                0.37718472,
                0.37785039,
                0.37824584,
                0.37846014,
                0.37856623,
                0.37861426,
                0.37863415,
                0.37864170,
                0.37864431,
                0.37864515,
                0.37864539,
                0.37864545,
                0.37864547,
                0.37864547,
                0.37864547,
                0.37864547,
                0.37864547,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_RGB_to_sd_Gaussian(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.recovery.gaussian.RGB_to_sd_Gaussian`
        definition domain and range scale support.
        """

        XYZ_i = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        RGB_i = xp_as_array(XYZ_to_RGB_Gaussian(XYZ_i), xp=xp)
        XYZ_o = sd_to_XYZ_integration(RGB_to_sd_Gaussian(RGB_i))

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    sd_to_XYZ_integration(
                        RGB_to_sd_Gaussian(RGB_i * xp_as_array(factor_a, xp=xp))
                    ),
                    XYZ_o * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
