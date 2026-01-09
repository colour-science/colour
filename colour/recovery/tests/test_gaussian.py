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
                        0.04093342,
                        0.04093339,
                        0.04093323,
                        0.04093255,
                        0.04093003,
                        0.04092166,
                        0.04089694,
                        0.04083193,
                        0.04067981,
                        0.04036356,
                        0.03978040,
                        0.03883019,
                        0.03747300,
                        0.03581055,
                        0.03419452,
                        0.03342317,
                        0.03516961,
                        0.04270913,
                        0.06155954,
                        0.09905171,
                        0.16055164,
                        0.24376093,
                        0.33477314,
                        0.41089882,
                        0.45121215,
                        0.45564371,
                        0.45596901,
                        0.45612625,
                        0.45619379,
                        0.45621959,
                        0.45622837,
                        0.45623103,
                        0.45623174,
                        0.45623191,
                        0.45623195,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                    ],
                    [
                        0.07188468,
                        0.07188547,
                        0.07188854,
                        0.07189952,
                        0.07193553,
                        0.07204380,
                        0.07234202,
                        0.07309397,
                        0.07482775,
                        0.07847803,
                        0.08548245,
                        0.09770092,
                        0.11700622,
                        0.14448552,
                        0.17942386,
                        0.21852697,
                        0.25597008,
                        0.28462366,
                        0.29821485,
                        0.29353371,
                        0.27067292,
                        0.23429183,
                        0.19199021,
                        0.15115702,
                        0.11695668,
                        0.09158408,
                        0.07472028,
                        0.06460815,
                        0.05911199,
                        0.05639522,
                        0.05517088,
                        0.05466685,
                        0.05447703,
                        0.05441154,
                        0.05439083,
                        0.05438482,
                        0.05438321,
                        0.05438282,
                        0.05438273,
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
                        0.28219223,
                        0.26103014,
                        0.22885373,
                        0.19118347,
                        0.15375271,
                        0.12100683,
                        0.09533529,
                        0.07712331,
                        0.06536248,
                        0.05842269,
                        0.05463344,
                        0.05079495,
                        0.04543130,
                        0.03913807,
                        0.03293312,
                        0.02768916,
                        0.02383114,
                        0.02133428,
                        0.01990281,
                        0.01917237,
                        0.01883953,
                        0.01870378,
                        0.01865414,
                        0.01863784,
                        0.01863303,
                        0.01863176,
                        0.01863145,
                        0.01863139,
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
                    0.04502016,
                    0.04502016,
                    0.04502011,
                    0.04501991,
                    0.04501907,
                    0.04501594,
                    0.04500555,
                    0.04497485,
                    0.04489411,
                    0.04470519,
                    0.04431242,
                    0.04358810,
                    0.04240741,
                    0.04071801,
                    0.03863221,
                    0.03652768,
                    0.03518988,
                    0.03610952,
                    0.04200427,
                    0.05729605,
                    0.08788588,
                    0.13800219,
                    0.20565314,
                    0.27952175,
                    0.34128296,
                    0.37409306,
                    0.37791491,
                    0.37831891,
                    0.37851419,
                    0.37859806,
                    0.37863011,
                    0.37864101,
                    0.37864431,
                    0.37864520,
                    0.37864542,
                    0.37864546,
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
