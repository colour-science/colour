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
                        0.04093343,
                        0.04093342,
                        0.04093330,
                        0.04093246,
                        0.04092787,
                        0.04090783,
                        0.04083766,
                        0.04063902,
                        0.04018051,
                        0.03931132,
                        0.03794983,
                        0.03618163,
                        0.03428077,
                        0.03261045,
                        0.03145765,
                        0.03091240,
                        0.03091507,
                        0.03213843,
                        0.03931777,
                        0.06491856,
                        0.12491871,
                        0.22019200,
                        0.32514323,
                        0.40560803,
                        0.44603402,
                        0.45583131,
                        0.45614928,
                        0.45621039,
                        0.45622739,
                        0.45623118,
                        0.45623186,
                        0.45623195,
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
                        0.07188445,
                        0.07188462,
                        0.07188690,
                        0.07190832,
                        0.07205438,
                        0.07278951,
                        0.07556424,
                        0.08354052,
                        0.10125962,
                        0.13207694,
                        0.17446317,
                        0.22075721,
                        0.26055712,
                        0.28639625,
                        0.29730778,
                        0.29875234,
                        0.29504436,
                        0.27936571,
                        0.24704844,
                        0.20008929,
                        0.14912286,
                        0.10614786,
                        0.07766445,
                        0.06283291,
                        0.05680935,
                        0.05492314,
                        0.05447419,
                        0.05439425,
                        0.05438377,
                        0.05438278,
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
                        0.28887274,
                        0.28616448,
                        0.27502141,
                        0.25333555,
                        0.22212523,
                        0.18510319,
                        0.14748046,
                        0.11421109,
                        0.08845818,
                        0.07098557,
                        0.06060709,
                        0.05522371,
                        0.05279334,
                        0.05184200,
                        0.04916908,
                        0.04186991,
                        0.03275244,
                        0.02534683,
                        0.02109498,
                        0.01932061,
                        0.01877701,
                        0.01865442,
                        0.01863408,
                        0.01863161,
                        0.01863139,
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
                    0.04502016,
                    0.04502015,
                    0.04502000,
                    0.04501896,
                    0.04501326,
                    0.04498837,
                    0.04490123,
                    0.04465453,
                    0.04408510,
                    0.04300563,
                    0.04131476,
                    0.03911879,
                    0.03675806,
                    0.03468365,
                    0.03325189,
                    0.03257228,
                    0.03252966,
                    0.03357340,
                    0.03961249,
                    0.06071533,
                    0.10966834,
                    0.18704004,
                    0.27211051,
                    0.33732259,
                    0.37015513,
                    0.37820957,
                    0.37854280,
                    0.37861868,
                    0.37863980,
                    0.37864451,
                    0.37864534,
                    0.37864546,
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
