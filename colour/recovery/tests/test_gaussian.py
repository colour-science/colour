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

        peak_wavelengths, fwhm = optimise_gaussian_basis_parameters()

        assert set(peak_wavelengths.keys()) == {"red", "green", "blue"}
        assert set(fwhm.keys()) == {"red", "green", "blue"}

        assert 580 < peak_wavelengths["red"] < 650
        assert 510 < peak_wavelengths["green"] < 570
        assert 420 < peak_wavelengths["blue"] < 480

        assert 20 < fwhm["red"] < 150
        assert 20 < fwhm["green"] < 150
        assert 20 < fwhm["blue"] < 150


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
                        0.04093343,
                        0.04093343,
                        0.04084525,
                        0.04012429,
                        0.03882177,
                        0.03719640,
                        0.03552793,
                        0.03404295,
                        0.03288422,
                        0.03215853,
                        0.03213766,
                        0.03377829,
                        0.03974954,
                        0.05574525,
                        0.09069168,
                        0.15332485,
                        0.24403290,
                        0.34640307,
                        0.42805033,
                        0.45623205,
                        0.45623198,
                        0.45623197,
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
                        0.07188700,
                        0.07189318,
                        0.07191223,
                        0.07196693,
                        0.07211287,
                        0.07247458,
                        0.07330686,
                        0.07508309,
                        0.07859523,
                        0.08486765,
                        0.09447427,
                        0.10910571,
                        0.13039297,
                        0.15888928,
                        0.19327721,
                        0.23001130,
                        0.26371592,
                        0.28838793,
                        0.29907506,
                        0.29340939,
                        0.27236240,
                        0.23990137,
                        0.20171291,
                        0.16355061,
                        0.12985387,
                        0.10306177,
                        0.08367645,
                        0.07082935,
                        0.06299757,
                        0.05859277,
                        0.05630225,
                        0.05519924,
                        0.05470676,
                        0.05450270,
                        0.05442416,
                        0.05439607,
                        0.05438673,
                        0.05438384,
                        0.05438300,
                        0.05438278,
                        0.05438273,
                        0.05438271,
                        0.05438271,
                    ],
                    [
                        0.28887709,
                        0.28887792,
                        0.28888048,
                        0.28888781,
                        0.28890738,
                        0.28895587,
                        0.28906746,
                        0.28930561,
                        0.28977650,
                        0.28828524,
                        0.27050461,
                        0.23801562,
                        0.19787879,
                        0.15755668,
                        0.12279327,
                        0.09646683,
                        0.07862931,
                        0.06741228,
                        0.06023255,
                        0.05479642,
                        0.04964374,
                        0.04422333,
                        0.03865246,
                        0.03336054,
                        0.02877976,
                        0.02516679,
                        0.02256133,
                        0.02083706,
                        0.01978655,
                        0.01919586,
                        0.01888874,
                        0.01874085,
                        0.01867482,
                        0.01864746,
                        0.01863693,
                        0.01863316,
                        0.01863191,
                        0.01863152,
                        0.01863141,
                        0.01863138,
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
                    0.04502017,
                    0.04502017,
                    0.04491066,
                    0.04401528,
                    0.04239764,
                    0.04037903,
                    0.03830670,
                    0.03646088,
                    0.03501204,
                    0.03405960,
                    0.03380749,
                    0.03500319,
                    0.03978677,
                    0.05276604,
                    0.08118748,
                    0.13214985,
                    0.20596348,
                    0.28926932,
                    0.35571195,
                    0.37864559,
                    0.37864550,
                    0.37864548,
                    0.37864548,
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
