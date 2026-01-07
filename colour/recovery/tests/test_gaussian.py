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
                        0.04093342,
                        0.04083031,
                        0.04011251,
                        0.03881528,
                        0.03719562,
                        0.03553169,
                        0.03404924,
                        0.03289109,
                        0.03216467,
                        0.03214275,
                        0.03378312,
                        0.03975588,
                        0.05575516,
                        0.09070558,
                        0.15333904,
                        0.24403917,
                        0.34639329,
                        0.42802401,
                        0.45622390,
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
                        0.45623196,
                        0.45623196,
                    ],
                    [
                        0.07188682,
                        0.07189255,
                        0.07191024,
                        0.07196103,
                        0.07209651,
                        0.07243232,
                        0.07320499,
                        0.07485402,
                        0.07811464,
                        0.08407932,
                        0.09415077,
                        0.10910492,
                        0.13040224,
                        0.15890398,
                        0.19329086,
                        0.23001799,
                        0.26371237,
                        0.28807463,
                        0.29851996,
                        0.29335716,
                        0.27234737,
                        0.23989389,
                        0.20171341,
                        0.16355736,
                        0.12986401,
                        0.10307254,
                        0.08368588,
                        0.07083653,
                        0.06300245,
                        0.05859575,
                        0.05630391,
                        0.05520009,
                        0.05470716,
                        0.05450287,
                        0.05442423,
                        0.05439609,
                        0.05438674,
                        0.05438384,
                        0.05438301,
                        0.05438278,
                        0.05438273,
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
                        0.28680935,
                        0.26989895,
                        0.23801414,
                        0.19789614,
                        0.15758421,
                        0.12281883,
                        0.09647935,
                        0.07862265,
                        0.06682572,
                        0.05919330,
                        0.05469864,
                        0.04961559,
                        0.04420932,
                        0.03865340,
                        0.03337317,
                        0.02879875,
                        0.02518694,
                        0.02257899,
                        0.02085052,
                        0.01979568,
                        0.01920145,
                        0.01889185,
                        0.01874243,
                        0.01867556,
                        0.01864778,
                        0.01863705,
                        0.01863321,
                        0.01863192,
                        0.01863153,
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
                    0.04502015,
                    0.04489210,
                    0.04400064,
                    0.04238958,
                    0.04037806,
                    0.03831136,
                    0.03646870,
                    0.03502057,
                    0.03406722,
                    0.03381382,
                    0.03500919,
                    0.03979465,
                    0.05277835,
                    0.08120475,
                    0.13216747,
                    0.20597126,
                    0.28925717,
                    0.35567926,
                    0.37863546,
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
