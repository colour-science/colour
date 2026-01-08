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

        # Primary colours
        assert 580 < peak_wavelengths["red"] < 650
        assert 510 < peak_wavelengths["green"] < 570
        assert 420 < peak_wavelengths["blue"] < 480

        # Secondary colours
        assert 460 < peak_wavelengths["cyan"] < 540  # below green
        assert 510 < peak_wavelengths["magenta"] < 570  # valley at green
        assert 550 < peak_wavelengths["yellow"] < 620  # above green

        for colour in peak_wavelengths:
            assert 20 < fwhm[colour] < 150


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
            0.040933350852,
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
                        0.04093335,
                        0.04093315,
                        0.04093252,
                        0.04093067,
                        0.04092560,
                        0.04091273,
                        0.04088247,
                        0.04081655,
                        0.04068367,
                        0.04043617,
                        0.04001108,
                        0.03933950,
                        0.03836715,
                        0.03708402,
                        0.03555432,
                        0.03393410,
                        0.03247530,
                        0.03157108,
                        0.03204886,
                        0.03613717,
                        0.04937096,
                        0.08213168,
                        0.14663226,
                        0.24548197,
                        0.35754213,
                        0.43883821,
                        0.45503720,
                        0.45556666,
                        0.45588681,
                        0.45606514,
                        0.45615684,
                        0.45620044,
                        0.45621964,
                        0.45622747,
                        0.45623044,
                        0.45623148,
                        0.45623182,
                        0.45623192,
                        0.45623195,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                        0.45623196,
                    ],
                    [
                        0.07188620,
                        0.07189063,
                        0.07190468,
                        0.07194608,
                        0.07205934,
                        0.07234675,
                        0.07302295,
                        0.07449660,
                        0.07746803,
                        0.08300385,
                        0.09251465,
                        0.10754426,
                        0.12931040,
                        0.15804153,
                        0.19231242,
                        0.22871022,
                        0.26215042,
                        0.28694386,
                        0.29794488,
                        0.29255845,
                        0.27179605,
                        0.23962099,
                        0.20169858,
                        0.16375987,
                        0.13022686,
                        0.10352913,
                        0.08416924,
                        0.07128968,
                        0.06338666,
                        0.05889359,
                        0.05651646,
                        0.05534045,
                        0.05479330,
                        0.05455218,
                        0.05445066,
                        0.05440941,
                        0.05439306,
                        0.05438668,
                        0.05438421,
                        0.05438327,
                        0.05438291,
                        0.05438278,
                        0.05438273,
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
                        0.28887675,
                        0.27696250,
                        0.24235571,
                        0.19547944,
                        0.14828895,
                        0.10946943,
                        0.08241825,
                        0.06616629,
                        0.05766433,
                        0.05300206,
                        0.04936830,
                        0.04570134,
                        0.04172851,
                        0.03758792,
                        0.03355567,
                        0.02989157,
                        0.02677058,
                        0.02426729,
                        0.02236973,
                        0.02100674,
                        0.02007719,
                        0.01947437,
                        0.01910221,
                        0.01888328,
                        0.01876048,
                        0.01869476,
                        0.01866118,
                        0.01864480,
                        0.01863717,
                        0.01863377,
                        0.01863232,
                        0.01863173,
                        0.01863150,
                        0.01863142,
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
                    0.04502007,
                    0.04501982,
                    0.04501904,
                    0.04501674,
                    0.04501044,
                    0.04499446,
                    0.04495688,
                    0.04487501,
                    0.04470998,
                    0.04440261,
                    0.04387468,
                    0.04304062,
                    0.04183304,
                    0.04023947,
                    0.03833948,
                    0.03632532,
                    0.03449895,
                    0.03328757,
                    0.03344854,
                    0.03681723,
                    0.04782283,
                    0.07473123,
                    0.12724675,
                    0.20735718,
                    0.29799751,
                    0.36379568,
                    0.37716167,
                    0.37781923,
                    0.37821682,
                    0.37843829,
                    0.37855218,
                    0.37860633,
                    0.37863017,
                    0.37863990,
                    0.37864358,
                    0.37864488,
                    0.37864530,
                    0.37864543,
                    0.37864546,
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
