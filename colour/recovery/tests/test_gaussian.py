"""Define the unit tests for the :mod:`colour.recovery.gaussian` module."""

from __future__ import annotations

import numpy as np

from colour.colorimetry import sd_to_XYZ_integration
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.recovery import RGB_to_sd_Gaussian
from colour.recovery.gaussian import (
    XYZ_to_RGB_Gaussian,
    optimise_gaussian_basis_parameters,
)
from colour.utilities import domain_range_scale

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestOptimiseGaussianBasisParameters",
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

        sd = RGB_to_sd_Gaussian(
            XYZ_to_RGB_Gaussian(np.array([0.21781186, 0.12541048, 0.04697113]))
        )
        np.testing.assert_allclose(
            [sd[w] for w in [360, 400, 450, 500, 550, 600, 650, 700, 750, 780]],
            np.array(
                [
                    0.03982193,
                    0.03982193,
                    0.03971491,
                    0.03145787,
                    0.03554526,
                    0.30858755,
                    0.40639599,
                    0.40639599,
                    0.40639599,
                    0.40639599,
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
