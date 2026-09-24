"""Define the unit tests for the :mod:`colour.colorimetry.photometry` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import Callable, ModuleType

import numpy as np
import pytest

from colour.colorimetry import (
    SDS_ILLUMINANTS,
    SDS_LIGHT_SOURCES,
    SpectralDistribution,
    luminous_efficacy,
    luminous_efficiency,
    luminous_flux,
    sd_zeros,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import xp_assert_close

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLuminousFlux",
    "TestLuminousEfficiency",
    "TestLuminousEfficacy",
    "TestPhotometryAutodiff",
]


class TestLuminousFlux:
    """
    Define :func:`colour.colorimetry.photometry.luminous_flux` definition unit
    tests methods.
    """

    def test_luminous_flux(self) -> None:
        """Test :func:`colour.colorimetry.photometry.luminous_flux` definition."""

        xp_assert_close(
            float(luminous_flux(SDS_ILLUMINANTS["FL2"].copy().normalise())),
            28588.73612977,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            float(luminous_flux(SDS_LIGHT_SOURCES["Neodimium Incandescent"])),
            23807.65552737,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            float(luminous_flux(SDS_LIGHT_SOURCES["F32T8/TL841 (Triphosphor)"])),
            13090.06759053,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestLuminousEfficiency:
    """
    Define :func:`colour.colorimetry.photometry.luminous_efficiency`
    definition unit tests methods.
    """

    def test_luminous_efficiency(self) -> None:
        """
        Test :func:`colour.colorimetry.photometry.luminous_efficiency`
        definition.
        """

        xp_assert_close(
            float(luminous_efficiency(SDS_ILLUMINANTS["FL2"].copy().normalise())),
            0.49317624,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            float(luminous_efficiency(SDS_LIGHT_SOURCES["Neodimium Incandescent"])),
            0.19943936,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            float(luminous_efficiency(SDS_LIGHT_SOURCES["F32T8/TL841 (Triphosphor)"])),
            0.51080919,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestLuminousEfficacy:
    """
    Define :func:`colour.colorimetry.photometry.luminous_efficacy`
    definition unit tests methods.
    """

    def test_luminous_efficacy(self) -> None:
        """
        Test :func:`colour.colorimetry.photometry.luminous_efficacy`
        definition.
        """

        xp_assert_close(
            float(luminous_efficacy(SDS_ILLUMINANTS["FL2"].copy().normalise())),
            336.83937176,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            float(luminous_efficacy(SDS_LIGHT_SOURCES["Neodimium Incandescent"])),
            136.21708032,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            float(luminous_efficacy(SDS_LIGHT_SOURCES["F32T8/TL841 (Triphosphor)"])),
            348.88267549,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        sd = sd_zeros()
        sd[555] = 1
        xp_assert_close(
            float(luminous_efficacy(sd)), 683.00000000, atol=TOLERANCE_ABSOLUTE_TESTS
        )


class TestPhotometryAutodiff:
    """
    Define automatic differentiation regression tests for the
    :mod:`colour.colorimetry.photometry` module.

    An ordinary :class:`colour.SpectralDistribution` carrying backend values but
    a NumPy wavelength axis previously detached through
    :func:`colour.utilities.xp_trapezoid`, whose NumPy fallback dropped the
    automatic differentiation graph.
    """

    @pytest.mark.parametrize(
        "function",
        [luminous_flux, luminous_efficiency, luminous_efficacy],
        ids=lambda function: function.__name__,
    )
    def test_autodiff_photometry(
        self, xp: ModuleType, autodiff: Callable, function: Callable
    ) -> None:
        """Test that a finite gradient reaches the spectral values."""

        wavelengths = np.arange(360.0, 831.0, 1.0)

        # ``sd.wavelengths`` remains a NumPy axis while ``sd.values`` is a
        # backend tensor, the exact mixed-namespace case.
        _result, (gradient,), _inputs = autodiff(
            lambda values: function(SpectralDistribution(values, wavelengths)),
            np.linspace(0.1, 1.0, wavelengths.size),
        )

        assert xp.isfinite(gradient).all()
        assert xp.any(gradient != 0)
