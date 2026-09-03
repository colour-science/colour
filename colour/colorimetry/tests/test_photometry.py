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
    "TestPhotometryAutograd",
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


class TestPhotometryAutograd:
    """
    Define autograd regression tests for the
    :mod:`colour.colorimetry.photometry` module under the *PyTorch* backend.

    An ordinary :class:`colour.SpectralDistribution` carrying backend values but
    a NumPy wavelength axis previously detached through
    :func:`colour.utilities.xp_trapezoid`, whose NumPy fallback dropped the
    autograd graph.
    """

    @pytest.mark.parametrize(
        "function",
        [luminous_flux, luminous_efficiency, luminous_efficacy],
        ids=lambda function: function.__name__,
    )
    def test_autograd_photometry(
        self, xp: ModuleType, function: Callable
    ) -> None:
        """Test that the definition preserves a finite gradient to spectral values."""

        if xp.__name__ != "torch":
            pytest.skip("Autograd preservation is only defined for *PyTorch*.")

        wavelengths = np.arange(360.0, 831.0, 1.0)
        values = xp.rand(wavelengths.size, requires_grad=True)
        sd = SpectralDistribution(values, wavelengths)

        # ``sd.wavelengths`` remains a NumPy axis while ``sd.values`` is a
        # backend tensor, the exact mixed-namespace case.
        result = function(sd)
        (gradient,) = xp.autograd.grad(xp.sum(result), values)

        assert result.grad_fn is not None
        assert xp.isfinite(gradient).all()
