"""Define the unit tests for the :mod:`colour.colorimetry.photometry` module."""


import numpy as np

from colour.colorimetry import (
    SDS_ILLUMINANTS,
    SDS_LIGHT_SOURCES,
    luminous_efficacy,
    luminous_efficiency,
    luminous_flux,
    sd_zeros,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS

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
]


class TestLuminousFlux:
    """
    Define :func:`colour.colorimetry.photometry.luminous_flux` definition unit
    tests methods.
    """

    def test_luminous_flux(self):
        """Test :func:`colour.colorimetry.photometry.luminous_flux` definition."""

        np.testing.assert_allclose(
            luminous_flux(SDS_ILLUMINANTS["FL2"].copy().normalise()),
            28588.73612977,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            luminous_flux(SDS_LIGHT_SOURCES["Neodimium Incandescent"]),
            23807.65552737,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            luminous_flux(SDS_LIGHT_SOURCES["F32T8/TL841 (Triphosphor)"]),
            13090.06759053,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestLuminousEfficiency:
    """
    Define :func:`colour.colorimetry.photometry.luminous_efficiency`
    definition unit tests methods.
    """

    def test_luminous_efficiency(self):
        """
        Test :func:`colour.colorimetry.photometry.luminous_efficiency`
        definition.
        """

        np.testing.assert_allclose(
            luminous_efficiency(SDS_ILLUMINANTS["FL2"].copy().normalise()),
            0.49317624,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            luminous_efficiency(SDS_LIGHT_SOURCES["Neodimium Incandescent"]),
            0.19943936,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            luminous_efficiency(SDS_LIGHT_SOURCES["F32T8/TL841 (Triphosphor)"]),
            0.51080919,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestLuminousEfficacy:
    """
    Define :func:`colour.colorimetry.photometry.luminous_efficacy`
    definition unit tests methods.
    """

    def test_luminous_efficacy(self):
        """
        Test :func:`colour.colorimetry.photometry.luminous_efficacy`
        definition.
        """

        np.testing.assert_allclose(
            luminous_efficacy(SDS_ILLUMINANTS["FL2"].copy().normalise()),
            336.83937176,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            luminous_efficacy(SDS_LIGHT_SOURCES["Neodimium Incandescent"]),
            136.21708032,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            luminous_efficacy(SDS_LIGHT_SOURCES["F32T8/TL841 (Triphosphor)"]),
            348.88267549,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        sd = sd_zeros()
        sd[555] = 1
        np.testing.assert_allclose(
            luminous_efficacy(sd), 683.00000000, atol=TOLERANCE_ABSOLUTE_TESTS
        )
