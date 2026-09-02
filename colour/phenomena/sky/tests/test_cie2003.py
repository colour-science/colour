"""Define the unit tests for the :mod:`colour.phenomena.sky.cie2003` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.phenomena.sky import (
    sky_luminance_distribution_CIE2003,
    sky_luminance_distribution_overcast_CIE2003,
    sky_luminance_gradation_CIE2003,
    sky_scattering_indicatrix_CIE2003,
)
from colour.utilities import (
    as_ndarray,
    ignore_numpy_errors,
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
    "TestSkyLuminanceGradationCIE2003",
    "TestSkyScatteringIndicatrixCIE2003",
    "TestSkyLuminanceDistributionCIE2003",
    "TestSkyLuminanceDistributionOvercastCIE2003",
]


class TestSkyLuminanceGradationCIE2003:
    """
    Define :func:`colour.phenomena.sky.sky_luminance_gradation_CIE2003`
    definition unit tests methods.
    """

    def test_sky_luminance_gradation_CIE2003(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.sky.\
sky_luminance_gradation_CIE2003` definition.
        """

        # Type 1: a=4.0, b=-0.70. At zenith (Z=0): 1 + 4*exp(-0.70).
        xp_assert_close(
            sky_luminance_gradation_CIE2003(xp_as_array([0.0], xp=xp), 4.0, -0.70),
            1 + 4.0 * np.exp(-0.70),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # At horizon (Z=pi/2): always 1.
        xp_assert_close(
            sky_luminance_gradation_CIE2003(
                xp_as_array([np.pi / 2], xp=xp), 4.0, -0.70
            ),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Type 5: a=0, b=-1.0. Uniform: phi(Z) = 1 for all Z.
        xp_assert_close(
            sky_luminance_gradation_CIE2003(xp_as_array([0.0], xp=xp), 0, -1.0),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sky_luminance_gradation_CIE2003(
                xp_as_array([np.radians(45)], xp=xp), 0, -1.0
            ),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_sky_luminance_gradation_CIE2003(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.phenomena.sky.\
sky_luminance_gradation_CIE2003` definition n-dimensional support.
        """

        Z = xp_as_array([0, np.radians(30), np.radians(60)], xp=xp)
        result = sky_luminance_gradation_CIE2003(Z, 4.0, -0.70)

        assert as_ndarray(result).shape == (3,)

    @ignore_numpy_errors
    def test_nan_sky_luminance_gradation_CIE2003(self) -> None:
        """
        Test :func:`colour.phenomena.sky.\
sky_luminance_gradation_CIE2003` definition nan support.
        """

        sky_luminance_gradation_CIE2003(np.nan, np.nan, np.nan)


class TestSkyScatteringIndicatrixCIE2003:
    """
    Define :func:`colour.phenomena.sky.sky_scattering_indicatrix_CIE2003`
    definition unit tests methods.
    """

    def test_sky_scattering_indicatrix_CIE2003(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.sky.\
sky_scattering_indicatrix_CIE2003` definition.
        """

        # When c=0, e=0: f(chi) = 1 for any chi.
        xp_assert_close(
            sky_scattering_indicatrix_CIE2003(
                xp_as_array([np.radians(45)], xp=xp), 0, -1.0, 0
            ),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # At chi=0 (looking at sun): f(0) = 1 + c*(1 - exp(d*pi/2)) + e.
        xp_assert_close(
            sky_scattering_indicatrix_CIE2003(
                xp_as_array([0.0], xp=xp), 10, -3.0, 0.45
            ),
            1 + 10 * (1 - np.exp(-3.0 * np.pi / 2)) + 0.45,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            sky_scattering_indicatrix_CIE2003(
                xp_as_array([np.radians(30)], xp=xp), 10, -3.0, 0.45
            ),
            3.32646285329632,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_sky_scattering_indicatrix_CIE2003(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.phenomena.sky.\
sky_scattering_indicatrix_CIE2003` definition n-dimensional support.
        """

        chi = xp_as_array([0, np.radians(45), np.radians(90)], xp=xp)
        result = sky_scattering_indicatrix_CIE2003(chi, 10, -3.0, 0.45)

        assert as_ndarray(result).shape == (3,)

    @ignore_numpy_errors
    def test_nan_sky_scattering_indicatrix_CIE2003(self) -> None:
        """
        Test :func:`colour.phenomena.sky.\
sky_scattering_indicatrix_CIE2003` definition nan support.
        """

        sky_scattering_indicatrix_CIE2003(np.nan, np.nan, np.nan, np.nan)


class TestSkyLuminanceDistributionCIE2003:
    """
    Define :func:`colour.phenomena.sky.\
sky_luminance_distribution_CIE2003` definition unit tests methods.
    """

    def test_sky_luminance_distribution_CIE2003(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.sky.\
sky_luminance_distribution_CIE2003` definition.
        """

        # Type 1 (CIE Standard Overcast Sky): at zenith, L/Lz = 1.
        xp_assert_close(
            sky_luminance_distribution_CIE2003(
                1,
                xp_as_array([0.0], xp=xp),
                xp_as_array([0.0], xp=xp),
                np.radians(30),
                np.radians(0),
            ),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Type 5 (Uniform sky): L/Lz = 1 everywhere.
        xp_assert_close(
            sky_luminance_distribution_CIE2003(
                5,
                xp_as_array([np.radians(45)], xp=xp),
                xp_as_array([np.radians(90)], xp=xp),
                np.radians(30),
                np.radians(0),
            ),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Type 1: overcast sky at specific position.
        xp_assert_close(
            sky_luminance_distribution_CIE2003(
                1,
                xp_as_array([np.radians(45)], xp=xp),
                xp_as_array([np.radians(180)], xp=xp),
                np.radians(30),
                np.radians(0),
            ),
            0.83258464279644,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Type 12: clear sky at specific position.
        xp_assert_close(
            sky_luminance_distribution_CIE2003(
                12,
                xp_as_array([np.radians(45)], xp=xp),
                xp_as_array([np.radians(180)], xp=xp),
                np.radians(30),
                np.radians(0),
            ),
            0.45445243068520,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_sky_type_validation(self) -> None:
        """
        Test :func:`colour.phenomena.sky.\
sky_luminance_distribution_CIE2003` definition sky type validation.
        """

        with pytest.raises(ValueError):
            sky_luminance_distribution_CIE2003(0, 0, 0, 0, 0)

        with pytest.raises(ValueError):
            sky_luminance_distribution_CIE2003(16, 0, 0, 0, 0)

    def test_n_dimensional_sky_luminance_distribution_CIE2003(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.phenomena.sky.\
sky_luminance_distribution_CIE2003` definition n-dimensional support.
        """

        Z = xp_as_array([0, np.radians(30), np.radians(60)], xp=xp)
        alpha = xp_as_array([0, np.radians(90), np.radians(180)], xp=xp)
        result = sky_luminance_distribution_CIE2003(
            1, Z, alpha, np.radians(30), np.radians(0)
        )

        assert as_ndarray(result).shape == (3,)

    @ignore_numpy_errors
    def test_nan_sky_luminance_distribution_CIE2003(self) -> None:
        """
        Test :func:`colour.phenomena.sky.\
sky_luminance_distribution_CIE2003` definition nan support.
        """

        sky_luminance_distribution_CIE2003(1, np.nan, np.nan, np.nan, np.nan)


class TestSkyLuminanceDistributionOvercastCIE2003:
    """
    Define :func:`colour.phenomena.sky.\
sky_luminance_distribution_overcast_CIE2003` definition unit tests methods.
    """

    def test_sky_luminance_distribution_overcast_CIE2003(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.sky.\
sky_luminance_distribution_overcast_CIE2003` definition.
        """

        # At zenith (Z=0, gamma=pi/2): (1 + 2*1) / 3 = 1.
        xp_assert_close(
            sky_luminance_distribution_overcast_CIE2003(xp_as_array([0.0], xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # At horizon (Z=pi/2, gamma=0): (1 + 0) / 3 = 1/3.
        xp_assert_close(
            sky_luminance_distribution_overcast_CIE2003(
                xp_as_array([np.pi / 2], xp=xp)
            ),
            1 / 3,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # At 45 degrees elevation (Z=pi/4, gamma=pi/4).
        xp_assert_close(
            sky_luminance_distribution_overcast_CIE2003(
                xp_as_array([np.pi / 4], xp=xp)
            ),
            (1 + 2 * np.sin(np.pi / 4)) / 3,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_sky_luminance_distribution_overcast_CIE2003(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.phenomena.sky.\
sky_luminance_distribution_overcast_CIE2003` definition n-dimensional \
support.
        """

        Z = xp_as_array([0, np.radians(30), np.radians(60)], xp=xp)
        result = sky_luminance_distribution_overcast_CIE2003(Z)

        assert as_ndarray(result).shape == (3,)

    @ignore_numpy_errors
    def test_nan_sky_luminance_distribution_overcast_CIE2003(self) -> None:
        """
        Test :func:`colour.phenomena.sky.\
sky_luminance_distribution_overcast_CIE2003` definition nan support.
        """

        sky_luminance_distribution_overcast_CIE2003(np.nan)
