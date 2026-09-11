"""Define the unit tests for the :mod:`colour.colorimetry.dominant` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.colorimetry import (
    CCS_ILLUMINANTS,
    MSDS_CMFS,
    colorimetric_purity,
    complementary_wavelength,
    dominant_wavelength,
    excitation_purity,
)
from colour.colorimetry.dominant import closest_spectral_locus_wavelength
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import XYZ_to_xy
from colour.utilities import (
    as_ndarray,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_assert_equal,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestClosestSpectralLocusWavelength",
    "TestDominantWavelength",
    "TestComplementaryWavelength",
    "TestExcitationPurity",
    "TestColorimetricPurity",
]


class TestClosestSpectralLocusWavelength:
    """
    Define :func:`colour.colorimetry.dominant.\
closest_spectral_locus_wavelength` definition unit tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._xy_s = XYZ_to_xy(MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].values)

        self._xy_D65 = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]

    def test_closest_spectral_locus_wavelength(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.dominant.\
closest_spectral_locus_wavelength` definition.
        """

        xy = xp_as_array([0.54369557, 0.32107944], xp=xp)
        xy_n = self._xy_D65
        i_wl, xy_wl = closest_spectral_locus_wavelength(xy, xy_n, self._xy_s)

        xp_assert_equal(i_wl, np.array(256))
        xp_assert_close(
            xy_wl,
            [0.68354746, 0.31628409],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xy = xp_as_array([0.37605506, 0.24452225], xp=xp)
        i_wl, xy_wl = closest_spectral_locus_wavelength(xy, xy_n, self._xy_s)

        xp_assert_equal(i_wl, np.array(248))
        xp_assert_close(
            xy_wl,
            [0.45723147, 0.13628148],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_closest_spectral_locus_wavelength(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.colorimetry.dominant.\
closest_spectral_locus_wavelength` definition n-dimensional arrays support.
        """

        xy = xp_as_array([0.54369557, 0.32107944], xp=xp)
        xy_n = self._xy_D65
        i_wl, xy_wl = closest_spectral_locus_wavelength(xy, xy_n, self._xy_s)
        i_wl_r, xy_wl_r = (
            xp_as_array(256, xp=xp),
            xp_as_array([0.68354746, 0.31628409], xp=xp),
        )
        xp_assert_close(i_wl, i_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        xy_n = xp.tile(xp_as_array(xy_n, xp=xp), (6, 1))
        i_wl, xy_wl = closest_spectral_locus_wavelength(xy, xy_n, self._xy_s)
        i_wl_r = xp.tile(xp_as_array(i_wl_r, xp=xp), (6,))
        xy_wl_r = xp.tile(xp_as_array(xy_wl_r, xp=xp), (6, 1))
        xp_assert_close(i_wl, i_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        xy_n = xp_reshape(xp_as_array(xy_n, xp=xp), (2, 3, 2), xp=xp)
        i_wl, xy_wl = closest_spectral_locus_wavelength(xy, xy_n, self._xy_s)
        i_wl_r = xp_reshape(xp_as_array(i_wl_r, xp=xp), (2, 3), xp=xp)
        xy_wl_r = xp_reshape(xp_as_array(xy_wl_r, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(i_wl, i_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_closest_spectral_locus_wavelength(self) -> None:
        """
        Test :func:`colour.colorimetry.dominant.\
closest_spectral_locus_wavelength` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            closest_spectral_locus_wavelength(case, case, self._xy_s)


class TestDominantWavelength:
    """
    Define :func:`colour.colorimetry.dominant.dominant_wavelength` definition
    unit tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._xy_D65 = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]

    def test_dominant_wavelength(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.dominant.dominant_wavelength`
        definition.
        """

        xy = xp_as_array([0.54369557, 0.32107944], xp=xp)
        xy_n = self._xy_D65
        wl, xy_wl, xy_cwl = dominant_wavelength(xy, xy_n)

        xp_assert_equal(wl, np.array(616.0))
        xp_assert_close(
            xy_wl,
            [0.68354746, 0.31628409],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            xy_cwl,
            [0.68354746, 0.31628409],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xy = xp_as_array([0.37605506, 0.24452225], xp=xp)
        i_wl, xy_wl, xy_cwl = dominant_wavelength(xy, xy_n)

        xp_assert_equal(i_wl, np.array(-509.0))
        xp_assert_close(
            xy_wl,
            [0.45723147, 0.13628148],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            xy_cwl,
            [0.01040962, 0.73207453],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_dominant_wavelength(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.dominant.dominant_wavelength`
        definition n-dimensional arrays support.
        """

        xy = xp_as_array([0.54369557, 0.32107944], xp=xp)
        xy_n = self._xy_D65
        wl, xy_wl, xy_cwl = dominant_wavelength(xy, xy_n)
        wl_r, xy_wl_r, xy_cwl_r = (
            np.array(616.0),
            np.array([0.68354746, 0.31628409]),
            np.array([0.68354746, 0.31628409]),
        )
        xp_assert_close(wl, wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_cwl, xy_cwl_r, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        xy_n = xp.tile(xp_as_array(xy_n, xp=xp), (6, 1))
        wl, xy_wl, xy_cwl = dominant_wavelength(xy, xy_n)
        wl_r = xp.tile(xp_as_array(wl_r, xp=xp), (6,))
        xy_wl_r = xp.tile(xp_as_array(xy_wl_r, xp=xp), (6, 1))
        xy_cwl_r = xp.tile(xp_as_array(xy_cwl_r, xp=xp), (6, 1))
        xp_assert_close(wl, wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_cwl, xy_cwl_r, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        xy_n = xp_reshape(xp_as_array(xy_n, xp=xp), (2, 3, 2), xp=xp)
        wl, xy_wl, xy_cwl = dominant_wavelength(xy, xy_n)
        wl_r = xp_reshape(xp_as_array(wl_r, xp=xp), (2, 3), xp=xp)
        xy_wl_r = xp_reshape(xp_as_array(xy_wl_r, xp=xp), (2, 3, 2), xp=xp)
        xy_cwl_r = xp_reshape(xp_as_array(xy_cwl_r, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(wl, wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_cwl, xy_cwl_r, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_dominant_wavelength(self) -> None:
        """
        Test :func:`colour.colorimetry.dominant.dominant_wavelength`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            dominant_wavelength(case, case)


class TestComplementaryWavelength:
    """
    Define :func:`colour.colorimetry.dominant.complementary_wavelength`
    definition unit tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._xy_D65 = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]

    def test_complementary_wavelength(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.dominant.complementary_wavelength`
        definition.
        """

        xy = xp_as_array([0.54369557, 0.32107944], xp=xp)
        xy_n = self._xy_D65
        wl, xy_wl, xy_cwl = complementary_wavelength(xy, xy_n)

        xp_assert_equal(wl, np.array(492.0))
        xp_assert_close(
            xy_wl,
            [0.03647950, 0.33847127],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            xy_cwl,
            [0.03647950, 0.33847127],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xy = xp_as_array([0.37605506, 0.24452225], xp=xp)
        i_wl, xy_wl, xy_cwl = complementary_wavelength(xy, xy_n)

        xp_assert_equal(i_wl, np.array(509.0))
        xp_assert_close(
            xy_wl,
            [0.01040962, 0.73207453],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            xy_cwl,
            [0.01040962, 0.73207453],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_complementary_wavelength(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.dominant.complementary_wavelength`
        definition n-dimensional arrays support.
        """

        xy = xp_as_array([0.54369557, 0.32107944], xp=xp)
        xy_n = self._xy_D65
        wl, xy_wl, xy_cwl = complementary_wavelength(xy, xy_n)
        wl_r, xy_wl_r, xy_cwl_r = (
            np.array(492.0),
            np.array([0.03647950, 0.33847127]),
            np.array([0.03647950, 0.33847127]),
        )
        xp_assert_close(wl, wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_cwl, xy_cwl_r, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        xy_n = xp.tile(xp_as_array(xy_n, xp=xp), (6, 1))
        wl, xy_wl, xy_cwl = complementary_wavelength(xy, xy_n)
        wl_r = xp.tile(xp_as_array(wl_r, xp=xp), (6,))
        xy_wl_r = xp.tile(xp_as_array(xy_wl_r, xp=xp), (6, 1))
        xy_cwl_r = xp.tile(xp_as_array(xy_cwl_r, xp=xp), (6, 1))
        xp_assert_close(wl, wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_cwl, xy_cwl_r, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        xy_n = xp_reshape(xp_as_array(xy_n, xp=xp), (2, 3, 2), xp=xp)
        wl, xy_wl, xy_cwl = complementary_wavelength(xy, xy_n)
        wl_r = xp_reshape(xp_as_array(wl_r, xp=xp), (2, 3), xp=xp)
        xy_wl_r = xp_reshape(xp_as_array(xy_wl_r, xp=xp), (2, 3, 2), xp=xp)
        xy_cwl_r = xp_reshape(xp_as_array(xy_cwl_r, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(wl, wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_wl, xy_wl_r, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(xy_cwl, xy_cwl_r, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_complementary_wavelength(self) -> None:
        """
        Test :func:`colour.colorimetry.dominant.complementary_wavelength`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            complementary_wavelength(case, case)


class TestExcitationPurity:
    """
    Define :func:`colour.colorimetry.dominant.excitation_purity` definition
    unit tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._xy_D65 = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]

    def test_excitation_purity(self, xp: ModuleType) -> None:
        """Test :func:`colour.colorimetry.dominant.excitation_purity` definition."""

        xy = xp_as_array([0.54369557, 0.32107944], xp=xp)
        xy_n = self._xy_D65

        xp_assert_close(
            excitation_purity(xy, xy_n),
            0.622885671878446,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xy = xp_as_array([0.37605506, 0.24452225], xp=xp)
        xp_assert_close(
            excitation_purity(xy, xy_n),
            0.438347859215887,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_excitation_purity(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.dominant.excitation_purity` definition
        n-dimensional arrays support.
        """

        xy = xp_as_array([0.54369557, 0.32107944], xp=xp)
        xy_n = self._xy_D65
        P_e = as_ndarray(excitation_purity(xy, xy_n))

        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        xy_n = xp.tile(xp_as_array(xy_n, xp=xp), (6, 1))
        P_e = xp.tile(xp_as_array(P_e, xp=xp), (6,))
        xp_assert_close(excitation_purity(xy, xy_n), P_e, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        xy_n = xp_reshape(xp_as_array(xy_n, xp=xp), (2, 3, 2), xp=xp)
        P_e = xp_reshape(xp_as_array(P_e, xp=xp), (2, 3), xp=xp)
        xp_assert_close(excitation_purity(xy, xy_n), P_e, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_excitation_purity_autodiff(
        self, xp: ModuleType, autodiff: typing.Callable
    ) -> None:
        """Test local numerical gradients on both locus-selection branches."""

        for xy in ([0.54369557, 0.32107944], [0.37605506, 0.24452225]):
            _P_e, (gradient_xy, gradient_xy_n), _inputs = autodiff(
                excitation_purity, xy, self._xy_D65
            )

            assert xp.isfinite(gradient_xy).all()
            assert xp.isfinite(gradient_xy_n).all()
            assert xp.any(gradient_xy != 0)
            assert xp.any(gradient_xy_n != 0)

    @ignore_numpy_errors
    def test_nan_excitation_purity(self) -> None:
        """
        Test :func:`colour.colorimetry.dominant.excitation_purity` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            excitation_purity(case, case)


class TestColorimetricPurity:
    """
    Define :func:`colour.colorimetry.dominant.colorimetric_purity` definition
    unit tests methods.
    """

    def setup_method(self) -> None:
        """Initialise the common tests attributes."""

        self._xy_D65 = CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"]

    def test_colorimetric_purity(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.dominant.colorimetric_purity`
        definition.
        """

        xy = xp_as_array([0.54369557, 0.32107944], xp=xp)
        xy_n = self._xy_D65

        xp_assert_close(
            colorimetric_purity(xy, xy_n),
            0.613582813175483,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xy = xp_as_array([0.37605506, 0.24452225], xp=xp)
        xp_assert_close(
            colorimetric_purity(xy, xy_n),
            0.244307811178847,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_colorimetric_purity(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.dominant.colorimetric_purity`
        definition n-dimensional arrays support.
        """

        xy = xp_as_array([0.54369557, 0.32107944], xp=xp)
        xy_n = self._xy_D65
        P_e = as_ndarray(colorimetric_purity(xy, xy_n))

        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        xy_n = xp.tile(xp_as_array(xy_n, xp=xp), (6, 1))
        P_e = xp.tile(xp_as_array(P_e, xp=xp), (6,))
        xp_assert_close(
            colorimetric_purity(xy, xy_n),
            P_e,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        xy_n = xp_reshape(xp_as_array(xy_n, xp=xp), (2, 3, 2), xp=xp)
        P_e = xp_reshape(xp_as_array(P_e, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            colorimetric_purity(xy, xy_n),
            P_e,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_colorimetric_purity_autodiff(
        self, xp: ModuleType, autodiff: typing.Callable
    ) -> None:
        """Test local numerical gradients on both locus-selection branches."""

        for xy in ([0.54369557, 0.32107944], [0.37605506, 0.24452225]):
            _P_c, (gradient_xy, gradient_xy_n), _inputs = autodiff(
                colorimetric_purity, xy, self._xy_D65
            )

            assert xp.isfinite(gradient_xy).all()
            assert xp.isfinite(gradient_xy_n).all()
            assert xp.any(gradient_xy != 0)
            assert xp.any(gradient_xy_n != 0)

    @ignore_numpy_errors
    def test_nan_colorimetric_purity(self) -> None:
        """
        Test :func:`colour.colorimetry.dominant.colorimetric_purity`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            colorimetric_purity(case, case)
