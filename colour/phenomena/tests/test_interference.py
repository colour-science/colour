"""Define the unit tests for the :mod:`colour.phenomena.interference` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.phenomena.interference import (
    light_water_molar_refraction_Schiebener1990,
    light_water_refractive_index_Schiebener1990,
    multilayer_tmm,
    thin_film_tmm,
)
from colour.utilities import (
    as_ndarray,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLightWaterMolarRefractionSchiebener1990",
    "TestLightWaterRefractiveIndexSchiebener1990",
    "TestThinFilmTmm",
    "TestMultilayerTmm",
]


class TestLightWaterMolarRefractionSchiebener1990:
    """
    Define :func:`colour.phenomena.interference.\
light_water_molar_refraction_Schiebener1990` definition unit tests methods.
    """

    def test_light_water_molar_refraction_Schiebener1990(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.interference.\
light_water_molar_refraction_Schiebener1990` definition.
        """

        xp_assert_close(
            light_water_molar_refraction_Schiebener1990(xp_as_array([589.0], xp=xp)),
            0.206211470522,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            light_water_molar_refraction_Schiebener1990(
                xp_as_array([400.0], xp=xp), 300, 1000
            ),
            0.211842881763,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            light_water_molar_refraction_Schiebener1990(
                xp_as_array([700.0], xp=xp), 280, 998
            ),
            0.204829756928,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_light_water_molar_refraction_Schiebener1990(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.phenomena.interference.\
light_water_molar_refraction_Schiebener1990` definition n-dimensional arrays support.
        """

        wl = 589
        LL = light_water_molar_refraction_Schiebener1990(wl)

        wl = xp.tile(xp_as_array(wl, xp=xp), (6,))
        LL = xp.tile(xp_as_array(LL, xp=xp), (6,))
        xp_assert_close(
            light_water_molar_refraction_Schiebener1990(wl),
            LL,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3), xp=xp)
        LL = xp_reshape(xp_as_array(LL, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            light_water_molar_refraction_Schiebener1990(wl),
            LL,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3, 1), xp=xp)
        LL = xp_reshape(xp_as_array(LL, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            light_water_molar_refraction_Schiebener1990(wl),
            LL,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_light_water_molar_refraction_Schiebener1990(self) -> None:
        """
        Test :func:`colour.phenomena.interference.\
light_water_molar_refraction_Schiebener1990` definition nan support.
        """

        light_water_molar_refraction_Schiebener1990(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLightWaterRefractiveIndexSchiebener1990:
    """
    Define :func:`colour.phenomena.interference.\
light_water_refractive_index_Schiebener1990` definition unit tests methods.
    """

    def test_light_water_refractive_index_Schiebener1990(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.interference.\
light_water_refractive_index_Schiebener1990` definition.
        """

        xp_assert_close(
            light_water_refractive_index_Schiebener1990(xp_as_array([400.0], xp=xp)),
            1.344143366618,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            light_water_refractive_index_Schiebener1990(xp_as_array([500.0], xp=xp)),
            1.337363795367,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            light_water_refractive_index_Schiebener1990(xp_as_array([600.0], xp=xp)),
            1.333585122179,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_light_water_refractive_index_Schiebener1990(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.phenomena.interference.\
light_water_refractive_index_Schiebener1990` definition n-dimensional arrays support.
        """

        wl = 400
        n = light_water_refractive_index_Schiebener1990(wl)

        wl = xp.tile(xp_as_array(wl, xp=xp), (6,))
        n = xp.tile(xp_as_array(n, xp=xp), (6,))
        xp_assert_close(
            light_water_refractive_index_Schiebener1990(wl),
            n,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3), xp=xp)
        n = xp_reshape(xp_as_array(n, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            light_water_refractive_index_Schiebener1990(wl),
            n,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3, 1), xp=xp)
        n = xp_reshape(xp_as_array(n, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            light_water_refractive_index_Schiebener1990(wl),
            n,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_light_water_refractive_index_Schiebener1990(self) -> None:
        """
        Test :func:`colour.phenomena.interference.\
light_water_refractive_index_Schiebener1990` definition nan support.
        """

        light_water_refractive_index_Schiebener1990(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestThinFilmTmm:
    """
    Define :func:`colour.phenomena.interference.thin_film_tmm`
    definition unit tests methods.
    """

    def test_thin_film_tmm(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.interference.thin_film_tmm`
        definition.
        """

        # Test single wavelength - returns (R, T) tuple
        R, T = thin_film_tmm([1.0, 1.5, 1.0], 250, xp_as_array(500.0, xp=xp))
        R, T = as_ndarray(R), as_ndarray(T)
        assert R.shape == (1, 1, 1, 2)  # (W, A, T, 2) - [R_s, R_p]
        assert T.shape == (1, 1, 1, 2)  # (W, A, T, 2) - [T_s, T_p]
        assert np.all((R >= 0) & (R <= 1))
        assert np.all((T >= 0) & (T <= 1))

        # Test energy conservation
        xp_assert_close(R + T, 1.0, atol=TOLERANCE_ABSOLUTE_TESTS * 10)

        # Test multiple wavelengths
        R, T = thin_film_tmm([1.0, 1.5, 1.0], 250, xp_as_array([400, 500, 600], xp=xp))
        R, T = as_ndarray(R), as_ndarray(T)
        assert R.shape == (3, 1, 1, 2)  # (W=3, A=1, T=1, 2) - Spectroscopy Convention
        assert T.shape == (3, 1, 1, 2)
        assert np.all((R >= 0) & (R <= 1))
        assert np.all((T >= 0) & (T <= 1))

        # Test that s and p polarisations are similar at normal incidence
        R_normal, _ = thin_film_tmm(
            [1.0, 1.5, 1.0], 250, xp_as_array(500.0, xp=xp), theta=0
        )
        R_normal = as_ndarray(R_normal)
        xp_assert_close(
            R_normal[0, 0, 0, 0],
            R_normal[0, 0, 0, 1],
            atol=TOLERANCE_ABSOLUTE_TESTS * 0.001,
        )

    def test_n_dimensional_thin_film_tmm(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.interference.thin_film_tmm`
        definition n-dimensional arrays support.
        """

        wl = 555
        R, T = thin_film_tmm([1.0, 1.5, 1.0], 250, wl)

        wl = xp.tile(xp_as_array(wl, xp=xp), (6,))
        R = xp.tile(xp_as_array(R, xp=xp), (6, 1, 1, 1))
        T = xp.tile(xp_as_array(T, xp=xp), (6, 1, 1, 1))
        R_array, T_array = thin_film_tmm([1.0, 1.5, 1.0], 250, wl)
        xp_assert_close(R_array, R, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(T_array, T, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_thin_film_tmm(self) -> None:
        """
        Test :func:`colour.phenomena.interference.thin_film_tmm`
        definition nan support.
        """

        thin_film_tmm(
            [1.0, 1.5, 1.0],
            250,
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]),
        )

    def test_thin_film_tmm_complex_n(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.interference.thin_film_tmm`
        with complex refractive indices (absorbing layers).
        """

        # Absorbing layer: n = 2.0 + 0.5j
        n_absorbing = 2.0 + 0.5j
        R, T = thin_film_tmm([1.0, n_absorbing, 1.0], 250, xp_as_array([500.0], xp=xp))

        assert R.shape == (1, 1, 1, 2)
        assert T.shape == (1, 1, 1, 2)
        assert np.all((as_ndarray(R) >= 0) & (as_ndarray(R) <= 1))
        assert np.all((as_ndarray(T) >= 0) & (as_ndarray(T) <= 1))

        # For absorbing media: R + T < 1 (absorption A = 1 - R - T > 0)
        R_avg = np.mean(as_ndarray(R))
        T_avg = np.mean(as_ndarray(T))
        A = 1 - R_avg - T_avg
        assert A > 0, f"Expected absorption > 0, got A = {A}"

        # Silver mirror: n ≈ 0.18 + 3.15j at 500nm
        n_silver = 0.18 + 3.15j
        R_silver, _ = thin_film_tmm(
            [1.0, n_silver, 1.0], 50, xp_as_array([500.0], xp=xp)
        )

        # Silver should have high reflectance
        assert np.mean(as_ndarray(R_silver)) > 0.5


class TestMultilayerTmm:
    """
    Define :func:`colour.phenomena.interference.multilayer_tmm`
    definition unit tests methods.
    """

    def test_multilayer_tmm(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.interference.multilayer_tmm`
        definition.
        """

        # Test single layer (should match thin_film_tmm)
        R_multi, T_multi = multilayer_tmm(
            [1.0, 1.5, 1.0], [250], xp_as_array(500.0, xp=xp)
        )
        R_single, T_single = thin_film_tmm(
            [1.0, 1.5, 1.0], 250, xp_as_array(500.0, xp=xp)
        )
        xp_assert_close(R_multi, as_ndarray(R_single), atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(T_multi, as_ndarray(T_single), atol=TOLERANCE_ABSOLUTE_TESTS)

        # Test multiple layers
        R, T = multilayer_tmm(
            [1.0, 1.5, 2.0, 1.0],
            [250, 150],
            xp_as_array([400, 500, 600], xp=xp),
        )
        R, T = as_ndarray(R), as_ndarray(T)
        assert R.shape == (3, 1, 1, 2)  # (W=3, A=1, T=1, 2) - Spectroscopy Convention
        assert T.shape == (3, 1, 1, 2)
        assert np.all((R >= 0) & (R <= 1))
        assert np.all((T >= 0) & (T <= 1))

        # Test energy conservation
        xp_assert_close(R + T, 1.0, atol=TOLERANCE_ABSOLUTE_TESTS * 10)

        # Test with different substrate
        R_sub, T_sub = multilayer_tmm([1.0, 1.5, 1.5], [250], xp_as_array(500.0, xp=xp))
        R_sub, T_sub = as_ndarray(R_sub), as_ndarray(T_sub)
        assert R_sub.shape == (1, 1, 1, 2)
        assert T_sub.shape == (1, 1, 1, 2)

    def test_n_dimensional_multilayer_tmm(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.interference.multilayer_tmm`
        definition n-dimensional arrays support.
        """

        wl = 555
        R, T = multilayer_tmm([1.0, 1.5, 2.0, 1.0], [250, 150], wl)

        wl = xp.tile(xp_as_array(wl, xp=xp), (6,))
        R = xp.tile(xp_as_array(R, xp=xp), (6, 1, 1, 1))
        T = xp.tile(xp_as_array(T, xp=xp), (6, 1, 1, 1))
        R_array, T_array = multilayer_tmm([1.0, 1.5, 2.0, 1.0], [250, 150], wl)
        xp_assert_close(R_array, R, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(T_array, T, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_multilayer_tmm(self) -> None:
        """
        Test :func:`colour.phenomena.interference.multilayer_tmm`
        definition nan support.
        """

        multilayer_tmm(
            [1.0, 1.5, 1.0],
            [250],
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]),
        )

    def test_multilayer_tmm_complex_n(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.interference.multilayer_tmm`
        with complex refractive indices.
        """

        # Stack of two absorbing layers: air | layer1 | layer2 | air
        n_layers = [1.0, 2.0 + 0.5j, 1.8 + 0.3j, 1.0]
        thicknesses = [200, 300]
        wavelengths = xp_as_array([400, 500, 600], xp=xp)

        R, T = multilayer_tmm(n_layers, thicknesses, wavelengths)
        R, T = as_ndarray(R), as_ndarray(T)

        # Check shapes and validity
        assert R.shape == (3, 1, 1, 2)  # (W=3, A=1, T=1, 2) - Spectroscopy Convention
        assert T.shape == (3, 1, 1, 2)
        assert np.all((R >= 0) & (R <= 1))
        assert np.all((T >= 0) & (T <= 1))

        # For absorbing media: R + T < 1
        assert np.all(R + T < 1.0 + 1e-6)

        # Glass + Silver + Glass structure: glass | glass | silver | glass | glass
        n_layers_metal = [1.5, 1.5, 0.18 + 3.15j, 1.5, 1.5]
        thicknesses_metal = [100, 50, 100]
        R_metal, _ = multilayer_tmm(n_layers_metal, thicknesses_metal, 500)

        # High reflectance expected for metal
        assert np.mean(R_metal) > 0.5

    def test_multilayer_tmm_mixed_structures(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.interference.multilayer_tmm`
        with mixed transparent and absorbing layers.
        """

        # Anti-reflection coating + absorbing layer + glass substrate
        # air | AR coating | absorber | glass
        n_ar = 1.38  # MgF2 (transparent)
        n_absorber = 2.0 + 0.5j  # Absorbing layer
        n_substrate = 1.5  # Glass

        n_layers = [1.0, n_ar, n_absorber, n_substrate]
        thicknesses = [100, 300]
        wavelength = 550

        R, T = multilayer_tmm(n_layers, thicknesses, wavelength)

        # Basic validity
        assert np.all((R >= 0) & (R <= 1))
        assert np.all((T >= 0) & (T <= 1))

        # For absorbing media: R + T < 1
        R_avg = np.mean(R)
        T_avg = np.mean(T)
        A = 1 - R_avg - T_avg
        assert A > 0, f"Expected absorption > 0, got A = {A}"

        # Three-layer stack: air | transparent | absorbing | transparent | air
        n_layers_mixed = [
            1.0,
            1.5,
            2.0 + 0.3j,
            1.7,
            1.0,
        ]  # air, Real, Complex, Real, air
        thicknesses_mixed = [150, 200, 250]
        wavelengths = xp_as_array([450, 550, 650], xp=xp)

        R_mixed, T_mixed = multilayer_tmm(
            n_layers_mixed, thicknesses_mixed, wavelengths
        )
        R_mixed, T_mixed = as_ndarray(R_mixed), as_ndarray(T_mixed)

        # Check shapes and validity
        assert R_mixed.shape == (
            3,
            1,
            1,
            2,
        )  # (W=3, A=1, T=1, 2) - Spectroscopy Convention
        assert T_mixed.shape == (3, 1, 1, 2)
        assert np.all((R_mixed >= 0) & (R_mixed <= 1))
        assert np.all((T_mixed >= 0) & (T_mixed <= 1))

        # Test with real refractive indices (lossless):
        # air | layer1 | layer2 | layer3 | air
        n_layers_real = [1.0, 1.38, 2.0, 1.7, 1.0]
        thicknesses_real = [100, 200, 150]
        R_real, T_real = multilayer_tmm(n_layers_real, thicknesses_real, 550)

        # For lossless media: R + T = 1
        xp_assert_close(R_real + T_real, 1.0, atol=TOLERANCE_ABSOLUTE_TESTS * 10)
