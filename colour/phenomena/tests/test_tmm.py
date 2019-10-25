"""Define the unit tests for the :mod:`colour.phenomena.tmm` module."""

from __future__ import annotations

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.phenomena.interference import matrix_transfer_tmm
from colour.phenomena.tmm import (
    polarised_light_magnitude_elements,
    polarised_light_reflection_amplitude,
    polarised_light_reflection_coefficient,
    polarised_light_transmission_amplitude,
    polarised_light_transmission_coefficient,
    snell_law,
)
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestSnellLaw",
    "TestPolarisedLightMagnitudeElements",
    "TestPolarisedLightReflectionAmplitude",
    "TestPolarisedLightReflectionCoefficient",
    "TestPolarisedLightTransmissionAmplitude",
    "TestPolarisedLightTransmissionCoefficient",
    "TestMatrixTransferTmm",
]


class TestSnellLaw:
    """
    Define :func:`colour.phenomena.tmm.snell_law` definition unit tests
    methods.
    """

    def test_snell_law(self) -> None:
        """Test :func:`colour.phenomena.tmm.snell_law` definition."""

        np.testing.assert_allclose(
            snell_law(1.0, 1.5, 30.0),
            19.4712206345,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            snell_law(1.0, 1.33, 45.0),
            32.117631278,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            snell_law(1.5, 1.0, 19.47),
            30.0,
            atol=0.01,
        )

        # Test normal incidence (0 degrees)
        np.testing.assert_allclose(
            snell_law(1.0, 1.5, 0.0),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_snell_law(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.snell_law` definition n-dimensional
        arrays support.
        """

        n_1 = 1.0
        n_2 = 1.5
        theta_i = 30.0
        theta_t = snell_law(n_1, n_2, theta_i)

        theta_i = np.tile(theta_i, 6)
        theta_t = np.tile(theta_t, 6)
        np.testing.assert_allclose(
            snell_law(n_1, n_2, theta_i),
            theta_t,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        theta_i = np.reshape(theta_i, (2, 3))
        theta_t = np.reshape(theta_t, (2, 3))
        np.testing.assert_allclose(
            snell_law(n_1, n_2, theta_i),
            theta_t,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_snell_law(self) -> None:
        """Test :func:`colour.phenomena.tmm.snell_law` definition nan support."""

        snell_law(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]),
            1.5,
            30.0,
        )


class TestPolarisedLightMagnitudeElements:
    """
    Define :func:`colour.phenomena.tmm.polarised_light_magnitude_elements`
    definition unit tests methods.
    """

    def test_polarised_light_magnitude_elements(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_magnitude_elements`
        definition.
        """

        result = polarised_light_magnitude_elements(1.0, 1.5, 0.0, 0.0)
        np.testing.assert_allclose(result[0], 1.0 + 0j, atol=TOLERANCE_ABSOLUTE_TESTS)
        np.testing.assert_allclose(result[1], 1.0 + 0j, atol=TOLERANCE_ABSOLUTE_TESTS)
        np.testing.assert_allclose(result[2], 1.5 + 0j, atol=TOLERANCE_ABSOLUTE_TESTS)
        np.testing.assert_allclose(result[3], 1.5 + 0j, atol=TOLERANCE_ABSOLUTE_TESTS)

        # Test at 45 degrees
        result_45 = polarised_light_magnitude_elements(1.0, 1.5, 45.0, 30.0)
        assert len(result_45) == 4

    def test_n_dimensional_polarised_light_magnitude_elements(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_magnitude_elements`
        definition n-dimensional arrays support.
        """

        n_1 = 1.0
        n_2 = 1.5
        theta_i = 0.0
        theta_t = 0.0
        m0, m1, m2, m3 = polarised_light_magnitude_elements(n_1, n_2, theta_i, theta_t)

        theta_i_array = np.tile(theta_i, 6)
        theta_t_array = np.tile(theta_t, 6)
        m0_array, m1_array, m2_array, m3_array = polarised_light_magnitude_elements(
            n_1, n_2, theta_i_array, theta_t_array
        )
        np.testing.assert_allclose(
            m0_array, np.tile(m0, 6), atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            m1_array, np.tile(m1, 6), atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            m2_array, np.tile(m2, 6), atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            m3_array, np.tile(m3, 6), atol=TOLERANCE_ABSOLUTE_TESTS
        )

        theta_i_array = np.reshape(theta_i_array, (2, 3))
        theta_t_array = np.reshape(theta_t_array, (2, 3))
        m0_array, m1_array, m2_array, m3_array = polarised_light_magnitude_elements(
            n_1, n_2, theta_i_array, theta_t_array
        )
        np.testing.assert_allclose(
            m0_array, np.tile(m0, 6).reshape(2, 3), atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            m1_array, np.tile(m1, 6).reshape(2, 3), atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            m2_array, np.tile(m2, 6).reshape(2, 3), atol=TOLERANCE_ABSOLUTE_TESTS
        )
        np.testing.assert_allclose(
            m3_array, np.tile(m3, 6).reshape(2, 3), atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_polarised_light_magnitude_elements(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_magnitude_elements`
        definition nan support.
        """

        polarised_light_magnitude_elements(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]),
            1.5,
            0.0,
            0.0,
        )


class TestPolarisedLightReflectionAmplitude:
    """
    Define :func:`colour.phenomena.tmm.polarised_light_reflection_amplitude`
    definition unit tests methods.
    """

    def test_polarised_light_reflection_amplitude(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_reflection_amplitude`
        definition.
        """

        np.testing.assert_allclose(
            polarised_light_reflection_amplitude(1.0, 1.5, 0.0, 0.0),
            np.array([-0.2 + 0j, -0.2 + 0j]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            polarised_light_reflection_amplitude(1.0, 1.5, 30.0, 19.47),
            np.array([-0.24041175 + 0j, -0.15889613 + 0j]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_polarised_light_reflection_amplitude(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_reflection_amplitude`
        definition n-dimensional arrays support.
        """

        n_1 = 1.0
        n_2 = 1.5
        theta_i = 0.0
        theta_t = 0.0
        r = polarised_light_reflection_amplitude(n_1, n_2, theta_i, theta_t)

        theta_i_array = np.tile(theta_i, 6)
        theta_t_array = np.tile(theta_t, 6)
        r_array = polarised_light_reflection_amplitude(
            n_1, n_2, theta_i_array, theta_t_array
        )
        np.testing.assert_allclose(
            r_array,
            np.tile(r, (6, 1)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        theta_i_array = np.reshape(theta_i_array, (2, 3))
        theta_t_array = np.reshape(theta_t_array, (2, 3))
        r_array = polarised_light_reflection_amplitude(
            n_1, n_2, theta_i_array, theta_t_array
        )
        np.testing.assert_allclose(
            r_array,
            np.tile(r, (6, 1)).reshape(2, 3, 2),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_polarised_light_reflection_amplitude(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_reflection_amplitude`
        definition nan support.
        """

        polarised_light_reflection_amplitude(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]),
            1.5,
            0.0,
            0.0,
        )


class TestPolarisedLightReflectionCoefficient:
    """
    Define :func:`colour.phenomena.tmm.polarised_light_reflection_coefficient`
    definition unit tests methods.
    """

    def test_polarised_light_reflection_coefficient(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_reflection_coefficient`
        definition.
        """

        np.testing.assert_allclose(
            polarised_light_reflection_coefficient(1.0, 1.5, 0.0, 0.0),
            np.array([0.04 + 0j, 0.04 + 0j]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test that reflectance is always between 0 and 1
        R = polarised_light_reflection_coefficient(1.0, 1.5, 30.0, 19.47)
        assert np.all(np.real(R) >= 0)
        assert np.all(np.real(R) <= 1)

    def test_n_dimensional_polarised_light_reflection_coefficient(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_reflection_coefficient`
        definition n-dimensional arrays support.
        """

        n_1 = 1.0
        n_2 = 1.5
        theta_i = 0.0
        theta_t = 0.0
        R = polarised_light_reflection_coefficient(n_1, n_2, theta_i, theta_t)

        theta_i_array = np.tile(theta_i, 6)
        theta_t_array = np.tile(theta_t, 6)
        R_array = polarised_light_reflection_coefficient(
            n_1, n_2, theta_i_array, theta_t_array
        )
        np.testing.assert_allclose(
            R_array,
            np.tile(R, (6, 1)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        theta_i_array = np.reshape(theta_i_array, (2, 3))
        theta_t_array = np.reshape(theta_t_array, (2, 3))
        R_array = polarised_light_reflection_coefficient(
            n_1, n_2, theta_i_array, theta_t_array
        )
        np.testing.assert_allclose(
            R_array,
            np.tile(R, (6, 1)).reshape(2, 3, 2),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_polarised_light_reflection_coefficient(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_reflection_coefficient`
        definition nan support.
        """

        polarised_light_reflection_coefficient(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]),
            1.5,
            0.0,
            0.0,
        )


class TestPolarisedLightTransmissionAmplitude:
    """
    Define :func:`colour.phenomena.tmm.polarised_light_transmission_amplitude`
    definition unit tests methods.
    """

    def test_polarised_light_transmission_amplitude(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_transmission_amplitude`
        definition.
        """

        np.testing.assert_allclose(
            polarised_light_transmission_amplitude(1.0, 1.5, 0.0, 0.0),
            np.array([0.8 + 0j, 0.8 + 0j]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_polarised_light_transmission_amplitude(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_transmission_amplitude`
        definition n-dimensional arrays support.
        """

        n_1 = 1.0
        n_2 = 1.5
        theta_i = 0.0
        theta_t = 0.0
        t = polarised_light_transmission_amplitude(n_1, n_2, theta_i, theta_t)

        theta_i_array = np.tile(theta_i, 6)
        theta_t_array = np.tile(theta_t, 6)
        t_array = polarised_light_transmission_amplitude(
            n_1, n_2, theta_i_array, theta_t_array
        )
        np.testing.assert_allclose(
            t_array,
            np.tile(t, (6, 1)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        theta_i_array = np.reshape(theta_i_array, (2, 3))
        theta_t_array = np.reshape(theta_t_array, (2, 3))
        t_array = polarised_light_transmission_amplitude(
            n_1, n_2, theta_i_array, theta_t_array
        )
        np.testing.assert_allclose(
            t_array,
            np.tile(t, (6, 1)).reshape(2, 3, 2),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_polarised_light_transmission_amplitude(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_transmission_amplitude`
        definition nan support.
        """

        polarised_light_transmission_amplitude(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]),
            1.5,
            0.0,
            0.0,
        )


class TestPolarisedLightTransmissionCoefficient:
    """
    Define :func:`colour.phenomena.tmm.polarised_light_transmission_coefficient`
    definition unit tests methods.
    """

    def test_polarised_light_transmission_coefficient(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_transmission_coefficient`
        definition.
        """

        np.testing.assert_allclose(
            polarised_light_transmission_coefficient(1.0, 1.5, 0.0, 0.0),
            np.array([0.96 + 0j, 0.96 + 0j]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test energy conservation: R + T = 1
        R = polarised_light_reflection_coefficient(1.0, 1.5, 0.0, 0.0)
        T = polarised_light_transmission_coefficient(1.0, 1.5, 0.0, 0.0)
        np.testing.assert_allclose(
            np.real(R + T), np.array([1.0, 1.0]), atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_polarised_light_transmission_coefficient(
        self,
    ) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_transmission_coefficient`
        definition n-dimensional arrays support.
        """

        n_1 = 1.0
        n_2 = 1.5
        theta_i = 0.0
        theta_t = 0.0
        T = polarised_light_transmission_coefficient(n_1, n_2, theta_i, theta_t)

        theta_i_array = np.tile(theta_i, 6)
        theta_t_array = np.tile(theta_t, 6)
        T_array = polarised_light_transmission_coefficient(
            n_1, n_2, theta_i_array, theta_t_array
        )
        np.testing.assert_allclose(
            T_array,
            np.tile(T, (6, 1)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        theta_i_array = np.reshape(theta_i_array, (2, 3))
        theta_t_array = np.reshape(theta_t_array, (2, 3))
        T_array = polarised_light_transmission_coefficient(
            n_1, n_2, theta_i_array, theta_t_array
        )
        np.testing.assert_allclose(
            T_array,
            np.tile(T, (6, 1)).reshape(2, 3, 2),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_polarised_light_transmission_coefficient(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_transmission_coefficient`
        definition nan support.
        """

        polarised_light_transmission_coefficient(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]),
            1.5,
            0.0,
            0.0,
        )


class TestMatrixTransferTmm:
    """
    Define :func:`colour.phenomena.tmm.matrix_transfer_tmm`
    definition unit tests methods.
    """

    def test_matrix_transfer_tmm(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.matrix_transfer_tmm`
        definition.
        """

        # Single layer structure
        result = matrix_transfer_tmm(
            n=[1.0, 1.5, 1.0], t=[250], theta=0, wavelength=550
        )

        # Check shapes - (W, A, T, 2, 2)
        assert result.M_s.shape == (
            1,
            1,
            1,
            2,
            2,
        )  # (wavelengths=1, angles=1, thickness=1, 2, 2)
        assert result.M_p.shape == (1, 1, 1, 2, 2)
        # theta has shape (angles, media)
        assert result.theta.shape == (1, 3)  # (angles=1, media=3)
        assert len(result.n) == 3  # incident, layer, substrate

        # Check refractive indices
        # n has shape (media_count, wavelengths_count)
        assert result.n.shape == (3, 1)
        np.testing.assert_allclose(
            result.n[:, 0], [1.0, 1.5, 1.0], atol=TOLERANCE_ABSOLUTE_TESTS
        )

        # Check angles (normal incidence)
        assert result.theta[0, 0] == 0.0  # incident
        assert result.theta[0, -1] == 0.0  # substrate (by Snell's law)

        # Check transfer matrix properties (should be 2x2 complex)
        assert result.M_s.dtype in [np.complex64, np.complex128]
        assert result.M_p.dtype in [np.complex64, np.complex128]

    def test_matrix_transfer_tmm_multilayer(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.matrix_transfer_tmm`
        with multiple layers.
        """

        # Two-layer structure
        result = matrix_transfer_tmm(
            n=[1.0, 1.5, 2.0, 1.5],
            t=[250, 150],
            theta=0,
            wavelength=550,
        )

        # Check shapes - (W, A, T, 2, 2)
        assert result.M_s.shape == (
            1,
            1,
            1,
            2,
            2,
        )  # (wavelengths=1, angles=1, thickness=1, 2, 2)
        assert result.M_p.shape == (1, 1, 1, 2, 2)
        # theta has shape (angles, media)
        assert result.theta.shape == (1, 4)  # (angles=1, media=4)
        assert len(result.n) == 4

        # Check refractive indices
        # n has shape (media_count, wavelengths_count)
        assert result.n.shape == (4, 1)
        np.testing.assert_allclose(
            result.n[:, 0], [1.0, 1.5, 2.0, 1.5], atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_matrix_transfer_tmm_multiple_wavelengths(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.matrix_transfer_tmm`
        with multiple wavelengths.
        """

        wavelengths = [400, 500, 600]
        result = matrix_transfer_tmm(
            n=[1.0, 1.5, 1.0], t=[250], theta=0, wavelength=wavelengths
        )

        # Check shapes - (W, A, T, 2, 2)
        assert result.M_s.shape == (
            3,
            1,
            1,
            2,
            2,
        )  # (wavelengths=3, angles=1, thickness=1, 2, 2)
        assert result.M_p.shape == (3, 1, 1, 2, 2)

        # theta has shape (angles, media)
        assert result.theta.shape == (1, 3)  # (angles=1, media=3)
        # n has shape (media_count, wavelengths_count)
        assert result.n.shape == (3, 3)

    def test_matrix_transfer_tmm_complex_n(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.matrix_transfer_tmm`
        with complex refractive indices.
        """

        # Absorbing layer
        n_absorbing = 2.0 + 0.5j
        result = matrix_transfer_tmm(
            n=[1.0, n_absorbing, 1.0], t=[250], theta=0, wavelength=550
        )

        # Check that complex n is preserved
        # n has shape (media_count, wavelengths_count)
        assert np.iscomplex(result.n[1, 0])
        np.testing.assert_allclose(
            result.n[1, 0], n_absorbing, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        # Transfer matrices should be complex
        assert np.iscomplexobj(result.M_s)
        assert np.iscomplexobj(result.M_p)

    def test_matrix_transfer_tmm_oblique_incidence(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.matrix_transfer_tmm`
        with oblique incidence.
        """

        theta_i = 30.0  # 30 degrees
        result = matrix_transfer_tmm(
            n=[1.0, 1.5, 1.0], t=[250], theta=theta_i, wavelength=550
        )

        # Check incident angle - theta has shape (angles, media)
        np.testing.assert_allclose(
            result.theta[0, 0], theta_i, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        # Check that angle changes in layer (Snell's law)
        assert result.theta[0, 1] != theta_i  # Should be refracted

        # s and p matrices should differ at oblique incidence
        assert not np.allclose(result.M_s, result.M_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_matrix_transfer_tmm_energy_consistency(self) -> None:
        """
        Test that transfer matrices from transfer_matrix_tmm give
        consistent R and T values.
        """

        # Build transfer matrix
        result = matrix_transfer_tmm(
            n=[1.0, 1.5, 1.0], t=[250], theta=0, wavelength=550
        )

        # Extract R and T manually - M_s has shape (W, A, T, 2, 2)
        r_s = result.M_s[0, 0, 0, 1, 0] / result.M_s[0, 0, 0, 0, 0]
        R_s = np.abs(r_s) ** 2

        t_s = 1.0 / result.M_s[0, 0, 0, 0, 0]
        theta_i_rad = np.radians(0.0)
        theta_f_rad = np.radians(result.theta[0, -1])

        # Extract incident and substrate from result.n
        n_incident = result.n[0, 0]
        n_substrate = result.n[-1, 0]

        angle_factor = np.real(n_substrate * np.cos(theta_f_rad)) / np.real(
            n_incident * np.cos(theta_i_rad)
        )
        T_s = np.abs(t_s) ** 2 * angle_factor

        # Energy conservation for lossless media: R + T = 1
        np.testing.assert_allclose(R_s + T_s, 1.0, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_n_dimensional_matrix_transfer_tmm(self) -> None:
        """
        Test :func:`colour.phenomena.tmm.matrix_transfer_tmm`
        definition n-dimensional arrays support.
        """

        wl = 555
        result = matrix_transfer_tmm(n=[1.0, 1.5, 1.0], t=[250], theta=0, wavelength=wl)

        wl_array = np.tile(wl, 6)
        result_array = matrix_transfer_tmm(
            n=[1.0, 1.5, 1.0], t=[250], theta=0, wavelength=wl_array
        )

        # Check shape - (W, A, T, 2, 2)
        assert result_array.M_s.shape == (
            6,
            1,
            1,
            2,
            2,
        )  # (wavelengths=6, angles=1, thickness=1, 2, 2)
        assert result_array.M_p.shape == (6, 1, 1, 2, 2)

        # theta shapes: result has (1, 3), result_array has (1, 3)
        assert result_array.theta.shape == result.theta.shape
        # n shapes: result has (3, 1), result_array has (3, 6)
        # For constant n, all wavelength columns should match
        assert result_array.n.shape == (3, 6)
        assert result.n.shape == (3, 1)
        np.testing.assert_allclose(
            result_array.n[:, 0],
            result.n[:, 0],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
