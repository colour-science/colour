"""Define the unit tests for the :mod:`colour.phenomena.tmm` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

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

    def test_snell_law(self, xp: ModuleType) -> None:
        """Test :func:`colour.phenomena.tmm.snell_law` definition."""

        xp_assert_close(
            snell_law(1.0, 1.5, xp_as_array([30.0], xp=xp)),
            19.4712206345,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            snell_law(1.0, 1.33, xp_as_array([45.0], xp=xp)),
            32.117631278,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            snell_law(1.5, 1.0, xp_as_array([19.47], xp=xp)),
            30.0,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

        # Test normal incidence (0 degrees)
        xp_assert_close(
            snell_law(1.0, 1.5, xp_as_array([0.0], xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_snell_law(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.tmm.snell_law` definition n-dimensional
        arrays support.
        """

        n_1 = 1.0
        n_2 = 1.5
        theta_i = 30.0
        theta_t = snell_law(n_1, n_2, theta_i)

        theta_i = xp.tile(xp_as_array(theta_i, xp=xp), (6,))
        theta_t = xp.tile(xp_as_array(theta_t, xp=xp), (6,))
        xp_assert_close(
            snell_law(n_1, n_2, theta_i),
            theta_t,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        theta_i = xp_reshape(xp_as_array(theta_i, xp=xp), (2, 3), xp=xp)
        theta_t = xp_reshape(xp_as_array(theta_t, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
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

    def test_polarised_light_magnitude_elements(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_magnitude_elements`
        definition.
        """

        result = polarised_light_magnitude_elements(
            1.0, 1.5, xp_as_array([0.0], xp=xp), xp_as_array([0.0], xp=xp)
        )
        xp_assert_close(result[0], 1.0 + 0j, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(result[1], 1.0 + 0j, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(result[2], 1.5 + 0j, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(result[3], 1.5 + 0j, atol=TOLERANCE_ABSOLUTE_TESTS)

        # Test at 45 degrees
        result_45 = polarised_light_magnitude_elements(
            1.0, 1.5, xp_as_array([45.0], xp=xp), xp_as_array([30.0], xp=xp)
        )
        assert len(result_45) == 4

    def test_n_dimensional_polarised_light_magnitude_elements(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_magnitude_elements`
        definition n-dimensional arrays support.
        """

        n_1 = 1.0
        n_2 = 1.5
        theta_i = 0.0
        theta_t = 0.0
        m0, m1, m2, m3 = polarised_light_magnitude_elements(n_1, n_2, theta_i, theta_t)

        theta_i_array = xp.tile(xp_as_array(theta_i, xp=xp), (6,))
        theta_t_array = xp.tile(xp_as_array(theta_t, xp=xp), (6,))
        m0_array, m1_array, m2_array, m3_array = polarised_light_magnitude_elements(
            n_1, n_2, theta_i_array, theta_t_array
        )
        xp_assert_close(
            m0_array,
            np.tile(as_ndarray(m0), (6,)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            m1_array,
            np.tile(as_ndarray(m1), (6,)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            m2_array,
            np.tile(as_ndarray(m2), (6,)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            m3_array,
            np.tile(as_ndarray(m3), (6,)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        theta_i_array = xp_reshape(xp_as_array(theta_i_array, xp=xp), (2, 3), xp=xp)
        theta_t_array = xp_reshape(xp_as_array(theta_t_array, xp=xp), (2, 3), xp=xp)
        m0_array, m1_array, m2_array, m3_array = polarised_light_magnitude_elements(
            n_1, n_2, theta_i_array, theta_t_array
        )
        xp_assert_close(
            m0_array,
            np.tile(as_ndarray(m0), (6,)).reshape(2, 3),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            m1_array,
            np.tile(as_ndarray(m1), (6,)).reshape(2, 3),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            m2_array,
            np.tile(as_ndarray(m2), (6,)).reshape(2, 3),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            m3_array,
            np.tile(as_ndarray(m3), (6,)).reshape(2, 3),
            atol=TOLERANCE_ABSOLUTE_TESTS,
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

    def test_polarised_light_reflection_amplitude(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_reflection_amplitude`
        definition.
        """

        xp_assert_close(
            polarised_light_reflection_amplitude(
                1.0, 1.5, xp_as_array([0.0], xp=xp), xp_as_array([0.0], xp=xp)
            ),
            [[-0.2 + 0j, -0.2 + 0j]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            polarised_light_reflection_amplitude(
                1.0,
                1.5,
                xp_as_array([30.0], xp=xp),
                xp_as_array([19.47], xp=xp),
            ),
            [[-0.24041175 + 0j, -0.15889613 + 0j]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_polarised_light_reflection_amplitude(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_reflection_amplitude`
        definition n-dimensional arrays support.
        """

        n_1 = 1.0
        n_2 = 1.5
        theta_i = 0.0
        theta_t = 0.0
        r = polarised_light_reflection_amplitude(n_1, n_2, theta_i, theta_t)

        theta_i_array = xp.tile(xp_as_array(theta_i, xp=xp), (6,))
        theta_t_array = xp.tile(xp_as_array(theta_t, xp=xp), (6,))
        r_array = polarised_light_reflection_amplitude(
            n_1, n_2, theta_i_array, theta_t_array
        )
        xp_assert_close(
            r_array,
            np.tile(as_ndarray(r), (6, 1)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        theta_i_array = xp_reshape(xp_as_array(theta_i_array, xp=xp), (2, 3), xp=xp)
        theta_t_array = xp_reshape(xp_as_array(theta_t_array, xp=xp), (2, 3), xp=xp)
        r_array = polarised_light_reflection_amplitude(
            n_1, n_2, theta_i_array, theta_t_array
        )
        xp_assert_close(
            r_array,
            np.tile(as_ndarray(r), (6, 1)).reshape(2, 3, 2),
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

    def test_polarised_light_reflection_coefficient(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_reflection_coefficient`
        definition.
        """

        xp_assert_close(
            polarised_light_reflection_coefficient(
                1.0, 1.5, xp_as_array([0.0], xp=xp), xp_as_array([0.0], xp=xp)
            ),
            [[0.04 + 0j, 0.04 + 0j]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test that reflectance is always between 0 and 1
        R = as_ndarray(
            polarised_light_reflection_coefficient(
                1.0,
                1.5,
                xp_as_array([30.0], xp=xp),
                xp_as_array([19.47], xp=xp),
            )
        )
        assert np.all(np.real(R) >= 0)
        assert np.all(np.real(R) <= 1)

    def test_n_dimensional_polarised_light_reflection_coefficient(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_reflection_coefficient`
        definition n-dimensional arrays support.
        """

        n_1 = 1.0
        n_2 = 1.5
        theta_i = 0.0
        theta_t = 0.0
        R = polarised_light_reflection_coefficient(n_1, n_2, theta_i, theta_t)

        theta_i_array = xp.tile(xp_as_array(theta_i, xp=xp), (6,))
        theta_t_array = xp.tile(xp_as_array(theta_t, xp=xp), (6,))
        R_array = polarised_light_reflection_coefficient(
            n_1, n_2, theta_i_array, theta_t_array
        )
        xp_assert_close(
            R_array,
            np.tile(as_ndarray(R), (6, 1)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        theta_i_array = xp_reshape(xp_as_array(theta_i_array, xp=xp), (2, 3), xp=xp)
        theta_t_array = xp_reshape(xp_as_array(theta_t_array, xp=xp), (2, 3), xp=xp)
        R_array = polarised_light_reflection_coefficient(
            n_1, n_2, theta_i_array, theta_t_array
        )
        xp_assert_close(
            R_array,
            np.tile(as_ndarray(R), (6, 1)).reshape(2, 3, 2),
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

    def test_polarised_light_transmission_amplitude(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_transmission_amplitude`
        definition.
        """

        xp_assert_close(
            polarised_light_transmission_amplitude(
                1.0, 1.5, xp_as_array([0.0], xp=xp), xp_as_array([0.0], xp=xp)
            ),
            [[0.8 + 0j, 0.8 + 0j]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_polarised_light_transmission_amplitude(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_transmission_amplitude`
        definition n-dimensional arrays support.
        """

        n_1 = 1.0
        n_2 = 1.5
        theta_i = 0.0
        theta_t = 0.0
        t = polarised_light_transmission_amplitude(n_1, n_2, theta_i, theta_t)

        theta_i_array = xp.tile(xp_as_array(theta_i, xp=xp), (6,))
        theta_t_array = xp.tile(xp_as_array(theta_t, xp=xp), (6,))
        t_array = polarised_light_transmission_amplitude(
            n_1, n_2, theta_i_array, theta_t_array
        )
        xp_assert_close(
            t_array,
            np.tile(as_ndarray(t), (6, 1)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        theta_i_array = xp_reshape(xp_as_array(theta_i_array, xp=xp), (2, 3), xp=xp)
        theta_t_array = xp_reshape(xp_as_array(theta_t_array, xp=xp), (2, 3), xp=xp)
        t_array = polarised_light_transmission_amplitude(
            n_1, n_2, theta_i_array, theta_t_array
        )
        xp_assert_close(
            t_array,
            np.tile(as_ndarray(t), (6, 1)).reshape(2, 3, 2),
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

    def test_polarised_light_transmission_coefficient(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.tmm.polarised_light_transmission_coefficient`
        definition.
        """

        xp_assert_close(
            polarised_light_transmission_coefficient(
                1.0, 1.5, xp_as_array([0.0], xp=xp), xp_as_array([0.0], xp=xp)
            ),
            [[0.96 + 0j, 0.96 + 0j]],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test energy conservation: R + T = 1
        R = as_ndarray(
            polarised_light_reflection_coefficient(
                1.0, 1.5, xp_as_array([0.0], xp=xp), xp_as_array([0.0], xp=xp)
            )
        )
        T = as_ndarray(
            polarised_light_transmission_coefficient(
                1.0, 1.5, xp_as_array([0.0], xp=xp), xp_as_array([0.0], xp=xp)
            )
        )
        xp_assert_close(np.real(R + T), [[1.0, 1.0]], atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_n_dimensional_polarised_light_transmission_coefficient(
        self, xp: ModuleType
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

        theta_i_array = xp.tile(xp_as_array(theta_i, xp=xp), (6,))
        theta_t_array = xp.tile(xp_as_array(theta_t, xp=xp), (6,))
        T_array = polarised_light_transmission_coefficient(
            n_1, n_2, theta_i_array, theta_t_array
        )
        xp_assert_close(
            T_array,
            np.tile(as_ndarray(T), (6, 1)),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        theta_i_array = xp_reshape(xp_as_array(theta_i_array, xp=xp), (2, 3), xp=xp)
        theta_t_array = xp_reshape(xp_as_array(theta_t_array, xp=xp), (2, 3), xp=xp)
        T_array = polarised_light_transmission_coefficient(
            n_1, n_2, theta_i_array, theta_t_array
        )
        xp_assert_close(
            T_array,
            np.tile(as_ndarray(T), (6, 1)).reshape(2, 3, 2),
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

    def test_matrix_transfer_tmm(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.tmm.matrix_transfer_tmm`
        definition.
        """

        # Single layer structure
        result = matrix_transfer_tmm(
            n=[1.0, 1.5, 1.0],
            t=[250],
            theta=0,
            wavelength=xp_as_array([550], xp=xp),
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
        xp_assert_close(result.n[:, 0], [1.0, 1.5, 1.0], atol=TOLERANCE_ABSOLUTE_TESTS)

        # Check angles (normal incidence)
        assert result.theta[0, 0] == 0.0  # incident
        assert result.theta[0, -1] == 0.0  # substrate (by Snell's law)

        # Check transfer matrix properties (should be 2x2 complex)
        assert "complex" in str(result.M_s.dtype)
        assert "complex" in str(result.M_p.dtype)

    def test_matrix_transfer_tmm_multilayer(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.tmm.matrix_transfer_tmm`
        with multiple layers.
        """

        # Two-layer structure
        result = matrix_transfer_tmm(
            n=[1.0, 1.5, 2.0, 1.5],
            t=[250, 150],
            theta=xp_as_array([0.0], xp=xp),
            wavelength=xp_as_array([550.0], xp=xp),
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
        xp_assert_close(
            result.n[:, 0],
            [1.0, 1.5, 2.0, 1.5],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_matrix_transfer_tmm_multiple_wavelengths(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.tmm.matrix_transfer_tmm`
        with multiple wavelengths.
        """

        wavelengths = xp_as_array([400.0, 500.0, 600.0], xp=xp)
        result = matrix_transfer_tmm(
            n=[1.0, 1.5, 1.0],
            t=[250],
            theta=xp_as_array([0.0], xp=xp),
            wavelength=wavelengths,
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

    def test_matrix_transfer_tmm_complex_n(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.tmm.matrix_transfer_tmm`
        with complex refractive indices.
        """

        # Absorbing layer
        n_absorbing = 2.0 + 0.5j
        result = matrix_transfer_tmm(
            n=[1.0, n_absorbing, 1.0],
            t=[250],
            theta=xp_as_array([0.0], xp=xp),
            wavelength=xp_as_array([550.0], xp=xp),
        )

        # Check that complex n is preserved
        # n has shape (media_count, wavelengths_count)
        assert np.iscomplex(as_ndarray(result.n)[1, 0])
        xp_assert_close(result.n[1, 0], n_absorbing, atol=TOLERANCE_ABSOLUTE_TESTS)

        # Transfer matrices should be complex
        assert np.iscomplexobj(as_ndarray(result.M_s))
        assert np.iscomplexobj(as_ndarray(result.M_p))

    def test_matrix_transfer_tmm_oblique_incidence(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.tmm.matrix_transfer_tmm`
        with oblique incidence.
        """

        theta_i = 30.0  # 30 degrees
        result = matrix_transfer_tmm(
            n=[1.0, 1.5, 1.0],
            t=[250],
            theta=xp_as_array([theta_i], xp=xp),
            wavelength=xp_as_array([550.0], xp=xp),
        )

        # Check incident angle - theta has shape (angles, media)
        xp_assert_close(result.theta[0, 0], theta_i, atol=TOLERANCE_ABSOLUTE_TESTS)

        # Check that angle changes in layer (Snell's law)
        assert as_ndarray(result.theta)[0, 1] != theta_i  # Should be refracted

        # s and p matrices should differ at oblique incidence
        assert not np.allclose(
            as_ndarray(result.M_s),
            as_ndarray(result.M_p),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_matrix_transfer_tmm_energy_consistency(self, xp: ModuleType) -> None:
        """
        Test that transfer matrices from transfer_matrix_tmm give
        consistent R and T values.
        """

        # Build transfer matrix
        result = matrix_transfer_tmm(
            n=[1.0, 1.5, 1.0],
            t=[250],
            theta=xp_as_array([0.0], xp=xp),
            wavelength=xp_as_array([550.0], xp=xp),
        )

        # Extract R and T manually - M_s has shape (W, A, T, 2, 2)
        M_s = as_ndarray(result.M_s)
        r_s = M_s[0, 0, 0, 1, 0] / M_s[0, 0, 0, 0, 0]
        R_s = np.abs(r_s) ** 2

        t_s = 1.0 / M_s[0, 0, 0, 0, 0]
        theta_i_rad = np.radians(0.0)
        theta_f_rad = np.radians(as_ndarray(result.theta)[0, -1])

        # Extract incident and substrate from result.n
        n_incident = as_ndarray(result.n)[0, 0]
        n_substrate = as_ndarray(result.n)[-1, 0]

        angle_factor = np.real(n_substrate * np.cos(theta_f_rad)) / np.real(
            n_incident * np.cos(theta_i_rad)
        )
        T_s = np.abs(t_s) ** 2 * angle_factor

        # Energy conservation for lossless media: R + T = 1
        xp_assert_close(R_s + T_s, 1.0, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_n_dimensional_matrix_transfer_tmm(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.phenomena.tmm.matrix_transfer_tmm`
        definition n-dimensional arrays support.
        """

        wl = 555
        result = matrix_transfer_tmm(n=[1.0, 1.5, 1.0], t=[250], theta=0, wavelength=wl)

        wl_array = xp.tile(xp_as_array(wl, xp=xp), (6,))
        result_array = matrix_transfer_tmm(
            n=[1.0, 1.5, 1.0], t=[250], theta=0, wavelength=wl_array
        )

        # Check shape - (W, A, T, 2, 2)
        assert as_ndarray(result_array.M_s).shape == (
            6,
            1,
            1,
            2,
            2,
        )  # (wavelengths=6, angles=1, thickness=1, 2, 2)
        assert as_ndarray(result_array.M_p).shape == (6, 1, 1, 2, 2)

        # theta shapes: result has (1, 3), result_array has (1, 3)
        assert as_ndarray(result_array.theta).shape == result.theta.shape
        # n shapes: result has (3, 1), result_array has (3, 6)
        # For constant n, all wavelength columns should match
        assert as_ndarray(result_array.n).shape == (3, 6)
        assert result.n.shape == (3, 1)
        xp_assert_close(
            result_array.n[:, 0],
            result.n[:, 0],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
