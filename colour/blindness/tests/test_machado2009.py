"""Define the unit tests for the :mod:`colour.blindness.machado2009` module."""

from __future__ import annotations

import typing
from contextlib import nullcontext

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import numpy as np
import pytest

from colour.blindness import (
    CVD_MATRICES_MACHADO2010,
    matrix_anomalous_trichromacy_Machado2009,
    matrix_cvd_Machado2009,
    msds_cmfs_anomalous_trichromacy_Machado2009,
)
from colour.blindness.machado2009 import MATRIX_LMS_TO_WSYBRG, matrix_RGB_to_WSYBRG
from colour.characterisation import MSDS_DISPLAY_PRIMARIES
from colour.colorimetry import MSDS_CMFS_LMS, SpectralShape
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    array_namespace,
    ignore_numpy_errors,
    xp_as_array,
    xp_as_float_array,
    xp_assert_close,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestMsdsCmfsAnomalousTrichromacyMachado2009",
    "TestMatrixAnomalousTrichromacyMachado2009",
    "TestMatrixCvdMachado2009",
]


class TestMsdsCmfsAnomalousTrichromacyMachado2009:
    """
    Define :func:`colour.blindness.machado2009.\
msds_cmfs_anomalous_trichromacy_Machado2009` definition unit tests methods.
    """

    def test_msds_cmfs_anomalous_trichromacy_Machado2009(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.blindness.machado2009.\
msds_cmfs_anomalous_trichromacy_Machado2009` definition.
        """

        cmfs = MSDS_CMFS_LMS["Smith & Pokorny 1975 Normal Trichromats"]
        xp_assert_close(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                xp_as_array([0, 0, 0], xp=xp),
            )[450],
            cmfs[450],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                xp_as_array([1, 0, 0], xp=xp),
            )[450],
            [0.03631700, 0.06350000, 0.91000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                xp_as_array([0, 1, 0], xp=xp),
            )[450],
            [0.03430000, 0.06178404, 0.91000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                xp_as_array([0, 0, 1], xp=xp),
            )[450],
            [0.03430000, 0.06350000, 0.92270240],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                xp_as_array([10, 0, 0], xp=xp),
            )[450],
            [0.05447001, 0.06350000, 0.91000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                xp_as_array([0, 10, 0], xp=xp),
            )[450],
            [0.03430000, 0.04634036, 0.91000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            msds_cmfs_anomalous_trichromacy_Machado2009(
                cmfs,
                xp_as_array([0, 0, 10], xp=xp),
            )[450],
            [0.03430000, 0.06350000, 1.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestMatrixAnomalousTrichromacyMachado2009:
    """
    Define :func:`colour.blindness.machado2009.\
matrix_anomalous_trichromacy_Machado2009` definition unit tests methods.
    """

    def test_matrix_anomalous_trichromacy_Machado2009(
        self, xp: ModuleType, request: pytest.FixtureRequest
    ) -> None:
        """
        Test :func:`colour.blindness.machado2009.\
matrix_anomalous_trichromacy_Machado2009` definition.
        """

        cmfs = MSDS_CMFS_LMS["Smith & Pokorny 1975 Normal Trichromats"]
        primaries = MSDS_DISPLAY_PRIMARIES["Typical CRT Brainard 1997"]
        xp_assert_close(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, xp_as_array([0, 0, 0], xp=xp)
            ),
            np.identity(3),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, xp_as_array([2, 0, 0], xp=xp)
            ),
            CVD_MATRICES_MACHADO2010["Protanomaly"][0.1],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.0001),
        )

        xp_assert_close(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, xp_as_array([10, 0, 0], xp=xp)
            ),
            CVD_MATRICES_MACHADO2010["Protanomaly"][0.5],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.0001),
        )

        xp_assert_close(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, xp_as_array([20, 0, 0], xp=xp)
            ),
            CVD_MATRICES_MACHADO2010["Protanomaly"][1.0],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.0001),
        )

        xp_assert_close(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, xp_as_array([0, 2, 0], xp=xp)
            ),
            CVD_MATRICES_MACHADO2010["Deuteranomaly"][0.1],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.0001),
        )

        xp_assert_close(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, xp_as_array([0, 10, 0], xp=xp)
            ),
            CVD_MATRICES_MACHADO2010["Deuteranomaly"][0.5],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.0001),
        )

        xp_assert_close(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, xp_as_array([0, 20, 0], xp=xp)
            ),
            CVD_MATRICES_MACHADO2010["Deuteranomaly"][1.0],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.0001),
        )

        xp_assert_close(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, xp_as_array([0, 0, 5.00056688094503], xp=xp)
            ),
            CVD_MATRICES_MACHADO2010["Tritanomaly"][0.1],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.0001),
        )

        xp_assert_close(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, xp_as_array([0, 0, 29.002939088780934], xp=xp)
            ),
            CVD_MATRICES_MACHADO2010["Tritanomaly"][0.5],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.0001),
        )

        xp_assert_close(
            matrix_anomalous_trichromacy_Machado2009(
                cmfs, primaries, xp_as_array([0, 0, 59.00590434857581], xp=xp)
            ),
            CVD_MATRICES_MACHADO2010["Tritanomaly"][1.0],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.001),
        )

        # Numeric spectral samples can be native independently of the host
        # wavelength/shape metadata and of the other spectrum.
        for native_cmfs, native_primaries in (
            (True, False),
            (False, True),
            (True, True),
        ):
            cones = cmfs.copy()
            display = primaries.copy().align(cones.shape)
            expected = matrix_anomalous_trichromacy_Machado2009(
                cones, display, [2, 0, 0]
            )
            reference = xp_as_array(cones.values, xp=xp)
            if native_cmfs:
                cones.values = reference
            if native_primaries:
                display.values = xp_as_array(display.values, xp=xp)
                reference = display.values
            with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
                actual = matrix_anomalous_trichromacy_Machado2009(
                    cones, display, [2, 0, 0]
                )
            assert array_namespace(actual) is array_namespace(reference)
            assert getattr(actual, "device", None) == getattr(reference, "device", None)
            xp_assert_close(actual, expected, atol=TOLERANCE_ABSOLUTE_TESTS)

        # Use well-conditioned smooth spectra to compare every derivative
        # with the trapezoidal integral and row-normalisation quotient rule.
        cones = cmfs.copy().align(SpectralShape(400, 700, 10))
        display = primaries.copy().align(cones.shape)
        t = np.linspace(0, 1, len(cones.wavelengths))
        cone_values = np.column_stack([2 + t, 0.3 + t**2, 0.1 + 0.2 * t])
        display_values = np.column_stack([0.2 + t, 0.3 + t**2, 0.4 + t**3])
        wavelengths = cones.wavelengths
        weights = np.empty_like(wavelengths)
        weights[0] = (wavelengths[1] - wavelengths[0]) / 2
        weights[-1] = (wavelengths[-1] - wavelengths[-2]) / 2
        weights[1:-1] = (wavelengths[2:] - wavelengths[:-2]) / 2
        opponent = cone_values @ MATRIX_LMS_TO_WSYBRG.T
        integrals = np.einsum("nk,nj,n->kj", opponent, display_values, weights)
        expected = integrals / integrals.sum(axis=1, keepdims=True)
        loss_weights = np.arange(9).reshape(3, 3) / 8
        quotient = (
            loss_weights - (expected * loss_weights).sum(axis=1, keepdims=True)
        ) / integrals.sum(axis=1, keepdims=True)
        expected_gradients = (
            np.einsum(
                "kj,nj,n,kl->nl",
                quotient,
                display_values,
                weights,
                MATRIX_LMS_TO_WSYBRG,
            ),
            np.einsum("kj,nk,n->nj", quotient, opponent, weights),
        )

        def matrix(cone_samples: typing.Any, display_samples: typing.Any) -> typing.Any:
            cones.values = cone_samples
            display.values = display_samples
            return matrix_RGB_to_WSYBRG(cones, display)

        def objective(
            cone_samples: typing.Any, display_samples: typing.Any
        ) -> typing.Any:
            result = matrix(cone_samples, display_samples)
            return xp.sum(result * xp_as_float_array(loss_weights, xp=xp, like=result))

        for native_cmfs, native_primaries in (
            (True, False),
            (False, True),
            (True, True),
        ):
            inputs = (
                xp_as_array(cone_values, xp=xp) if native_cmfs else cone_values,
                xp_as_array(display_values, xp=xp)
                if native_primaries
                else display_values,
            )
            reference = inputs[0] if native_cmfs else inputs[1]
            with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
                actual = matrix(*inputs)
                assert array_namespace(actual) is array_namespace(reference)
                assert getattr(actual, "device", None) == getattr(
                    reference, "device", None
                )
                xp_assert_close(actual, expected, atol=TOLERANCE_ABSOLUTE_TESTS)
                if xp.__name__ != "numpy":
                    if native_cmfs and native_primaries:
                        _, gradients, _ = request.getfixturevalue("autodiff")(
                            objective, *inputs
                        )
                        for gradient, expected_gradient in zip(
                            gradients, expected_gradients, strict=True
                        ):
                            assert bool(xp.any(gradient != 0))
                            xp_assert_close(
                                gradient,
                                expected_gradient,
                                atol=TOLERANCE_ABSOLUTE_TESTS,
                            )
                        if xp.__name__.startswith("jax"):
                            jax = pytest.importorskip("jax")
                            for differentiate in (jax.jacfwd, jax.jacrev):
                                derivatives = differentiate(objective, argnums=(0, 1))(
                                    *inputs
                                )
                                for derivative, expected_gradient in zip(
                                    derivatives, expected_gradients, strict=True
                                ):
                                    assert derivative.device == reference.device
                                    xp_assert_close(
                                        derivative,
                                        expected_gradient,
                                        atol=TOLERANCE_ABSOLUTE_TESTS,
                                    )
                    for index, native in enumerate((native_cmfs, native_primaries)):
                        if not native:
                            continue

                        def function(
                            value: typing.Any,
                            index: int = index,
                        ) -> typing.Any:
                            arguments = [cone_values, display_values]
                            arguments[index] = value
                            return objective(*arguments)

                        _, (gradient,), _ = request.getfixturevalue("autodiff")(
                            function, inputs[index]
                        )
                        assert bool(xp.any(gradient != 0))
                        xp_assert_close(
                            gradient,
                            expected_gradients[index],
                            atol=TOLERANCE_ABSOLUTE_TESTS,
                        )
                        if xp.__name__.startswith("jax"):
                            jax = pytest.importorskip("jax")

                            forward = jax.jacfwd(function)(inputs[index])
                            reverse = jax.jacrev(function)(inputs[index])
                            for derivative in (forward, reverse):
                                assert derivative.device == reference.device
                                xp_assert_close(
                                    derivative,
                                    expected_gradients[index],
                                    atol=TOLERANCE_ABSOLUTE_TESTS,
                                )


class TestMatrixCvdMachado2009:
    """
    Define :func:`colour.blindness.machado2009.matrix_cvd_Machado2009`
    definition unit tests methods.
    """

    def test_matrix_cvd_Machado2009(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.blindness.machado2009.matrix_cvd_Machado2009`
        definition.
        """

        xp_assert_close(
            matrix_cvd_Machado2009("Protanomaly", 0.0),
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_cvd_Machado2009("Deuteranomaly", 0.1),
            [
                [0.86643500, 0.17770400, -0.04413900],
                [0.04956700, 0.93906300, 0.01137000],
                [-0.00345300, 0.00723300, 0.99622000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_cvd_Machado2009("Tritanomaly", 1.0),
            [
                [1.25552800, -0.07674900, -0.17877900],
                [-0.07841100, 0.93080900, 0.14760200],
                [0.00473300, 0.69136700, 0.30390000],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_cvd_Machado2009("Tritanomaly", 0.55),
            [
                [1.06088700, -0.01504350, -0.04584350],
                [-0.01895750, 0.96774750, 0.05121150],
                [0.00317700, 0.27513700, 0.72168600],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_matrix_cvd_Machado2009(self) -> None:
        """
        Test :func:`colour.blindness.machado2009.matrix_cvd_Machado2009`
        definition nan support.
        """

        for case in [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]:
            matrix_cvd_Machado2009("Tritanomaly", case)
