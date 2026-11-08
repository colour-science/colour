"""
Pytest Configuration
====================

Configure *pytest* with array backend fixtures for *Array API* testing.
"""

from __future__ import annotations

import contextlib
import os
import sys
import typing

import numpy as np
import pytest

if typing.TYPE_CHECKING:
    from colour.hints import Generator, ModuleType

from colour.constants import (
    DTYPE_COMPLEX_DEFAULT,
    DTYPE_FLOAT_DEFAULT,
    TOLERANCE_ABSOLUTE_TESTS,
)
from colour.utilities import (
    array_api_enable,
    as_ndarray,
    set_default_complex_dtype,
    set_default_float_dtype,
    xp_as_float_array,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "autodiff",
    "xp",
]

try:
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

try:
    import torch

    torch.set_default_dtype(torch.float64)
except ImportError:
    torch = None


_TEST_BACKENDS: str | None = os.environ.get("COLOUR_SCIENCE__TEST_BACKENDS")
"""
Optional comma-separated list of backend parameter ids (``numpy``, ``jax``,
``jax-cuda``, ``torch``, ``torch-mps``, ``torch-cuda``) restricting the
:func:`xp` fixture parametrisation. Unset yields every installed backend.
"""

_AUTODIFF_FULL_FINITE_DIFFERENCE_SIZE: int = 64
"""Maximum input size for component-wise finite-difference checking."""

_AUTODIFF_DIRECTIONAL_CHECKS: int = 2
"""Directional finite-difference checks used for larger inputs."""

_AUTODIFF_DIRECTION_SEED: int = 0
"""Seed ensuring deterministic finite-difference directions."""

_AUTODIFF_RELATIVE_TOLERANCE: float = 5e-4
"""Minimum relative tolerance for finite-difference comparisons."""

_AUTODIFF_ABSOLUTE_TOLERANCE: float = 1e-6
"""Minimum absolute tolerance for finite-difference comparisons."""


def _test_backend_requested(backend: str) -> bool:
    """Return whether the specified test backend was requested."""

    if _TEST_BACKENDS is None:
        return True

    return backend in {
        token.strip() for token in _TEST_BACKENDS.split(",") if token.strip()
    }


def _jax_device(platform: str) -> object | None:
    """Return the first available *JAX* device for the specified platform."""

    if jax is None:
        return None

    try:
        devices = jax.devices(platform)
    except RuntimeError:
        return None

    return devices[0] if devices else None


_JAX_CPU_DEVICE = _jax_device("cpu") if _test_backend_requested("jax") else None
_JAX_CUDA_DEVICE = _jax_device("gpu") if _test_backend_requested("jax-cuda") else None


def _make_backend_parameters() -> list:
    """Build the parametrised backend list."""

    params = [pytest.param((np, "numpy"), id="numpy")]

    if jnp is not None:
        params.append(pytest.param((jnp, "jax"), id="jax"))
        if _JAX_CUDA_DEVICE is not None:
            params.append(pytest.param((jnp, "jax-cuda"), id="jax-cuda"))

    if torch is not None:
        params.append(pytest.param((torch, "torch"), id="torch"))
        if torch.backends.mps.is_available():
            params.append(pytest.param((torch, "torch-mps"), id="torch-mps"))
        if torch.cuda.is_available():
            params.append(pytest.param((torch, "torch-cuda"), id="torch-cuda"))

    if _TEST_BACKENDS is None:
        return params

    requested = [token.strip() for token in _TEST_BACKENDS.split(",") if token.strip()]
    available = {str(parameter.id): parameter for parameter in params}

    missing = [backend for backend in requested if backend not in available]
    if missing:
        pytest.exit(
            f"COLOUR_SCIENCE__TEST_BACKENDS requests unavailable backend(s): "
            f"{', '.join(missing)}; available: {', '.join(available)}.",
            returncode=1,
        )

    return [available[backend] for backend in requested]


@pytest.fixture(params=_make_backend_parameters())
def xp(request: pytest.FixtureRequest) -> Generator[ModuleType, None, None]:
    """
    Parametrised array namespace fixture.

    Yields :mod:`numpy` and, when available, :mod:`jax.numpy` and
    :mod:`torch`. Non-NumPy backends automatically enable Array API dispatch
    for the duration of the test. The ``jax-cuda``, ``torch-mps`` and
    ``torch-cuda`` variants additionally set the matching default device. The
    ``torch-mps`` variant also sets the default dtype to ``float32``.
    """

    backend, variant = request.param

    if variant == "numpy":
        yield backend
    elif variant in ("jax", "jax-cuda"):
        device = _JAX_CUDA_DEVICE if variant == "jax-cuda" else _JAX_CPU_DEVICE
        default_device = jax.default_device(device)  # pyright: ignore
        with array_api_enable(True), default_device:
            yield backend
    elif variant == "torch-cuda":
        with array_api_enable(True):
            default_device = torch.get_default_device()  # pyright: ignore
            torch.set_default_device("cuda")  # pyright: ignore

            try:
                yield backend
            finally:
                torch.set_default_device(default_device)  # pyright: ignore
    elif variant == "torch-mps":
        with array_api_enable(True):
            default_dtype = torch.get_default_dtype()  # pyright: ignore
            torch.set_default_dtype(torch.float32)  # pyright: ignore
            torch.set_default_device("mps")  # pyright: ignore
            set_default_float_dtype(np.float32)
            set_default_complex_dtype(np.complex64)

            # Relax test tolerance for float32 precision. A per-test
            # ``@pytest.mark.mps_tolerance_absolute(value)`` marker overrides
            # the ``5e-4`` default for tests whose float32 deltas need more
            # headroom. Tests that thread
            # :attr:`colour.constants.TOLERANCE_ABSOLUTE_TESTS` honour it, as
            # do ``xp_assert_close`` calls relying on the default tolerances
            # (resolved at call time); hard-coded tolerance literals do not.
            marker = request.node.get_closest_marker("mps_tolerance_absolute")
            tolerance = marker.args[0] if marker else 5e-4
            # The original tolerance is snapshotted into a local BEFORE the
            # sweep: the sweep patches every module holding the constant,
            # including this ``conftest`` module, so restoring from the
            # module-level name in the ``finally`` would restore the patched
            # value and leak the relaxed tolerance across the whole worker.
            tolerance_original = TOLERANCE_ABSOLUTE_TESTS
            for module in sys.modules.values():
                if hasattr(module, "TOLERANCE_ABSOLUTE_TESTS"):
                    module.TOLERANCE_ABSOLUTE_TESTS = tolerance  # pyright: ignore

            # Tests that cannot pass at any sane tolerance under float32
            # (large-magnitude radiometry, divergent solvers, hard-coded
            # tolerance literals) opt in to a strict expected failure via
            # ``@pytest.mark.mps_xfail("reason")``. ``strict=True`` makes
            # an unexpected pass a CI failure, so the marker stays honest
            # as *MPS* support improves.
            xfail_marker = request.node.get_closest_marker("mps_xfail")
            if xfail_marker is not None:
                request.node.add_marker(
                    pytest.mark.xfail(
                        reason=xfail_marker.args[0]
                        if xfail_marker.args
                        else "MPS float32",
                        raises=(AssertionError, RuntimeError, TypeError),
                        strict=True,
                    )
                )

            try:
                yield backend
            finally:
                torch.set_default_device("cpu")  # pyright: ignore
                torch.set_default_dtype(default_dtype)  # pyright: ignore
                set_default_float_dtype(np.float64)
                set_default_complex_dtype(np.complex128)

                for module in sys.modules.values():
                    if hasattr(module, "TOLERANCE_ABSOLUTE_TESTS"):
                        module.TOLERANCE_ABSOLUTE_TESTS = (  # pyright: ignore
                            tolerance_original
                        )
    else:
        with array_api_enable(True):
            yield backend


@pytest.fixture
def autodiff(xp: ModuleType) -> typing.Callable:
    """
    Return a backend-aware automatic differentiation verification helper.

    Reverse-mode gradients are compared with central finite differences and
    checked for device retention. Small inputs receive a component-wise
    comparison; larger inputs use deterministic directional derivatives to
    keep spectral regression tests tractable. Inconclusive float32 differences
    receive an independent float64 CPU reference using the same callable and
    input values. Outputs, analytical gradients and device checks remain native,
    and comparison tolerances are unchanged.
    """

    if xp.__name__ == "numpy":
        pytest.skip("Automatic differentiation requires *JAX* or *PyTorch*.")

    def evaluate(function: typing.Callable, *inputs: typing.Any) -> tuple:
        """Evaluate a function and verify its summed-output gradients."""

        variables: tuple[typing.Any, ...] = tuple(
            xp_as_float_array(value, xp=xp) for value in inputs
        )

        if xp.__name__ == "torch":
            variables = tuple(
                variable.detach().clone().requires_grad_(True) for variable in variables
            )
            result = function(*variables)
            gradients = tuple(xp.autograd.grad(xp.sum(result), variables))
        else:

            def objective_with_result(*arguments: object) -> tuple:
                """Return the summed objective and original result."""

                value = function(*arguments)

                return xp.sum(value), value

            (_objective, result), gradients = jax.value_and_grad(  # pyright: ignore
                objective_with_result,
                argnums=tuple(range(len(variables))),
                has_aux=True,
            )(*variables)

        # Outputs and analytical gradients always come from the requested
        # backend/device. A higher-precision reference below is verification
        # only and never substitutes for these gradients.
        device = getattr(variables[0], "device", None)
        for value in (result, *gradients):
            if not bool(xp.all(xp.isfinite(value))):
                pytest.fail("Automatic differentiation produced a nonfinite value.")
            if device is not None and getattr(value, "device", None) != device:
                pytest.fail(
                    "Automatic differentiation moved a value off its input device."
                )

        def central_difference(
            arguments: tuple, index: int, direction: typing.Any, step: float
        ) -> tuple[float, float]:
            """Return the difference and its output-rounding error scale."""

            plus = list(arguments)
            minus = list(arguments)
            plus[index] = arguments[index] + step * direction
            minus[index] = arguments[index] - step * direction
            try:
                with (
                    xp.no_grad() if xp.__name__ == "torch" else contextlib.nullcontext()
                ):
                    output_plus = function(*plus)
                    output_minus = function(*minus)
            except (TypeError, RuntimeError, ValueError) as error:
                if reference_active:
                    pytest.fail(
                        "The float64 CPU reference could not evaluate the same "
                        f"callable; check captured device-specific state: {error}"
                    )
                raise
            if reference_active:
                for output in (output_plus, output_minus):
                    if (
                        str(output.dtype) not in ("torch.float64", "float64")
                        or "cpu" not in str(output.device).lower()
                    ):
                        pytest.fail(
                            "The float64 CPU reference retained captured "
                            "device-specific state or could not evaluate in the "
                            "requested precision."
                        )
            epsilon = float(xp.finfo(output_plus.dtype).eps)
            plus_values = as_ndarray(output_plus).astype(np.float64)
            minus_values = as_ndarray(output_minus).astype(np.float64)
            # Subtract outputs before summing to avoid cancellation against
            # the float32 summed objective's potentially large offset.
            derivative = float(np.sum(plus_values - minus_values) / (2 * step))
            roundoff = float(
                epsilon
                * np.sum(np.abs(plus_values) + np.abs(minus_values))
                / (2 * step)
            )

            return derivative, roundoff

        rtol = max(_AUTODIFF_RELATIVE_TOLERANCE, 10 * TOLERANCE_ABSOLUTE_TESTS)
        atol = max(_AUTODIFF_ABSOLUTE_TOLERANCE, TOLERANCE_ABSOLUTE_TESTS)
        reference_active = False
        reference_checks = []
        for index, (variable, gradient) in enumerate(
            zip(variables, gradients, strict=True)
        ):
            values = as_ndarray(variable)
            epsilon = float(xp.finfo(variable.dtype).eps)
            rng = np.random.default_rng(_AUTODIFF_DIRECTION_SEED + index)
            count = (
                values.size
                if values.size <= _AUTODIFF_FULL_FINITE_DIFFERENCE_SIZE
                else _AUTODIFF_DIRECTIONAL_CHECKS
            )
            for element in range(count):
                if values.size <= _AUTODIFF_FULL_FINITE_DIFFERENCE_SIZE:
                    direction_values = np.zeros_like(values)
                    direction_values.flat[element] = 1
                    scale = max(1, abs(float(values.flat[element])))
                else:
                    direction_values = rng.uniform(-1, 1, variable.shape)
                    direction_values /= np.max(np.abs(direction_values))
                    scale = max(1, float(np.max(np.abs(values))))
                direction = xp.asarray(
                    direction_values, dtype=variable.dtype, device=variable.device
                )
                analytical = float(as_ndarray(xp.sum(gradient * direction)))
                step = epsilon ** (1 / 3) * scale
                numerical, roundoff = central_difference(
                    variables, index, direction, step
                )
                allowance = atol + rtol * abs(numerical)
                if epsilon >= np.finfo(np.float32).eps:
                    refined, refined_roundoff = central_difference(
                        variables, index, direction, step / 2
                    )
                    uncertainty = abs(refined - numerical) + max(
                        roundoff, refined_roundoff
                    )
                    # Cancellation or branch-crossing can make native float32
                    # differences inconclusive. Resolve them independently at
                    # float64 precision rather than enlarging tolerances.
                    if uncertainty > allowance or abs(
                        analytical - refined
                    ) > atol + rtol * abs(refined):
                        reference_checks.append(
                            (index, direction_values, scale, analytical)
                        )
                        continue
                    numerical = refined
                np.testing.assert_allclose(analytical, numerical, rtol=rtol, atol=atol)

        if reference_checks:
            dtype_original = DTYPE_FLOAT_DEFAULT
            complex_original = DTYPE_COMPLEX_DEFAULT
            torch_dtype = xp.get_default_dtype() if xp.__name__ == "torch" else None
            if xp.__name__ == "torch":
                context = xp.device("cpu")
            else:
                jax_backend = jax
                if jax_backend is None:
                    error = "The JAX numerical reference backend is unavailable."
                    raise RuntimeError(error)
                context = jax_backend.default_device(jax_backend.devices("cpu")[0])
            try:
                with context:
                    if xp.__name__ == "torch":
                        xp.set_default_dtype(xp.float64)
                    set_default_float_dtype(np.float64)
                    set_default_complex_dtype(np.complex128)
                    reference_variables = tuple(
                        xp.asarray(as_ndarray(value), dtype=xp.float64, device="cpu")
                        if xp.__name__ == "torch"
                        else xp.asarray(as_ndarray(value), dtype=xp.float64)
                        for value in variables
                    )
                    reference_active = True
                    for index, direction_values, scale, analytical in reference_checks:
                        direction = xp.asarray(direction_values, dtype=xp.float64)
                        step = np.finfo(np.float64).eps ** (1 / 3) * scale
                        numerical, _ = central_difference(
                            reference_variables, index, direction, step
                        )
                        refined, _ = central_difference(
                            reference_variables, index, direction, step / 2
                        )
                        np.testing.assert_allclose(
                            numerical, refined, rtol=rtol, atol=atol
                        )
                        np.testing.assert_allclose(
                            analytical, refined, rtol=rtol, atol=atol
                        )
            finally:
                reference_active = False
                set_default_float_dtype(dtype_original)
                set_default_complex_dtype(complex_original)
                if torch_dtype is not None:
                    xp.set_default_dtype(torch_dtype)

        return result, gradients, variables

    return evaluate
