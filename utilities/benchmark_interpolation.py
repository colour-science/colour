"""
Interpolation Backend Benchmark
===============================

Compare, across data lengths and devices, the *SciPy* reference interpolators
against the :mod:`colour.algebra.interpolation` dispatching
interpolators used by colour. They delegate *NumPy* inputs to *SciPy* and
*PyTorch* / *JAX* inputs to native, automatic-differentiation-preserving
implementations dispatched by the input array namespace.

Configurations:

-   ``scipy/numpy`` -- the raw *SciPy* interpolator (reference),
-   ``xp/numpy`` -- the dispatching interpolator on *NumPy* (delegates to
    *SciPy*; shows the dispatch overhead),
-   ``xp/torch-cpu`` -- the dispatching interpolator on a *PyTorch* CPU tensor
    (native),
-   ``xp/torch-gpu`` -- the dispatching interpolator on a *PyTorch* GPU tensor
    (native), when a device is available,
-   ``xp/jax-cpu`` -- the dispatching interpolator on a *JAX* CPU array (native),
-   ``xp/jax-gpu`` -- the dispatching interpolator on a *JAX* GPU array (native),
    when a device is available.

Construction and evaluation are timed separately: the native cubic spline
solves its interpolation system once at construction. Asynchronous backends
(*JAX*, *CUDA* / *ROCm*) are synchronised inside the timed region so the reported
time is computation, not dispatch latency.

Usage
-----
    python utilities/benchmark_interpolation.py
"""

from __future__ import annotations

import time
import typing

import numpy as np
import scipy.interpolate

from colour.algebra import interpolation as xp_interpolation

if typing.TYPE_CHECKING:
    from collections.abc import Callable

    from colour.hints import NDArrayFloat

try:
    import torch

    torch.set_default_dtype(torch.float64)
except ImportError:  # pragma: no cover
    torch = None

try:
    import jax
    import jax.numpy as jnp

    # Match the *float64* precision used by the *NumPy* / *PyTorch* configurations.
    jax.config.update("jax_enable_x64", True)
except ImportError:  # pragma: no cover
    jax = None
    jnp = None

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "DATA_LENGTHS",
    "QUERY_FACTOR",
    "measure",
    "benchmark_interpolator",
    "main",
]

DATA_LENGTHS: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048)
"""Numbers of knots used for the independent and dependent variables."""

QUERY_FACTOR: int = 4
"""Number of evaluation points expressed as a multiple of the knot count."""

_RUNS: int = 7
"""Timed repetitions retained per measurement."""

_WARMUP: int = 2
"""Untimed repetitions preceding each measurement."""


def measure(
    function: Callable[[], object],
    synchronize: Callable[[object], object] = lambda _result: None,
) -> float:
    """
    Return the minimum wall-clock time of the specified callable in seconds.

    The minimum rejects scheduler and cache noise. ``synchronize`` receives the
    result of ``function`` and is awaited inside the timed region so an
    asynchronous backend (*JAX*, *ROCm* / *CUDA*) reports computation time rather
    than enqueue latency.
    """

    for _ in range(_WARMUP):
        synchronize(function())

    times = []
    for _ in range(_RUNS):
        start = time.perf_counter()
        synchronize(function())
        times.append(time.perf_counter() - start)

    return min(times)


def _sample_data(length: int) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Return strictly increasing knots, smooth values and evaluation points."""

    x = np.linspace(0.0, 1.0, length)
    y = np.sin(12.0 * x) + 0.25 * np.cos(37.0 * x)
    x_e = np.linspace(x[0], x[-1], length * QUERY_FACTOR)

    return x, y, x_e


def _jax_synchronize(result: object) -> None:
    """
    Block until the specified *JAX* result is ready.

    ``result`` is either a *JAX* array (evaluation) or a dispatching interpolator
    (construction); in the latter case the interpolant's solved on-device arrays
    are awaited so construction timing includes the solve, not just its dispatch.
    """

    if jax is None:  # pragma: no cover
        return

    ready = getattr(result, "block_until_ready", None)
    if ready is not None:
        ready()
        return

    implementation = getattr(result, "_implementation", result)
    jax.block_until_ready(
        [
            value
            for value in vars(implementation).values()
            if hasattr(value, "block_until_ready")
        ]
    )


def _configurations(
    scipy_constructor: Callable[[NDArrayFloat, NDArrayFloat], object],
    native_constructor: Callable[..., object],
) -> list[tuple[str, Callable, Callable, Callable]]:
    """Return the ``(name, constructor, cast, synchronize)`` configurations."""

    identity: Callable[[NDArrayFloat], object] = lambda a: a  # noqa: E731
    noop: Callable[[object], object] = lambda _result: None  # noqa: E731

    configurations: list = [
        ("scipy/numpy", scipy_constructor, identity, noop),
        ("xp/numpy", native_constructor, identity, noop),
    ]

    if torch is not None:
        torch_ = torch
        configurations.append(
            (
                "xp/torch-cpu",
                native_constructor,
                lambda a: torch_.asarray(a, device="cpu"),
                noop,
            )
        )
        if torch_.cuda.is_available():
            configurations.append(
                (
                    "xp/torch-gpu",
                    native_constructor,
                    lambda a: torch_.asarray(a, device="cuda"),
                    lambda _result: torch_.cuda.synchronize(),
                )
            )

    if jax is not None and jnp is not None:
        jax_, jnp_ = jax, jnp
        cpu_device = jax_.devices("cpu")[0]
        configurations.append(
            (
                "xp/jax-cpu",
                native_constructor,
                lambda a, d=cpu_device: jax_.device_put(jnp_.asarray(a), d),
                _jax_synchronize,
            )
        )

        try:
            gpu_device = jax_.devices("gpu")[0]
        except RuntimeError:
            gpu_device = None
        if gpu_device is not None:
            configurations.append(
                (
                    "xp/jax-gpu",
                    native_constructor,
                    lambda a, d=gpu_device: jax_.device_put(jnp_.asarray(a), d),
                    _jax_synchronize,
                )
            )

    return configurations


def _print_table(
    title: str, names: list[str], rows: list[tuple[int, list[float]]]
) -> None:
    """Print a table of per-configuration timings in milliseconds."""

    print(f"\n{title}")  # noqa: T201
    print("-" * len(title))  # noqa: T201
    print(f"{'knots':>7} | " + " ".join(f"{name:>18}" for name in names))  # noqa: T201
    for length, values in rows:
        cells = " ".join(f"{value * 1e3:>18.4f}" for value in values)
        print(f"{length:>7} | {cells}")  # noqa: T201


def benchmark_interpolator(
    label: str,
    scipy_constructor: Callable[[NDArrayFloat, NDArrayFloat], object],
    native_constructor: Callable[..., object],
) -> None:
    """
    Benchmark the *SciPy* reference against the dispatching interpolators
    across configurations.
    """

    configurations = _configurations(scipy_constructor, native_constructor)
    names = [name for name, *_ in configurations]

    build_rows: list[tuple[int, list[float]]] = []
    call_rows: list[tuple[int, list[float]]] = []
    for length in DATA_LENGTHS:
        x_np, y_np, x_e_np = _sample_data(length)

        builds, calls = [], []
        for _name, constructor, cast, synchronize in configurations:
            x, y, x_e = cast(x_np), cast(y_np), cast(x_e_np)
            synchronize(x_e)  # Settle the device transfer before timing.

            builds.append(measure(lambda c=constructor, x=x, y=y: c(x, y), synchronize))

            interpolator = constructor(x, y)
            synchronize(interpolator)
            calls.append(measure(lambda f=interpolator, x_e=x_e: f(x_e), synchronize))

        build_rows.append((length, builds))
        call_rows.append((length, calls))

    _print_table(f"{label} - construction (ms, min of {_RUNS} runs)", names, build_rows)
    _print_table(
        f"{label} - evaluation (ms, {QUERY_FACTOR}x knots query points)",
        names,
        call_rows,
    )


def main() -> None:
    """Run the interpolation backend benchmark."""

    if torch is None:  # pragma: no cover
        print('"PyTorch" is not installed; only NumPy configurations are shown.')  # noqa: T201

    benchmark_interpolator(
        "Cubic spline",
        lambda x, y: scipy.interpolate.interp1d(x, y, kind="cubic"),
        xp_interpolation.CubicSplineInterpolator,
    )
    benchmark_interpolator(
        "PCHIP",
        scipy.interpolate.PchipInterpolator,
        xp_interpolation.PchipInterpolator,
    )


if __name__ == "__main__":
    main()
