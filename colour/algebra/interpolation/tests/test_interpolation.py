"""
Unit tests for the :mod:`colour.algebra.interpolation` dispatching
interpolators.

Backend-parametrised: each interpolator is exercised on every backend whose
framework is installed *and* whose ``backends`` submodule exists, so
coverage grows automatically as backends are added.
"""

from __future__ import annotations

import importlib
import inspect
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import scipy.interpolate

from colour.algebra import Extrapolator
from colour.algebra.interpolation import (
    CubicSplineInterpolator,
    KernelInterpolator,
    LinearInterpolator,
    NearestNeighbourInterpolator,
    NullInterpolator,
    PchipInterpolator,
    SpragueInterpolator,
    kernel_linear,
)
from colour.algebra.interpolation._dispatch import detect_backend
from colour.algebra.interpolation.backends._core import reflect_indices
from colour.constants import DTYPE_FLOAT_DEFAULT
from colour.utilities import array_api_enable

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import torch

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

# ``x`` used across the tests: strictly increasing, >= 4 points (cubic).
_X = np.array([0.0, 0.4, 1.1, 2.0, 3.5, 5.0])
_Y = np.sin(1.7 * _X) + 0.3 * np.cos(2.9 * _X)
_Y2 = np.stack([_Y, np.cos(1.3 * _X)], axis=-1)
_X_E = np.linspace(_X[0], _X[-1], 23)

# Uniformly spaced data for the Sprague interpolator (requires >= 6 points).
_X_UNIFORM = np.arange(7.0)
_Y_UNIFORM = np.array([5.92, 9.37, 10.8135, 4.51, 69.59, 27.8007, 86.05])
_X_E_UNIFORM = np.linspace(0.0, 6.0, 25)


def _to_numpy(a: Any) -> np.ndarray:
    """Return the specified array materialised as a *NumPy* array."""

    return np.asarray(a, dtype=np.float64)


def _available_backends() -> list[tuple[str, Callable[[Any], Any]]]:
    """Return ``(name, cast)`` for each usable backend."""

    backends: list[tuple[str, Callable[[Any], Any]]] = [("numpy", _to_numpy)]

    try:
        import torch  # noqa: PLC0415

        backends.append(("torch", lambda a: torch.asarray(_to_numpy(a))))
    except ImportError:
        pass

    try:
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        jax.config.update("jax_enable_x64", True)
        backends.append(("jax", lambda a: jnp.asarray(_to_numpy(a))))
    except ImportError:
        pass

    # Only keep backends whose implementation submodule exists.
    usable = []
    for name, cast in backends:
        try:
            importlib.import_module(f"colour.algebra.interpolation.backends.{name}")
        except ImportError:
            continue
        usable.append((name, cast))

    return usable


_BACKENDS = _available_backends()

# Non-*NumPy* backends, used by the tests that exercise backend switching (which
# only happens under Array API dispatch for framework data).
_FRAMEWORK_BACKENDS = [(name, cast) for name, cast in _BACKENDS if name != "numpy"]


@pytest.fixture(params=[name for name, _ in _BACKENDS])
def backend(
    request: pytest.FixtureRequest,
) -> Generator[tuple[str, Callable[[Any], Any]], None, None]:
    """
    Yield ``(name, cast)`` for each available backend.

    Non-*NumPy* backends enable Array API dispatch for the duration of the test,
    matching the project-wide ``xp`` fixture: a backend is only engaged when the
    caller has opted into dispatch.
    """

    item = next(item for item in _BACKENDS if item[0] == request.param)
    if item[0] == "numpy":
        yield item
    else:
        with array_api_enable(True):
            yield item


def test_detect_backend_numpy() -> None:
    """Test backend detection for *NumPy* inputs and Python scalars."""

    assert detect_backend(_X, _Y) == "numpy"
    assert detect_backend([0.0, 1.0], 2.0) == "numpy"


# ``x``, ``y`` per interpolator; 2.5 is in-range and off-knot for both datasets.
_INTERPOLATOR_CASES = [
    ("linear", LinearInterpolator, _X, _Y),
    ("nearest", NearestNeighbourInterpolator, _X, _Y),
    ("null", NullInterpolator, _X, _Y),
    ("sprague", SpragueInterpolator, _X_UNIFORM, _Y_UNIFORM),
    ("cubic", CubicSplineInterpolator, _X, _Y),
    ("pchip", PchipInterpolator, _X, _Y),
    # The kernel (convolution) interpolator assumes uniformly spaced data.
    ("kernel", KernelInterpolator, _X_UNIFORM, _Y_UNIFORM),
]


@pytest.mark.parametrize(
    ("interpolator_type", "x", "y"),
    [pytest.param(t, x, y, id=i) for i, t, x, y in _INTERPOLATOR_CASES],
)
def test_scalar_query(
    backend: tuple[str, Callable[[Any], Any]],
    interpolator_type: type,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    """Test that a 0-d (scalar) query returns a 0-d result."""

    _name, cast = backend

    interpolator = interpolator_type(cast(x), cast(y))
    result = interpolator(cast(2.5))

    assert _to_numpy(result).shape == ()


@pytest.mark.parametrize(
    ("interpolator_type", "x", "y"),
    [pytest.param(t, x, y, id=i) for i, t, x, y in _INTERPOLATOR_CASES],
)
def test_empty_query(
    backend: tuple[str, Callable[[Any], Any]],
    interpolator_type: type,
    x: np.ndarray,
    y: np.ndarray,
) -> None:
    """Test that an empty (size-0) query returns an empty result."""

    _name, cast = backend

    interpolator = interpolator_type(cast(x), cast(y))
    result = interpolator(cast(np.array([])))

    assert _to_numpy(result).shape == (0,)


def test_cubic_spline_matches_scipy(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test the dispatching cubic spline against the *SciPy* reference."""

    name, cast = backend
    interpolator = CubicSplineInterpolator(cast(_X), cast(_Y))

    assert interpolator.backend == name

    reference = scipy.interpolate.interp1d(_X, _Y, kind="cubic")(_X_E)
    np.testing.assert_allclose(
        _to_numpy(interpolator(cast(_X_E))), reference, atol=1e-10
    )


def test_cubic_spline_rank_2(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching cubic spline on rank-2 dependent data."""

    _name, cast = backend
    interpolator = CubicSplineInterpolator(cast(_X), cast(_Y2), axis=0)

    reference = scipy.interpolate.interp1d(_X, _Y2, kind="cubic", axis=0)(_X_E)
    np.testing.assert_allclose(
        _to_numpy(interpolator(cast(_X_E))), reference, atol=1e-10
    )


def test_pchip_matches_scipy(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching PCHIP interpolator against *SciPy*."""

    name, cast = backend
    interpolator = PchipInterpolator(cast(_X), cast(_Y))

    assert interpolator.backend == name

    reference = scipy.interpolate.PchipInterpolator(_X, _Y)(_X_E)
    np.testing.assert_allclose(
        _to_numpy(interpolator(cast(_X_E))), reference, atol=1e-10
    )


def test_y_setter(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test that assigning ``y`` re-solves the interpolant."""

    _name, cast = backend
    interpolator = CubicSplineInterpolator(cast(_X), cast(_Y))

    y_new = np.cos(0.9 * _X)
    interpolator.y = cast(y_new)

    reference = scipy.interpolate.interp1d(_X, y_new, kind="cubic")(_X_E)
    np.testing.assert_allclose(
        _to_numpy(interpolator(cast(_X_E))), reference, atol=1e-10
    )


def test_x_setter(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test that assigning ``x`` rebuilds the interpolant and shifts evaluation."""

    _name, cast = backend
    interpolator = LinearInterpolator(cast(_X), cast(_Y))

    x_new = _X * 2.0
    interpolator.x = cast(x_new)

    query = np.linspace(x_new[0], x_new[-1], 11)
    reference = LinearInterpolator(x_new, _Y)(query)
    np.testing.assert_allclose(
        _to_numpy(interpolator(cast(query))), _to_numpy(reference), atol=1e-10
    )


def test_kernel_setters(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test that assigning kernel parameters rebuilds the interpolant."""

    _name, cast = backend
    interpolator = KernelInterpolator(cast(_X_UNIFORM), cast(_Y_UNIFORM))

    interpolator.window = 4
    interpolator.kernel = kernel_linear
    assert interpolator.window == 4
    assert interpolator.kernel is kernel_linear

    reference = KernelInterpolator(
        _X_UNIFORM, _Y_UNIFORM, window=4, kernel=kernel_linear
    )(_X_E_UNIFORM)
    np.testing.assert_allclose(
        _to_numpy(interpolator(cast(_X_E_UNIFORM))), _to_numpy(reference), atol=1e-8
    )


def test_null_setters(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """
    Test that assigning ``default`` and the tolerances changes evaluation.

    Regression: a bare attribute write previously shadowed the implementation
    through :meth:`__getattr__` and was silently ignored by evaluation.
    """

    _name, cast = backend
    interpolator = NullInterpolator(cast(_X), cast(_Y))

    interpolator.default = 42.0
    interpolator.absolute_tolerance = 0.1
    interpolator.relative_tolerance = 0.2
    assert interpolator.default == 42.0
    assert interpolator.absolute_tolerance == 0.1
    assert interpolator.relative_tolerance == 0.2

    # A query away from any knot does not match within tolerance: the default.
    off_knot = np.array([_X[0] + 0.123])
    np.testing.assert_allclose(_to_numpy(interpolator(cast(off_knot))), 42.0)


def test_setattr_rejects_shadowing(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """
    Test that assigning an attribute without a writable property raises rather
    than silently shadowing the delegated implementation.
    """

    _name, cast = backend
    interpolator = NullInterpolator(cast(_X), cast(_Y))

    with pytest.raises(AttributeError):
        interpolator.nonexistent = 1.0

    # ``backend`` is a read-only property and likewise not writable.
    with pytest.raises(AttributeError):
        interpolator.backend = "numpy"  # pyright: ignore[reportAttributeAccessIssue]


def test_scipy_members(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """
    Test that the wrapped *SciPy* members are exposed on the *NumPy* backend.

    ``PchipInterpolator`` methods and ``interp1d.fill_value`` wrap *SciPy* and
    exist on the *NumPy* backend only.
    """

    name, cast = backend
    cubic = CubicSplineInterpolator(cast(_X), cast(_Y))
    pchip = PchipInterpolator(cast(_X), cast(_Y))
    pchip_methods = ("derivative", "integrate", "roots", "solve")

    if name == "numpy":
        assert hasattr(cubic, "fill_value")
        for method in pchip_methods:
            assert callable(getattr(pchip, method))

        reference = scipy.interpolate.PchipInterpolator(_X, _Y)
        np.testing.assert_allclose(
            pchip.integrate(_X[0], _X[-1]), reference.integrate(_X[0], _X[-1])
        )
    else:
        assert not hasattr(cubic, "fill_value")
        for method in pchip_methods:
            assert not hasattr(pchip, method)


def test_linear(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching linear interpolator across backends."""

    name, cast = backend
    interpolator = LinearInterpolator(cast(_X), cast(_Y))
    assert interpolator.backend == name

    reference = LinearInterpolator(_X, _Y)(_X_E)
    np.testing.assert_allclose(
        _to_numpy(interpolator(cast(_X_E))), _to_numpy(reference), atol=1e-10
    )


def test_nearest_neighbour(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching nearest-neighbour interpolator."""

    _name, cast = backend
    query = np.array([0.3, 1.2, 2.6, 4.9])
    reference = NearestNeighbourInterpolator(_X, _Y)(query)
    np.testing.assert_allclose(
        _to_numpy(NearestNeighbourInterpolator(cast(_X), cast(_Y))(cast(query))),
        _to_numpy(reference),
        atol=1e-10,
    )


def test_null(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching null interpolator."""

    _name, cast = backend
    query = np.array([0.0, 0.5, 1.1, 2.0])  # 0.0, 1.1, 2.0 are knots
    reference = NullInterpolator(_X, _Y)(query)
    np.testing.assert_allclose(
        _to_numpy(NullInterpolator(cast(_X), cast(_Y))(cast(query))),
        _to_numpy(reference),
        atol=1e-10,
        equal_nan=True,
    )


def test_sprague(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching *Sprague (1880)* interpolator."""

    name, cast = backend
    interpolator = SpragueInterpolator(cast(_X_UNIFORM), cast(_Y_UNIFORM))
    assert interpolator.backend == name

    reference = SpragueInterpolator(_X_UNIFORM, _Y_UNIFORM)(_X_E_UNIFORM)
    np.testing.assert_allclose(
        _to_numpy(interpolator(cast(_X_E_UNIFORM))), _to_numpy(reference), atol=1e-8
    )


def test_null_rank_2(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching null interpolator on rank-2 dependent data."""

    _name, cast = backend
    y2 = np.stack([_Y, 2.0 * _Y], axis=-1)
    query = np.array([_X[0], 0.5, _X[2], 3.0])  # knots and non-knots, in range
    reference = NullInterpolator(_X, y2)(query)
    np.testing.assert_allclose(
        _to_numpy(NullInterpolator(cast(_X), cast(y2))(cast(query))),
        _to_numpy(reference),
        atol=1e-10,
        equal_nan=True,
    )


def test_null_attributes(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test that null interpolator tolerance attributes are exposed."""

    _name, cast = backend
    interpolator = NullInterpolator(
        cast(_X), cast(_Y), absolute_tolerance=0.1, relative_tolerance=0.2, default=0.0
    )
    assert interpolator.absolute_tolerance == 0.1
    assert interpolator.relative_tolerance == 0.2
    assert interpolator.default == 0.0


@pytest.mark.parametrize(
    "interpolator",
    [
        LinearInterpolator,
        NearestNeighbourInterpolator,
        NullInterpolator,
        SpragueInterpolator,
        KernelInterpolator,
    ],
)
def test_dimension_validation(
    backend: tuple[str, Callable[[Any], Any]], interpolator: type
) -> None:
    """
    Test that mismatched lengths and a non-1-D ``x`` raise at construction.

    Regression: the nearest-neighbour interpolator accepted invalid inputs and
    failed later with backend-dependent indexing errors.
    """

    _name, cast = backend
    with pytest.raises(ValueError):
        interpolator(cast(_X), cast(_Y[:-1]))

    with pytest.raises(ValueError):
        interpolator(cast(np.stack([_X, _X])), cast(_Y))


@array_api_enable(True)
def test_torch_autodiff() -> None:
    """Test that *PyTorch* autograd propagates through the interpolators."""

    torch = pytest.importorskip("torch")

    x = torch.as_tensor(_X)
    y = torch.as_tensor(_Y).clone().requires_grad_(True)
    x_e = torch.as_tensor(_X_E[1:-1]).clone().requires_grad_(True)

    for interpolator in (
        CubicSplineInterpolator(x, y),
        PchipInterpolator(x, y),
    ):
        assert interpolator.backend == "torch"
        gradient_y, gradient_x = torch.autograd.grad(
            interpolator(x_e).sum(), (y, x_e), retain_graph=True
        )
        assert torch.isfinite(gradient_y).all() and bool((gradient_y != 0).any())
        assert torch.isfinite(gradient_x).all() and bool((gradient_x != 0).any())


@array_api_enable(True)
def test_jax_autodiff() -> None:
    """Test that *JAX* autodiff propagates through the interpolators."""

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    x = jnp.asarray(_X)
    y = jnp.asarray(_Y)
    x_e = jnp.asarray(_X_E[1:-1])

    # ``fill_value="extrapolate"`` avoids the ``bounds_error`` branch, which is
    # a data-dependent raise incompatible with ``jax`` tracing.
    def cubic_of_y(values: Any) -> Any:
        return CubicSplineInterpolator(x, values, fill_value="extrapolate")(x_e).sum()

    def cubic_of_x(query: Any) -> Any:
        return CubicSplineInterpolator(x, y, fill_value="extrapolate")(query).sum()

    def pchip_of_y(values: Any) -> Any:
        return PchipInterpolator(x, values)(x_e).sum()

    def pchip_of_x(query: Any) -> Any:
        return PchipInterpolator(x, y)(query).sum()

    assert CubicSplineInterpolator(x, y).backend == "jax"

    for of_y, of_x in ((cubic_of_y, cubic_of_x), (pchip_of_y, pchip_of_x)):
        gradient_y = jax.grad(of_y)(y)
        gradient_x = jax.grad(of_x)(x_e)
        assert bool(jnp.isfinite(gradient_y).all()) and bool((gradient_y != 0).any())
        assert bool(jnp.isfinite(gradient_x).all()) and bool((gradient_x != 0).any())


def test_detect_backend_frameworks(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test backend detection directly for every available backend's arrays."""

    name, cast = backend

    assert detect_backend(cast(_X), cast(_Y)) == name
    # A single non-*NumPy* array is enough to select its backend.
    assert detect_backend(_X, cast(_Y)) == name


def test_kernel(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching kernel interpolator against the *NumPy* reference."""

    name, cast = backend
    interpolator = KernelInterpolator(cast(_X_UNIFORM), cast(_Y_UNIFORM))
    assert interpolator.backend == name

    reference = KernelInterpolator(_X_UNIFORM, _Y_UNIFORM)(_X_E_UNIFORM)
    np.testing.assert_allclose(
        _to_numpy(interpolator(cast(_X_E_UNIFORM))), _to_numpy(reference), atol=1e-8
    )


def test_kernel_rank_2(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching kernel interpolator on rank-2 dependent data."""

    _name, cast = backend
    y2 = np.stack([_Y_UNIFORM, _Y_UNIFORM * 2.0], axis=-1)
    reference = KernelInterpolator(_X_UNIFORM, y2)(_X_E_UNIFORM)
    np.testing.assert_allclose(
        _to_numpy(KernelInterpolator(cast(_X_UNIFORM), cast(y2))(cast(_X_E_UNIFORM))),
        _to_numpy(reference),
        atol=1e-8,
    )


def test_kernel_properties() -> None:
    """Test the kernel interpolator exposes the expected accessors."""

    interpolator = KernelInterpolator(
        _X_UNIFORM, _Y_UNIFORM, window=4, kernel=kernel_linear, kernel_kwargs={"a": 2}
    )

    assert interpolator.window == 4
    assert interpolator.kernel is kernel_linear
    assert interpolator.kernel_kwargs == {"a": 2}
    assert interpolator.padding_kwargs["mode"] == "reflect"


def test_kernel_raises() -> None:
    """Test the kernel interpolator raises on mismatched dimensions."""

    with pytest.raises(ValueError):
        KernelInterpolator(np.linspace(0, 1, 10), np.linspace(0, 1, 15))


def test_kernel_reflect_padding_wider_than_data(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """
    Test that a window wider than the data reflects correctly and agrees across
    backends.

    Regression: the default ``window`` is wider than a two-sample input, which
    used to raise on *PyTorch* and silently clamp (diverging numerically) on
    *JAX*.
    """

    _name, cast = backend
    query = np.array([0.5])

    reference = KernelInterpolator(np.array([0.0, 1.0]), np.array([2.0, 4.0]))(query)
    got = KernelInterpolator(cast(np.array([0.0, 1.0])), cast(np.array([2.0, 4.0])))(
        cast(query)
    )
    np.testing.assert_allclose(_to_numpy(got), _to_numpy(reference), atol=1e-10)


@pytest.mark.parametrize("window", [1, 2, 3, 5, 9])
def test_kernel_reflect_padding_matches_numpy(
    backend: tuple[str, Callable[[Any], Any]], window: int
) -> None:
    """Test that reflect padding matches ``np.pad`` for arbitrary window widths."""

    _name, cast = backend
    x = np.arange(4.0)
    y = np.array([2.0, 4.0, 8.0, 16.0])
    query = np.linspace(0.0, 3.0, 13)

    reference = KernelInterpolator(x, y, window=window)(query)
    got = KernelInterpolator(cast(x), cast(y), window=window)(cast(query))
    np.testing.assert_allclose(_to_numpy(got), _to_numpy(reference), atol=1e-8)


@pytest.mark.parametrize(("n", "window"), [(1, 2), (2, 3), (4, 1), (4, 7), (5, 2)])
def test_reflect_indices_match_numpy_pad(n: int, window: int) -> None:
    """Test that ``reflect_indices`` reproduces ``np.pad`` reflect, incl. ``n == 1``."""

    y = np.arange(float(n)) + 1.0
    reference = np.pad(y, (window, window), mode="reflect" if n > 1 else "edge")
    np.testing.assert_array_equal(y[reflect_indices(n, window)], reference)


@pytest.mark.parametrize(
    "padding_kwargs",
    [{"mode": "mean"}, {"pad_width": (5, 5), "mode": "reflect"}],
)
def test_kernel_padding_kwargs_rejected(
    backend: tuple[str, Callable[[Any], Any]], padding_kwargs: dict
) -> None:
    """Test that unsupported padding is rejected the same way on every backend."""

    _name, cast = backend
    with pytest.raises(ValueError):
        KernelInterpolator(
            cast(_X_UNIFORM), cast(_Y_UNIFORM), padding_kwargs=padding_kwargs
        )


def test_extrapolator_rank_2(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the extrapolator on rank-2 data, including a scalar query."""

    _name, cast = backend
    x = np.array([3.0, 4.0, 5.0])
    y = np.stack([np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])], axis=-1)

    for query in (1.0, np.array([1.0, 3.5, 8.0])):  # scalar (below) and vector
        reference = Extrapolator(LinearInterpolator(x, y))(query)
        got = _to_numpy(Extrapolator(LinearInterpolator(cast(x), cast(y)))(cast(query)))
        np.testing.assert_allclose(got, _to_numpy(reference), atol=1e-10)

    # A scalar query returns one value per signal, not a collapsed scalar.
    scalar = _to_numpy(Extrapolator(LinearInterpolator(cast(x), cast(y)))(cast(1.0)))
    assert scalar.shape == (2,)


def test_extrapolator(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching extrapolator against the *NumPy* reference."""

    name, cast = backend
    x = np.array([3.0, 4.0, 5.0])
    y = np.array([1.0, 2.0, 3.0])
    query = np.array([1.0, 3.5, 8.0])

    for method, left, right in (
        ("Linear", None, None),
        ("Constant", None, None),
        ("Constant", 0.0, 9.0),
    ):
        reference = Extrapolator(
            LinearInterpolator(x, y), method=method, left=left, right=right
        )(query)
        extrapolator = Extrapolator(
            LinearInterpolator(cast(x), cast(y)),
            method=method,
            left=left,
            right=right,
        )
        assert extrapolator.backend == name
        np.testing.assert_allclose(
            _to_numpy(extrapolator(cast(query))), _to_numpy(reference), atol=1e-10
        )


def test_extrapolator_properties() -> None:
    """Test the extrapolator exposes the expected accessors."""

    x = np.array([3.0, 4.0, 5.0])
    interpolator = LinearInterpolator(x, np.array([1.0, 2.0, 3.0]))
    extrapolator = Extrapolator(interpolator, method="Constant", left=0.0, right=9.0)

    assert extrapolator.interpolator is interpolator
    assert extrapolator.method == "constant"
    assert extrapolator.left == 0.0
    assert extrapolator.right == 9.0


@array_api_enable(True)
def test_kernel_torch_autodiff() -> None:
    """Test *PyTorch* autograd propagates through the kernel interpolator."""

    torch = pytest.importorskip("torch")

    x = torch.as_tensor(_X_UNIFORM)
    y = torch.as_tensor(_Y_UNIFORM).clone().requires_grad_(True)
    x_e = torch.as_tensor(_X_E_UNIFORM[1:-1]).clone().requires_grad_(True)

    interpolator = KernelInterpolator(x, y)
    assert interpolator.backend == "torch"
    gradient_y, gradient_x = torch.autograd.grad(
        interpolator(x_e).sum(), (y, x_e), retain_graph=True
    )
    assert torch.isfinite(gradient_y).all() and bool((gradient_y != 0).any())
    assert torch.isfinite(gradient_x).all() and bool((gradient_x != 0).any())


@array_api_enable(True)
def test_extrapolator_torch_autodiff() -> None:
    """Test *PyTorch* autograd propagates through the extrapolator."""

    torch = pytest.importorskip("torch")

    x = torch.as_tensor(_X)
    y = torch.as_tensor(_Y).clone().requires_grad_(True)
    query = torch.as_tensor(np.array([-1.0, 2.0, 6.0]))

    extrapolator = Extrapolator(CubicSplineInterpolator(x, y, fill_value="extrapolate"))
    assert extrapolator.backend == "torch"
    (gradient_y,) = torch.autograd.grad(extrapolator(query).sum(), (y,))
    assert torch.isfinite(gradient_y).all() and bool((gradient_y != 0).any())


def _torch_accelerator() -> str | None:
    """Return an available non-*CPU* *PyTorch* device, or *None*."""

    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    return None


_TORCH_ACCELERATOR = _torch_accelerator()

# ``(interpolator, x, y)`` triples; *Sprague* and the kernel require uniform
# spacing, so they use the uniform sample.
_TORCH_DEVICE_INTERPOLATORS = [
    (CubicSplineInterpolator, _X, _Y),
    (PchipInterpolator, _X, _Y),
    (LinearInterpolator, _X, _Y),
    (NearestNeighbourInterpolator, _X, _Y),
    (NullInterpolator, _X, _Y),
    (SpragueInterpolator, _X_UNIFORM, _Y_UNIFORM),
    (KernelInterpolator, _X_UNIFORM, _Y_UNIFORM),
]


@pytest.mark.skipif(
    _TORCH_ACCELERATOR is None,
    reason="Requires a non-CPU PyTorch device (CUDA or MPS).",
)
@pytest.mark.parametrize(
    ("cls", "x", "y"),
    _TORCH_DEVICE_INTERPOLATORS,
    ids=[cls.__name__ for cls, _, _ in _TORCH_DEVICE_INTERPOLATORS],
)
def test_torch_colocates_host_knots_with_device_values(
    cls: Any, x: np.ndarray, y: np.ndarray
) -> None:
    """
    Test host knot coordinates co-locate with device-resident values.

    Host ``x`` paired with device ``y`` must not raise a device mismatch; both
    construction and data reassignment move ``x`` onto the values' device. The
    query is evaluated at interior knots, which every interpolator (including
    the exact-match :class:`NullInterpolator`) resolves to a finite value.
    """

    torch = pytest.importorskip("torch")
    device = _TORCH_ACCELERATOR

    def _device_values() -> torch.Tensor:
        return torch.as_tensor(y.astype(np.float32), device=device)

    def _assert_colocated(interpolator: Any) -> None:
        assert interpolator.x.device == interpolator.y.device
        result = interpolator(
            torch.as_tensor(x[1:-1].astype(np.float32), device=device)
        )
        assert result.device.type == torch.device(device).type
        assert bool(torch.isfinite(result).all())

    with array_api_enable(True):
        interpolator = cls(x, _device_values(), dtype=np.float32)
        _assert_colocated(interpolator)

        # Reassigning host ``x`` and device ``y`` keeps both on one device.
        interpolator.x = x
        interpolator.y = _device_values()
        _assert_colocated(interpolator)


@array_api_enable(True)
def test_kernel_jax_autodiff() -> None:
    """Test *JAX* autodiff propagates through the kernel interpolator."""

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    x = jnp.asarray(_X_UNIFORM)
    x_e = jnp.asarray(_X_E_UNIFORM[1:-1])

    def kernel_of_y(values: Any) -> Any:
        return KernelInterpolator(x, values)(x_e).sum()

    gradient_y = jax.grad(kernel_of_y)(jnp.asarray(_Y_UNIFORM))
    assert bool(jnp.isfinite(gradient_y).all()) and bool((gradient_y != 0).any())


@array_api_enable(True)
def test_extrapolator_jax_autodiff() -> None:
    """Test *JAX* autodiff propagates through the extrapolator."""

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    x = jnp.asarray(_X)
    query = jnp.asarray(np.array([-1.0, 2.0, 6.0]))

    def extrapolate_of_y(values: Any) -> Any:
        interpolator = CubicSplineInterpolator(x, values, fill_value="extrapolate")
        return Extrapolator(interpolator)(query).sum()

    gradient_y = jax.grad(extrapolate_of_y)(jnp.asarray(_Y))
    assert bool(jnp.isfinite(gradient_y).all()) and bool((gradient_y != 0).any())


def test_cubic_spline_fill_value(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test scalar and 2-tuple ``fill_value`` outside the interpolation range."""

    _name, cast = backend
    query = np.array([-1.0, 2.5, 6.0])  # below, in-range, above

    # ``fill_value`` takes effect only when out-of-range queries are not an
    # error, matching ``scipy.interpolate.interp1d``.
    scalar = CubicSplineInterpolator(
        cast(_X), cast(_Y), fill_value=0.0, bounds_error=False
    )(cast(query))
    scalar = _to_numpy(scalar)
    assert scalar[0] == 0.0
    assert scalar[-1] == 0.0

    tupled = CubicSplineInterpolator(
        cast(_X), cast(_Y), fill_value=(-1.0, 9.0), bounds_error=False
    )(cast(query))
    tupled = _to_numpy(tupled)
    assert tupled[0] == -1.0
    assert tupled[-1] == 9.0


def test_cubic_spline_bounds_error(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test ``bounds_error`` raises outside the interpolation range."""

    _name, cast = backend
    interpolator = CubicSplineInterpolator(cast(_X), cast(_Y), bounds_error=True)

    with pytest.raises(ValueError):
        interpolator(cast(np.array([-1.0])))


def test_cubic_spline_extrapolate_and_raise() -> None:
    """Test combining ``fill_value="extrapolate"`` with ``bounds_error``."""

    with pytest.raises(ValueError):
        CubicSplineInterpolator(_X, _Y, fill_value="extrapolate", bounds_error=True)


def test_pchip_extrapolate_false(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test PCHIP returns *NaN* outside the range when not extrapolating."""

    _name, cast = backend
    values = PchipInterpolator(cast(_X), cast(_Y), extrapolate=False)(
        cast(np.array([-1.0, 2.5, 6.0]))
    )
    values = _to_numpy(values)

    assert np.isnan(values[0])
    assert np.isnan(values[-1])
    assert np.isfinite(values[1])


def test_pchip_derivatives(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test PCHIP derivative orders against the *SciPy* reference."""

    _name, cast = backend
    interpolator = PchipInterpolator(cast(_X), cast(_Y))
    reference = scipy.interpolate.PchipInterpolator(_X, _Y)

    for nu in (1, 2, 3):
        np.testing.assert_allclose(
            _to_numpy(interpolator(cast(_X_E), nu=nu)),
            reference(_X_E, nu=nu),
            atol=1e-8,
        )

    # Orders beyond the cubic degree are identically zero.
    np.testing.assert_allclose(
        _to_numpy(interpolator(cast(_X_E), nu=4)), np.zeros_like(_X_E), atol=1e-12
    )


def test_cubic_spline_assume_sorted(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test the cubic spline sorts unsorted knots when ``assume_sorted`` is off."""

    _name, cast = backend
    order = np.array([2, 0, 4, 1, 5, 3])
    x_shuffled = _X[order]
    y_shuffled = _Y[order]

    interpolator = CubicSplineInterpolator(
        cast(x_shuffled), cast(y_shuffled), assume_sorted=False
    )
    reference = scipy.interpolate.interp1d(_X, _Y, kind="cubic")(_X_E)
    np.testing.assert_allclose(
        _to_numpy(interpolator(cast(_X_E))), reference, atol=1e-10
    )


def test_null_tolerance(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the null interpolator honours tolerance and the default value."""

    _name, cast = backend
    interpolator = NullInterpolator(
        cast(_X), cast(_Y), absolute_tolerance=1e-6, default=-1.0
    )
    # A knot returns its value; a point away from any knot returns the default.
    values = _to_numpy(interpolator(cast(np.array([_X[2], 0.75]))))

    np.testing.assert_allclose(values[0], _Y[2], atol=1e-8)
    assert values[1] == -1.0


def test_validation_errors(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the interpolators raise on invalid construction inputs."""

    _name, cast = backend

    with pytest.raises(ValueError):  # Too few points for the cubic spline.
        CubicSplineInterpolator(cast(_X[:3]), cast(_Y[:3]))

    with pytest.raises(ValueError):  # Length mismatch.
        PchipInterpolator(cast(_X), cast(_Y[:-1]))

    with pytest.raises(ValueError):  # Non-increasing knots.
        CubicSplineInterpolator(cast(_X[::-1].copy()), cast(_Y), assume_sorted=True)

    with pytest.raises(ValueError):  # Fewer than six points for Sprague.
        SpragueInterpolator(cast(_X[:5]), cast(_Y[:5]))


@array_api_enable(True)
def test_jax_jit() -> None:
    """Test the *JAX* cubic spline and PCHIP evaluate under ``jax.jit``."""

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    x = jnp.asarray(_X)
    x_e = jnp.asarray(_X_E)

    # Construction validates knots with data-dependent predicates, so build the
    # interpolant eagerly and ``jit`` only its evaluation (the on-device path).
    interpolator = CubicSplineInterpolator(x, jnp.asarray(_Y), fill_value="extrapolate")
    evaluate = jax.jit(interpolator.__call__)

    reference = scipy.interpolate.interp1d(_X, _Y, kind="cubic")(_X_E)
    np.testing.assert_allclose(_to_numpy(evaluate(x_e)), reference, atol=1e-8)


@array_api_enable(True)
def test_jax_jit_range_checked_evaluation() -> None:
    """
    Test that *JAX* interpolators whose evaluation validates the query range
    evaluate under ``jax.jit`` and match eager evaluation.

    The range check raises eagerly but is skipped under ``jit``, where the query
    values are unknown at trace time.
    """

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    x = jnp.asarray(_X)
    y = jnp.asarray(_Y)
    x_e = jnp.asarray(_X_E)

    interpolators = [
        LinearInterpolator(x, y),
        NearestNeighbourInterpolator(x, y),
        CubicSplineInterpolator(x, y),
        KernelInterpolator(jnp.asarray(_X_UNIFORM), jnp.asarray(_Y_UNIFORM)),
    ]
    for interpolator in interpolators:
        query = (
            jnp.asarray(_X_E_UNIFORM)
            if isinstance(interpolator, KernelInterpolator)
            else x_e
        )
        np.testing.assert_allclose(
            _to_numpy(jax.jit(interpolator.__call__)(query)),
            _to_numpy(interpolator(query)),
            atol=1e-8,
        )


@array_api_enable(True)
def test_jax_jit_extrapolator() -> None:
    """
    Test that a *JAX* :class:`Extrapolator` evaluates under ``jax.jit``, over a
    query spanning below, within and above the interpolation range, and matches
    eager evaluation.
    """

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp  # noqa: PLC0415

    jax.config.update("jax_enable_x64", True)
    x = jnp.asarray(_X)
    y = jnp.asarray(_Y)
    query = jnp.asarray(np.linspace(_X[0] - 1.0, _X[-1] + 1.0, 29))

    extrapolator = Extrapolator(CubicSplineInterpolator(x, y, fill_value="extrapolate"))
    np.testing.assert_allclose(
        _to_numpy(jax.jit(extrapolator)(query)),
        _to_numpy(extrapolator(query)),
        atol=1e-8,
    )


def test_backend_interpolators_standalone(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """
    Test that a backend submodule's interpolators are usable directly, with the
    same writable surface and mutation behaviour as a fresh instance.
    """

    name, cast = backend
    module = importlib.import_module(f"colour.algebra.interpolation.backends.{name}")

    kernel = module.KernelInterpolator(cast(_X_UNIFORM), cast(_Y_UNIFORM))
    kernel.window = 4
    kernel.kernel = kernel_linear
    assert kernel.window == 4
    assert kernel.kernel is kernel_linear

    reference = module.KernelInterpolator(
        cast(_X_UNIFORM), cast(_Y_UNIFORM), window=4, kernel=kernel_linear
    )
    np.testing.assert_allclose(
        _to_numpy(kernel(cast(_X_E_UNIFORM))),
        _to_numpy(reference(cast(_X_E_UNIFORM))),
        atol=1e-8,
    )

    null = module.NullInterpolator(cast(_X), cast(_Y))
    null.absolute_tolerance = 0.1
    null.default = 7.0
    assert null.absolute_tolerance == 0.1
    assert null.default == 7.0


def test_mutate_between_calls_in_place(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """
    Test that mutating a forwarded attribute between evaluations takes effect on
    the next call and reuses the cached backend implementation in place.
    """

    _name, cast = backend
    interpolator = KernelInterpolator(cast(_X_UNIFORM), cast(_Y_UNIFORM))

    interpolator(cast(_X_E_UNIFORM))
    cached = interpolator._impl  # noqa: SLF001

    interpolator.window = 4
    assert interpolator._impl is cached  # noqa: SLF001

    reference = KernelInterpolator(cast(_X_UNIFORM), cast(_Y_UNIFORM), window=4)
    np.testing.assert_allclose(
        _to_numpy(interpolator(cast(_X_E_UNIFORM))),
        _to_numpy(reference(cast(_X_E_UNIFORM))),
        atol=1e-8,
    )


def test_set_raises_and_preserves_on_rejected_value(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """
    Test that a write the backend rejects raises, keeps the previous value and
    still evaluates (the rejected write leaves no half-applied state).
    """

    _name, cast = backend
    interpolator = KernelInterpolator(cast(_X_UNIFORM), cast(_Y_UNIFORM))
    reference = _to_numpy(interpolator(cast(_X_E_UNIFORM)))
    cached = interpolator._impl  # noqa: SLF001

    with pytest.raises(ValueError):
        interpolator.padding_kwargs = {"mode": "edge", "pad_width": (3, 3)}

    # The rejected write recovers in place, so a wrapper holding the
    # implementation by reference (e.g. an Extrapolator) stays attached.
    assert interpolator._impl is cached  # noqa: SLF001
    assert interpolator.padding_kwargs["mode"] == "reflect"
    np.testing.assert_allclose(
        _to_numpy(interpolator(cast(_X_E_UNIFORM))), reference, atol=1e-8
    )


def test_rejected_write_is_not_recorded(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """
    Test that a write the backend rejects is not recorded, so it is not replayed
    when the implementation is later rebuilt.
    """

    _name, cast = backend
    interpolator = KernelInterpolator(cast(_X_UNIFORM), cast(_Y_UNIFORM))

    with pytest.raises(ValueError):
        interpolator.padding_kwargs = {"mode": "edge", "pad_width": (3, 3)}

    assert "padding_kwargs" not in interpolator._overrides  # noqa: SLF001

    # Rebuilding replays only recorded writes, so the rejected one is gone and
    # the rebuild succeeds.
    interpolator._impl = interpolator._build_implementation(  # noqa: SLF001
        interpolator._backend  # noqa: SLF001
    )


@pytest.mark.parametrize("interpolator", [CubicSplineInterpolator, PchipInterpolator])
def test_x_setter_cubic_pchip(
    backend: tuple[str, Callable[[Any], Any]], interpolator: type
) -> None:
    """
    Test that the ``x`` setter rebuilds the Cubic/Pchip interpolant, both
    standalone and through the dispatcher.

    Regression: forwarding only the *NumPy* backend's setters dropped the
    writable ``x`` these interpolators previously exposed.
    """

    name, cast = backend
    module = importlib.import_module(f"colour.algebra.interpolation.backends.{name}")

    new_x = _X_UNIFORM * 2.0
    query = cast(np.array([5.0]))
    reference = interpolator(cast(new_x), cast(_Y_UNIFORM))

    dispatching = interpolator(cast(_X_UNIFORM), cast(_Y_UNIFORM))
    standalone = getattr(module, interpolator.__name__)(
        cast(_X_UNIFORM), cast(_Y_UNIFORM)
    )
    for interp in (dispatching, standalone):
        interp.x = cast(new_x)
        np.testing.assert_allclose(
            _to_numpy(interp(query)), _to_numpy(reference(query)), atol=1e-8
        )


def test_extrapolator_sees_interpolator_mutation(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """
    Test that mutating a wrapped interpolator in place is seen by the
    extrapolator, which holds the interpolator's implementation object.
    """

    _name, cast = backend
    x = cast(np.array([3.0, 4.0, 5.0]))
    interpolator = LinearInterpolator(x, cast(np.array([1.0, 2.0, 3.0])))
    extrapolator = Extrapolator(interpolator)

    interpolator.y = cast(np.array([10.0, 20.0, 30.0]))

    reference = Extrapolator(LinearInterpolator(x, cast(np.array([10.0, 20.0, 30.0]))))
    query = cast(np.array([1.0]))
    np.testing.assert_allclose(
        _to_numpy(extrapolator(query)), _to_numpy(reference(query)), atol=1e-10
    )


@pytest.mark.parametrize(("name", "cast"), _FRAMEWORK_BACKENDS)
@pytest.mark.parametrize("axis", ["x", "y"])
def test_set_data_switches_backend(
    name: str, cast: Callable[[Any], Any], axis: str
) -> None:
    """
    Test that assigning a framework array to ``x`` or ``y`` of a *NumPy*-built
    interpolator switches the backend, resolving it from the incoming value
    combined with the retained other axis.
    """

    with array_api_enable(True):
        interpolator = LinearInterpolator(_X, _Y)  # *NumPy* data.
        assert interpolator.backend == "numpy"

        setattr(interpolator, axis, cast(_X if axis == "x" else _Y))
        assert interpolator.backend == name

        reference = LinearInterpolator(cast(_X), cast(_Y))
        query = cast(np.array([2.5]))
        np.testing.assert_allclose(
            _to_numpy(interpolator(query)), _to_numpy(reference(query)), atol=1e-8
        )


@pytest.mark.parametrize(("name", "cast"), _FRAMEWORK_BACKENDS)
def test_switch_replays_recorded_overrides(
    name: str, cast: Callable[[Any], Any]
) -> None:
    """
    Test that an option written before a backend switch is replayed onto the
    implementation rebuilt on the new backend.
    """

    with array_api_enable(True):
        interpolator = NullInterpolator(_X, _Y)  # *NumPy* data.
        interpolator.default = 7.0
        assert interpolator.backend == "numpy"

        interpolator.x = cast(_X)  # Switch to the framework backend.
        assert interpolator.backend == name
        assert float(interpolator.default) == 7.0


@pytest.mark.parametrize("cast", [cast for _, cast in _FRAMEWORK_BACKENDS])
def test_switch_rollback_on_failed_rebuild(cast: Callable[[Any], Any]) -> None:
    """
    Test that a backend switch whose rebuild fails leaves the interpolator
    unchanged and still evaluable (no half-applied state).
    """

    with array_api_enable(True):
        interpolator = LinearInterpolator(_X, _Y)  # *NumPy* data.
        query = np.array([2.5])
        reference = _to_numpy(interpolator(query))

        with pytest.raises(ValueError):
            interpolator.x = cast(_X[:-1])  # Framework array, mismatched length.

        assert interpolator.backend == "numpy"
        np.testing.assert_allclose(_to_numpy(interpolator(query)), reference, atol=1e-8)


@pytest.mark.skipif(
    {"torch", "jax"} - {name for name, _ in _FRAMEWORK_BACKENDS} != set(),
    reason="Requires both PyTorch and JAX.",
)
def test_mixed_namespaces_raise() -> None:
    """
    Test that data spanning two framework namespaces has no single backend and
    raises, both at construction and on reassignment.
    """

    casts = dict(_FRAMEWORK_BACKENDS)
    torch_cast, jax_cast = casts["torch"], casts["jax"]

    with array_api_enable(True):
        with pytest.raises(ValueError):
            LinearInterpolator(torch_cast(_X), jax_cast(_Y))

        interpolator = LinearInterpolator(torch_cast(_X), torch_cast(_Y))
        with pytest.raises(ValueError):
            interpolator.y = jax_cast(_Y)

        # The rejected reassignment left the interpolator on its backend.
        assert interpolator.backend == "torch"


@pytest.mark.parametrize(("name", "cast"), _FRAMEWORK_BACKENDS)
def test_extrapolator_tracks_wrapped_backend_switch(
    name: str, cast: Callable[[Any], Any]
) -> None:
    """
    Test that an extrapolator tracks a backend switch of its wrapped
    interpolator mutated in place after wrapping.
    """

    with array_api_enable(True):
        interpolator = LinearInterpolator(_X, _Y)  # *NumPy* data.
        extrapolator = Extrapolator(interpolator)
        assert extrapolator.backend == "numpy"

        interpolator.x = cast(_X)  # Switch the wrapped interpolator's backend.
        assert interpolator.backend == name

        query = cast(np.array([-1.0, 6.0]))  # Outside the domain: extrapolated.
        result = extrapolator(query)
        assert extrapolator.backend == name
        assert detect_backend(result) == name

        reference = Extrapolator(LinearInterpolator(cast(_X), cast(_Y)))
        np.testing.assert_allclose(
            _to_numpy(result), _to_numpy(reference(query)), atol=1e-8
        )


# Backend class names shared across every backend submodule.
_CLASS_NAMES = [t.__name__ for _, t, _, _ in _INTERPOLATOR_CASES]


@pytest.mark.parametrize("class_name", _CLASS_NAMES)
def test_backend_property_symmetry(class_name: str) -> None:
    """
    Test that every backend exposes the same property surface for a class.

    :func:`colour.algebra.interpolation._dispatch._forward_properties` mirrors
    the *NumPy* backend's properties onto the dispatcher, so a backend drifting
    out of parallel would silently break the forwarded API. ``dtype`` is part of
    the shared surface.
    """

    property_sets = {}
    for name, _ in _BACKENDS:
        module = importlib.import_module(
            f"colour.algebra.interpolation.backends.{name}"
        )
        property_sets[name] = {
            member
            for member, value in inspect.getmembers(
                getattr(module, class_name), lambda v: isinstance(v, property)
            )
        }

    reference = property_sets["numpy"]
    assert "dtype" in reference
    for name, properties in property_sets.items():
        assert properties == reference, (name, properties ^ reference)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    ("interpolator_type", "x", "y"),
    [pytest.param(t, x, y, id=i) for i, t, x, y in _INTERPOLATOR_CASES],
)
def test_dtype_honoured(
    backend: tuple[str, Callable[[Any], Any]],
    interpolator_type: type,
    x: np.ndarray,
    y: np.ndarray,
    dtype: type,
) -> None:
    """Test that ``dtype`` sets the stored-data and output precision."""

    _name, cast = backend

    interpolator = interpolator_type(cast(x), cast(y), dtype=dtype)

    assert np.asarray(interpolator.x).dtype == np.dtype(dtype)
    assert np.asarray(interpolator.y).dtype == np.dtype(dtype)
    assert np.asarray(interpolator(cast(2.5))).dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    "interpolator_type", [LinearInterpolator, NearestNeighbourInterpolator]
)
def test_dtype_positional(
    backend: tuple[str, Callable[[Any], Any]],
    interpolator_type: type,
) -> None:
    """Test that ``dtype`` may be passed positionally, matching the prior API."""

    _name, cast = backend

    interpolator = interpolator_type(cast(_X), cast(_Y), np.float32)

    assert np.asarray(interpolator.x).dtype == np.dtype(np.float32)


def test_default_dtype_numpy() -> None:
    """Test that the *NumPy* backend defaults to the library float precision."""

    interpolator = LinearInterpolator(_X, _Y)

    assert interpolator.dtype == np.dtype(DTYPE_FLOAT_DEFAULT)
    assert interpolator.x.dtype == np.dtype(DTYPE_FLOAT_DEFAULT)


def test_kernel_positional_window(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test that the kernel ``window`` may be passed positionally."""

    _name, cast = backend

    interpolator = KernelInterpolator(cast(_X_UNIFORM), cast(_Y_UNIFORM), 4)

    assert interpolator.window == 4


def test_setter_preserves_dtype(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test that assigning ``y`` keeps the construction ``dtype``."""

    _name, cast = backend

    interpolator = LinearInterpolator(cast(_X), cast(_Y), dtype=np.float32)
    interpolator.y = cast(_Y)  # ``cast`` yields float64 data.

    assert np.asarray(interpolator.y).dtype == np.float32
