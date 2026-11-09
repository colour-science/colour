"""
Unit tests for the :mod:`colour.algebra.interpolation` dispatching
interpolators.

Backend-parametrised: each interpolator is exercised on every backend whose
framework is installed *and* whose ``backends`` submodule exists, so
coverage grows automatically as backends are added.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import typing

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
from colour.constants import DTYPE_FLOAT_DEFAULT, TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    array_api_enable,
    array_namespace,
    as_ndarray,
    xp_as_float_array,
    xp_assert_close,
    xp_assert_equal,
)

if typing.TYPE_CHECKING:
    import torch

    from colour.hints import Any, Callable, ModuleType

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


def _available_backends() -> list[tuple[str, Callable[[Any], Any]]]:
    """Return ``(name, cast)`` for each usable backend."""

    backends: list[tuple[str, Callable[[Any], Any]]] = [("numpy", as_ndarray)]

    try:
        import torch  # noqa: PLC0415

        backends.append(("torch", lambda a: torch.asarray(as_ndarray(a))))
    except ImportError:
        pass

    try:
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        jax.config.update("jax_enable_x64", True)
        backends.append(("jax", lambda a: jnp.asarray(as_ndarray(a))))
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


@pytest.fixture
def backend(xp: ModuleType) -> tuple[str, Callable[[Any], Any]]:
    """Return the backend and converter selected by the shared fixture."""

    name = xp.__name__.split(".")[0]
    return name, lambda value: xp_as_float_array(value, xp=xp)


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

    assert as_ndarray(result).shape == ()


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

    assert as_ndarray(result).shape == (0,)


def test_cubic_spline_matches_scipy(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test the dispatching cubic spline against the *SciPy* reference."""

    name, cast = backend
    interpolator = CubicSplineInterpolator(cast(_X), cast(_Y))

    assert interpolator.backend == name

    reference = scipy.interpolate.interp1d(_X, _Y, kind="cubic")(_X_E)
    xp_assert_close(
        as_ndarray(interpolator(cast(_X_E))),
        reference,
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.001,
    )


def test_cubic_spline_rank_2(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test the dispatching cubic spline on rank-2 dependent data."""

    _name, cast = backend
    interpolator = CubicSplineInterpolator(cast(_X), cast(_Y2), axis=0)

    reference = scipy.interpolate.interp1d(_X, _Y2, kind="cubic", axis=0)(_X_E)
    xp_assert_close(
        as_ndarray(interpolator(cast(_X_E))),
        reference,
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.001,
    )


def test_pchip_matches_scipy(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test the dispatching PCHIP interpolator against *SciPy*."""

    name, cast = backend
    interpolator = PchipInterpolator(cast(_X), cast(_Y))

    assert interpolator.backend == name

    reference = scipy.interpolate.PchipInterpolator(_X, _Y)(_X_E)
    xp_assert_close(
        as_ndarray(interpolator(cast(_X_E))),
        reference,
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.001,
    )

    interpolator.extrapolate = False
    assert np.isnan(as_ndarray(interpolator(cast(-1.0)))).all()
    interpolator.extrapolate = True
    xp_assert_close(
        interpolator(cast(-1.0)),
        scipy.interpolate.PchipInterpolator(_X, _Y)(-1.0),
        atol=TOLERANCE_ABSOLUTE_TESTS,
    )


def test_y_setter(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test that assigning ``y`` re-solves the interpolant."""

    _name, cast = backend
    interpolator = CubicSplineInterpolator(cast(_X), cast(_Y))

    y_new = np.cos(0.9 * _X)
    interpolator.y = cast(y_new)

    reference = scipy.interpolate.interp1d(_X, y_new, kind="cubic")(_X_E)
    xp_assert_close(
        as_ndarray(interpolator(cast(_X_E))),
        reference,
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.001,
    )


def test_x_setter(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test that assigning ``x`` rebuilds the interpolant and shifts evaluation."""

    _name, cast = backend
    interpolator = LinearInterpolator(cast(_X), cast(_Y))

    x_new = _X * 2.0
    interpolator.x = cast(x_new)

    query = np.linspace(x_new[0], x_new[-1], 11)
    reference = LinearInterpolator(x_new, _Y)(query)
    xp_assert_close(
        as_ndarray(interpolator(cast(query))),
        as_ndarray(reference),
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.001,
    )


def test_kernel_setters(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
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
    xp_assert_close(
        as_ndarray(interpolator(cast(_X_E_UNIFORM))),
        as_ndarray(reference),
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
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
    xp_assert_close(as_ndarray(interpolator(cast(off_knot))), 42.0)


def test_setattr_rejects_shadowing(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
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

    ``PchipInterpolator`` calculus methods wrap *SciPy* on the *NumPy*
    backend. The cubic ``fill_value`` property is shared by all backends.
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
        xp_assert_close(
            pchip.integrate(_X[0], _X[-1]), reference.integrate(_X[0], _X[-1])
        )
    else:
        assert hasattr(cubic, "fill_value")
        for method in pchip_methods:
            assert not hasattr(pchip, method)


def test_linear(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching linear interpolator across backends."""

    name, cast = backend
    interpolator = LinearInterpolator(cast(_X), cast(_Y))
    assert interpolator.backend == name

    reference = LinearInterpolator(_X, _Y)(_X_E)
    xp_assert_close(
        as_ndarray(interpolator(cast(_X_E))),
        as_ndarray(reference),
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.001,
    )

    class CustomLinear(LinearInterpolator):
        """Define a user-specialised linear interpolator."""

    interpolator = CustomLinear(cast(_X), cast(_Y))
    xp_assert_close(interpolator(cast(_X_E)), reference, atol=TOLERANCE_ABSOLUTE_TESTS)
    interpolator.y = cast(2 * _Y)
    xp_assert_close(
        interpolator(cast(_X_E)), 2 * reference, atol=TOLERANCE_ABSOLUTE_TESTS
    )


def test_nearest_neighbour(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test the dispatching nearest-neighbour interpolator."""

    _name, cast = backend
    query = np.array([0.3, 1.2, 2.6, 4.9])
    reference = NearestNeighbourInterpolator(_X, _Y)(query)
    xp_assert_close(
        as_ndarray(NearestNeighbourInterpolator(cast(_X), cast(_Y))(cast(query))),
        as_ndarray(reference),
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.001,
    )


def test_null(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching null interpolator."""

    _name, cast = backend
    query = np.array([0.0, 0.5, 1.1, 2.0])  # 0.0, 1.1, 2.0 are knots
    reference = NullInterpolator(_X, _Y)(query)
    xp_assert_close(
        as_ndarray(NullInterpolator(cast(_X), cast(_Y))(cast(query))),
        as_ndarray(reference),
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.001,
    )


def test_sprague(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching *Sprague (1880)* interpolator."""

    name, cast = backend
    interpolator = SpragueInterpolator(cast(_X_UNIFORM), cast(_Y_UNIFORM))
    assert interpolator.backend == name

    reference = SpragueInterpolator(_X_UNIFORM, _Y_UNIFORM)(_X_E_UNIFORM)
    xp_assert_close(
        as_ndarray(interpolator(cast(_X_E_UNIFORM))),
        as_ndarray(reference),
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
    )


def test_null_rank_2(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test the dispatching null interpolator on rank-2 dependent data."""

    _name, cast = backend
    y2 = np.stack([_Y, 2.0 * _Y], axis=-1)
    query = np.array([_X[0], 0.5, _X[2], 3.0])  # knots and non-knots, in range
    reference = NullInterpolator(_X, y2)(query)
    xp_assert_close(
        as_ndarray(NullInterpolator(cast(_X), cast(y2))(cast(query))),
        as_ndarray(reference),
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.001,
    )


def test_null_attributes(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
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
    xp_assert_close(
        as_ndarray(interpolator(cast(_X_E_UNIFORM))),
        as_ndarray(reference),
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
    )

    query = np.array([[0.25, 0.5], [1.25, 2.5]])
    result = interpolator(cast(query))
    assert result.shape == query.shape
    xp_assert_close(
        result,
        KernelInterpolator(_X_UNIFORM, _Y_UNIFORM)(query.ravel()).reshape(query.shape),
        atol=TOLERANCE_ABSOLUTE_TESTS,
    )


def test_kernel_rank_2(
    backend: tuple[str, Callable[[Any], Any]], request: pytest.FixtureRequest
) -> None:
    """Test kernel interpolation preserves every trailing dependent-data axis."""

    _name, cast = backend
    y2 = np.stack([_Y_UNIFORM, _Y_UNIFORM * 2.0], axis=-1)
    reference = KernelInterpolator(_X_UNIFORM, y2)(_X_E_UNIFORM)
    xp_assert_close(
        as_ndarray(KernelInterpolator(cast(_X_UNIFORM), cast(y2))(cast(_X_E_UNIFORM))),
        as_ndarray(reference),
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
    )

    query = np.array([[0.25, 0.5], [1.25, 2.5]])
    interpolator = KernelInterpolator(
        cast(_X_UNIFORM), cast(y2), padding_kwargs={"mode": "reflect", "pad_width": 3}
    )
    result = interpolator(cast(query))
    assert result.shape == (*query.shape, 2)
    xp_assert_close(
        result,
        KernelInterpolator(_X_UNIFORM, y2)(query.ravel()).reshape((*query.shape, 2)),
        atol=TOLERANCE_ABSOLUTE_TESTS,
    )

    for trailing_shape in ((2, 3), (2, 1, 3)):
        values = np.linspace(0.2, 1.4, len(_X_UNIFORM) * 6).reshape(
            (len(_X_UNIFORM), *trailing_shape)
        )
        native_values = cast(values)
        interpolator = KernelInterpolator(_X_UNIFORM, native_values)
        for points in (np.array(1.25), query, np.empty((2, 0))):
            # Every component is evaluated independently by the scalar NumPy
            # implementation, without the batched padding or broadcasting.
            expected = np.stack(
                [
                    KernelInterpolator(
                        _X_UNIFORM, values.reshape(len(values), -1)[:, i]
                    )(points)
                    for i in range(6)
                ],
                axis=-1,
            ).reshape((*points.shape, *trailing_shape))
            actual = interpolator(cast(points))
            assert actual.shape == (*points.shape, *trailing_shape)
            assert actual.dtype == native_values.dtype
            assert array_namespace(actual) is array_namespace(native_values)
            if hasattr(native_values, "device"):
                assert actual.device == native_values.device
            xp_assert_close(actual, expected, atol=TOLERANCE_ABSOLUTE_TESTS)

        if _name != "numpy":
            autodiff = request.getfixturevalue("autodiff")
            for points in (np.array(1.25), query):
                autodiff(
                    lambda ordinates, coordinates: KernelInterpolator(
                        _X_UNIFORM, ordinates
                    )(coordinates),
                    values,
                    points,
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
    xp_assert_close(
        as_ndarray(got), as_ndarray(reference), atol=TOLERANCE_ABSOLUTE_TESTS * 0.001
    )


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
    xp_assert_close(
        as_ndarray(got), as_ndarray(reference), atol=TOLERANCE_ABSOLUTE_TESTS * 0.1
    )


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


def test_extrapolator_rank_2(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test the extrapolator on rank-2 data, including a scalar query."""

    _name, cast = backend
    x = np.array([3.0, 4.0, 5.0])
    y = np.stack([np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0])], axis=-1)

    for query in (1.0, np.array([1.0, 3.5, 8.0])):  # scalar (below) and vector
        reference = Extrapolator(LinearInterpolator(x, y))(query)
        got = as_ndarray(
            Extrapolator(LinearInterpolator(cast(x), cast(y)))(cast(query))
        )
        xp_assert_close(
            got, as_ndarray(reference), atol=TOLERANCE_ABSOLUTE_TESTS * 0.001
        )

    # A scalar query returns one value per signal, not a collapsed scalar.
    scalar = as_ndarray(Extrapolator(LinearInterpolator(cast(x), cast(y)))(cast(1.0)))
    assert scalar.shape == (2,)


def test_extrapolator(
    backend: tuple[str, Callable[[Any], Any]], request: pytest.FixtureRequest
) -> None:
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
        xp_assert_close(
            as_ndarray(extrapolator(cast(query))),
            as_ndarray(reference),
            atol=TOLERANCE_ABSOLUTE_TESTS * 0.001,
        )

    # A fixed boundary value is constant in the finite ordinates, even at an
    # infinite query. Check each one-sided override as well as the pair.
    for left, right, points, expected in (
        (7.0, None, [-np.inf, -1e20], [7.0, 7.0]),
        (None, 9.0, [np.inf, 1e20], [9.0, 9.0]),
        (7.0, 9.0, [-np.inf, np.inf], [7.0, 9.0]),
    ):
        for values in (y, np.stack([y, y * 2], axis=-1)):
            expected_values = (
                expected
                if values.ndim == 1
                else np.repeat(np.asarray(expected)[:, None], 2, axis=1)
            )

            def fixed_extrapolation(
                values: Any,
                left: float | None = left,
                right: float | None = right,
                points: Any = points,
            ) -> Any:
                return Extrapolator(
                    LinearInterpolator(x, values), left=left, right=right
                )(points)

            result = fixed_extrapolation(cast(values))
            xp_assert_equal(result, expected_values)
            if name != "numpy":
                autodiff = request.getfixturevalue("autodiff")
                _result, (gradient,), _variables = autodiff(fixed_extrapolation, values)
                xp_assert_equal(gradient, np.zeros_like(values))


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

        if cls is KernelInterpolator:
            # Explicit device arrays must also work with a CPU default device.
            with torch.device("cpu"):
                for trailing_shape in ((2, 3), (2, 1, 3)):
                    values = np.linspace(0.2, 1.4, len(x) * 6).reshape(
                        (len(x), *trailing_shape)
                    )
                    native_values = torch.as_tensor(
                        values, dtype=torch.float32, device=device
                    )
                    interpolator = cls(x, native_values, dtype=np.float32)
                    _assert_colocated(interpolator)
                    result = interpolator(x[1:-1])
                    assert result.shape == (len(x) - 2, *trailing_shape)
                    assert result.device == native_values.device
                    xp_assert_close(result, values[1:-1], atol=TOLERANCE_ABSOLUTE_TESTS)


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
    request: pytest.FixtureRequest,
) -> None:
    """Test scalar and 2-tuple ``fill_value`` outside the interpolation range."""

    _name, cast = backend
    query = np.array([-1.0, 2.5, 6.0])  # below, in-range, above

    # ``fill_value`` takes effect only when out-of-range queries are not an
    # error, matching ``scipy.interpolate.interp1d``.
    scalar = CubicSplineInterpolator(
        cast(_X), cast(_Y), fill_value=0.0, bounds_error=False
    )(cast(query))
    scalar = as_ndarray(scalar)
    assert scalar[0] == 0.0
    assert scalar[-1] == 0.0

    tupled = CubicSplineInterpolator(
        cast(_X), cast(_Y), fill_value=(-1.0, 9.0), bounds_error=False
    )(cast(query))
    tupled = as_ndarray(tupled)
    assert tupled[0] == -1.0
    assert tupled[-1] == 9.0

    # Fill values broadcast over the data dimensions, independently of the
    # shape of the query and the location of the interpolation axis.
    for shape, axis in (((2, 6), -1), ((2, 6, 3), 1), ((6, 2, 3), 0)):
        y = np.arange(np.prod(shape), dtype=float).reshape(shape)
        fill_shape = shape[: axis % len(shape)] + shape[axis % len(shape) + 1 :]
        below = np.arange(np.prod(fill_shape), dtype=float).reshape(fill_shape)
        above = below + 10
        for fill in (below, (below, above), (np.nan, above)):
            reference = scipy.interpolate.interp1d(
                _X, y, kind="cubic", axis=axis, bounds_error=False, fill_value=fill
            )
            interpolator = CubicSplineInterpolator(
                cast(_X), cast(y), axis=axis, bounds_error=False, fill_value=fill
            )
            for points in (-1.0, 2.5, 6.0, query, query.reshape(3, 1)):
                xp_assert_close(
                    interpolator(cast(points)),
                    reference(points),
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
            interpolator.fill_value = (below, above)
            reference.fill_value = (below, above)
            xp_assert_close(
                interpolator(cast(query)),
                reference(query),
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    knots = cast(np.arange(4.0))
    ordinates = cast([0.0, 1.0, 0.0, 2.0])
    huge = 1e20 if as_ndarray(ordinates).dtype == np.float32 else 1e200
    outside = np.array([-np.inf, -huge, -1.0, 4.0, huge, np.inf])
    filled = CubicSplineInterpolator(
        knots, ordinates, bounds_error=False, fill_value=0.0
    )
    xp_assert_equal(filled(cast(outside)), np.zeros(6))
    xp_assert_equal(filled(cast(np.nan)), np.nan)

    if _name != "numpy":
        autodiff = request.getfixturevalue("autodiff")
        # Finite offsets let the derivative oracle perturb queries at infinity
        # without introducing undefined infinity-minus-infinity differences.
        _result, gradients, _variables = autodiff(
            lambda values, offsets: CubicSplineInterpolator(
                np.arange(4.0), values, bounds_error=False, fill_value=0.0
            )(
                xp_as_float_array(outside, xp=array_namespace(offsets), like=offsets)
                + offsets
            ),
            ordinates,
            np.zeros(6),
        )
        for gradient in gradients:
            xp_assert_equal(gradient, np.zeros(gradient.shape))
        mixed = np.array([-huge, -1.0, 0.5, 1.5, 2.5, 4.0, huge])
        result, _gradients, _variables = autodiff(
            lambda values, offsets: CubicSplineInterpolator(
                np.arange(4.0), values, bounds_error=False, fill_value=0.0
            )(
                xp_as_float_array(mixed, xp=array_namespace(offsets), like=offsets)
                + offsets
            ),
            ordinates,
            np.zeros(7),
        )
        reference = scipy.interpolate.interp1d(
            np.arange(4.0), [0.0, 1.0, 0.0, 2.0], kind="cubic"
        )
        expected = np.zeros(7)
        expected[2:5] = reference([0.5, 1.5, 2.5])
        xp_assert_close(result, expected, atol=TOLERANCE_ABSOLUTE_TESTS)
        xp_assert_close(
            filled(cast([0.0, 1.5, 3.0])),
            reference([0.0, 1.5, 3.0]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


def test_cubic_spline_bounds_error(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test ``bounds_error`` raises outside the interpolation range."""

    _name, cast = backend
    interpolator = CubicSplineInterpolator(cast(_X), cast(_Y), bounds_error=True)

    with pytest.raises(ValueError):
        interpolator(cast(np.array([-1.0])))

    with pytest.raises(ValueError):
        interpolator.fill_value = "extrapolate"

    interpolator.bounds_error = False
    assert not interpolator.bounds_error
    interpolator.fill_value = -1.0
    xp_assert_close(
        interpolator(cast([-1.0, 6.0])), [-1.0, -1.0], atol=TOLERANCE_ABSOLUTE_TESTS
    )
    interpolator.bounds_error = True
    assert interpolator.bounds_error
    with pytest.raises(ValueError):
        interpolator(cast([6.0]))


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
    values = as_ndarray(values)

    assert np.isnan(values[0])
    assert np.isnan(values[-1])
    assert np.isfinite(values[1])


def test_pchip_derivatives(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test PCHIP derivative orders against the *SciPy* reference."""

    _name, cast = backend
    interpolator = PchipInterpolator(cast(_X), cast(_Y))
    reference = scipy.interpolate.PchipInterpolator(_X, _Y)

    for nu in (1, 2, 3):
        xp_assert_close(
            as_ndarray(interpolator(cast(_X_E), nu=nu)),
            reference(_X_E, nu=nu),
            atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
        )

    # Orders beyond the cubic degree are identically zero.
    xp_assert_close(
        as_ndarray(interpolator(cast(_X_E), nu=4)),
        np.zeros_like(_X_E),
        atol=TOLERANCE_ABSOLUTE_TESTS * 1e-05,
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
    xp_assert_close(
        as_ndarray(interpolator(cast(_X_E))),
        reference,
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.001,
    )


def test_null_tolerance(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """Test the null interpolator honours tolerance and the default value."""

    _name, cast = backend
    interpolator = NullInterpolator(
        cast(_X), cast(_Y), absolute_tolerance=1e-6, default=-1.0
    )
    # A knot returns its value; a point away from any knot returns the default.
    values = as_ndarray(interpolator(cast(np.array([_X[2], 0.75]))))

    xp_assert_close(values[0], _Y[2], atol=TOLERANCE_ABSOLUTE_TESTS * 0.1)
    assert values[1] == -1.0


def test_validation_errors(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
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

    # Check the original lengths before a sorting gather can discard values or
    # manufacture out-of-range entries on a native backend.
    for constructor in (CubicSplineInterpolator, PchipInterpolator):
        for values in (_Y[:-1], np.append(_Y, 1.0)):
            with pytest.raises(ValueError):
                constructor(cast(_X), cast(values))
            with pytest.raises(ValueError):
                constructor(cast(_X), cast(np.stack([values, values])), axis=1)
        for knots, values in (
            (np.array(1.0), _Y),
            (_X[None, :], _Y),
            (_X, np.array(1.0)),
        ):
            with pytest.raises((ValueError, TypeError, IndexError, ZeroDivisionError)):
                constructor(cast(knots), cast(values))

    with pytest.raises(IndexError):
        CubicSplineInterpolator(cast(_X), cast(_Y), axis=1)

    for invalid in (float("nan"), float("inf"), -float("inf")):
        knots = _X.copy()
        knots[-1] = invalid
        values = _Y.copy()
        values[2] = invalid
        for x, y in ((knots, _Y), (_X, values)):
            with pytest.raises(ValueError):
                PchipInterpolator(cast(x), cast(y))

    # interp1d accepts nonfinite cubic data; do not impose PCHIP's finite-data
    # requirement on this class.
    values = _Y.copy()
    values[2] = np.nan
    assert np.all(
        np.isnan(
            as_ndarray(CubicSplineInterpolator(cast(_X), cast(values))(cast(_X_E)))
        )
    )


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
    xp_assert_close(
        as_ndarray(evaluate(x_e)), reference, atol=TOLERANCE_ABSOLUTE_TESTS * 0.1
    )


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
        xp_assert_close(
            as_ndarray(jax.jit(interpolator.__call__)(query)),
            as_ndarray(interpolator(query)),
            atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
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
    xp_assert_close(
        as_ndarray(jax.jit(extrapolator)(query)),
        as_ndarray(extrapolator(query)),
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
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
    xp_assert_close(
        as_ndarray(kernel(cast(_X_E_UNIFORM))),
        as_ndarray(reference(cast(_X_E_UNIFORM))),
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
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
    xp_assert_close(
        as_ndarray(interpolator(cast(_X_E_UNIFORM))),
        as_ndarray(reference(cast(_X_E_UNIFORM))),
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
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
    reference = as_ndarray(interpolator(cast(_X_E_UNIFORM)))
    cached = interpolator._impl  # noqa: SLF001

    with pytest.raises(ValueError):
        interpolator.padding_kwargs = {"mode": "edge", "pad_width": (3, 3)}

    # The rejected write recovers in place, so a wrapper holding the
    # implementation by reference (e.g. an Extrapolator) stays attached.
    assert interpolator._impl is cached  # noqa: SLF001
    assert interpolator.padding_kwargs["mode"] == "reflect"
    xp_assert_close(
        as_ndarray(interpolator(cast(_X_E_UNIFORM))),
        reference,
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
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
        xp_assert_close(
            as_ndarray(interp(query)),
            as_ndarray(reference(query)),
            atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
        )

        invalid_x = [new_x[:-1], np.array(1.0), new_x[None, :]]
        duplicates = new_x.copy()
        duplicates[2] = duplicates[1]
        invalid_x.append(duplicates)
        invalid_y = [_Y_UNIFORM[:-1], np.append(_Y_UNIFORM, 1.0), np.array(1.0)]
        if interpolator is PchipInterpolator:
            invalid_x.append(new_x[::-1].copy())
            for nonfinite in (float("nan"), float("inf"), -float("inf")):
                knots = new_x.copy()
                knots[-1] = nonfinite
                values = _Y_UNIFORM.copy()
                values[2] = nonfinite
                invalid_x.append(knots)
                invalid_y.append(values)

        # The dispatcher and standalone native classes both retain their last
        # valid data and evaluation after a rejected replacement.
        if interp is dispatching or name != "numpy":
            expected_x = as_ndarray(interp.x).copy()
            expected_y = as_ndarray(interp.y).copy()
            expected = as_ndarray(interp(query)).copy()
            for attribute, replacements in (("x", invalid_x), ("y", invalid_y)):
                for values in replacements:
                    with pytest.raises(
                        (ValueError, TypeError, IndexError, ZeroDivisionError)
                    ):
                        setattr(interp, attribute, cast(values))
                    xp_assert_equal(interp.x, expected_x)
                    xp_assert_equal(interp.y, expected_y)
                    xp_assert_close(interp(query), expected)

        if interpolator is CubicSplineInterpolator:
            if interp is dispatching or name != "numpy":
                strict = type(interp)(cast(new_x), cast(_Y_UNIFORM), assume_sorted=True)
                expected = as_ndarray(strict(query)).copy()
                with pytest.raises(ValueError):
                    strict.x = cast(new_x[::-1].copy())
                xp_assert_equal(strict.x, new_x)
                xp_assert_close(strict(query), expected)
            # Default cubic sorting remains supported by data setters.
            interp.x = cast(new_x[::-1].copy())
            sorted_reference = interpolator(cast(new_x[::-1].copy()), cast(_Y_UNIFORM))
            xp_assert_close(interp(query), sorted_reference(query))
            values = _Y_UNIFORM.copy()
            values[2] = np.nan
            interp.y = cast(values)
            assert np.all(np.isnan(as_ndarray(interp(query))))
        elif name != "numpy":
            for extrapolate in (False, None):
                interp.extrapolate = extrapolate
                interp.y = cast(_Y_UNIFORM)
                assert interp.extrapolate is extrapolate
                assert np.all(np.isnan(as_ndarray(interp(cast(new_x[-1] + 1)))))


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
    xp_assert_close(
        as_ndarray(extrapolator(query)),
        as_ndarray(reference(query)),
        atol=TOLERANCE_ABSOLUTE_TESTS * 0.001,
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
        xp_assert_close(
            as_ndarray(interpolator(query)),
            as_ndarray(reference(query)),
            atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
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
def test_switch_rollback_on_failed_rebuild(
    cast: Callable[[Any], Any],
) -> None:
    """
    Test that a backend switch whose rebuild fails leaves the interpolator
    unchanged and still evaluable (no half-applied state).
    """

    with array_api_enable(True):
        interpolator = LinearInterpolator(_X, _Y)  # *NumPy* data.
        query = np.array([2.5])
        reference = as_ndarray(interpolator(query))

        with pytest.raises(ValueError):
            interpolator.x = cast(_X[:-1])  # Framework array, mismatched length.

        assert interpolator.backend == "numpy"
        xp_assert_close(
            as_ndarray(interpolator(query)),
            reference,
            atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
        )


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
        xp_assert_close(
            as_ndarray(result),
            as_ndarray(reference(query)),
            atol=TOLERANCE_ABSOLUTE_TESTS * 0.1,
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

    dtype = (
        np.float32
        if str(getattr(interpolator.x, "device", "")).startswith("mps")
        else dtype
    )
    assert as_ndarray(interpolator.x).dtype == np.dtype(dtype)
    assert as_ndarray(interpolator.y).dtype == np.dtype(dtype)
    assert as_ndarray(interpolator(cast(2.5))).dtype == np.dtype(dtype)

    if _name == "jax" and interpolator_type is CubicSplineInterpolator:
        for x_dtype, y_dtype in ((np.float32, np.float64), (np.float64, np.float32)):
            knots = cast(x.astype(x_dtype))
            ordinates = cast(y.astype(y_dtype))
            mixed = interpolator_type(knots, ordinates)
            expected_dtype = np.result_type(x_dtype, y_dtype)
            assert as_ndarray(mixed.x).dtype == expected_dtype
            assert as_ndarray(mixed.y).dtype == expected_dtype
            xp_assert_close(
                mixed(2.5),
                interpolator_type(x, y)(2.5),
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )
            mixed = interpolator_type(
                cast(x.astype(np.float32)), cast(y.astype(np.float32))
            )
            mixed.y = ordinates
            mixed.x = knots
            assert as_ndarray(mixed.x).dtype == expected_dtype
            assert as_ndarray(mixed.y).dtype == expected_dtype
            xp_assert_close(
                mixed(2.5),
                interpolator_type(x, y)(2.5),
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    if _name == "torch" and dtype == np.float64:
        import torch  # noqa: PLC0415

        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            precise_x = [1.0 + i * 1e-8 for i in range(6)]
            precise_y = torch.arange(6, dtype=torch.float64, device="cpu")
            precise = interpolator_type(precise_x, precise_y, dtype=np.float64)
            assert bool(torch.all(precise.x[1:] > precise.x[:-1]))
            xp_assert_close(precise.x, np.array(precise_x), atol=0, rtol=0)
            query = torch.tensor(1.0 + 2.5e-8, dtype=torch.float64, device="cpu")
            expected = interpolator_type(np.array(precise_x), np.arange(6.0))(
                float(query)
            )
            xp_assert_close(precise(query), expected, atol=TOLERANCE_ABSOLUTE_TESTS)
            implicit = interpolator_type(precise_x, precise_y)
            xp_assert_close(implicit.x, np.array(precise_x), atol=0, rtol=0)
            precise.x = precise_x
            precise.y = [float(i) + 1e-8 for i in range(6)]
            xp_assert_close(precise.y, np.arange(6.0) + 1e-8, atol=0, rtol=0)
        finally:
            torch.set_default_dtype(original_dtype)

    # Host views must remain valid in mixed constructors, setters and queries.
    x_view = np.asarray(x)[::-1].copy()[::-1]
    y_view = np.asarray(y)[::-1]
    query_view = np.array([2.5, 2.5])[::-1]
    # Compare views with contiguous data on the same backend: this checks
    # layout handling independently of platform-specific float32 rounding.
    reference = interpolator_type(
        cast(np.ascontiguousarray(x_view)),
        cast(np.ascontiguousarray(y_view)),
        dtype=dtype,
    )(cast(np.ascontiguousarray(query_view)))
    for knots, values in ((cast(x_view), y_view), (x_view, cast(y_view))):
        viewed = interpolator_type(knots, values, dtype=dtype)
        xp_assert_close(viewed(query_view), reference, atol=TOLERANCE_ABSOLUTE_TESTS)
        viewed.x = x_view
        viewed.y = y_view
        xp_assert_close(viewed(query_view), reference, atol=TOLERANCE_ABSOLUTE_TESTS)

    if interpolator_type is PchipInterpolator:
        # Nonuniform intervals distinguish the harmonic-mean weights when
        # either adjacent secant is much smaller than its neighbour.
        devices = [None]
        if _name == "torch":
            import torch  # noqa: PLC0415

            devices = [torch.device("cpu")]
            if dtype == np.float32 and torch.backends.mps.is_available():
                devices.append(torch.device("mps"))

        for device in devices:

            def native_pchip(a: Any, device_: Any = device) -> Any:
                if _name == "torch":
                    return torch.as_tensor(
                        a, dtype=getattr(torch, np.dtype(dtype).name), device=device_
                    )
                return cast(a).astype(dtype)

            for grid in ([0.0, 1.0, 3.0], [0.0, 1.0, 2.0]):
                for raw, interval_index in (
                    ([-1e20, -1.0, 0.0], 1),
                    ([0.0, 1.0, 1e20], 0),
                    ([1e20, 1.0, 0.0], 1),
                    ([0.0, -1.0, -1e20], 0),
                    ([0.0, 1.0, 3.0], 0),
                    ([0.0, 2.0, 1.0], 1),
                    ([0.0, 0.0, 1.0], 1),
                ):
                    rounded = np.asarray(raw, dtype=dtype).astype(np.float64)
                    knots = native_pchip(grid)
                    ordinates = native_pchip(rounded)
                    control = scipy.interpolate.PchipInterpolator(grid, rounded)
                    left, right = grid[interval_index : interval_index + 2]
                    points = left + np.array([0.25, 0.5]) * (right - left)
                    magnitude = max(1.0, float(np.max(np.abs(control(points)))))
                    query_scale = (right - left) / 2
                    for query_values in (points[0], points.reshape(1, 2)):
                        query = native_pchip(query_values)
                        with (
                            torch.device("cpu")
                            if _name == "torch"
                            else contextlib.nullcontext()
                        ):
                            actual = PchipInterpolator(knots, ordinates, dtype=dtype)
                            for order in range(4):
                                result = actual(query, nu=order)
                                assert result.shape == np.shape(query_values)
                                assert result.dtype == ordinates.dtype
                                assert getattr(result, "device", None) == getattr(
                                    ordinates, "device", None
                                )
                                xp_assert_close(
                                    as_ndarray(result).astype(np.float64)
                                    * query_scale**order
                                    / magnitude,
                                    control(query_values, nu=order)
                                    * query_scale**order
                                    / magnitude,
                                    atol=TOLERANCE_ABSOLUTE_TESTS,
                                    err_msg=f"{grid=} {raw=} {order=}",
                                )
                        if _name == "numpy":
                            continue

                        # Normalize ordinate perturbations component-wise: a
                        # unit direction must resolve both the 1e20 and unit
                        # ordinates without changing the active slope branch.
                        scales = np.maximum(np.abs(rounded), 1.0)
                        parameters = native_pchip(rounded / scales)
                        native_scales = native_pchip(scales)
                        # Canonical interval coordinates keep query tangents
                        # comparable across the unequal interval widths.
                        query = native_pchip((query_values - left) / query_scale - 1)

                        def normalized_pchip(
                            parameters_: Any,
                            query_: Any,
                            knots_: Any = knots,
                            scales_: Any = native_scales,
                            magnitude_: float = magnitude,
                            left_: float = left,
                            query_scale_: float = query_scale,
                        ) -> Any:
                            return (
                                PchipInterpolator(
                                    knots_, parameters_ * scales_, dtype=dtype
                                )(left_ + (query_ + 1) * query_scale_)
                                / magnitude_
                            )

                        with (
                            torch.device("cpu")
                            if _name == "torch"
                            else contextlib.nullcontext()
                        ):
                            if _name == "torch":
                                parameters = parameters.clone().requires_grad_(True)
                                query = query.clone().requires_grad_(True)
                                gradients = [
                                    torch.autograd.grad(
                                        normalized_pchip(parameters, query).sum(),
                                        (parameters, query),
                                    )
                                ]
                            else:
                                import jax  # noqa: PLC0415
                                import jax.numpy as jnp  # noqa: PLC0415

                                def objective(parameters_: Any, query_: Any) -> Any:
                                    return jnp.sum(
                                        normalized_pchip(parameters_, query_)
                                    )

                                gradients = [
                                    differentiate(objective, argnums=(0, 1))(
                                        parameters, query
                                    )
                                    for differentiate in (jax.jacfwd, jax.jacrev)
                                ]
                        expected_query = (
                            control(query_values, nu=1) * query_scale / magnitude
                        )
                        # A plateau's ordinate Jacobian is not defined at the
                        # branch boundary; query derivatives remain defined.
                        expected_parameters = []
                        step = 1e-4
                        for index in range(3):
                            perturbation = np.zeros(3)
                            perturbation[index] = step * scales[index]
                            plus = scipy.interpolate.PchipInterpolator(
                                grid, rounded + perturbation
                            )(query_values)
                            minus = scipy.interpolate.PchipInterpolator(
                                grid, rounded - perturbation
                            )(query_values)
                            expected_parameters.append(
                                np.sum(plus - minus) / (2 * step * magnitude)
                            )
                        for gradient_y, gradient_query in gradients:
                            assert (
                                getattr(gradient_y, "device", None) == parameters.device
                            )
                            assert (
                                getattr(gradient_query, "device", None) == query.device
                            )
                            if raw[0] != raw[1]:
                                xp_assert_close(
                                    gradient_y,
                                    expected_parameters,
                                    atol=TOLERANCE_ABSOLUTE_TESTS,
                                    err_msg=f"{grid=} {raw=}",
                                )
                            xp_assert_close(
                                gradient_query,
                                expected_query,
                                atol=TOLERANCE_ABSOLUTE_TESTS,
                                err_msg=f"{grid=} {raw=}",
                            )

    if dtype == np.float32 and interpolator_type in (
        PchipInterpolator,
        SpragueInterpolator,
    ):
        # Reference calculations use the exact rounded float32 ordinates in
        # float64, so the comparison does not hide input quantization.
        knots = np.arange(6, dtype=np.float32)
        queries = np.array([1.25, 2.5], dtype=np.float32)
        offset_values = (2e38 + np.arange(6) * 1e36).astype(np.float32)
        profiles = (
            np.full(6, 2e38, dtype=np.float32),
            offset_values,
            np.full(6, 1e36, dtype=np.float32),
            np.array([-2e38, -1e38, 0, 1e38, 2e38, 2.5e38], dtype=np.float32),
        )
        devices: list[Any] = [None]
        if _name == "torch":
            import torch  # noqa: PLC0415

            devices = [torch.device("cpu")]
            if torch.backends.mps.is_available():
                devices.append(torch.device("mps"))

        for device in devices:

            def native(a: Any, device_: Any = device) -> Any:
                if _name == "torch":
                    return torch.as_tensor(a, dtype=torch.float32, device=device_)
                result = cast(a)
                return result.astype(np.float32) if _name == "jax" else result

            for values in profiles:
                expected = interpolator_type(knots, values, dtype=np.float64)(queries)
                ordinates = native(values)
                large = interpolator_type(native(knots), ordinates)
                result = large(native(queries))
                assert result.dtype == ordinates.dtype
                assert getattr(result, "device", None) == getattr(
                    ordinates, "device", None
                )
                xp_assert_close(
                    as_ndarray(result) / 1e38,
                    as_ndarray(expected) / 1e38,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

            if _name == "numpy":
                continue

            # A scalar objective avoids overflowing a sum of large outputs;
            # raw ordinate derivatives remain ordinary interpolation weights.
            query = native(np.float32(1.25))
            ordinates = native(offset_values)
            x_native = native(knots)

            def evaluate(values: Any, point: Any, knots_: Any = x_native) -> Any:
                return interpolator_type(knots_, values)(point)

            values_64 = offset_values.astype(np.float64)
            reference = interpolator_type(knots, values_64, dtype=np.float64)
            step = 1e31
            weights = []
            for index in range(len(values_64)):
                perturbation = np.zeros_like(values_64)
                perturbation[index] = step
                weights.append(
                    (
                        interpolator_type(
                            knots, values_64 + perturbation, dtype=np.float64
                        )(1.25)
                        - interpolator_type(
                            knots, values_64 - perturbation, dtype=np.float64
                        )(1.25)
                    )
                    / (2 * step)
                )
            query_derivative = (
                scipy.interpolate.PchipInterpolator(knots, values_64)(1.25, nu=1)
                if interpolator_type is PchipInterpolator
                else (reference(1.251) - reference(1.249)) / 0.002
            )
            if _name == "torch":
                ordinates = ordinates.clone().requires_grad_(True)
                query = query.clone().requires_grad_(True)
                offset_result = interpolator_type(x_native, ordinates)(query)
                gradients = [torch.autograd.grad(offset_result, (ordinates, query))]
            else:
                import jax  # noqa: PLC0415

                gradients = [
                    jax.jacfwd(evaluate, argnums=(0, 1))(ordinates, query),
                    jax.jacrev(evaluate, argnums=(0, 1))(ordinates, query),
                ]
            for gradient_y, gradient_query in gradients:
                assert getattr(gradient_y, "device", None) == ordinates.device
                assert getattr(gradient_query, "device", None) == query.device
                xp_assert_close(gradient_y, weights, atol=TOLERANCE_ABSOLUTE_TESTS)
                xp_assert_close(
                    as_ndarray(gradient_query) / 1e36,
                    query_derivative / 1e36,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

            if interpolator_type is SpragueInterpolator:
                knot_profiles = []
                for index in (0, 2, 5):
                    values = np.full(6, 1e38, dtype=np.float32)
                    values[index] = 1e-35
                    knot_profiles.append(
                        (values, (float(index), min(index + 0.75, 4.75)))
                    )
                knot_profiles.extend(
                    (values, (1.25, 4.75)) for values in (offset_values, profiles[-1])
                )
                for values, points in knot_profiles:
                    for rank_2 in (False, True):
                        data = (
                            np.stack([values, values / 2], axis=-1)
                            if rank_2
                            else values
                        )
                        ordinates = native(data)
                        short_knots = native(knots)

                        def sprague(
                            data_: Any,
                            point_: Any,
                            knots_: Any = short_knots,
                            rank_2_: bool = rank_2,
                        ) -> Any:
                            result_ = SpragueInterpolator(knots_, data_)(point_)
                            return result_[0] if rank_2_ else result_

                        # Unit-basis interpolation is an independent ordinate
                        # Jacobian oracle for this linear-in-data interpolator.
                        basis = np.eye(6)
                        control = SpragueInterpolator(
                            knots, data.astype(np.float64), dtype=np.float64
                        )
                        query_control = SpragueInterpolator(
                            knots, values.astype(np.float64) / 1e38, dtype=np.float64
                        )
                        for point in points:
                            query = native(np.float32(point))
                            result = SpragueInterpolator(short_knots, ordinates)(query)
                            assert getattr(result, "device", None) == ordinates.device
                            if point in knots:
                                xp_assert_equal(result, ordinates[int(point)])
                            else:
                                xp_assert_close(
                                    as_ndarray(result).astype(np.float64) / 1e38,
                                    control(point) / 1e38,
                                    atol=TOLERANCE_ABSOLUTE_TESTS,
                                )
                            weights = np.array(
                                [
                                    SpragueInterpolator(
                                        knots, column, dtype=np.float64
                                    )(point)
                                    for column in basis
                                ]
                            )
                            if rank_2:
                                weights = np.stack(
                                    [weights, np.zeros_like(weights)], axis=-1
                                )
                            step = 1e-5
                            if point == 0:
                                query_derivative = (
                                    -3 * query_control(point)
                                    + 4 * query_control(point + step)
                                    - query_control(point + 2 * step)
                                ) / (2 * step)
                            elif point == 5:
                                query_derivative = (
                                    3 * query_control(point)
                                    - 4 * query_control(point - step)
                                    + query_control(point - 2 * step)
                                ) / (2 * step)
                            else:
                                query_derivative = (
                                    query_control(point + step)
                                    - query_control(point - step)
                                ) / (2 * step)
                            if _name == "torch":
                                tracked = ordinates.clone().requires_grad_(True)
                                tracked_query = query.clone().requires_grad_(True)
                                gradients = [
                                    torch.autograd.grad(
                                        sprague(tracked, tracked_query),
                                        (tracked, tracked_query),
                                    )
                                ]
                            else:
                                gradients = [
                                    jax.jacfwd(sprague, argnums=(0, 1))(
                                        ordinates, query
                                    ),
                                    jax.jacrev(sprague, argnums=(0, 1))(
                                        ordinates, query
                                    ),
                                ]
                            for gradient_y, gradient_query in gradients:
                                assert (
                                    getattr(gradient_y, "device", None)
                                    == ordinates.device
                                )
                                assert (
                                    getattr(gradient_query, "device", None)
                                    == query.device
                                )
                                xp_assert_close(
                                    gradient_y, weights, atol=TOLERANCE_ABSOLUTE_TESTS
                                )
                                xp_assert_close(
                                    as_ndarray(gradient_query).astype(np.float64)
                                    / 1e38,
                                    query_derivative,
                                    atol=TOLERANCE_ABSOLUTE_TESTS,
                                )

            if interpolator_type is PchipInterpolator:
                # Tiny/huge adjacent slopes must not invert an underflowed
                # ratio. The weights follow from the Hermite basis at t=1/2.
                slope_cases = (
                    ([0, 1, 2], [0, 1e-20, 1e38], 0.5, [0.75, 0.25, 0], 1e-20),
                    ([0, 1, 2], [-1e38, -1e-20, 0], 1.5, [0, 0.25, 0.75], 1e-20),
                    ([0, 1, 2], [0, 1e-20, 2e-20], 0.5, [0.375, 0.75, -0.125], 1e-20),
                    ([0, 1, 2], [0, 1e20, 2e20], 0.5, [0.375, 0.75, -0.125], 1e20),
                    ([0, 1, 2], [-2e38, 0, 2e38], 0.5, [0.375, 0.75, -0.125], 1e38),
                    ([0, 1, 3], [0, 2e38, -2e38], 0.5, [1 / 3, 0.6875, -1 / 48], 1e38),
                    ([0, 1, 2], [0, 1e-35, 1e38], 0.5, [0.75, 0.25, 0], 1e-35),
                    ([0, 1, 2], [1e-35, 1e38, 2e38], 0.0, [1, 0, 0], 1e-35),
                    ([0, 1, 2], [2e38, 1e38, 1e-35], 2.0, [0, 0, 1], 1e-35),
                    ([0, 1, 2], [-2e38, 1e-35, 2e38], 1.0, [0, 1, 0], 1e-35),
                )
                peak_cases = []
                for span in (np.float32(1), np.float32(1e-6)):
                    grid = (np.arange(3, dtype=np.float32) * span).tolist()
                    points = (
                        float(span * np.float32(0.75)),
                        float(span),
                        float(span * np.float32(1.25)),
                        float(np.nextafter(span, np.float32(0))),
                        float(np.nextafter(span, np.float32(2))),
                    )
                    if span != 1:
                        # Away from this narrow peak the true query derivative
                        # exceeds float32; keep representable knot/neighbor
                        # controls.
                        points = points[1:2] + points[3:]
                    for point in points:
                        t = point / float(span)
                        if t <= 1:
                            weights = [
                                0.5 * t**3 - 1.5 * t + 1,
                                -(t**2) + 2 * t,
                                -0.5 * t**3 + t**2 - 0.5 * t,
                            ]
                        else:
                            u = t - 1
                            weights = [
                                0.5 * (u**3 - u**2),
                                1 - u**2,
                                -0.5 * u**3 + 1.5 * u**2,
                            ]
                        peak_cases.append(
                            (grid, [-2e38, 2e38, -2e38], point, weights, 1e38)
                        )
                slope_cases = (*slope_cases, *peak_cases)
                for grid, raw, point, expected_weights, magnitude in slope_cases:
                    rounded = np.asarray(raw, dtype=np.float32)
                    ordinates = native(rounded)
                    query = native(np.float32(point))
                    short_knots = native(np.asarray(grid, dtype=np.float32))

                    def pchip(
                        values: Any, point: Any, knots_: Any = short_knots
                    ) -> Any:
                        return PchipInterpolator(knots_, values)(point)

                    control = scipy.interpolate.PchipInterpolator(
                        grid, rounded.astype(np.float64)
                    )
                    value = pchip(ordinates, query)
                    assert getattr(value, "device", None) == ordinates.device
                    # At an exact knot the supplied ordinate is the oracle;
                    # even float64 polynomial evaluation can cancel tiny values.
                    expected_value = (
                        float(
                            rounded[
                                next(
                                    index
                                    for index, knot in enumerate(grid)
                                    if point == knot
                                )
                            ]
                        )
                        if point in grid
                        else control(point)
                    )
                    xp_assert_close(
                        as_ndarray(value).astype(np.float64) / magnitude,
                        expected_value / magnitude,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )
                    for order in (1, 2, 3):
                        expected = control(point, nu=order)
                        if abs(expected) <= np.finfo(np.float32).max:
                            derivative = PchipInterpolator(short_knots, ordinates)(
                                query, nu=order
                            )
                            xp_assert_close(
                                as_ndarray(derivative).astype(np.float64) / magnitude,
                                expected / magnitude,
                                atol=TOLERANCE_ABSOLUTE_TESTS,
                                err_msg=f"{grid=} {raw=} {point=} {order=}",
                            )
                    if _name == "torch":
                        ordinates = ordinates.clone().requires_grad_(True)
                        query = query.clone().requires_grad_(True)
                        gradients = [
                            torch.autograd.grad(
                                pchip(ordinates, query), (ordinates, query)
                            )
                        ]
                    else:
                        gradients = [
                            jax.jacfwd(pchip, argnums=(0, 1))(ordinates, query),
                            jax.jacrev(pchip, argnums=(0, 1))(ordinates, query),
                        ]
                    for gradient_y, gradient_query in gradients:
                        assert getattr(gradient_y, "device", None) == ordinates.device
                        assert getattr(gradient_query, "device", None) == query.device
                        xp_assert_close(
                            gradient_y,
                            expected_weights,
                            atol=TOLERANCE_ABSOLUTE_TESTS,
                            err_msg=f"{grid=} {raw=} {point=}",
                        )
                        query_reference = control(point, nu=1)
                        query_scale = magnitude
                        if raw[0] == raw[2] == -raw[1]:
                            span = float(grid[1])
                            query_reference = (
                                -4 * float(rounded[1]) * (point - span) / span**2
                            )
                            query_scale = (
                                abs(query_reference) if query_reference else magnitude
                            )
                        xp_assert_close(
                            as_ndarray(gradient_query).astype(np.float64) / query_scale,
                            query_reference / query_scale,
                            atol=TOLERANCE_ABSOLUTE_TESTS,
                            err_msg=f"{grid=} {raw=} {point=}",
                        )


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

    assert as_ndarray(interpolator.x).dtype == np.dtype(np.float32)


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

    assert as_ndarray(interpolator.y).dtype == np.float32
