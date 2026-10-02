"""
Unit tests for the :mod:`colour.algebra.interpolation` dispatching
interpolators.

Backend-parametrised: each interpolator is exercised on every backend whose
framework is installed *and* whose ``backends`` submodule exists, so
coverage grows automatically as backends are added.
"""

from __future__ import annotations

import importlib
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
from colour.utilities import array_api_enable

if TYPE_CHECKING:
    from collections.abc import Callable

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


@pytest.fixture(params=[name for name, _ in _BACKENDS])
def backend(request: pytest.FixtureRequest) -> tuple[str, Callable[[Any], Any]]:
    """Yield ``(name, cast)`` for each available backend."""

    return next(item for item in _BACKENDS if item[0] == request.param)


def test_detect_backend_numpy() -> None:
    """Test backend detection for *NumPy* inputs and Python scalars."""

    assert detect_backend(_X, _Y) == "numpy"
    assert detect_backend([0.0, 1.0], 2.0) == "numpy"


def test_numpy_data_backend_query_namespace(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """
    Test that a *NumPy*-data interpolator or extrapolator evaluated at a backend
    query returns the result in the query's namespace.

    Sampling *NumPy* data (e.g. a colour matching function) at a *PyTorch* /
    *JAX* query must not drop back to *NumPy*: the output follows the union of
    the data and query namespaces.
    """

    name, cast = backend

    interpolator = LinearInterpolator(_X, _Y)  # *NumPy* data.
    assert interpolator.backend == "numpy"

    # A backend result follows the query only while Array API dispatch is
    # enabled, matching how the ``colour`` interpolators are used on backends.
    with array_api_enable(name != "numpy"):
        assert detect_backend(interpolator(cast(_X_E))) == name

        extrapolator = Extrapolator(LinearInterpolator(_X, _Y), left=0.0)
        assert detect_backend(extrapolator(cast(_X_E))) == name


def test_numpy_data_backend_query_flag_disabled(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """
    Test that re-dispatch to the query backend is gated on Array API dispatch.

    With dispatch disabled the *NumPy* backend is retained and a (concrete)
    backend query returns a *NumPy* result.
    """

    _name, cast = backend

    interpolator = LinearInterpolator(_X, _Y)
    with array_api_enable(False):
        assert detect_backend(interpolator(cast(_X_E))) == "numpy"


def _query_gradient(
    name: str, function: Callable[[Any], Any], x: np.ndarray
) -> np.ndarray:
    """Return ``d(sum(function(q)))/dq`` at ``x`` via ``name``'s autodiff."""

    x = np.atleast_1d(np.asarray(x, dtype=np.float64))

    if name == "torch":
        import torch  # noqa: PLC0415

        query = torch.tensor(x, requires_grad=True, dtype=torch.float64)
        function(query).sum().backward()

        return _to_numpy(query.grad)

    if name == "jax":
        import jax  # noqa: PLC0415
        import jax.numpy as jnp  # noqa: PLC0415

        return _to_numpy(jax.grad(lambda value: function(value).sum())(jnp.asarray(x)))

    error = f'Unsupported differentiable backend: "{name}".'
    raise ValueError(error)


def _finite_difference(interpolator: Any, x: np.ndarray, h: float = 1e-5) -> np.ndarray:
    """Return the central finite-difference derivative of ``interpolator``."""

    x = np.atleast_1d(np.asarray(x, dtype=np.float64))

    return (_to_numpy(interpolator(x + h)) - _to_numpy(interpolator(x - h))) / (2 * h)


@pytest.mark.parametrize(
    "interpolator_type",
    [LinearInterpolator, CubicSplineInterpolator, PchipInterpolator],
)
def test_numpy_data_differentiable_query(
    backend: tuple[str, Callable[[Any], Any]],
    interpolator_type: type,
) -> None:
    """
    Test that a *NumPy*-data interpolator is differentiable at a backend query.

    A gradient-tracked *PyTorch* or *JAX* query must not drop back to *NumPy*
    (which would raise on the conversion and sever the graph); evaluation
    re-dispatches to the query backend, and the gradient matches a *NumPy*
    finite-difference reference.
    """

    name, _cast = backend
    if name == "numpy":
        pytest.skip("Differentiability is only defined for non-NumPy backends.")

    x = np.array([2.5])  # Between knots: the linear interpolant is smooth here.

    with array_api_enable(True):
        interpolator = interpolator_type(_X, _Y)  # *NumPy* data.
        gradient = _query_gradient(name, interpolator, x)

    reference = _finite_difference(interpolator_type(_X, _Y), x)
    np.testing.assert_allclose(gradient, reference, atol=1e-4)


def test_extrapolator_differentiable_query(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """
    Test that a *NumPy*-data extrapolator is differentiable at a backend query.

    The out-of-range *Linear* gradient equals the boundary-pair slope.
    """

    name, _cast = backend
    if name == "numpy":
        pytest.skip("Differentiability is only defined for non-NumPy backends.")

    x_i = np.array([3.0, 4.0, 5.0])
    y_i = np.array([1.0, 2.0, 3.0])
    slope = (y_i[1] - y_i[0]) / (x_i[1] - x_i[0])

    with array_api_enable(True):
        extrapolator = Extrapolator(LinearInterpolator(x_i, y_i))
        gradient = _query_gradient(name, extrapolator, np.array([1.0, 6.0]))

    np.testing.assert_allclose(gradient, [slope, slope], atol=1e-6)


def test_extrapolator_differentiable_query_interpolator_mutated(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """
    Test that a cross-backend query reflects in-place mutation of the wrapped
    interpolator after the cross-backend implementation has been cached.
    """

    name, _cast = backend
    if name == "numpy":
        pytest.skip("Differentiability is only defined for non-NumPy backends.")

    x_i = np.array([3.0, 4.0, 5.0])

    with array_api_enable(True):
        extrapolator = Extrapolator(LinearInterpolator(x_i, np.array([1.0, 2.0, 3.0])))
        # Cache the cross-backend implementation, then mutate the wrapped
        # interpolator in place through the live getter.
        _query_gradient(name, extrapolator, np.array([1.0, 6.0]))
        extrapolator.interpolator.y = np.array([1.0, 6.0, 11.0])
        gradient = _query_gradient(name, extrapolator, np.array([1.0, 6.0]))

    np.testing.assert_allclose(gradient, [5.0, 5.0], atol=1e-6)


def test_cross_backend_query_not_implemented(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """
    Test that a non-*NumPy* interpolator raises for a different backend query.
    """

    name, cast = backend
    if name == "numpy":
        pytest.skip("The NumPy backend re-dispatches instead of raising.")

    foreign = "jax" if name != "jax" else "torch"

    class _ForeignArray:
        pass

    _ForeignArray.__module__ = foreign

    interpolator = LinearInterpolator(cast(_X), cast(_Y))
    with array_api_enable(True), pytest.raises(NotImplementedError):
        interpolator(_ForeignArray())


def test_numpy_data_concrete_query_promoted(
    backend: tuple[str, Callable[[Any], Any]],
) -> None:
    """
    Test that a concrete backend query is evaluated with the *NumPy* backend and
    promoted to the query namespace.

    This preserves *NumPy*-only features (e.g. non-"reflect" kernel padding) that
    the query backend does not implement, while a differentiable query
    re-dispatches (see :func:`test_numpy_data_differentiable_query`).
    """

    name, cast = backend
    if name == "numpy":
        pytest.skip("Promotion only applies to non-NumPy queries.")

    with array_api_enable(True):
        interpolator = CubicSplineInterpolator(_X, _Y)  # *NumPy* data.
        result = interpolator(cast(_X_E))  # Concrete backend query.

    assert detect_backend(result) == name
    reference = scipy.interpolate.interp1d(_X, _Y, kind="cubic")(_X_E)
    np.testing.assert_allclose(_to_numpy(result), reference, atol=1e-10)


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

    name, cast = backend

    with array_api_enable(name != "numpy"):
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

    name, cast = backend

    with array_api_enable(name != "numpy"):
        interpolator = interpolator_type(cast(x), cast(y))
        result = interpolator(cast(np.array([])))

    assert _to_numpy(result).shape == (0,)


@pytest.mark.parametrize(
    "interpolator_type",
    [LinearInterpolator, CubicSplineInterpolator, PchipInterpolator],
)
def test_scalar_differentiable_query(
    backend: tuple[str, Callable[[Any], Any]],
    interpolator_type: type,
) -> None:
    """Test that a 0-d gradient-tracked query re-dispatches and is 0-d."""

    name, _cast = backend
    if name == "numpy":
        pytest.skip("Differentiability is only defined for non-NumPy backends.")

    with array_api_enable(True):
        interpolator = interpolator_type(_X, _Y)  # *NumPy* data.

        if name == "torch":
            import torch  # noqa: PLC0415

            query = torch.tensor(2.5, requires_grad=True, dtype=torch.float64)
            result = interpolator(query)
            assert result.ndim == 0
            result.backward()
            assert query.grad is not None
            gradient = float(query.grad)
        else:
            import jax  # noqa: PLC0415
            import jax.numpy as jnp  # noqa: PLC0415

            gradient = float(jax.grad(interpolator)(jnp.asarray(2.5)))

    reference = _finite_difference(interpolator_type(_X, _Y), np.array([2.5]))[0]
    np.testing.assert_allclose(gradient, reference, atol=1e-4)


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


def test_dimension_validation(backend: tuple[str, Callable[[Any], Any]]) -> None:
    """Test that mismatched variable dimensions raise a ``ValueError``."""

    _name, cast = backend
    with pytest.raises(ValueError):
        LinearInterpolator(cast(_X), cast(_Y[:-1]))


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
