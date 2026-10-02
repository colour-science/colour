"""
Backend detection and dispatching interpolators.

The dispatching interpolators inspect the array namespace of their input data
at construction and delegate to the matching backend submodule
(``.backends.<backend>``), which provides an identically named,
backend-specialised interpolator.

Evaluation resolves the backend from the union of the construction data and the
query namespace: when Array API dispatch is enabled, a *NumPy*-data interpolator
evaluated at a *PyTorch* or *JAX* query re-dispatches to the query backend, so
the computation stays in the query namespace and preserves automatic
differentiation. See :class:`_BackendDispatcher`.
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "detect_backend",
    "LinearInterpolator",
    "NearestNeighbourInterpolator",
    "NullInterpolator",
    "SpragueInterpolator",
    "CubicSplineInterpolator",
    "PchipInterpolator",
    "KernelInterpolator",
]

# Root module name of an array's type mapped to the backend key. Detection is
# by module name so that neither *PyTorch* nor *JAX* is imported unless one of
# their arrays is actually passed in.
_ROOT_MODULE_BACKENDS: dict[str, str] = {
    "torch": "torch",
    "jax": "jax",
    "jaxlib": "jax",
    "numpy": "numpy",
}

_BACKEND_MODULES: dict[str, str] = {
    "numpy": ".backends.numpy",
    "torch": ".backends.torch",
    "jax": ".backends.jax",
}


def detect_backend(*arrays: Any) -> str:
    """
    Return the backend name for the specified arrays.

    The first array belonging to a non-*NumPy* backend (*PyTorch* or *JAX*)
    determines the result; otherwise the *NumPy* backend is used, including for
    Python scalars and sequences.
    """

    for array in arrays:
        if array is None:
            continue

        root = type(array).__module__.partition(".")[0]
        backend = _ROOT_MODULE_BACKENDS.get(root)
        if backend is not None and backend != "numpy":
            return backend

    return "numpy"


def _backend_module(backend: str) -> ModuleType:
    """Return the imported backend submodule for the specified backend."""

    return importlib.import_module(_BACKEND_MODULES[backend], __package__)


def _as_numpy_query(x: Any) -> np.ndarray | None:
    """
    Return the query materialised as a *NumPy* array, or ``None`` when it cannot
    be converted, e.g. a gradient-tracked *PyTorch* tensor or a *JAX* tracer.
    """

    try:
        return np.asarray(x)
    except Exception:  # noqa: BLE001
        return None


def _promote_to_query(result: Any, x: Any) -> Any:
    """Return a *NumPy* ``result`` promoted to the query's array namespace."""

    from colour.utilities import array_namespace, xp_as_array  # noqa: PLC0415

    return xp_as_array(result, xp=array_namespace(x), like=x)


class _BackendDispatcher(ABC):
    """
    Resolve the array backend from the union of the construction data and the
    query, then delegate evaluation to the matching backend implementation.

    The construction data fixes a base backend. When Array API dispatch is
    enabled (:func:`colour.utilities.is_array_api_enabled`) and a *NumPy*-backed
    object is evaluated at a *PyTorch* or *JAX* query, evaluation re-dispatches to
    the query backend so the computation stays in the query namespace and
    preserves automatic differentiation. Backend implementations are built lazily
    from the construction data and cached per backend.

    Subclasses provide :meth:`_build_implementation` and seed the base backend
    with :meth:`_init_dispatch`.
    """

    _backend: str
    _implementations: dict[str, Any]

    def _init_dispatch(self, backend: str, implementation: Any) -> None:
        """Seed the dispatcher with the base backend implementation."""

        self._backend = backend
        self._implementations = {backend: implementation}

    @abstractmethod
    def _build_implementation(self, backend: str) -> Any:
        """Build the implementation for the specified backend."""

    def _resolve_query_backend(self, x: Any) -> str | None:
        """
        Return the query backend requiring cross-namespace handling, else
        ``None`` to evaluate with the base backend.
        """

        from colour.utilities import is_array_api_enabled  # noqa: PLC0415

        if not is_array_api_enabled():
            return None

        query_backend = detect_backend(x)
        if query_backend in ("numpy", self._backend):
            return None

        if self._backend != "numpy":
            error = (
                f'Cannot evaluate a "{self._backend}"-backed '
                f'{type(self).__name__} at a "{query_backend}" query.'
            )
            raise NotImplementedError(error)

        return query_backend

    def _implementation_for(self, backend: str) -> Any:
        """Return the cached or freshly built implementation for a backend."""

        implementation = self._implementations.get(backend)
        if implementation is None:
            implementation = self._build_implementation(backend)
            self._implementations[backend] = implementation

        return implementation

    def _invalidate_cross_backend(self) -> None:
        """
        Drop cached cross-backend implementations, keeping the base implementation
        mutated in place.

        For a dispatcher whose base implementation exposes setters (e.g.
        :class:`colour.Extrapolator`), the base is mutated directly and only the
        cross-backend snapshots are stale.
        """

        self._implementations = {self._backend: self._base_implementation}

    def _rebuild(self) -> None:
        """
        Rebuild the base implementation from the current construction inputs and
        drop any cached cross-backend implementations.

        For a dispatcher whose base implementation has no setter for a mutated
        attribute, reconstruction from the stored inputs is the only way to apply
        the change.
        """

        self._implementations = {
            self._backend: self._build_implementation(self._backend)
        }

    @property
    def _base_implementation(self) -> Any:
        """Return the implementation built from the construction data."""

        return self._implementations[self._backend]

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate at the specified point(s) in the query's namespace."""

        query_backend = self._resolve_query_backend(x)
        if query_backend is None:
            return self._base_implementation(x, *args, **kwargs)

        # A *NumPy*-data object evaluated at a *PyTorch* / *JAX* query. A
        # concrete query is evaluated with the (feature-complete) *NumPy* backend
        # and promoted to the query namespace; a query that cannot be
        # materialised to *NumPy* (gradient-tracked or traced) is evaluated
        # natively on the query backend so automatic differentiation is
        # preserved.
        x_numpy = _as_numpy_query(x)
        if x_numpy is None:
            return self._implementation_for(query_backend)(x, *args, **kwargs)

        result = self._base_implementation(x_numpy, *args, **kwargs)

        return _promote_to_query(result, x)

    @property
    def backend(self) -> str:
        """Getter for the base backend name resolved from the data."""

        return self._backend


class _DispatchingInterpolator(_BackendDispatcher):
    """
    Construct the backend-specialised implementation matching the namespace of
    the input data and delegate evaluation and attribute access to it.

    Subclasses carry no behaviour; their name selects the identically named
    interpolator in the resolved backend submodule.
    """

    def __init__(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:
        self._x_source = x
        self._y_source = y
        self._args = args
        self._kwargs = kwargs

        backend = detect_backend(x, y)
        implementation = getattr(_backend_module(backend), type(self).__name__)(
            x, y, *args, **kwargs
        )
        self._init_dispatch(backend, implementation)

    def _build_implementation(self, backend: str) -> Any:
        return getattr(_backend_module(backend), type(self).__name__)(
            self._x_source, self._y_source, *self._args, **self._kwargs
        )

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the base backend implementation."""

        if name in ("_implementations", "_backend"):
            raise AttributeError(name)

        return getattr(self._base_implementation, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Set an internal attribute or a writable property, rejecting anything
        else.

        A public name without a backing property would otherwise create an
        instance attribute that shadows :meth:`__getattr__` without reaching the
        implementation, leaving a stale value that evaluation never reads.
        """

        if name.startswith("_") or isinstance(
            getattr(type(self), name, None), property
        ):
            super().__setattr__(name, value)

            return

        error = (
            f'Cannot set attribute "{name}" on {type(self).__name__}; it is not a '
            f"writable property."
        )
        raise AttributeError(error)

    def _set_construction_kwarg(self, name: str, value: Any) -> None:
        """Update a construction keyword argument and rebuild."""

        self._kwargs[name] = value
        self._rebuild()

    @property
    def x(self) -> Any:
        """Getter and setter for the independent :math:`x` variable."""

        return self._base_implementation.x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property; rebuilds the implementation."""

        self._x_source = value
        self._rebuild()

    @property
    def y(self) -> Any:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._base_implementation.y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; rebuilds the implementation."""

        self._y_source = value
        self._rebuild()


class LinearInterpolator(_DispatchingInterpolator):
    """
    Perform linear interpolation of a 1-D function.

    Resolve the backend from the input data and delegate to the matching
    backend-specialised implementation, raising for evaluation points outside
    the interpolation range.
    """


class NearestNeighbourInterpolator(_DispatchingInterpolator):
    """
    Perform nearest-neighbour interpolation on discrete data.

    Resolve the backend from the input data and delegate to the matching
    backend-specialised implementation, selecting the closest known data point
    for each query position.
    """


class NullInterpolator(_DispatchingInterpolator):
    """
    Implement 1-D function null interpolation.

    Return the dependent value when the query matches a knot within tolerance,
    else the default value.
    """

    @property
    def absolute_tolerance(self) -> Any:
        """Getter and setter for the absolute matching tolerance."""

        return self._base_implementation.absolute_tolerance

    @absolute_tolerance.setter
    def absolute_tolerance(self, value: Any) -> None:
        """Setter for the **self.absolute_tolerance** property."""

        self._set_construction_kwarg("absolute_tolerance", value)

    @property
    def relative_tolerance(self) -> Any:
        """Getter and setter for the relative matching tolerance."""

        return self._base_implementation.relative_tolerance

    @relative_tolerance.setter
    def relative_tolerance(self, value: Any) -> None:
        """Setter for the **self.relative_tolerance** property."""

        self._set_construction_kwarg("relative_tolerance", value)

    @property
    def default(self) -> Any:
        """Getter and setter for the value returned when no knot matches."""

        return self._base_implementation.default

    @default.setter
    def default(self, value: Any) -> None:
        """Setter for the **self.default** property."""

        self._set_construction_kwarg("default", value)


class SpragueInterpolator(_DispatchingInterpolator):
    """
    Perform fifth-order polynomial interpolation using the *Sprague (1880)*
    method for uniformly spaced data.

    A minimum of 6 data points is required.

    References
    ----------
    :cite:`CIETC1-382005f`, :cite:`Westland2012h`
    """


class CubicSplineInterpolator(_DispatchingInterpolator):
    """
    Perform *not-a-knot* cubic spline interpolation on one-dimensional data.

    Provide smooth interpolation through specified data points using piecewise
    cubic polynomials. *NumPy* arrays are evaluated with *SciPy*, other array
    namespaces use an equivalent native *not-a-knot* cubic spline that preserves
    automatic differentiation graphs.

    On the *NumPy* backend, attributes absent here fall through to the wrapped
    *SciPy* ``interp1d``, exposing its members such as ``fill_value``. ``axis``
    and ``bounds_error`` are available on every backend.
    """


class PchipInterpolator(_DispatchingInterpolator):
    """
    Interpolate a 1-D function using Piecewise Cubic Hermite Interpolating
    Polynomial (PCHIP) interpolation.

    *NumPy* arrays are evaluated with *SciPy*, other array namespaces use an
    equivalent native implementation that preserves automatic differentiation
    graphs.

    On the *NumPy* backend the wrapped *SciPy* ``PchipInterpolator`` methods
    ``derivative``, ``integrate``, ``roots`` and ``solve`` are exposed; they are
    unavailable for other backends.
    """


class KernelInterpolator(_DispatchingInterpolator):
    """
    Perform kernel-based (convolution) interpolation of a 1-D function.

    Reconstruct a continuous signal from discrete samples as the convolution of
    the data with a continuous interpolation kernel.

    References
    ----------
    :cite:`Burger2009b`, :cite:`Wikipedia2005b`
    """

    @property
    def window(self) -> Any:
        """Getter and setter for the interpolation window size."""

        return self._base_implementation.window

    @window.setter
    def window(self, value: Any) -> None:
        """Setter for the **self.window** property; rebuilds the implementation."""

        self._set_construction_kwarg("window", value)

    @property
    def kernel(self) -> Any:
        """Getter and setter for the interpolation kernel callable."""

        return self._base_implementation.kernel

    @kernel.setter
    def kernel(self, value: Any) -> None:
        """Setter for the **self.kernel** property; rebuilds the implementation."""

        self._set_construction_kwarg("kernel", value)

    @property
    def kernel_kwargs(self) -> Any:
        """Getter and setter for the kernel keyword arguments."""

        return self._base_implementation.kernel_kwargs

    @kernel_kwargs.setter
    def kernel_kwargs(self, value: Any) -> None:
        """Setter for the **self.kernel_kwargs** property; rebuilds."""

        self._set_construction_kwarg("kernel_kwargs", value)

    @property
    def padding_kwargs(self) -> Any:
        """Getter and setter for the padding keyword arguments."""

        return self._base_implementation.padding_kwargs

    @padding_kwargs.setter
    def padding_kwargs(self, value: Any) -> None:
        """Setter for the **self.padding_kwargs** property; rebuilds."""

        self._set_construction_kwarg("padding_kwargs", value)
