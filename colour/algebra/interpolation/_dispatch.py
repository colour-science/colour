"""
Backend detection and dispatching interpolators.

The dispatching interpolator inspect the array namespace of their input data
and delegate to the matching backend submodule
(``.backends.<backend>``), which provides an identically named,
backend-specialised interpolator.
"""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
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


def _resolve_backend(*arrays: Any) -> str:
    """
    Return the single backend selected for the specified data, gated on Array
    API dispatch.

    With dispatch disabled the *NumPy* backend is used for any data, matching
    :func:`colour.utilities.array_namespace`. Enabled, the backend is the sole
    non-*NumPy* namespace present (:func:`detect_backend`), or *NumPy* when none
    is. Data spanning two different non-*NumPy* namespaces (e.g. *PyTorch* and
    *JAX*) has no single backend and raises.
    """

    from colour.utilities import is_array_api_enabled  # noqa: PLC0415

    if not is_array_api_enabled():
        return "numpy"

    frameworks = {
        backend
        for array in arrays
        if array is not None and (backend := detect_backend(array)) != "numpy"
    }
    if len(frameworks) > 1:
        error = (
            "Cannot resolve a single backend for data spanning multiple array "
            f"namespaces: {sorted(frameworks)}."
        )
        raise ValueError(error)

    return frameworks.pop() if frameworks else "numpy"


class _BackendDispatcher(ABC):
    """
    Hold a single backend implementation and forward evaluation, attribute
    reads and writes to it.

    The backend is resolved from the construction data (:func:`detect_backend`)
    and delegated to the matching ``.backends.<backend>`` implementation. A
    query is passed straight to that implementation, which coerces it into its
    own namespace or raises; the backend is never resolved from the query.

    Subclasses provide :meth:`_construct` and seed the implementation with
    :meth:`_init_dispatch`. A write to a forwarded option is applied in place
    and recorded in :attr:`_overrides`, so a rebuild on another backend replays
    it.
    """

    _backend: str
    _impl: Any
    _overrides: dict[str, Any]

    def _init_dispatch(self, backend: str, implementation: Any) -> None:
        """Seed the dispatcher with the backend implementation."""

        self._backend = backend
        self._impl = implementation

    @abstractmethod
    def _construct(self, backend: str) -> Any:
        """Build a fresh implementation for the backend from construction inputs."""

    def _build_implementation(self, backend: str) -> Any:
        """Build an implementation and replay the recorded writes onto it."""

        implementation = self._construct(backend)
        for name, value in self._overrides.items():
            setattr(implementation, name, value)

        return implementation

    def _apply_in_place(self, mutate: Callable[[], None]) -> None:
        """
        Apply an in-place mutation to the live implementation.

        A mutation that raises rebuilds a clean implementation from the current
        inputs, so a backend setter that partially mutated before raising leaves
        no half-applied state behind, then re-raises.
        """

        try:
            mutate()
        except Exception:
            self._impl = self._build_implementation(self._backend)

            raise

    def _set_forwarded(self, name: str, value: Any) -> None:
        """
        Apply a write in place to the implementation and record it for replay.

        A write the backend rejects leaves no half-applied state and is not
        recorded (see :meth:`_apply_in_place`), then re-raises.
        """

        self._apply_in_place(lambda: setattr(self._impl, name, value))
        self._overrides[name] = value

    def _reassign(
        self, backend: str, source_attr: str, value: Any, mutate: Callable[[], None]
    ) -> None:
        """
        Reassign a data source, switching backend when it changed.

        Same backend: apply ``mutate`` to the live implementation (which a
        wrapper may hold by reference), then store ``value`` in ``source_attr``.
        Changed backend: store ``value``, build a fresh implementation (restoring
        the previous source if the build fails) and swap it in. The stored source
        is kept consistent with the live implementation throughout.
        """

        previous = getattr(self, source_attr)
        if backend == self._backend:
            self._apply_in_place(mutate)
            setattr(self, source_attr, value)

            return

        setattr(self, source_attr, value)
        try:
            implementation = self._build_implementation(backend)
        except Exception:
            setattr(self, source_attr, previous)

            raise

        self._backend = backend
        self._impl = implementation

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate at the specified point(s), delegating to the backend."""

        return self._impl(x, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown public attributes to the backend implementation."""

        if name.startswith("_"):
            raise AttributeError(name)

        return getattr(self._impl, name)

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

    @property
    def backend(self) -> str:
        """Getter for the backend name resolved from the data."""

        return self._backend


_C = TypeVar("_C", bound=type)


def _forwarded_property(name: str, *, writable: bool) -> property:
    """
    Return a property that reads ``name`` off the backend implementation.

    When ``writable``, the setter records the write and applies it in place
    through :meth:`_BackendDispatcher._set_forwarded`; otherwise the property is
    read-only.
    """

    def getter(self: _BackendDispatcher) -> Any:
        return getattr(self._impl, name)

    if not writable:
        return property(getter, doc=f"Getter for the **self.{name}** property.")

    def setter(self: _BackendDispatcher, value: Any) -> None:
        self._set_forwarded(name, value)

    return property(
        getter, setter, doc=f"Getter and setter for the **self.{name}** property."
    )


def _forward_properties(cls: _C) -> _C:
    """
    Install forwarding properties on ``cls`` mirroring the *NumPy* backend's
    properties for the identically named class.

    The *NumPy* backend is the reference surface: each of its properties becomes
    a forwarding property on the dispatcher, read/write when the backend exposes
    a setter and read-only otherwise, so the API tracks the backend without
    hand-written boilerplate. Properties already defined on the dispatcher (e.g.
    the data accessors ``x`` and ``y``, or the :class:`Extrapolator`
    ``interpolator``) are left untouched, and non-property members (e.g. the
    wrapped *SciPy* ``fill_value``) still reach the caller through
    :meth:`_BackendDispatcher.__getattr__`.
    """

    backend_cls = getattr(_backend_module("numpy"), cls.__name__)
    for name, member in inspect.getmembers(
        backend_cls, lambda member: isinstance(member, property)
    ):
        if not hasattr(cls, name):
            setattr(
                cls, name, _forwarded_property(name, writable=member.fset is not None)
            )

    return cls


class _DispatchingInterpolator(_BackendDispatcher):
    """
    Construct the backend-specialised implementation matching the namespace of
    the input data and delegate evaluation and attribute access to it.

    A write to a forwarded option is applied in place to the implementation and
    recorded in :attr:`_overrides`. Writing ``x`` or ``y`` re-detects the
    backend: an unchanged backend applies the write in place, a changed backend
    rebuilds the implementation on the new backend and replays the recorded
    writes. Subclasses carry no behaviour; their name selects the identically
    named interpolator in the resolved backend submodule and the *NumPy*
    backend's writable surface.
    """

    def __init__(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:
        self._x_source = x
        self._y_source = y
        self._args = args
        self._kwargs = kwargs
        self._overrides = {}

        backend = _resolve_backend(x, y)
        self._init_dispatch(backend, self._construct(backend))

    def _construct(self, backend: str) -> Any:
        """Build a fresh implementation from the construction inputs."""

        return getattr(_backend_module(backend), type(self).__name__)(
            self._x_source, self._y_source, *self._args, **self._kwargs
        )

    def _set_data(self, name: str, value: Any) -> None:
        """
        Assign ``x`` or ``y``, re-detecting the backend from the incoming value
        combined with the retained other axis.

        An unchanged backend applies the write in place, preserving the
        implementation object a wrapper (e.g. an :class:`colour.Extrapolator`)
        may hold. A changed backend rebuilds on the new backend and replays the
        recorded writes.
        """

        x = value if name == "x" else self._x_source
        y = value if name == "y" else self._y_source
        backend = _resolve_backend(x, y)

        self._reassign(
            backend, f"_{name}_source", value, lambda: setattr(self._impl, name, value)
        )

    @property
    def x(self) -> Any:
        """Getter and setter for the independent :math:`x` variable."""

        return self._impl.x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property; may re-detect the backend."""

        self._set_data("x", value)

    @property
    def y(self) -> Any:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._impl.y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; may re-detect the backend."""

        self._set_data("y", value)


@_forward_properties
class LinearInterpolator(_DispatchingInterpolator):
    """
    Perform linear interpolation of a 1-D function.

    Resolve the backend from the input data and delegate to the matching
    backend-specialised implementation, raising for evaluation points outside
    the interpolation range.
    """


@_forward_properties
class NearestNeighbourInterpolator(_DispatchingInterpolator):
    """
    Perform nearest-neighbour interpolation on discrete data.

    Resolve the backend from the input data and delegate to the matching
    backend-specialised implementation, selecting the closest known data point
    for each query position.
    """


@_forward_properties
class NullInterpolator(_DispatchingInterpolator):
    """
    Implement 1-D function null interpolation.

    Return the dependent value when the query matches a knot within tolerance,
    else the default value. ``absolute_tolerance``, ``relative_tolerance`` and
    ``default`` are writable.
    """


@_forward_properties
class SpragueInterpolator(_DispatchingInterpolator):
    """
    Perform fifth-order polynomial interpolation using the *Sprague (1880)*
    method for uniformly spaced data.

    A minimum of 6 data points is required.

    References
    ----------
    :cite:`CIETC1-382005f`, :cite:`Westland2012h`
    """


@_forward_properties
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


@_forward_properties
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


@_forward_properties
class KernelInterpolator(_DispatchingInterpolator):
    """
    Perform kernel-based (convolution) interpolation of a 1-D function.

    Reconstruct a continuous signal from discrete samples as the convolution of
    the data with a continuous interpolation kernel. ``window``, ``kernel``,
    ``kernel_kwargs`` and ``padding_kwargs`` are writable. Only reflect padding
    with a ``pad_width`` equal to ``window`` is supported; other
    ``padding_kwargs`` raise.

    References
    ----------
    :cite:`Burger2009b`, :cite:`Wikipedia2005b`
    """
