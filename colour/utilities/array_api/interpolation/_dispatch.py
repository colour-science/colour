"""
Backend detection and dispatching interpolators.

The dispatching interpolators inspect the array namespace of their input data
at construction and delegate to the matching backend submodule
(``.backends.<backend>``), which provides an identically named,
backend-specialised interpolator.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

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
    "Extrapolator",
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


class _DispatchingInterpolator:
    """
    Construct the backend-specialised implementation matching the namespace of
    the input data and delegate evaluation and attribute access to it.

    Subclasses carry no behaviour; their name selects the identically named
    interpolator in the resolved backend submodule.
    """

    def __init__(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:
        self._backend = detect_backend(x, y)
        implementation = getattr(_backend_module(self._backend), type(self).__name__)
        self._implementation = implementation(x, y, *args, **kwargs)

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the interpolant at the specified point(s)."""

        return self._implementation(x, *args, **kwargs)

    @property
    def backend(self) -> str:
        """Getter for the resolved backend name."""

        return self._backend

    @property
    def x(self) -> Any:
        """Getter for the independent :math:`x` variable."""

        return self._implementation.x

    @property
    def y(self) -> Any:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._implementation.y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property."""

        self._implementation.y = value


class LinearInterpolator(_DispatchingInterpolator):
    """Dispatching linear interpolator (raises outside the interpolation range)."""


class NearestNeighbourInterpolator(_DispatchingInterpolator):
    """Dispatching nearest-neighbour interpolator."""


class NullInterpolator(_DispatchingInterpolator):
    """Dispatching null interpolator (exact matches within tolerance)."""


class SpragueInterpolator(_DispatchingInterpolator):
    """Dispatching *Sprague (1880)* fifth-order interpolator (uniform spacing)."""


class CubicSplineInterpolator(_DispatchingInterpolator):
    """
    Dispatching *not-a-knot* cubic spline interpolator.

    Mirrors the ``colour.algebra.interpolation.CubicSplineInterpolator`` API and
    delegates to the backend-specialised implementation for the input data.
    """


class PchipInterpolator(_DispatchingInterpolator):
    """
    Dispatching PCHIP interpolator.

    Mirrors the ``colour.algebra.interpolation.PchipInterpolator`` API and
    delegates to the backend-specialised implementation for the input data.
    """


class KernelInterpolator(_DispatchingInterpolator):
    """
    Dispatching kernel-based (convolution) interpolator.

    Adds the kernel/window/padding accessors on top of the shared ``x`` / ``y``
    forwarding of :class:`_DispatchingInterpolator`.
    """

    @property
    def window(self) -> Any:
        """Getter for the interpolation window size."""

        return self._implementation.window

    @property
    def kernel(self) -> Any:
        """Getter for the interpolation kernel callable."""

        return self._implementation.kernel

    @property
    def kernel_kwargs(self) -> Any:
        """Getter for the kernel keyword arguments."""

        return self._implementation.kernel_kwargs

    @property
    def padding_kwargs(self) -> Any:
        """Getter for the padding keyword arguments."""

        return self._implementation.padding_kwargs


class Extrapolator:
    """
    Dispatching extrapolator wrapping an interpolator.

    Unlike the interpolators, the constructor takes an ``interpolator`` (not
    ``x`` / ``y``); the backend is resolved from the wrapped interpolator's data
    and evaluation is delegated to the matching backend implementation.
    """

    def __init__(self, interpolator: Any = None, *args: Any, **kwargs: Any) -> None:
        self._backend = detect_backend(
            getattr(interpolator, "x", None), getattr(interpolator, "y", None)
        )
        implementation = _backend_module(self._backend).Extrapolator
        self._implementation = implementation(interpolator, *args, **kwargs)

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the extrapolator at the specified point(s)."""

        return self._implementation(x, *args, **kwargs)

    @property
    def backend(self) -> str:
        """Getter for the resolved backend name."""

        return self._backend

    @property
    def interpolator(self) -> Any:
        """Getter for the wrapped interpolator."""

        return self._implementation.interpolator

    @property
    def method(self) -> Any:
        """Getter for the extrapolation method."""

        return self._implementation.method

    @property
    def left(self) -> Any:
        """Getter for the left boundary value."""

        return self._implementation.left

    @property
    def right(self) -> Any:
        """Getter for the right boundary value."""

        return self._implementation.right
