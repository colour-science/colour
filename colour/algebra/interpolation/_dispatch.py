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


def _match_query_namespace(result: Any, x: Any, backend: str) -> Any:
    """
    Return the evaluation ``result`` in the query's array namespace.

    The backend is resolved from the interpolator data at construction, so a
    *NumPy*-backed interpolator (NumPy data) evaluated at a *PyTorch* or *JAX*
    query returns a *NumPy* result. Promote it to the query namespace so the
    output follows the union of the data and query namespaces, as expected when
    e.g. sampling a *NumPy* colour matching function at a backend wavelength.
    """

    if backend != "numpy" or detect_backend(x) == "numpy":
        return result

    from colour.utilities import array_namespace, xp_as_array  # noqa: PLC0415

    return xp_as_array(result, xp=array_namespace(x), like=x)


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

        result = self._implementation(x, *args, **kwargs)

        return _match_query_namespace(result, x, self._backend)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the backend implementation."""

        if name == "_implementation":
            raise AttributeError(name)

        return getattr(self._implementation, name)

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
    """


class PchipInterpolator(_DispatchingInterpolator):
    """
    Interpolate a 1-D function using Piecewise Cubic Hermite Interpolating
    Polynomial (PCHIP) interpolation.

    *NumPy* arrays are evaluated with *SciPy*, other array namespaces use an
    equivalent native implementation that preserves automatic differentiation
    graphs.
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
