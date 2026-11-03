"""
Extrapolation
=============

Define the :class:`colour.Extrapolator` class for extending 1-D function values
beyond their original interpolation range.

References
----------
-   :cite:`Sastanina` : sastanin. (n.d.). How to make scipy.interpolate give an
    extrapolated result beyond the input range? Retrieved August 8, 2014, from
    http://stackoverflow.com/a/2745496/931625
-   :cite:`Westland2012i` : Westland, S., Ripamonti, C., & Cheung, V. (2012).
    Extrapolation Methods. In Computational Colour Science Using MATLAB (2nd
    ed., p. 38). ISBN:978-0-470-66569-5
"""

from __future__ import annotations

from typing import Any

import numpy as np  # noqa: F401  (used by the doctests)

from colour.algebra.interpolation._dispatch import (
    _backend_module,
    _BackendDispatcher,
    _DispatchingInterpolator,
    _forward_properties,
    _resolve_backend,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Extrapolator",
]


@_forward_properties
class Extrapolator(_BackendDispatcher):
    """
    Extrapolate 1-D function values beyond a wrapped interpolator's domain.

    Resolve the backend from the wrapped interpolator's data (see
    :class:`colour.algebra.interpolation._dispatch._BackendDispatcher`) and
    delegate to the matching backend-specialised implementation. The query is
    forwarded to that implementation unchanged. Two methods are supported:

    -   *Linear*: extend using the boundary-pair slope, ``(xi[0], xi[1])`` for
        ``x < xi[0]`` and ``(xi[-1], xi[-2])`` for ``x > xi[-1]``.
    -   *Constant*: assign the boundary value ``yi[0]`` / ``yi[-1]``.

    ``left`` / ``right`` override the method for points outside the domain. The
    wrapped interpolator must expose ``x`` and ``y``.

    Parameters
    ----------
    interpolator
        Interpolator object.
    method
        Extrapolation method, ``"Linear"`` or ``"Constant"``.
    left
        Value to return for ``x < xi[0]``.
    right
        Value to return for ``x > xi[-1]``.

    References
    ----------
    :cite:`Sastanina`, :cite:`Westland2012i`

    Examples
    --------
    Extrapolating a single numeric variable:

    >>> from colour.algebra import LinearInterpolator
    >>> x = np.array([3, 4, 5])
    >>> y = np.array([1, 2, 3])
    >>> extrapolator = Extrapolator(LinearInterpolator(x, y))
    >>> extrapolator(1)
    np.float64(-1.0)

    Extrapolating an `ArrayLike` variable:

    >>> extrapolator(np.array([6, 7, 8]))
    array([4., 5., 6.])

    Using the *Constant* extrapolation method:

    >>> extrapolator = Extrapolator(LinearInterpolator(x, y), method="Constant")
    >>> extrapolator(np.array([0.1, 0.2, 8, 9]))
    array([1., 1., 3., 3.])

    Using a defined *left* boundary and the *Constant* method:

    >>> extrapolator = Extrapolator(LinearInterpolator(x, y), method="Constant", left=0)
    >>> extrapolator(np.array([0.1, 0.2, 8, 9]))
    array([0., 0., 3., 3.])
    """

    def __init__(
        self,
        interpolator: Any = None,
        method: str = "Linear",
        left: float | None = None,
        right: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._interpolator_source = interpolator
        self._method = method
        self._left = left
        self._right = right
        self._args = args
        self._kwargs = kwargs
        self._overrides = {}

        backend = _resolve_backend(
            getattr(interpolator, "x", None), getattr(interpolator, "y", None)
        )
        self._init_dispatch(backend, self._construct(backend))

    def _resolve_wrapped(self, interpolator: Any) -> Any:
        """
        Return the implementation the backend extrapolator wraps.

        A dispatching interpolator yields its backend implementation so the
        native extrapolator wraps a native interpolator. A plain backend
        interpolator (or ``None``) passes through unchanged.
        """

        if isinstance(interpolator, _DispatchingInterpolator):
            return interpolator._impl  # noqa: SLF001

        return interpolator

    def _construct(self, backend: str) -> Any:
        """Build a fresh implementation from the construction inputs."""

        return _backend_module(backend).Extrapolator(
            self._resolve_wrapped(self._interpolator_source),
            self._method,
            self._left,
            self._right,
            *self._args,
            **self._kwargs,
        )

    def _resolved_backend(self) -> str:
        """Return the backend of the wrapped interpolator's current data."""

        interpolator = self._interpolator_source

        return _resolve_backend(
            getattr(interpolator, "x", None), getattr(interpolator, "y", None)
        )

    def _sync_backend(self) -> None:
        """
        Rebuild the implementation when the wrapped interpolator's backend has
        changed.

        The wrapped interpolator may be mutated in place after being wrapped
        (e.g. ``extrapolator.interpolator.y = <tensor>``), switching its backend
        without passing through a setter here. Re-resolving on evaluation keeps
        the extrapolator on the wrapped interpolator's current backend.
        """

        backend = self._resolved_backend()
        if backend != self._backend:
            self._backend = backend
            self._impl = self._build_implementation(backend)

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate at the specified point(s), tracking the wrapped backend."""

        self._sync_backend()

        return self._impl(x, *args, **kwargs)

    @property
    def interpolator(self) -> Any:
        """Getter and setter for the wrapped interpolator."""

        return self._interpolator_source

    @interpolator.setter
    def interpolator(self, value: Any) -> None:
        """Setter for the **self.interpolator** property; may re-detect backend."""

        backend = _resolve_backend(getattr(value, "x", None), getattr(value, "y", None))

        self._reassign(
            backend,
            "_interpolator_source",
            value,
            lambda: setattr(self._impl, "interpolator", self._resolve_wrapped(value)),
        )
