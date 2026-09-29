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
    _BackendDispatcher,
    _DispatchingInterpolator,
    _backend_module,
    detect_backend,
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


class Extrapolator(_BackendDispatcher):
    """
    Extrapolate 1-D function values beyond a wrapped interpolator's domain.

    Resolve the backend from the union of the wrapped interpolator's data and the
    query (see :class:`colour.algebra.interpolation._dispatch._BackendDispatcher`)
    and delegate to the matching backend-specialised implementation. Two methods
    are supported:

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
        self._method_arg = method
        self._left_arg = left
        self._right_arg = right
        self._args = args
        self._kwargs = kwargs

        backend = detect_backend(
            getattr(interpolator, "x", None), getattr(interpolator, "y", None)
        )
        implementation = _backend_module(backend).Extrapolator(
            interpolator, method, left, right, *args, **kwargs
        )
        self._init_dispatch(backend, implementation)

    def _build_implementation(self, backend: str) -> Any:
        interpolator = self._interpolator_source
        if isinstance(interpolator, _DispatchingInterpolator):
            interpolator = interpolator._implementation_for(backend)  # noqa: SLF001
        elif interpolator is not None:
            error = (
                f'Cannot re-dispatch the extrapolator to the "{backend}" backend: '
                "the wrapped interpolator is not a colour dispatching interpolator."
            )
            raise NotImplementedError(error)

        return _backend_module(backend).Extrapolator(
            interpolator,
            self._method_arg,
            self._left_arg,
            self._right_arg,
            *self._args,
            **self._kwargs,
        )

    def _implementation_for(self, backend: str) -> Any:
        """
        Return the cross-backend implementation, rebuilding it when the wrapped
        interpolator has been mutated in place.

        The base backend reads the wrapped interpolator's data live, so it needs
        no invalidation. A cross-backend implementation instead wraps a snapshot
        of the interpolator's per-backend implementation; the dispatching
        interpolator rebuilds (a new object) on in-place mutation, so a cached
        implementation wrapping a superseded object is stale and rebuilt.
        """

        if backend == self._backend:
            return self._base_implementation

        interpolator = self._interpolator_source
        if isinstance(interpolator, _DispatchingInterpolator):
            current = interpolator._implementation_for(backend)  # noqa: SLF001
            cached = self._implementations.get(backend)
            if cached is None or cached.interpolator is not current:
                cached = self._build_implementation(backend)
                self._implementations[backend] = cached

            return cached

        return super()._implementation_for(backend)

    @property
    def interpolator(self) -> Any:
        """Getter and setter for the wrapped interpolator."""

        return self._base_implementation.interpolator

    @interpolator.setter
    def interpolator(self, value: Any) -> None:
        """Setter for the **self.interpolator** property."""

        self._interpolator_source = value
        self._base_implementation.interpolator = value
        self._invalidate_cross_backend()

    @property
    def method(self) -> Any:
        """Getter and setter for the extrapolation method."""

        return self._base_implementation.method

    @method.setter
    def method(self, value: Any) -> None:
        """Setter for the **self.method** property."""

        self._method_arg = value
        self._base_implementation.method = value
        self._invalidate_cross_backend()

    @property
    def left(self) -> Any:
        """Getter and setter for the left boundary value."""

        return self._base_implementation.left

    @left.setter
    def left(self, value: Any) -> None:
        """Setter for the **self.left** property."""

        self._left_arg = value
        self._base_implementation.left = value
        self._invalidate_cross_backend()

    @property
    def right(self) -> Any:
        """Getter and setter for the right boundary value."""

        return self._base_implementation.right

    @right.setter
    def right(self, value: Any) -> None:
        """Setter for the **self.right** property."""

        self._right_arg = value
        self._base_implementation.right = value
        self._invalidate_cross_backend()
