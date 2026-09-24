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
    _match_query_namespace,
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


class Extrapolator:
    """
    Extrapolate 1-D function values beyond a wrapped interpolator's domain.

    Resolve the backend from the wrapped interpolator's data and delegate to the
    matching backend-specialised implementation. Two methods are supported:

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

    def __init__(self, interpolator: Any = None, *args: Any, **kwargs: Any) -> None:
        self._backend = detect_backend(
            getattr(interpolator, "x", None), getattr(interpolator, "y", None)
        )
        implementation = _backend_module(self._backend).Extrapolator
        self._implementation = implementation(interpolator, *args, **kwargs)

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> Any:
        """Evaluate the extrapolator at the specified point(s)."""

        result = self._implementation(x, *args, **kwargs)

        return _match_query_namespace(result, x, self._backend)

    @property
    def backend(self) -> str:
        """Getter for the resolved backend name."""

        return self._backend

    @property
    def interpolator(self) -> Any:
        """Getter and setter for the wrapped interpolator."""

        return self._implementation.interpolator

    @interpolator.setter
    def interpolator(self, value: Any) -> None:
        """Setter for the **self.interpolator** property."""

        self._implementation.interpolator = value

    @property
    def method(self) -> Any:
        """Getter and setter for the extrapolation method."""

        return self._implementation.method

    @method.setter
    def method(self, value: Any) -> None:
        """Setter for the **self.method** property."""

        self._implementation.method = value

    @property
    def left(self) -> Any:
        """Getter and setter for the left boundary value."""

        return self._implementation.left

    @left.setter
    def left(self, value: Any) -> None:
        """Setter for the **self.left** property."""

        self._implementation.left = value

    @property
    def right(self) -> Any:
        """Getter and setter for the right boundary value."""

        return self._implementation.right

    @right.setter
    def right(self, value: Any) -> None:
        """Setter for the **self.right** property."""

        self._implementation.right = value
