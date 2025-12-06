"""
Extrapolation
=============

Define classes for extrapolating one-dimensional functions beyond their
original domain.

-   :class:`colour.Extrapolator`: Extrapolate 1-D functions using various
    methods to extend function values beyond the original interpolation range.

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

import typing

import numpy as np

from colour.algebra import NullInterpolator, sdiv, sdiv_mode
from colour.constants import DTYPE_FLOAT_DEFAULT

if typing.TYPE_CHECKING:
    from colour.hints import (
        Any,
        ArrayLike,
        DTypeReal,
        Literal,
        NDArrayFloat,
        ProtocolInterpolator,
        Real,
        Type,
    )

from colour.utilities import (
    as_float,
    as_float_array,
    attest,
    is_numeric,
    optional,
    validate_method,
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
    Extrapolate 1-D function values beyond the specified interpolator's
    domain boundaries.

    The :class:`colour.Extrapolator` class wraps a specified *Colour* or
    *scipy* interpolator instance with compatible signature to provide
    controlled extrapolation behaviour. Two extrapolation methods are
    supported:

    -   *Linear*: Extrapolate values linearly using the slope defined by
        boundary points (xi[0], xi[1]) for x < xi[0] and (xi[-1], xi[-2])
        for x > xi[-1].
    -   *Constant*: Assign boundary values xi[0] for x < xi[0] and xi[-1]
        for x > xi[-1].

    Specifying *left* and *right* arguments overrides the chosen
    extrapolation method, assigning these values to points outside the
    interpolator's domain.

    Parameters
    ----------
    interpolator
        Interpolator object.
    method
        Extrapolation method.
    left
        Value to return for x < xi[0].
    right
        Value to return for x > xi[-1].
    dtype
        Data type used for internal conversions.

    Methods
    -------
    -   :meth:`~colour.Extrapolator.__init__`
    -   :meth:`~colour.Extrapolator.__class__`

    Notes
    -----
    -   The interpolator must define ``x`` and ``y`` properties.

    References
    ----------
    :cite:`Sastanina`, :cite:`Westland2012i`

    Examples
    --------
    Extrapolating a single numeric variable:

    >>> from colour.algebra import LinearInterpolator
    >>> x = np.array([3, 4, 5])
    >>> y = np.array([1, 2, 3])
    >>> interpolator = LinearInterpolator(x, y)
    >>> extrapolator = Extrapolator(interpolator)
    >>> extrapolator(1)
    -1.0

    Extrapolating an `ArrayLike` variable:

    >>> extrapolator(np.array([6, 7, 8]))
    array([ 4.,  5.,  6.])

    Using the *Constant* extrapolation method:

    >>> x = np.array([3, 4, 5])
    >>> y = np.array([1, 2, 3])
    >>> interpolator = LinearInterpolator(x, y)
    >>> extrapolator = Extrapolator(interpolator, method="Constant")
    >>> extrapolator(np.array([0.1, 0.2, 8, 9]))
    array([ 1.,  1.,  3.,  3.])

    Using defined *left* boundary and *Constant* extrapolation method:

    >>> x = np.array([3, 4, 5])
    >>> y = np.array([1, 2, 3])
    >>> interpolator = LinearInterpolator(x, y)
    >>> extrapolator = Extrapolator(interpolator, method="Constant", left=0)
    >>> extrapolator(np.array([0.1, 0.2, 8, 9]))
    array([ 0.,  0.,  3.,  3.])
    """

    def __init__(
        self,
        interpolator: ProtocolInterpolator | None = None,
        method: Literal["Linear", "Constant"] | str = "Linear",
        left: Real | None = None,
        right: Real | None = None,
        dtype: Type[DTypeReal] | None = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        dtype = optional(dtype, DTYPE_FLOAT_DEFAULT)

        self._interpolator: ProtocolInterpolator = NullInterpolator(
            np.array([-np.inf, np.inf]), np.array([-np.inf, np.inf])
        )
        self.interpolator = optional(interpolator, self._interpolator)
        self._method: Literal["Linear", "Constant"] | str = "Linear"
        self.method = optional(method, self._method)
        self._right: Real | None = None
        self.right = right
        self._left: Real | None = None
        self.left = left

        self._dtype: Type[DTypeReal] = dtype

    @property
    def interpolator(self) -> ProtocolInterpolator:
        """
        Getter and setter for the interpolator.

        The interpolator must implement the interpolator protocol with an
        `x` attribute containing the independent variable data.

        Parameters
        ----------
        value
            Value to set the interpolator instance implementing the required
            protocol with an `x` attribute for wavelength or frequency values
            with.

        Returns
        -------
        ProtocolInterpolator
            Interpolator instance implementing the required protocol with
            an `x` attribute for wavelength or frequency values.
        """

        return self._interpolator

    @interpolator.setter
    def interpolator(self, value: ProtocolInterpolator) -> None:
        """Setter for the **self.interpolator** property."""

        attest(
            hasattr(value, "x"),
            f'"{value}" interpolator has no "x" attribute!',
        )

        attest(
            hasattr(value, "y"),
            f'"{value}" interpolator has no "y" attribute!',
        )

        self._interpolator = value

    @property
    def method(self) -> Literal["Linear", "Constant"] | str:
        """
        Getter and setter for the extrapolation method for the interpolator.

        This property controls the behaviour of the interpolator when
        extrapolating values outside the interpolation domain. The method
        determines how values are computed beyond the specified boundaries.

        Parameters
        ----------
        value
            Value to set the extrapolation method to use, either ``'Linear'``
            for linear extrapolation or ``'Constant'`` for constant value
            extrapolation at the boundaries.

        Returns
        -------
        :class:`str`
            Extrapolation method to use.
        """

        return self._method

    @method.setter
    def method(self, value: Literal["Linear", "Constant"] | str) -> None:
        """Setter for the **self.method** property."""

        attest(
            isinstance(value, str),
            f'"method" property: "{value}" type is not "str"!',
        )

        value = validate_method(value, ("Linear", "Constant"))

        self._method = value

    @property
    def left(self) -> Real | None:
        """
        Getter and setter for the left boundary value.

        Specifies the value to return when evaluating the interpolant at
        points beyond the leftmost data point ( x < xi[0]).

        Parameters
        ----------
        value
            Value to return for x < xi[0] for extrapolation beyond the
            leftmost data point.

        Returns
        -------
        Real or :py:data:`None`
            Value to return for x < xi[0] for extrapolation beyond the
            leftmost data point.
        """

        return self._left

    @left.setter
    def left(self, value: Real | None) -> None:
        """Setter for the **self.left** property."""

        if value is not None:
            attest(
                is_numeric(value),
                f'"left" property: "{value}" is not a "number"!',
            )

            self._left = value

    @property
    def right(self) -> Real | None:
        """
        Getter and setter for the right boundary value.

        Specifies the value to return when evaluating the interpolant at
        points beyond the rightmost data point (x > xi[-1]).

        Parameters
        ----------
        value
            Value to return for x > xi[-1] for extrapolation beyond the
            rightmost data point.

        Returns
        -------
        :class:`numbers.Real` or :py:data:`None`
            Value to return for x > xi[-1] for extrapolation beyond the
            rightmost data point.
        """

        return self._right

    @right.setter
    def right(self, value: Real | None) -> None:
        """Setter for the **self.right** property."""

        if value is not None:
            attest(
                is_numeric(value),
                f'"right" property: "{value}" is not a "number"!',
            )

            self._right = value

    def __call__(self, x: ArrayLike) -> NDArrayFloat:
        """
        Evaluate the extrapolator at specified point(s).

        Parameters
        ----------
        x
            Point(s) to evaluate the extrapolator at.

        Returns
        -------
        :class:`numpy.ndarray`
            Extrapolated point value(s).
        """

        x = as_float_array(x)

        xe = self._evaluate(x)

        return as_float(xe)

    def _evaluate(self, x: NDArrayFloat) -> NDArrayFloat:
        """
        Perform the extrapolating evaluation at specified points.

        Parameters
        ----------
        x
            Points to evaluate the extrapolator at.

        Returns
        -------
        :class:`numpy.ndarray`
            Extrapolated point values.
        """

        xi = self._interpolator.x
        yi = self._interpolator.y

        y = np.empty_like(x)

        if self._method == "linear":
            with sdiv_mode():
                y[x < xi[0]] = yi[0] + (x[x < xi[0]] - xi[0]) * sdiv(
                    yi[1] - yi[0], xi[1] - xi[0]
                )
                y[x > xi[-1]] = yi[-1] + (x[x > xi[-1]] - xi[-1]) * sdiv(
                    yi[-1] - yi[-2], xi[-1] - xi[-2]
                )
        elif self._method == "constant":
            y[x < xi[0]] = yi[0]
            y[x > xi[-1]] = yi[-1]

        if self._left is not None:
            y[x < xi[0]] = self._left
        if self._right is not None:
            y[x > xi[-1]] = self._right

        in_range = np.logical_and(x >= xi[0], x <= xi[-1])
        y[in_range] = self._interpolator(x[in_range])

        return y
