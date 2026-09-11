"""
NumPy backend interpolators.

Evaluated with *SciPy*, which is the numeric reference for the *NumPy* path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.interpolate

from colour.constants import DTYPE_FLOAT_DEFAULT

from ._core import (
    SPRAGUE_A_COEFFICIENTS,
    SPRAGUE_C_COEFFICIENTS,
    validate_dimensions,
    validate_extrapolation_method,
)
from ._kernels import kernel_lanczos

if TYPE_CHECKING:
    from collections.abc import Callable

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "LinearInterpolator",
    "NearestNeighbourInterpolator",
    "NullInterpolator",
    "SpragueInterpolator",
    "CubicSplineInterpolator",
    "PchipInterpolator",
    "KernelInterpolator",
    "Extrapolator",
]


def _validate_range(x: np.ndarray, x_i: np.ndarray) -> None:
    """Raise if any query value lies outside the interpolation range."""

    if bool(np.any(x < x_i[0])) or bool(np.any(x > x_i[-1])):
        error = "A value in x is outside the interpolation range."
        raise ValueError(error)


class CubicSplineInterpolator:
    """
    *Not-a-knot* cubic spline over *NumPy* inputs, backed by
    :class:`scipy.interpolate.interp1d` with ``kind="cubic"``.
    """

    def __init__(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["kind"] = "cubic"
        kwargs.setdefault("axis", 0)

        self._x = np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)
        self._y = np.asarray(y, dtype=DTYPE_FLOAT_DEFAULT)
        self._args = args
        self._kwargs = kwargs
        self._build()

    def _build(self) -> None:
        self._interpolator = scipy.interpolate.interp1d(
            self._x, self._y, *self._args, **self._kwargs
        )

    def __call__(self, x: Any) -> np.ndarray:
        """Evaluate the cubic spline at the specified point(s)."""

        # SciPy evaluates in float64; honour the configured default precision.
        return self._interpolator(np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)).astype(
            DTYPE_FLOAT_DEFAULT
        )

    @property
    def x(self) -> np.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> np.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; rebuilds the spline."""

        self._y = np.asarray(value, dtype=DTYPE_FLOAT_DEFAULT)
        self._build()


class PchipInterpolator:
    """
    PCHIP interpolant over *NumPy* inputs, backed by
    :class:`scipy.interpolate.PchipInterpolator`.
    """

    def __init__(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:
        self._x = np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)
        self._y = np.asarray(y, dtype=DTYPE_FLOAT_DEFAULT)
        self._args = args
        self._kwargs = kwargs
        self._build()

    def _build(self) -> None:
        self._interpolator = scipy.interpolate.PchipInterpolator(
            self._x, self._y, *self._args, **self._kwargs
        )

    def __call__(self, x: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        """Evaluate the piecewise cubic interpolant at the specified point(s)."""

        # SciPy evaluates in float64; honour the configured default precision.
        return self._interpolator(
            np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT), *args, **kwargs
        ).astype(DTYPE_FLOAT_DEFAULT)

    @property
    def x(self) -> np.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> np.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; rebuilds the interpolant."""

        self._y = np.asarray(value, dtype=DTYPE_FLOAT_DEFAULT)
        self._build()


class LinearInterpolator:
    """Linear interpolant over *NumPy* inputs; raises outside the range."""

    def __init__(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self._x = np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)
        self._y = np.asarray(y, dtype=DTYPE_FLOAT_DEFAULT)
        validate_dimensions(self._x, self._y)

    def __call__(self, x: Any) -> np.ndarray:
        """Evaluate the linear interpolant at the specified point(s)."""

        x = np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)
        _validate_range(x, self._x)

        if self._y.ndim == 1:
            # ``numpy.interp`` always returns float64; honour the default.
            return np.interp(x, self._x, self._y).astype(DTYPE_FLOAT_DEFAULT)

        i = np.clip(np.searchsorted(self._x, x) - 1, 0, len(self._x) - 2)
        t = (x - self._x[i]) / (self._x[i + 1] - self._x[i])

        return self._y[i] + (self._y[i + 1] - self._y[i]) * t[..., None]

    @property
    def x(self) -> np.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> np.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property."""

        self._y = np.asarray(value, dtype=DTYPE_FLOAT_DEFAULT)


class NearestNeighbourInterpolator:
    """Nearest-neighbour interpolant over *NumPy* inputs."""

    def __init__(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self._x = np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)
        self._y = np.asarray(y, dtype=DTYPE_FLOAT_DEFAULT)

    def __call__(self, x: Any) -> np.ndarray:
        """Evaluate the nearest-neighbour interpolant at the specified point(s)."""

        x = np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)
        _validate_range(x, self._x)

        right = np.clip(np.searchsorted(self._x, x), 0, len(self._x) - 1)
        left = np.clip(right - 1, 0, len(self._x) - 1)
        choose_left = np.abs(x - self._x[left]) <= np.abs(self._x[right] - x)

        return self._y[np.where(choose_left, left, right)]

    @property
    def x(self) -> np.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> np.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property."""

        self._y = np.asarray(value, dtype=DTYPE_FLOAT_DEFAULT)


class NullInterpolator:
    """
    Null interpolant over *NumPy* inputs: return the dependent value when the
    query matches a knot within tolerance, else the default value.
    """

    def __init__(
        self,
        x: Any,
        y: Any,
        absolute_tolerance: float = 1e-8,
        relative_tolerance: float = 1e-8,
        default: float = float("nan"),
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        self._x = np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)
        self._y = np.asarray(y, dtype=DTYPE_FLOAT_DEFAULT)
        validate_dimensions(self._x, self._y)
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.default = default

    def __call__(self, x: Any) -> np.ndarray:
        """Evaluate the null interpolant at the specified point(s)."""

        x = np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)
        _validate_range(x, self._x)

        right = np.clip(np.searchsorted(self._x, x), 0, len(self._x) - 1)
        left = np.clip(right - 1, 0, len(self._x) - 1)
        distance_left = np.abs(x - self._x[left])
        distance_right = np.abs(self._x[right] - x)
        choose_left = distance_left <= distance_right
        nearest = np.where(choose_left, left, right)
        distance = np.where(choose_left, distance_left, distance_right)

        tolerance = self.absolute_tolerance + self.relative_tolerance * np.abs(
            self._x[nearest]
        )
        matched = distance <= tolerance
        if self._y.ndim > 1:
            matched = matched[..., None]

        return np.where(matched, self._y[nearest], self.default)

    @property
    def x(self) -> np.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> np.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property."""

        self._y = np.asarray(value, dtype=DTYPE_FLOAT_DEFAULT)


class SpragueInterpolator:
    """Fifth-order *Sprague (1880)* interpolant over uniformly spaced *NumPy* data."""

    def __init__(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self._x = np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)
        self._y = np.asarray(y, dtype=DTYPE_FLOAT_DEFAULT)
        validate_dimensions(self._x, self._y)

        if len(self._y) < 6:
            error = "Sprague interpolation requires at least 6 points."
            raise ValueError(error)

        self._build()

    def _build(self) -> None:
        x = self._x
        y = self._y
        h = (x[-1] - x[0]) / (len(x) - 1)
        self._x_p = np.concatenate(
            [[x[0] - 2 * h, x[0] - h], x, [x[-1] + h, x[-1] + 2 * h]]
        )

        coefficients = np.asarray(SPRAGUE_C_COEFFICIENTS, dtype=DTYPE_FLOAT_DEFAULT)
        if y.ndim == 2:
            coefficients = coefficients[..., None]
        y_boundary = (
            np.sum(coefficients * np.stack([y[0:6], y[0:6], y[-6:], y[-6:]]), axis=1)
            / 209
        )
        self._y_p = np.concatenate([y_boundary[:2], y, y_boundary[2:]], axis=0)

    def __call__(self, x: Any) -> np.ndarray:
        """Evaluate the *Sprague (1880)* interpolant at the specified point(s)."""

        x = np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)
        _validate_range(np.atleast_1d(x), self._x)

        x_flat = np.atleast_1d(x).reshape(-1)
        i = np.searchsorted(self._x_p, x_flat) - 1
        t = (x_flat - self._x_p[i]) / (self._x_p[i + 1] - self._x_p[i])

        windows = np.stack([self._y_p[i + k] for k in (-2, -1, 0, 1, 2, 3)])
        weights = np.asarray(SPRAGUE_A_COEFFICIENTS, dtype=DTYPE_FLOAT_DEFAULT)
        windows_shape = windows.shape
        a = (weights @ windows.reshape(windows_shape[0], -1) / 24).reshape(
            weights.shape[0], *windows_shape[1:]
        )

        # An integer exponent array would promote ``t`` to float64.
        basis = t ** np.array([[1], [2], [3], [4], [5]], dtype=DTYPE_FLOAT_DEFAULT)
        if self._y.ndim == 2:
            basis = basis[..., None]
        values = self._y_p[i] + (a * basis).sum(axis=0)

        return values.reshape((*np.shape(x), *self._y.shape[1:]))

    @property
    def x(self) -> np.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> np.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; rebuilds the boundary extension."""

        self._y = np.asarray(value, dtype=DTYPE_FLOAT_DEFAULT)
        self._build()


class KernelInterpolator:
    """
    Kernel-based (convolution) interpolant over *NumPy* inputs.

    Reconstruct a continuous signal from discrete samples as the convolution of
    the data with a continuous interpolation kernel. Uniform ``x`` spacing is
    assumed.
    """

    def __init__(
        self,
        x: Any,
        y: Any,
        *,
        window: float = 3,
        kernel: Callable = kernel_lanczos,
        kernel_kwargs: dict | None = None,
        padding_kwargs: dict | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        self._window = window
        self._kernel = kernel
        self._kernel_kwargs = {} if kernel_kwargs is None else dict(kernel_kwargs)
        self._padding_kwargs = (
            {"pad_width": (window, window), "mode": "reflect"}
            if padding_kwargs is None
            else dict(padding_kwargs)
        )

        self._x = np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)
        self._y = np.asarray(y, dtype=DTYPE_FLOAT_DEFAULT)

        if self._x.ndim != 1:
            error = '"x" independent variable must have exactly one dimension!'
            raise ValueError(error)
        if len(self._x) != len(self._y):
            error = (
                '"x" independent and "y" dependent variables have different '
                f'dimensions: "{len(self._x)}", "{len(self._y)}"'
            )
            raise ValueError(error)

        self._build()

    def _build(self) -> None:
        w = self._window
        interval = float(self._x[1] - self._x[0])
        self._interval = interval
        self._x_p = np.pad(
            self._x,
            (int(w), int(w)),
            "linear_ramp",
            end_values=(
                float(self._x.min()) - w * interval,
                float(self._x.max()) + w * interval,
            ),
        )

        pad_kwargs = dict(self._padding_kwargs)
        if self._y.ndim == 2:
            pad_width = pad_kwargs["pad_width"]
            pad_kwargs["pad_width"] = (pad_width, (0, 0))
        self._y_p = np.pad(self._y, **pad_kwargs)

    def __call__(self, x: Any) -> np.ndarray:
        """Evaluate the interpolator at the specified point(s)."""

        x = np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)
        values = self._evaluate(np.atleast_1d(x))

        return values[0] if x.ndim == 0 else values

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the interpolating convolution at the specified point(s)."""

        if bool(np.any(x < self._x[0])) or bool(np.any(x > self._x[-1])):
            error = f'"{x}" is outside the interpolation range.'
            raise ValueError(error)

        interval = self._interval
        x_p_min = float(self._x_p.min())
        clip_l = x_p_min / interval
        clip_h = float(self._x_p.max()) / interval

        windows = np.floor(x / interval)[..., None] + np.arange(
            -self._window + 1, self._window + 1
        )
        windows = np.clip(windows, clip_l, clip_h) - clip_l
        windows = np.round(windows).astype(int)

        weights = self._kernel(
            x[..., None] / interval - windows - x_p_min / interval,
            **self._kernel_kwargs,
        )
        if self._y.ndim == 2:
            weights = weights[..., None]

        return np.sum(self._y_p[windows] * weights, axis=1)

    @property
    def x(self) -> np.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> np.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; rebuilds the padded values."""

        self._y = np.asarray(value, dtype=DTYPE_FLOAT_DEFAULT)
        self._build()

    @property
    def window(self) -> float:
        """Getter for the interpolation window size."""

        return self._window

    @property
    def kernel(self) -> Callable:
        """Getter for the interpolation kernel callable."""

        return self._kernel

    @property
    def kernel_kwargs(self) -> dict:
        """Getter for the kernel keyword arguments."""

        return self._kernel_kwargs

    @property
    def padding_kwargs(self) -> dict:
        """Getter for the padding keyword arguments."""

        return self._padding_kwargs


class Extrapolator:
    """
    Extrapolate a wrapped interpolator beyond its domain over *NumPy* inputs.

    ``"Linear"`` extends using the boundary-pair slope; ``"Constant"`` holds the
    boundary value. ``left`` / ``right`` override the method outside the domain.
    """

    def __init__(
        self,
        interpolator: Any = None,
        method: str = "Linear",
        left: float | None = None,
        right: float | None = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        if interpolator is None:
            interpolator = NullInterpolator(
                np.array([-np.inf, np.inf]), np.array([-np.inf, np.inf])
            )
        self._interpolator = interpolator
        self.method = method
        self._left = left
        self._right = right

    @property
    def interpolator(self) -> Any:
        """Getter and setter for the wrapped interpolator."""

        return self._interpolator

    @interpolator.setter
    def interpolator(self, value: Any) -> None:
        """Setter for the **self.interpolator** property."""

        self._interpolator = value

    @property
    def method(self) -> str:
        """Getter and setter for the extrapolation method."""

        return self._method

    @method.setter
    def method(self, value: Any) -> None:
        """Setter for the **self.method** property."""

        self._method = validate_extrapolation_method(value)

    @property
    def left(self) -> float | None:
        """Getter and setter for the left boundary value (``x < xi[0]``)."""

        return self._left

    @left.setter
    def left(self, value: float | None) -> None:
        """Setter for the **self.left** property."""

        self._left = value

    @property
    def right(self) -> float | None:
        """Getter and setter for the right boundary value (``x > xi[-1]``)."""

        return self._right

    @right.setter
    def right(self, value: float | None) -> None:
        """Setter for the **self.right** property."""

        self._right = value

    def __call__(self, x: Any) -> np.ndarray:
        """Evaluate the extrapolator at the specified point(s)."""

        x = np.asarray(x, dtype=DTYPE_FLOAT_DEFAULT)

        return self._evaluate(x)

    def _evaluate(self, x: np.ndarray) -> np.ndarray:
        """Perform the extrapolating evaluation at the specified point(s)."""

        xi = np.asarray(self._interpolator.x, dtype=DTYPE_FLOAT_DEFAULT)
        yi = np.asarray(self._interpolator.y, dtype=DTYPE_FLOAT_DEFAULT)

        input_rank = yi.ndim
        if input_rank == 1:
            yi = yi[..., None]

        below = x < xi[0]
        above = x > xi[-1]
        in_range = np.logical_and(x >= xi[0], x <= xi[-1])

        y = np.zeros_like(x[..., None] + yi[0])
        below_b = below[..., None]
        above_b = above[..., None]
        x_offset_low = (x - xi[0])[..., None]
        x_offset_high = (x - xi[-1])[..., None]

        if self._method == "linear":
            y = np.where(
                below_b,
                yi[0] + x_offset_low * _safe_divide(yi[1] - yi[0], xi[1] - xi[0]),
                y,
            )
            y = np.where(
                above_b,
                yi[-1] + x_offset_high * _safe_divide(yi[-1] - yi[-2], xi[-1] - xi[-2]),
                y,
            )
        elif self._method == "constant":
            y = np.where(below_b, yi[0], y)
            y = np.where(above_b, yi[-1], y)

        if self._left is not None:
            y = np.where(below_b, self._left, y)
        if self._right is not None:
            y = np.where(above_b, self._right, y)

        if bool(np.any(in_range)):
            x_ravel = np.reshape(x, (-1,))
            in_range_ravel = np.reshape(in_range, (-1,))
            y_ravel = np.reshape(y, (-1, yi.shape[1]))

            interpolated = np.atleast_1d(self._interpolator(x_ravel[in_range_ravel]))
            if interpolated.ndim == 1:
                interpolated = interpolated[..., None]

            dense_idx = np.cumsum(in_range_ravel.astype(int)) - 1
            safe_idx = np.clip(dense_idx, 0, interpolated.shape[0] - 1)
            y_ravel = np.where(
                in_range_ravel[..., None], interpolated[safe_idx], y_ravel
            )
            y = np.reshape(y_ravel, (*x.shape, yi.shape[1]))

        if input_rank == 1:
            y = y[..., 0]

        return y[()] if x.ndim == 0 else y


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Divide ``a`` by ``b``, returning 0 where ``b`` is 0 (degenerate knots)."""

    return np.where(b == 0, np.zeros_like(a + b), a / np.where(b == 0, 1, b))
