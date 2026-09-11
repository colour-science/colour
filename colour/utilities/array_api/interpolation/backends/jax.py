"""
JAX backend interpolators.

Native *not-a-knot* cubic spline and PCHIP implementations using ``jax.numpy``.
Evaluation is ``jax.jit``-friendly, stays on device (``jnp.searchsorted``), and
preserves automatic differentiation with respect to the dependent values and
the evaluation points. The known knot locations are treated as structural.
"""

# ``jax`` is an optional backend. When it is not installed, static analysers
# resolve this module's ``import jax`` to the module itself and cannot verify
# any ``jax`` attribute; the attribute checks are only meaningful when *JAX* is
# present.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

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


def _validate_range(x: jnp.ndarray, x_i: jnp.ndarray) -> None:
    """Raise if any query value lies outside the interpolation range."""

    if bool(jnp.any(x < x_i[0])) or bool(jnp.any(x > x_i[-1])):
        error = "A value in x is outside the interpolation range."
        raise ValueError(error)


def _as_float(a: Any) -> jnp.ndarray:
    """Return the array promoted to a floating dtype."""

    a = jnp.asarray(a)
    if jnp.issubdtype(a.dtype, jnp.floating):
        return a

    return a.astype(float)


def _axis_to_front(y: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Move the interpolation axis of the values to the front."""

    axis %= y.ndim
    if axis == 0:
        return y

    return jnp.transpose(y, (axis, *range(axis), *range(axis + 1, y.ndim)))


def _restore_axis(y: jnp.ndarray, query_ndim: int, axis: int) -> jnp.ndarray:
    """Restore the interpolation axis placement in evaluated values."""

    if axis == 0 or query_ndim == 0:
        return y

    axes = (
        *range(query_ndim, query_ndim + axis),
        *range(query_ndim),
        *range(query_ndim + axis, y.ndim),
    )

    return jnp.transpose(y, axes)


def _interval_indices(x: jnp.ndarray, x_i: jnp.ndarray) -> jnp.ndarray:
    """Return clipped interpolation interval indices for query values."""

    return jnp.clip(jnp.searchsorted(x_i, x) - 1, 0, x_i.shape[0] - 2)


def _not_a_knot_second_derivatives(h: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Return the *not-a-knot* cubic-spline second derivatives, shape ``(n, k)``.

    The moment system is tridiagonal apart from the two *not-a-knot* boundary
    rows. Eliminating the boundary moments ``M[0]`` and ``M[-1]`` with the
    third-derivative continuity conditions yields a tridiagonal system over the
    interior moments, solved on device with
    :func:`jax.lax.linalg.tridiagonal_solve`. The reduction stays stable for
    uniform spacing, where a direct elimination of the boundary rows would
    produce a zero pivot; the boundary moments are recovered afterwards.
    """

    slope = (y[1:] - y[:-1]) / h[:, None]
    rhs = 6.0 * (slope[1:] - slope[:-1])

    sub = h[:-1]
    main = 2.0 * (h[:-1] + h[1:])
    sup = h[1:]

    main = main.at[0].set(3.0 * h[0] + 2.0 * h[1] + h[0] ** 2 / h[1])
    sup = sup.at[0].set((h[1] ** 2 - h[0] ** 2) / h[1])
    sub = sub.at[0].set(0.0)

    main = main.at[-1].set((h[-2] + h[-1]) * (2.0 * h[-2] + h[-1]) / h[-2])
    sub = sub.at[-1].set((h[-2] ** 2 - h[-1] ** 2) / h[-2])
    sup = sup.at[-1].set(0.0)

    inner = jax.lax.linalg.tridiagonal_solve(sub, main, sup, rhs)

    m_0 = (1.0 + h[0] / h[1]) * inner[0] - (h[0] / h[1]) * inner[1]
    m_n = (1.0 + h[-1] / h[-2]) * inner[-1] - (h[-1] / h[-2]) * inner[-2]

    return jnp.concatenate([m_0[None], inner, m_n[None]], axis=0)


class CubicSplineInterpolator:
    """Native *not-a-knot* cubic spline over *JAX* inputs."""

    def __init__(
        self,
        x: Any,
        y: Any,
        *,
        axis: int = 0,
        bounds_error: bool | None = None,
        fill_value: Any = float("nan"),
        assume_sorted: bool = False,
    ) -> None:
        x = _as_float(x)
        y = _as_float(y)
        self.axis = int(axis % y.ndim)

        if not assume_sorted:
            order = jnp.argsort(x)
            y = jnp.take(y, order, axis=self.axis)
            x = x[order]

        if x.ndim != 1:
            error = "`x` must be 1-dimensional."
            raise ValueError(error)
        if y.shape[self.axis] != x.shape[0]:
            error = "`x` and `y` must match in length along the interpolation axis."
            raise ValueError(error)
        if x.shape[0] < 4:
            error = "Cubic spline interpolation requires at least 4 points."
            raise ValueError(error)
        if bool(jnp.any(jnp.diff(x) <= 0)):
            error = "`x` must be strictly increasing."
            raise ValueError(error)

        self._x = x
        self._y = y

        self._extrapolate = isinstance(fill_value, str) and fill_value == "extrapolate"
        if self._extrapolate and bounds_error:
            error = "Cannot extrapolate and raise at the same time."
            raise ValueError(error)
        self.bounds_error = (
            not self._extrapolate if bounds_error is None else bounds_error
        )
        if isinstance(fill_value, tuple) and len(fill_value) == 2:
            self._fill_value_below, self._fill_value_above = fill_value
        else:
            self._fill_value_below = self._fill_value_above = fill_value

        self._solve()

    def _solve(self) -> None:
        """
        Solve the not-a-knot second-derivative system for the current data.

        The moment system is reduced to a tridiagonal system over the interior
        second derivatives and solved with
        :func:`jax.lax.linalg.tridiagonal_solve`, keeping the whole solve on
        device and ``jit``-compatible.
        """

        values = _axis_to_front(self._y, self.axis)
        self._values_shape = tuple(values.shape)
        self._values = values.reshape(self._values_shape[0], -1)

        h = jnp.diff(self._x)
        self._intervals = h
        self._second_derivatives = _not_a_knot_second_derivatives(h, self._values)

    def __call__(self, x: Any) -> jnp.ndarray:
        """Evaluate the cubic spline at the specified point(s)."""

        x = jnp.asarray(x, dtype=self._values.dtype)
        x_i = self._x
        h = self._intervals
        y_i = self._values
        second = self._second_derivatives

        x_flat = x.reshape(-1)
        indices = _interval_indices(x_flat, x_i)
        h_interval = h[indices]
        a = (x_i[indices + 1] - x_flat) / h_interval
        b = (x_flat - x_i[indices]) / h_interval
        values = (
            a[:, None] * y_i[indices]
            + b[:, None] * y_i[indices + 1]
            + (
                (a**3 - a)[:, None] * second[indices]
                + (b**3 - b)[:, None] * second[indices + 1]
            )
            * (h_interval**2)[:, None]
            / 6
        )
        values = values.reshape(*x.shape, *self._values_shape[1:])
        values = _restore_axis(values, x.ndim, self.axis)

        below = x < x_i[0]
        above = x > x_i[-1]
        if self.bounds_error and bool(jnp.any(below | above)):
            error = "A value in x_new is outside the interpolation range."
            raise ValueError(error)
        if not self._extrapolate:
            remaining_ndim = self._y.ndim - 1
            below = below.reshape(*below.shape, *((1,) * remaining_ndim))
            above = above.reshape(*above.shape, *((1,) * remaining_ndim))
            below = _restore_axis(below, x.ndim, self.axis)
            above = _restore_axis(above, x.ndim, self.axis)
            values = jnp.where(
                below,
                jnp.asarray(self._fill_value_below, dtype=values.dtype),
                values,
            )
            values = jnp.where(
                above,
                jnp.asarray(self._fill_value_above, dtype=values.dtype),
                values,
            )

        return values

    @property
    def x(self) -> jnp.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; re-solves the spline."""

        self._y = _as_float(value)
        self._solve()


class PchipInterpolator:
    """Native PCHIP interpolant over *JAX* inputs."""

    def __init__(
        self,
        x: Any,
        y: Any,
        axis: int = 0,
        extrapolate: bool | None = None,
    ) -> None:
        x = _as_float(x)
        y = _as_float(y)
        self.axis = int(axis % y.ndim)
        self._extrapolate = True if extrapolate is None else extrapolate

        if x.ndim != 1:
            error = "`x` must be 1-dimensional."
            raise ValueError(error)
        if y.shape[self.axis] != x.shape[0]:
            error = "`x` and `y` must match in length along the interpolation axis."
            raise ValueError(error)
        if x.shape[0] < 2:
            error = "PCHIP interpolation requires at least 2 points."
            raise ValueError(error)
        if bool(jnp.any(jnp.diff(x) <= 0)):
            error = "`x` must be strictly increasing."
            raise ValueError(error)

        self._x = x
        self._y = y
        self._solve()

    def _solve(self) -> None:
        """Compute the shape-preserving derivatives for the current data."""

        values = _axis_to_front(self._y, self.axis)
        self._values_shape = tuple(values.shape)
        self._values = values.reshape(self._values_shape[0], -1)

        h = jnp.diff(self._x)
        self._intervals = h
        slopes = (self._values[1:] - self._values[:-1]) / h[:, None]

        if self._x.shape[0] == 2:
            self._derivatives = jnp.stack([slopes[0], slopes[0]])
            return

        previous = slopes[:-1]
        following = slopes[1:]
        use_zero = (
            (previous == 0)
            | (following == 0)
            | (jnp.sign(previous) != jnp.sign(following))
        )
        safe_previous = jnp.where(use_zero, jnp.ones_like(previous), previous)
        safe_following = jnp.where(use_zero, jnp.ones_like(following), following)
        w_1 = 2 * h[1:, None] + h[:-1, None]
        w_2 = h[1:, None] + 2 * h[:-1, None]
        interior = jnp.where(
            use_zero,
            jnp.zeros_like(previous),
            (w_1 + w_2) / (w_1 / safe_previous + w_2 / safe_following),
        )

        def endpoint(
            h_0: jnp.ndarray,
            h_1: jnp.ndarray,
            slope_0: jnp.ndarray,
            slope_1: jnp.ndarray,
        ) -> jnp.ndarray:
            derivative = ((2 * h_0 + h_1) * slope_0 - h_0 * slope_1) / (h_0 + h_1)
            opposite = jnp.sign(derivative) != jnp.sign(slope_0)
            limited = (jnp.sign(slope_0) != jnp.sign(slope_1)) & (
                jnp.abs(derivative) > 3 * jnp.abs(slope_0)
            )
            return jnp.where(
                opposite,
                jnp.zeros_like(derivative),
                jnp.where(limited, 3 * slope_0, derivative),
            )

        self._derivatives = jnp.concatenate(
            [
                endpoint(h[0], h[1], slopes[0], slopes[1])[None, ...],
                interior,
                endpoint(h[-1], h[-2], slopes[-1], slopes[-2])[None, ...],
            ],
            axis=0,
        )

    def __call__(
        self, x: Any, nu: int = 0, extrapolate: bool | None = None
    ) -> jnp.ndarray:
        """Evaluate the piecewise cubic interpolant at the specified point(s)."""

        x = jnp.asarray(x, dtype=self._values.dtype)
        x_i = self._x
        h = self._intervals
        y_i = self._values
        derivatives = self._derivatives

        x_flat = x.reshape(-1)
        indices = _interval_indices(x_flat, x_i)
        h_interval = h[indices, None]
        t = (x_flat - x_i[indices])[:, None] / h_interval

        coefficient_3 = (
            2 * y_i[indices]
            - 2 * y_i[indices + 1]
            + h_interval * (derivatives[indices] + derivatives[indices + 1])
        )
        coefficient_2 = (
            -3 * y_i[indices]
            + 3 * y_i[indices + 1]
            - h_interval * (2 * derivatives[indices] + derivatives[indices + 1])
        )
        coefficient_1 = h_interval * derivatives[indices]

        if nu == 0:
            values = (
                coefficient_3 * t**3
                + coefficient_2 * t**2
                + coefficient_1 * t
                + y_i[indices]
            )
        elif nu == 1:
            values = (
                3 * coefficient_3 * t**2 + 2 * coefficient_2 * t + coefficient_1
            ) / h_interval
        elif nu == 2:
            values = (6 * coefficient_3 * t + 2 * coefficient_2) / h_interval**2
        elif nu == 3:
            values = 6 * coefficient_3 / h_interval**3
        else:
            values = jnp.zeros_like(coefficient_3)

        values = values.reshape(*x.shape, *self._values_shape[1:])
        values = _restore_axis(values, x.ndim, self.axis)

        extrapolate = self._extrapolate if extrapolate is None else extrapolate
        if not extrapolate:
            outside = (x < x_i[0]) | (x > x_i[-1])
            remaining_ndim = self._y.ndim - 1
            outside = outside.reshape(*outside.shape, *((1,) * remaining_ndim))
            outside = _restore_axis(outside, x.ndim, self.axis)
            values = jnp.where(
                outside, jnp.asarray(jnp.nan, dtype=values.dtype), values
            )

        return values

    @property
    def x(self) -> jnp.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; recomputes the derivatives."""

        self._y = _as_float(value)
        self._solve()


class LinearInterpolator:
    """Linear interpolant over *JAX* inputs; raises outside the range."""

    def __init__(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self._x = _as_float(x)
        self._y = _as_float(y)
        validate_dimensions(self._x, self._y)

    def __call__(self, x: Any) -> jnp.ndarray:
        """Evaluate the linear interpolant at the specified point(s)."""

        x = jnp.asarray(x, dtype=self._y.dtype)
        _validate_range(x, self._x)

        i = jnp.clip(jnp.searchsorted(self._x, x) - 1, 0, len(self._x) - 2)
        t = (x - self._x[i]) / (self._x[i + 1] - self._x[i])
        if self._y.ndim > 1:
            t = t[..., None]

        return self._y[i] + (self._y[i + 1] - self._y[i]) * t

    @property
    def x(self) -> jnp.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property."""

        self._y = _as_float(value)


class NearestNeighbourInterpolator:
    """Nearest-neighbour interpolant over *JAX* inputs."""

    def __init__(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self._x = _as_float(x)
        self._y = _as_float(y)

    def __call__(self, x: Any) -> jnp.ndarray:
        """Evaluate the nearest-neighbour interpolant at the specified point(s)."""

        x = jnp.asarray(x, dtype=self._y.dtype)
        _validate_range(x, self._x)

        right = jnp.clip(jnp.searchsorted(self._x, x), 0, len(self._x) - 1)
        left = jnp.clip(right - 1, 0, len(self._x) - 1)
        choose_left = jnp.abs(x - self._x[left]) <= jnp.abs(self._x[right] - x)

        return self._y[jnp.where(choose_left, left, right)]

    @property
    def x(self) -> jnp.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property."""

        self._y = _as_float(value)


class NullInterpolator:
    """Null interpolant over *JAX* inputs (exact matches within tolerance)."""

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
        self._x = _as_float(x)
        self._y = _as_float(y)
        validate_dimensions(self._x, self._y)
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)
        self.default = default

    def __call__(self, x: Any) -> jnp.ndarray:
        """Evaluate the null interpolant at the specified point(s)."""

        x = jnp.asarray(x, dtype=self._y.dtype)
        _validate_range(x, self._x)

        right = jnp.clip(jnp.searchsorted(self._x, x), 0, len(self._x) - 1)
        left = jnp.clip(right - 1, 0, len(self._x) - 1)
        distance_left = jnp.abs(x - self._x[left])
        distance_right = jnp.abs(self._x[right] - x)
        choose_left = distance_left <= distance_right
        nearest = jnp.where(choose_left, left, right)
        distance = jnp.where(choose_left, distance_left, distance_right)

        tolerance = self.absolute_tolerance + self.relative_tolerance * jnp.abs(
            self._x[nearest]
        )
        matched = distance <= tolerance
        if self._y.ndim > 1:
            matched = matched[..., None]

        return jnp.where(
            matched,
            self._y[nearest],
            jnp.asarray(self.default, dtype=self._y.dtype),
        )

    @property
    def x(self) -> jnp.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property."""

        self._y = _as_float(value)


class SpragueInterpolator:
    """Fifth-order *Sprague (1880)* interpolant over uniformly spaced data."""

    def __init__(self, x: Any, y: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        self._x = _as_float(x)
        self._y = _as_float(y)
        validate_dimensions(self._x, self._y)

        if self._y.shape[0] < 6:
            error = "Sprague interpolation requires at least 6 points."
            raise ValueError(error)

        self._solve()

    def _solve(self) -> None:
        x = self._x
        y = self._y
        h = (x[-1] - x[0]) / (x.shape[0] - 1)
        self._x_p = jnp.concatenate(
            [
                jnp.stack([x[0] - 2 * h, x[0] - h]),
                x,
                jnp.stack([x[-1] + h, x[-1] + 2 * h]),
            ]
        )

        coefficients = jnp.asarray(SPRAGUE_C_COEFFICIENTS, dtype=y.dtype)
        if y.ndim == 2:
            coefficients = coefficients[..., None]
        windows = jnp.stack([y[0:6], y[0:6], y[-6:], y[-6:]])
        y_boundary = (coefficients * windows).sum(axis=1) / 209
        self._y_p = jnp.concatenate([y_boundary[:2], y, y_boundary[2:]], axis=0)

    def __call__(self, x: Any) -> jnp.ndarray:
        """Evaluate the *Sprague (1880)* interpolant at the specified point(s)."""

        x = jnp.asarray(x, dtype=self._y.dtype)
        _validate_range(jnp.atleast_1d(x), self._x)

        x_flat = jnp.atleast_1d(x).reshape(-1)
        i = jnp.searchsorted(self._x_p, x_flat) - 1
        t = (x_flat - self._x_p[i]) / (self._x_p[i + 1] - self._x_p[i])

        windows = jnp.stack([self._y_p[i + k] for k in (-2, -1, 0, 1, 2, 3)])
        weights = jnp.asarray(SPRAGUE_A_COEFFICIENTS, dtype=self._y.dtype)
        windows_shape = windows.shape
        a = (weights @ windows.reshape(windows_shape[0], -1) / 24).reshape(
            weights.shape[0], *windows_shape[1:]
        )

        powers = jnp.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=self._y.dtype)
        basis = t**powers
        if self._y.ndim == 2:
            basis = basis[..., None]
        values = self._y_p[i] + (a * basis).sum(axis=0)

        return values.reshape((*x.shape, *self._y.shape[1:]))

    @property
    def x(self) -> jnp.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; rebuilds the boundary extension."""

        self._y = _as_float(value)
        self._solve()


def _reflect_pad(y: jnp.ndarray, window: int) -> jnp.ndarray:
    """
    Reflect-pad the leading axis of ``y`` by ``window`` on each side.

    Reflection excludes the edge sample (``np.pad`` ``"reflect"`` semantics) and
    is expressed as an integer-index gather so the padded values keep their
    autodiff graph with respect to ``y``.
    """

    n = y.shape[0]
    left = jnp.arange(window, 0, -1)
    middle = jnp.arange(n)
    right = jnp.arange(n - 2, n - 2 - window, -1)

    return y[jnp.concatenate([left, middle, right])]


class KernelInterpolator:
    """
    Native kernel-based (convolution) interpolant over *JAX* inputs.

    Evaluation preserves autodiff with respect to the dependent values (through
    the gathered padded values). Uniform ``x`` spacing is assumed. Only
    ``"reflect"`` padding is supported on device.
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
        if self._padding_kwargs.get("mode", "reflect") != "reflect":
            error = 'Only "reflect" padding is supported by the JAX backend.'
            raise ValueError(error)

        self._x = _as_float(x)
        self._y = _as_float(y)

        if self._x.ndim != 1:
            error = '"x" independent variable must have exactly one dimension!'
            raise ValueError(error)
        if self._x.shape[0] != self._y.shape[0]:
            error = (
                '"x" independent and "y" dependent variables have different '
                f'dimensions: "{self._x.shape[0]}", "{self._y.shape[0]}"'
            )
            raise ValueError(error)

        self._build()

    def _build(self) -> None:
        w = int(self._window)
        interval = float(self._x[1] - self._x[0])
        self._interval = interval
        self._x_p_min = float(self._x.min()) - w * interval
        self._x_p_max = float(self._x.max()) + w * interval
        self._y_p = _reflect_pad(self._y, w)

    def __call__(self, x: Any) -> jnp.ndarray:
        """Evaluate the interpolator at the specified point(s)."""

        x = jnp.asarray(x, dtype=self._y.dtype)
        values = self._evaluate(jnp.atleast_1d(x))

        return values[0] if x.ndim == 0 else values

    def _evaluate(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluate the interpolating convolution at the specified point(s)."""

        if bool(jnp.any(x < self._x[0])) or bool(jnp.any(x > self._x[-1])):
            error = "A value in x is outside the interpolation range."
            raise ValueError(error)

        w = int(self._window)
        interval = self._interval
        clip_l = self._x_p_min / interval
        clip_h = self._x_p_max / interval

        offsets = jnp.arange(-w + 1, w + 1, dtype=x.dtype)
        windows = jnp.floor(x / interval)[..., None] + offsets
        windows = jnp.clip(windows, clip_l, clip_h) - clip_l
        windows_i = jnp.round(windows).astype(int)

        weights = self._kernel(
            x[..., None] / interval
            - windows_i.astype(x.dtype)
            - self._x_p_min / interval,
            **self._kernel_kwargs,
        )
        if self._y.ndim == 2:
            weights = weights[..., None]

        return jnp.sum(self._y_p[windows_i] * weights, axis=1)

    @property
    def x(self) -> jnp.ndarray:
        """Getter for the independent :math:`x` variable."""

        return self._x

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; rebuilds the padded values."""

        self._y = _as_float(value)
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


def _safe_divide(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Divide ``a`` by ``b``, returning 0 where ``b`` is 0 (degenerate knots)."""

    b_safe = jnp.where(b == 0, jnp.ones_like(b), b)

    return jnp.where(b == 0, jnp.zeros_like(a + b), a / b_safe)


class Extrapolator:
    """
    Native extrapolator wrapping an interpolator over *JAX* inputs.

    ``"Linear"`` extends using the boundary-pair slope; ``"Constant"`` holds the
    boundary value; ``left`` / ``right`` override the method outside the domain.
    Autodiff is preserved through both the boundary slope and the wrapped
    interpolator's in-range values.
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
                jnp.array([-jnp.inf, jnp.inf]), jnp.array([-jnp.inf, jnp.inf])
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

    def __call__(self, x: Any) -> jnp.ndarray:
        """Evaluate the extrapolator at the specified point(s)."""

        xi = self._interpolator.x
        yi = self._interpolator.y
        x = jnp.asarray(x, dtype=yi.dtype)

        return self._evaluate(x, xi, yi)

    def _evaluate(
        self, x: jnp.ndarray, xi: jnp.ndarray, yi: jnp.ndarray
    ) -> jnp.ndarray:
        """Perform the extrapolating evaluation at the specified point(s)."""

        input_rank = yi.ndim
        if input_rank == 1:
            yi = yi[..., None]

        below = x < xi[0]
        above = x > xi[-1]
        in_range = jnp.logical_and(x >= xi[0], x <= xi[-1])

        y = jnp.zeros_like(x[..., None] + yi[0])
        below_b = below[..., None]
        above_b = above[..., None]
        x_offset_low = (x - xi[0])[..., None]
        x_offset_high = (x - xi[-1])[..., None]

        if self._method == "linear":
            y = jnp.where(
                below_b,
                yi[0] + x_offset_low * _safe_divide(yi[1] - yi[0], xi[1] - xi[0]),
                y,
            )
            y = jnp.where(
                above_b,
                yi[-1] + x_offset_high * _safe_divide(yi[-1] - yi[-2], xi[-1] - xi[-2]),
                y,
            )
        elif self._method == "constant":
            y = jnp.where(below_b, yi[0], y)
            y = jnp.where(above_b, yi[-1], y)

        if self._left is not None:
            y = jnp.where(below_b, jnp.asarray(self._left, dtype=y.dtype), y)
        if self._right is not None:
            y = jnp.where(above_b, jnp.asarray(self._right, dtype=y.dtype), y)

        if bool(jnp.any(in_range)):
            x_ravel = x.reshape(-1)
            in_range_ravel = in_range.reshape(-1)
            y_ravel = y.reshape(-1, yi.shape[1])

            interpolated = jnp.atleast_1d(self._interpolator(x_ravel[in_range_ravel]))
            if interpolated.ndim == 1:
                interpolated = interpolated[..., None]

            dense_idx = jnp.cumsum(in_range_ravel.astype(int)) - 1
            safe_idx = jnp.clip(dense_idx, 0, interpolated.shape[0] - 1)
            y_ravel = jnp.where(
                in_range_ravel[..., None], interpolated[safe_idx], y_ravel
            )
            y = y_ravel.reshape(*x.shape, yi.shape[1])

        if input_rank == 1:
            y = y[..., 0]

        return y
