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

import typing

import jax
import jax.numpy as jnp
import numpy as np

from ._core import (
    SPRAGUE_A_COEFFICIENTS,
    SPRAGUE_C_COEFFICIENTS,
    reflect_indices,
    validate_dimensions,
    validate_extrapolation_method,
    validate_padding_kwargs,
)
from ._kernels import kernel_lanczos

if typing.TYPE_CHECKING:
    from colour.hints import Any, Callable, Literal

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


def _any_true(mask: jnp.ndarray) -> bool:
    """
    Return whether any element of ``mask`` is true.

    Return ``False`` for an abstract tracer (under ``jax.jit``), where the query
    values are unknown at trace time and the predicate cannot be evaluated;
    concrete tracers (under ``jax.grad``) are evaluated as usual.
    """

    try:
        return bool(jnp.any(mask))
    except jax.errors.TracerBoolConversionError:
        return False


def _validate_range(x: jnp.ndarray, x_i: jnp.ndarray) -> None:
    """Raise if any query value lies outside the interpolation range."""

    if _any_true((x < x_i[0]) | (x > x_i[-1])):
        error = "A value in x is outside the interpolation range."
        raise ValueError(error)


def _as_float(a: Any, dtype: Any = None) -> jnp.ndarray:
    """
    Return ``a`` as a float array.

    A *NumPy* ``dtype`` casts to it (subject to the *JAX* ``x64`` configuration),
    preserving the autodiff graph. Unspecified, an existing floating dtype is
    kept and an integer array promoted to the default float.
    """

    a = jnp.asarray(a)
    if dtype is not None:
        return a.astype(np.dtype(dtype))

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
        axis: int = -1,
        bounds_error: bool | None = None,
        fill_value: Any = float("nan"),
        assume_sorted: bool = False,
        dtype: Any = None,
    ) -> None:
        x = _as_float(x, dtype)
        y = _as_float(y, dtype)
        if x.ndim != 1:
            error = "`x` must be 1-dimensional."
            raise ValueError(error)
        if y.ndim == 0:
            error = "`y` must have at least one dimension."
            raise ValueError(error)
        if not -y.ndim <= axis < y.ndim:
            error = "Interpolation axis is out of bounds for `y`."
            raise IndexError(error)
        self.axis = int(axis % y.ndim)
        if y.shape[self.axis] != x.shape[0]:
            error = "`x` and `y` must match in length along the interpolation axis."
            raise ValueError(error)

        if not assume_sorted:
            order = jnp.argsort(x)
            y = jnp.take(y, order, axis=self.axis)
            x = x[order]

        if x.shape[0] < 4:
            error = "Cubic spline interpolation requires at least 4 points."
            raise ValueError(error)
        if bool(jnp.any(jnp.diff(x) <= 0)):
            error = "`x` must be strictly increasing."
            raise ValueError(error)

        self._dtype_arg = dtype
        self._assume_sorted = assume_sorted
        self._x = x
        self._y = y

        self._fill_value = fill_value
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

        dtype = jnp.result_type(self._x, self._y)
        self._x = self._x.astype(dtype)
        self._y = self._y.astype(dtype)
        values = _axis_to_front(self._y, self.axis)
        self._values_shape = tuple(values.shape)
        self._values = values.reshape(self._values_shape[0], -1)

        h = jnp.diff(self._x)
        self._intervals = h
        self._second_derivatives = _not_a_knot_second_derivatives(h, self._values)

    @property
    def bounds_error(self) -> bool:
        """Getter and setter for out-of-bounds error handling."""

        return self._bounds_error

    @bounds_error.setter
    def bounds_error(self, value: bool) -> None:
        """Setter for the **self.bounds_error** property."""

        self._bounds_error = value

    @property
    def fill_value(self) -> Any:
        """Getter and setter for the out-of-bounds fill value."""

        return self._fill_value

    @fill_value.setter
    def fill_value(self, value: Any) -> None:
        """Setter for the **self.fill_value** property."""

        extrapolate = isinstance(value, str) and value == "extrapolate"
        if extrapolate and self.bounds_error:
            error = "Cannot extrapolate and raise at the same time."
            raise ValueError(error)
        self._fill_value = value
        self._extrapolate = extrapolate
        self._fill_value_below, self._fill_value_above = (
            value if isinstance(value, tuple) and len(value) == 2 else (value, value)
        )

    def __call__(self, x: Any) -> jnp.ndarray:
        """Evaluate the cubic spline at the specified point(s)."""

        x = jnp.asarray(x, dtype=self._values.dtype)
        x_i = self._x
        h = self._intervals
        y_i = self._values
        second = self._second_derivatives

        below = x < x_i[0]
        above = x > x_i[-1]
        # Filled queries must not evaluate an overflowing inactive polynomial:
        # masking its result afterwards can still contaminate the gradients.
        evaluated_x = x if self._extrapolate else jnp.where(below | above, x_i[0], x)
        x_flat = evaluated_x.reshape(-1)
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
        values = values.reshape((*x.shape, *self._values_shape[1:]))

        if self.bounds_error and _any_true(below | above):
            error = "A value in x_new is outside the interpolation range."
            raise ValueError(error)
        if not self._extrapolate:
            remaining_ndim = self._y.ndim - 1
            below = below.reshape((*below.shape, *((1,) * remaining_ndim)))
            above = above.reshape((*above.shape, *((1,) * remaining_ndim)))
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

        return _restore_axis(values, x.ndim, self.axis)

    @property
    def x(self) -> jnp.ndarray:
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property; re-solves the spline."""

        implementation = type(self)(
            _as_float(value, self._dtype_arg),
            self._y,
            axis=self.axis,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
            assume_sorted=self._assume_sorted,
            dtype=self._dtype_arg,
        )
        vars(self).update(vars(implementation))

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; re-solves the spline."""

        implementation = type(self)(
            self._x,
            _as_float(value, self._dtype_arg),
            axis=self.axis,
            bounds_error=self.bounds_error,
            fill_value=self.fill_value,
            assume_sorted=self._assume_sorted,
            dtype=self._dtype_arg,
        )
        vars(self).update(vars(implementation))

    @property
    def dtype(self) -> jnp.dtype:
        """Getter for the data dtype."""

        return self._y.dtype


class PchipInterpolator:
    """Native PCHIP interpolant over *JAX* inputs."""

    def __init__(
        self,
        x: Any,
        y: Any,
        axis: int = 0,
        extrapolate: bool | Literal["periodic"] | None = None,
        *,
        dtype: Any = None,
    ) -> None:
        x = _as_float(x, dtype)
        y = _as_float(y, dtype)
        if y.ndim == 0:
            error = "`y` must have at least one dimension."
            raise ValueError(error)
        self.axis = int(axis % y.ndim)
        self._extrapolate: bool | Literal["periodic"] | None = (
            True if extrapolate is None else extrapolate
        )

        if x.ndim != 1:
            error = "`x` must be 1-dimensional."
            raise ValueError(error)
        if y.shape[self.axis] != x.shape[0]:
            error = "`x` and `y` must match in length along the interpolation axis."
            raise ValueError(error)
        if x.shape[0] < 2:
            error = "PCHIP interpolation requires at least 2 points."
            raise ValueError(error)
        if _any_true(~jnp.isfinite(x)) or _any_true(~jnp.isfinite(y)):
            error = "`x` and `y` must contain only finite values."
            raise ValueError(error)
        if bool(jnp.any(jnp.diff(x) <= 0)):
            error = "`x` must be strictly increasing."
            raise ValueError(error)

        self._dtype_arg = dtype
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
        # Scale each difference locally so opposite-sign ordinates cannot
        # overflow before division, while unrelated tiny intervals stay intact.
        difference_scale = jnp.where(
            jnp.maximum(jnp.abs(self._values[1:]), jnp.abs(self._values[:-1]))
            > (jnp.finfo(self._values.dtype).max / 65536)
            * jnp.minimum(jnp.abs(h[:, None]), 1),
            jnp.full_like(self._values[:-1], 2.0**64),
            jnp.ones_like(self._values[:-1]),
        )
        difference_scale = difference_scale.astype(self._values.dtype)
        slopes = (
            self._values[1:] / difference_scale - self._values[:-1] / difference_scale
        ) / h[:, None]

        if self._x.shape[0] == 2:
            self._derivatives = jnp.stack([slopes[0], slopes[0]])
            self._derivative_scales = jnp.stack(
                [difference_scale[0], difference_scale[0]]
            )
            return

        previous_original, following_original = slopes[:-1], slopes[1:]
        previous_scale, following_scale = difference_scale[:-1], difference_scale[1:]
        pair_scale = jnp.maximum(previous_scale, following_scale)
        previous = previous_original * (previous_scale / pair_scale)
        following = following_original * (following_scale / pair_scale)
        use_zero = (
            (previous_original == 0)
            | (following_original == 0)
            | (jnp.sign(previous_original) != jnp.sign(following_original))
        )
        previous_smaller = jnp.abs(previous) <= jnp.abs(following)
        maximum = jnp.maximum(jnp.abs(previous), jnp.abs(following))
        minimum = jnp.minimum(jnp.abs(previous), jnp.abs(following))
        disparate = (
            minimum <= maximum * (jnp.finfo(previous.dtype).eps / 16)
        ) & ~use_zero
        ratio_disparate = jnp.where(disparate, minimum, 0) / jnp.where(
            disparate, maximum, 1
        )
        small_original = jnp.where(
            previous_smaller, previous_original, following_original
        )
        small_scale = jnp.where(previous_smaller, previous_scale, following_scale)
        disparate_smaller = jnp.where(disparate, small_original, 0)
        safe_previous = jnp.where(
            use_zero | disparate, jnp.ones_like(previous), previous
        )
        safe_following = jnp.where(
            use_zero | disparate, jnp.ones_like(following), following
        )
        w_1 = 2 * h[1:, None] + h[:-1, None]
        w_2 = h[1:, None] + 2 * h[:-1, None]
        # Anchor the harmonic mean at the smaller slope. Its ratio to the
        # larger slope may underflow harmlessly, but is never inverted.
        # Keep the original ordering: guarded disparate slopes are both one,
        # but their harmonic mean still needs the original interval weight.
        smaller = jnp.where(previous_smaller, safe_previous, safe_following)
        larger = jnp.where(previous_smaller, safe_following, safe_previous)
        # Normalize the ratio with a normal binary factor: JAX division
        # derivatives otherwise lose inverse squares, and MPS flushes factors
        # at the smallest normal exponent. Keep two bits of exponent headroom.
        raw_smaller = smaller
        _, exponent = jnp.frexp(jnp.abs(larger))
        limit = -int(np.log2(jnp.finfo(larger.dtype).tiny)) - 2
        factor = jnp.ldexp(jnp.ones_like(larger), -jnp.clip(exponent, -limit, limit))
        ratio = (smaller * factor) / (larger * factor)
        denominator = jnp.where(previous_smaller, w_1 + w_2 * ratio, w_2 + w_1 * ratio)
        harmonic = raw_smaller * ((w_1 + w_2) / denominator)
        # At the very top of the dtype range, even the exact ratio tangent
        # is subnormal. Factor ab/D through fourth roots instead, keeping
        # tangent intermediates normal. The min-slope path protects disparity.
        root_mode = (
            jnp.maximum(jnp.abs(safe_previous), jnp.abs(safe_following))
            > (1 / jnp.finfo(previous.dtype).tiny) / pair_scale
        ) & (
            jnp.minimum(jnp.abs(safe_previous), jnp.abs(safe_following))
            >= jnp.maximum(jnp.abs(safe_previous), jnp.abs(safe_following))
            * (jnp.finfo(previous.dtype).eps / 16)
        )
        root_previous = jnp.where(root_mode, safe_previous, 1)
        root_following = jnp.where(root_mode, safe_following, 1)
        mean = (w_1 / (w_1 + w_2)) * root_following + (
            w_2 / (w_1 + w_2)
        ) * root_previous
        fourth_root = jnp.sqrt(jnp.sqrt(jnp.abs(mean)))
        root_harmonic = (
            jnp.sign(mean)
            * ((root_previous / fourth_root) / fourth_root)
            * ((root_following / fourth_root) / fourth_root)
        )
        disparate_denominator = jnp.where(
            previous_smaller, w_1 + w_2 * ratio_disparate, w_2 + w_1 * ratio_disparate
        )
        disparate_harmonic = disparate_smaller * ((w_1 + w_2) / disparate_denominator)
        interior = jnp.where(
            use_zero,
            jnp.zeros_like(previous),
            jnp.where(
                disparate,
                disparate_harmonic,
                jnp.where(root_mode, root_harmonic, harmonic),
            ),
        )
        interior_scale = jnp.where(
            use_zero, 1, jnp.where(disparate, small_scale, pair_scale)
        )

        def endpoint(
            h_0: jnp.ndarray,
            h_1: jnp.ndarray,
            slope_0: jnp.ndarray,
            slope_1: jnp.ndarray,
            scale_0: jnp.ndarray,
            scale_1: jnp.ndarray,
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            scale = jnp.maximum(scale_0, scale_1)
            normalized_0 = slope_0 * (scale_0 / scale)
            normalized_1 = slope_1 * (scale_1 / scale)
            derivative = normalized_0 + (h_0 / (h_0 + h_1)) * (
                normalized_0 - normalized_1
            )
            opposite = jnp.sign(derivative) != jnp.sign(slope_0)
            limited = (jnp.sign(slope_0) != jnp.sign(slope_1)) & (
                jnp.abs(derivative) > 3 * jnp.abs(normalized_0)
            )
            result = jnp.where(opposite, 0, jnp.where(limited, 3 * slope_0, derivative))
            result_scale = jnp.where(opposite, 1, jnp.where(limited, scale_0, scale))
            return result, result_scale

        first, first_scale = endpoint(
            h[0], h[1], slopes[0], slopes[1], difference_scale[0], difference_scale[1]
        )
        last, last_scale = endpoint(
            h[-1],
            h[-2],
            slopes[-1],
            slopes[-2],
            difference_scale[-1],
            difference_scale[-2],
        )
        self._derivatives = jnp.concatenate(
            [first[None, ...], interior, last[None, ...]], axis=0
        )
        self._derivative_scales = jnp.concatenate(
            [first_scale[None, ...], interior_scale, last_scale[None, ...]], axis=0
        )

    def __call__(
        self, x: Any, nu: int = 0, extrapolate: bool | Literal["periodic"] | None = None
    ) -> jnp.ndarray:
        """Evaluate the piecewise cubic interpolant at the specified point(s)."""

        x = jnp.asarray(x, dtype=self._values.dtype)
        x_i = self._x
        h = self._intervals
        y_i = self._values
        derivatives = self._derivatives
        extrapolate = self._extrapolate if extrapolate is None else extrapolate
        if extrapolate == "periodic":
            x = x_i[0] + (x - x_i[0]) % (x_i[-1] - x_i[0])

        x_flat = x.reshape(-1)
        indices = _interval_indices(x_flat, x_i)
        h_interval = h[indices, None]
        t = (x_flat - x_i[indices])[:, None] / h_interval

        y_left, y_right = y_i[indices], y_i[indices + 1]
        d_left, d_right = derivatives[indices], derivatives[indices + 1]
        left_scale = self._derivative_scales[indices]
        right_scale = self._derivative_scales[indices + 1]
        scale = jnp.maximum(
            jnp.where(
                jnp.maximum(jnp.abs(y_left), jnp.abs(y_right))
                > jnp.finfo(y_i.dtype).max / 65536,
                jnp.full_like(y_left, 2.0**64),
                jnp.ones_like(y_left),
            ),
            jnp.maximum(left_scale, right_scale),
        )
        y_left, y_right = y_left / scale, y_right / scale
        d_left, d_right = d_left * (left_scale / scale), d_right * (right_scale / scale)
        delta = y_right - y_left
        coefficient_1 = h_interval * d_left
        right_step = h_interval * d_right
        coefficient_3 = (coefficient_1 - delta) + (right_step - delta)
        coefficient_2 = delta - coefficient_1 - coefficient_3
        remaining = jnp.where(
            scale > 1, (x_i[indices + 1] - x_flat)[:, None] / h_interval, 1 - t
        )
        right_coefficient_2 = -delta + right_step + coefficient_3

        if nu == 0:
            # Anchor at the nearer original ordinate: tiny knot values must
            # survive next to huge neighbours without subtractive cancellation.
            ordinary = jnp.where(
                t <= 0.5,
                coefficient_3 * t**3 + coefficient_2 * t**2 + coefficient_1 * t,
                -coefficient_3 * remaining**3
                + right_coefficient_2 * remaining**2
                - h_interval * d_right * remaining,
            )
            u = jnp.where(t <= 0.5, t, remaining) * 256
            first_coefficient = jnp.where(
                t <= 0.5, coefficient_1, -h_interval * d_right
            )
            second_coefficient = jnp.where(t <= 0.5, coefficient_2, right_coefficient_2)
            third_coefficient = jnp.where(t <= 0.5, coefficient_3, -coefficient_3)
            horner = (
                (third_coefficient / 256**3 * u + second_coefficient / 256**2) * u
                + first_coefficient / 256
            ) * u
            values = jnp.where(scale > 1, horner, ordinary)
        elif nu == 1:
            values = (
                jnp.where(
                    t <= 0.5,
                    (3 * coefficient_3 * t + 2 * coefficient_2) * t + coefficient_1,
                    -(
                        (-3 * coefficient_3 * remaining + 2 * right_coefficient_2)
                        * remaining
                        - right_step
                    ),
                )
                / h_interval
            )
        elif nu == 2:
            values = (
                jnp.where(
                    t <= 0.5,
                    6 * coefficient_3 * t + 2 * coefficient_2,
                    -6 * coefficient_3 * remaining + 2 * right_coefficient_2,
                )
                / h_interval**2
            )
        elif nu == 3:
            values = 6 * coefficient_3 / h_interval**3
        else:
            values = jnp.zeros_like(coefficient_3)

        values = values * scale
        if nu == 0:
            values = values + jnp.where(t <= 0.5, y_i[indices], y_i[indices + 1])
        values = values.reshape((*x.shape, *self._values_shape[1:]))
        values = _restore_axis(values, x.ndim, self.axis)

        if not extrapolate:
            outside = (x < x_i[0]) | (x > x_i[-1])
            remaining_ndim = self._y.ndim - 1
            outside = outside.reshape((*outside.shape, *((1,) * remaining_ndim)))
            outside = _restore_axis(outside, x.ndim, self.axis)
            values = jnp.where(
                outside, jnp.asarray(jnp.nan, dtype=values.dtype), values
            )

        return values

    @property
    def x(self) -> jnp.ndarray:
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property; recomputes the derivatives."""

        implementation = type(self)(
            _as_float(value, self._dtype_arg),
            self._y,
            axis=self.axis,
            extrapolate=self.extrapolate,
            dtype=self._dtype_arg,
        )
        # An assigned None masks extrapolation, unlike the constructor default.
        implementation.extrapolate = self.extrapolate
        vars(self).update(vars(implementation))

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; recomputes the derivatives."""

        implementation = type(self)(
            self._x,
            _as_float(value, self._dtype_arg),
            axis=self.axis,
            extrapolate=self.extrapolate,
            dtype=self._dtype_arg,
        )
        # An assigned None masks extrapolation, unlike the constructor default.
        implementation.extrapolate = self.extrapolate
        vars(self).update(vars(implementation))

    @property
    def extrapolate(self) -> bool | Literal["periodic"] | None:
        """Getter and setter for extrapolation outside the interpolation range."""

        return self._extrapolate

    @extrapolate.setter
    def extrapolate(self, value: bool | Literal["periodic"] | None) -> None:
        """Set extrapolation; assigned *None* masks out-of-domain queries."""

        self._extrapolate = value

    @property
    def dtype(self) -> jnp.dtype:
        """Getter for the data dtype."""

        return self._y.dtype


class LinearInterpolator:
    """Linear interpolant over *JAX* inputs; raises outside the range."""

    def __init__(
        self,
        x: Any,
        y: Any,
        dtype: Any = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        self._dtype_arg = dtype
        self._x = jnp.atleast_1d(_as_float(x, dtype))
        self._y = jnp.atleast_1d(_as_float(y, dtype))
        validate_dimensions(self._x, self._y)

    def __call__(self, x: Any) -> jnp.ndarray:
        """Evaluate the linear interpolant at the specified point(s)."""

        x = jnp.asarray(x, dtype=self._y.dtype)
        _validate_range(x, self._x)

        if len(self._x) == 1:
            return jnp.broadcast_to(self._y[0], (*x.shape, *self._y.shape[1:]))

        i = jnp.clip(jnp.searchsorted(self._x, x) - 1, 0, len(self._x) - 2)
        t = (x - self._x[i]) / (self._x[i + 1] - self._x[i])
        if self._y.ndim > 1:
            t = t[..., None]

        return self._y[i] + (self._y[i + 1] - self._y[i]) * t

    @property
    def x(self) -> jnp.ndarray:
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property."""

        x = jnp.atleast_1d(_as_float(value, self._dtype_arg))
        validate_dimensions(x, self._y)
        self._x = x

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property."""

        y = jnp.atleast_1d(_as_float(value, self._dtype_arg))
        validate_dimensions(self._x, y)
        self._y = y

    @property
    def dtype(self) -> jnp.dtype:
        """Getter for the data dtype."""

        return self._y.dtype


class NearestNeighbourInterpolator:
    """Nearest-neighbour interpolant over *JAX* inputs."""

    def __init__(
        self,
        x: Any,
        y: Any,
        dtype: Any = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        self._dtype_arg = dtype
        self._x = _as_float(x, dtype)
        self._y = _as_float(y, dtype)
        validate_dimensions(self._x, self._y)

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
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property."""

        x = _as_float(value, self._dtype_arg)
        validate_dimensions(x, self._y)
        self._x = x

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property."""

        y = _as_float(value, self._dtype_arg)
        validate_dimensions(self._x, y)
        self._y = y

    @property
    def dtype(self) -> jnp.dtype:
        """Getter for the data dtype."""

        return self._y.dtype


class NullInterpolator:
    """Null interpolant over *JAX* inputs (exact matches within tolerance)."""

    def __init__(
        self,
        x: Any,
        y: Any,
        absolute_tolerance: float = 1e-8,
        relative_tolerance: float = 1e-8,
        default: float = float("nan"),
        dtype: Any = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        self._dtype_arg = dtype
        self._x = jnp.atleast_1d(_as_float(x, dtype))
        self._y = jnp.atleast_1d(_as_float(y, dtype))
        validate_dimensions(self._x, self._y)
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
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
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property."""

        x = jnp.atleast_1d(_as_float(value, self._dtype_arg))
        validate_dimensions(x, self._y)
        self._x = x

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property."""

        y = jnp.atleast_1d(_as_float(value, self._dtype_arg))
        validate_dimensions(self._x, y)
        self._y = y

    @property
    def dtype(self) -> jnp.dtype:
        """Getter for the data dtype."""

        return self._y.dtype

    @property
    def absolute_tolerance(self) -> float:
        """Getter and setter for the absolute matching tolerance."""

        return self._absolute_tolerance

    @absolute_tolerance.setter
    def absolute_tolerance(self, value: Any) -> None:
        """Setter for the **self.absolute_tolerance** property."""

        self._absolute_tolerance = float(value)

    @property
    def relative_tolerance(self) -> float:
        """Getter and setter for the relative matching tolerance."""

        return self._relative_tolerance

    @relative_tolerance.setter
    def relative_tolerance(self, value: Any) -> None:
        """Setter for the **self.relative_tolerance** property."""

        self._relative_tolerance = float(value)

    @property
    def default(self) -> Any:
        """Getter and setter for the value returned when no knot matches."""

        return self._default

    @default.setter
    def default(self, value: Any) -> None:
        """Setter for the **self.default** property."""

        self._default = value


class SpragueInterpolator:
    """Fifth-order *Sprague (1880)* interpolant over uniformly spaced data."""

    def __init__(
        self,
        x: Any,
        y: Any,
        dtype: Any = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        self._dtype_arg = dtype
        self._x = _as_float(x, dtype)
        self._y = _as_float(y, dtype)
        validate_dimensions(self._x, self._y)

        if self._y.shape[0] < 6:
            error = "Sprague interpolation requires at least 6 points."
            raise ValueError(error)

        self._solve()

    def _solve(self) -> None:
        x = self._x
        y = self._y
        # A fixed binary scale protects both weighted sums and opposite-sign
        # differences, without subnormal derivatives of a data-sized scale.
        unit = jnp.ones((), dtype=y.dtype)
        self._y_scale = jnp.where(
            jnp.max(jnp.abs(y)) > jnp.finfo(y.dtype).max / 65536,
            unit * 65536,
            unit,
        )
        y = y / self._y_scale
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
        # Preserve the common ordinate offset before weighted cancellation.
        large = self._y_scale > 1
        origins = jnp.where(large, windows[:, 0], 0)
        slope = jnp.where(large, (windows[:, -1] - windows[:, 0]) / 5, 0)
        ramp = jnp.arange(6, dtype=y.dtype)
        positions = jnp.asarray([-2, -1, 6, 7], dtype=y.dtype)
        if y.ndim == 2:
            ramp = ramp[:, None]
            positions = positions[:, None]
        residual = windows - origins[:, None] - slope[:, None] * ramp[None, ...]
        y_boundary = (
            origins + slope * positions + (coefficients * residual).sum(axis=1) / 209
        )
        self._y_p = jnp.concatenate([y_boundary[:2], y, y_boundary[2:]], axis=0)

    def __call__(self, x: Any) -> jnp.ndarray:
        """Evaluate the *Sprague (1880)* interpolant at the specified point(s)."""

        x = jnp.asarray(x, dtype=self._y.dtype)
        _validate_range(jnp.atleast_1d(x), self._x)

        x_flat = jnp.atleast_1d(x).reshape(-1)
        large = self._y_scale > 1
        i = jnp.searchsorted(self._x_p, x_flat) - 1
        i = jnp.where(
            large,
            jnp.clip(
                jnp.searchsorted(self._x_p, x_flat, side="right") - 1, 2, len(self._x)
            ),
            i,
        )
        t = (x_flat - self._x_p[i]) / (self._x_p[i + 1] - self._x_p[i])
        reverse = large & (t > 0.5)
        anchor_index = jnp.where(reverse, i + 1, i)
        t = jnp.where(reverse, 1 - t, t)

        windows = jnp.stack([self._y_p[i + k] for k in (-2, -1, 0, 1, 2, 3)])
        reverse_b = reverse[..., None] if self._y.ndim == 2 else reverse
        windows = jnp.where(reverse_b[None, ...], jnp.flip(windows, axis=0), windows)
        weights = jnp.asarray(SPRAGUE_A_COEFFICIENTS, dtype=self._y.dtype)
        windows_shape = windows.shape
        large = self._y_scale > 1
        slope = jnp.where(large, (windows[3] - windows[1]) / 2, 0)
        curvature = jnp.where(large, (windows[3] - 2 * windows[2] + windows[1]) / 2, 0)
        offsets = jnp.asarray([-2, -1, 0, 1, 2, 3], dtype=windows.dtype)
        offsets = offsets.reshape((6,) + (1,) * (windows.ndim - 1))
        centered = (
            windows
            - jnp.where(large, windows[2], 0)[None, ...]
            - slope[None, ...] * offsets
            - curvature[None, ...] * offsets**2
        )
        a = (weights @ centered.reshape(windows_shape[0], -1) / 24).reshape(
            weights.shape[0], *windows_shape[1:]
        )

        a = jnp.concatenate(
            [a[:1] + slope[None, ...], a[1:2] + curvature[None, ...], a[2:]], axis=0
        )

        powers = jnp.asarray([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=self._y.dtype)
        basis = t**powers
        if self._y.ndim == 2:
            basis = basis[..., None]
        ordinary = self._y_p[i] + (a * basis).sum(axis=0)
        # Horner evaluation about the nearer original node keeps both tiny
        # knot values and reverse-mode query derivatives representable.
        t_b = (t[..., None] if self._y.ndim == 2 else t) * 256
        # Binary coordinate scaling keeps individual reverse-mode derivative
        # terms small until they have cancelled, then restores the query scale.
        coordinate_powers = jnp.asarray([256.0**k for k in range(1, 6)], dtype=a.dtype)
        coordinate_powers = coordinate_powers.reshape((5,) + (1,) * (a.ndim - 1))
        horner_a = a / coordinate_powers
        correction = horner_a[-1]
        for coefficient in reversed(horner_a[:-1]):
            correction = correction * t_b + coefficient
        correction = correction * t_b
        anchor = self._y[jnp.clip(anchor_index - 2, 0, len(self._y) - 1)]
        values = jnp.where(
            large, anchor + correction * self._y_scale, ordinary * self._y_scale
        )
        return values.reshape((*x.shape, *self._y.shape[1:]))

    @property
    def x(self) -> jnp.ndarray:
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property; rebuilds the boundary extension."""

        x = _as_float(value, self._dtype_arg)
        validate_dimensions(x, self._y)
        self._x = x
        self._solve()

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; rebuilds the boundary extension."""

        y = _as_float(value, self._dtype_arg)
        validate_dimensions(self._x, y)
        self._y = y
        self._solve()

    @property
    def dtype(self) -> jnp.dtype:
        """Getter for the data dtype."""

        return self._y.dtype


def _reflect_pad(y: jnp.ndarray, window: int) -> jnp.ndarray:
    """
    Reflect-pad the leading axis of ``y`` by ``window`` on each side, matching
    ``np.pad(mode="reflect")`` for any width.

    The integer-index gather keeps the padded values' autodiff graph with
    respect to ``y``.
    """

    return y[jnp.asarray(reflect_indices(y.shape[0], window))]


class KernelInterpolator:
    """
    Native kernel-based (convolution) interpolant over *JAX* inputs.

    Evaluation preserves autodiff with respect to the dependent values (through
    the gathered padded values). Uniform ``x`` spacing is assumed. Only reflect
    padding with a ``pad_width`` equal to ``window`` is supported; other
    ``padding_kwargs`` raise.
    """

    def __init__(
        self,
        x: Any,
        y: Any,
        window: float = 3,
        kernel: Callable = kernel_lanczos,
        kernel_kwargs: dict | None = None,
        padding_kwargs: dict | None = None,
        dtype: Any = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        self._dtype_arg = dtype
        self._window = window
        self._kernel = kernel
        self._kernel_kwargs = {} if kernel_kwargs is None else dict(kernel_kwargs)
        self._padding_kwargs_arg = padding_kwargs

        self._x = _as_float(x, dtype)
        self._y = _as_float(y, dtype)
        validate_dimensions(self._x, self._y)

        self._build()

    def _build(self) -> None:
        # The default padding tracks the current window, so changing the window
        # recomputes it to ``(window, window)`` to match a fresh construction.
        self._padding_kwargs = (
            {"pad_width": (self._window, self._window), "mode": "reflect"}
            if self._padding_kwargs_arg is None
            else dict(self._padding_kwargs_arg)
        )
        validate_padding_kwargs(self._padding_kwargs, self._window)

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

        _validate_range(x, self._x)

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
        if self._y.ndim > 1:
            weights = weights.reshape(weights.shape + (1,) * (self._y.ndim - 1))

        return jnp.sum(self._y_p[windows_i] * weights, axis=x.ndim)

    @property
    def x(self) -> jnp.ndarray:
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property; rebuilds the padded values."""

        x = _as_float(value, self._dtype_arg)
        validate_dimensions(x, self._y)
        self._x = x
        self._build()

    @property
    def y(self) -> jnp.ndarray:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; rebuilds the padded values."""

        y = _as_float(value, self._dtype_arg)
        validate_dimensions(self._x, y)
        self._y = y
        self._build()

    @property
    def window(self) -> float:
        """Getter and setter for the interpolation window size."""

        return self._window

    @window.setter
    def window(self, value: Any) -> None:
        """Setter for the **self.window** property; rebuilds the padded values."""

        self._window = value
        self._build()

    @property
    def kernel(self) -> Callable:
        """Getter and setter for the interpolation kernel callable."""

        return self._kernel

    @kernel.setter
    def kernel(self, value: Any) -> None:
        """Setter for the **self.kernel** property."""

        self._kernel = value

    @property
    def kernel_kwargs(self) -> dict:
        """Getter and setter for the kernel keyword arguments."""

        return self._kernel_kwargs

    @kernel_kwargs.setter
    def kernel_kwargs(self, value: Any) -> None:
        """Setter for the **self.kernel_kwargs** property."""

        self._kernel_kwargs = dict(value)

    @property
    def padding_kwargs(self) -> dict:
        """Getter and setter for the padding keyword arguments."""

        return self._padding_kwargs

    @padding_kwargs.setter
    def padding_kwargs(self, value: Any) -> None:
        """Setter for the **self.padding_kwargs** property; rebuilds the padding."""

        self._padding_kwargs_arg = value
        self._build()

    @property
    def dtype(self) -> jnp.dtype:
        """Getter for the data dtype."""

        return self._y.dtype


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
        # Fixed overrides and inactive sides must not record infinite offsets:
        # their zero reverse-mode cotangents would otherwise multiply infinity.
        x_offset_low = jnp.where(below & (self._left is None), x - xi[0], 0)[..., None]
        x_offset_high = jnp.where(above & (self._right is None), x - xi[-1], 0)[
            ..., None
        ]

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

        # Evaluate the wrapped interpolator on the query clamped into its domain
        # (fixed-shape, so the path traces under ``jax.jit``) and keep only the
        # in-range results. Clamping guarantees in-range values, so the wrapped
        # interpolator's own range check never fires.
        x_ravel = x.reshape(-1)
        clamped = jnp.clip(x_ravel, xi[0], xi[-1])
        interpolated = jnp.atleast_1d(self._interpolator(clamped))
        if interpolated.ndim == 1:
            interpolated = interpolated[..., None]

        y_ravel = jnp.where(
            in_range.reshape(-1)[..., None], interpolated, y.reshape(-1, yi.shape[1])
        )
        y = y_ravel.reshape((*x.shape, yi.shape[1]))

        if input_rank == 1:
            y = y[..., 0]

        return y
