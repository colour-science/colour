"""
PyTorch backend interpolators.

Native, on-device *not-a-knot* cubic spline and PCHIP implementations using
``torch`` operations. Evaluation preserves automatic differentiation with
respect to the dependent values and the evaluation points; the interval search
uses ``torch.searchsorted`` so the whole evaluation stays on device (no host
round-trip). The known knot locations are treated as structural.
"""

# ``torch`` is an optional backend. When it is not installed, static analysers
# resolve this module's ``import torch`` to the module itself and cannot verify
# any ``torch`` attribute; the attribute checks are only meaningful when
# *PyTorch* is present.
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

import typing

import numpy as np
import torch

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


def _validate_range(x: torch.Tensor, x_i: torch.Tensor) -> None:
    """Raise if any query value lies outside the interpolation range."""

    if bool(torch.any(x < x_i[0])) or bool(torch.any(x > x_i[-1])):
        error = "A value in x is outside the interpolation range."
        raise ValueError(error)


def _colocate(a: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return a.to(like.device)


def _as_float(
    a: Any, dtype: Any = None, *, like: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Return ``a`` as a float tensor.

    A *NumPy* ``dtype`` casts to the matching *PyTorch* dtype, preserving the
    autodiff graph and device. Unspecified, an existing floating dtype is kept
    and an integer tensor promoted to the default float. A ``like`` reference
    moves the result onto that tensor's device, mirroring the dispatch layer's
    :func:`colour.utilities.xp_as_float_array` ``like`` argument. Co-locating
    the knots with the values avoids a device mismatch when, for example,
    host coordinates are paired with device-resident values.
    """

    target_dtype = (
        getattr(torch, np.dtype(dtype).name)
        if dtype is not None
        else like.dtype
        if like is not None and torch.is_floating_point(like)
        else a.dtype
        if isinstance(a, torch.Tensor) and torch.is_floating_point(a)
        else getattr(torch, a.dtype.name)
        if isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating)
        else torch.get_default_dtype()
    )
    device = (
        like.device
        if like is not None
        else a.device
        if isinstance(a, torch.Tensor)
        else torch.device("cpu")
    )
    if device.type == "mps" and target_dtype == torch.float64:
        target_dtype = torch.float32
    if isinstance(a, np.ndarray) and any(stride < 0 for stride in a.strides):
        a = np.ascontiguousarray(a)
    return (
        a.to(device=device, dtype=target_dtype)
        if isinstance(a, torch.Tensor)
        else torch.as_tensor(a, dtype=target_dtype, device=device)
    )


def _axis_to_front(y: torch.Tensor, axis: int) -> torch.Tensor:
    """Move the interpolation axis of the values to the front."""

    axis %= y.ndim
    if axis == 0:
        return y

    return y.permute((axis, *range(axis), *range(axis + 1, y.ndim)))


def _restore_axis(y: torch.Tensor, query_ndim: int, axis: int) -> torch.Tensor:
    """Restore the interpolation axis placement in evaluated values."""

    if axis == 0 or query_ndim == 0:
        return y

    axes = (
        *range(query_ndim, query_ndim + axis),
        *range(query_ndim),
        *range(query_ndim + axis, y.ndim),
    )

    return y.permute(axes)


def _interval_indices(x: torch.Tensor, x_i: torch.Tensor) -> torch.Tensor:
    """Return clipped interpolation interval indices for query values."""

    return torch.clamp(torch.searchsorted(x_i, x) - 1, 0, x_i.shape[0] - 2)


def _not_a_knot_second_derivatives(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Return the *not-a-knot* cubic-spline second derivatives, shape ``(n, k)``.

    The moment system is tridiagonal apart from the two *not-a-knot* boundary
    rows. It is assembled densely and solved with the fused, autodiff-preserving
    :func:`torch.linalg.solve`. *PyTorch* has no native banded solver, and a
    hand-written sequential *Thomas* sweep is markedly slower here: at spectral
    sizes its per-step kernel-launch overhead loses several-fold to this single
    fused solve, so the dense solve is the fastest correct option on device.
    """

    n = h.shape[0] + 1
    matrix = torch.zeros((n, n), dtype=y.dtype, device=y.device)
    interior = torch.arange(1, n - 1, device=y.device)
    matrix[interior, interior - 1] = h[:-1]
    matrix[interior, interior] = 2 * (h[:-1] + h[1:])
    matrix[interior, interior + 1] = h[1:]
    matrix[0, 0], matrix[0, 1], matrix[0, 2] = h[1], -(h[0] + h[1]), h[0]
    matrix[-1, -3], matrix[-1, -2], matrix[-1, -1] = h[-1], -(h[-2] + h[-1]), h[-2]

    slope = (y[1:] - y[:-1]) / h[:, None]
    zeros = torch.zeros_like(y[:1])
    rhs = torch.cat([zeros, 6 * (slope[1:] - slope[:-1]), zeros], dim=0)

    return torch.linalg.solve(matrix, rhs)


class CubicSplineInterpolator:
    """Native *not-a-knot* cubic spline over *PyTorch* inputs."""

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
        self._dtype_arg = dtype
        self.axis = axis
        self._assume_sorted = assume_sorted
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

        self._set_data(x, y)

    def _set_data(
        self, x: Any, y: Any, *, reference: torch.Tensor | None = None
    ) -> None:
        """Validate and solve before replacing the interpolation data."""

        if reference is None and isinstance(x, torch.Tensor):
            reference = x
        y = _as_float(y, self._dtype_arg, like=reference)
        x = _as_float(x, self._dtype_arg, like=y)
        if x.ndim != 1:
            error = "`x` must be 1-dimensional."
            raise ValueError(error)
        if y.ndim == 0:
            error = "`y` must have at least one dimension."
            raise ValueError(error)
        if not -y.ndim <= self.axis < y.ndim:
            error = "Interpolation axis is out of bounds for `y`."
            raise IndexError(error)
        axis = int(self.axis % y.ndim)
        if y.shape[axis] != x.shape[0]:
            error = "`x` and `y` must match in length along the interpolation axis."
            raise ValueError(error)

        if not self._assume_sorted:
            order = torch.argsort(x)
            y = torch.index_select(y, axis, order)
            x = x[order]

        if x.shape[0] < 4:
            error = "Cubic spline interpolation requires at least 4 points."
            raise ValueError(error)
        if bool(torch.any(torch.diff(x) <= 0)):
            error = "`x` must be strictly increasing."
            raise ValueError(error)

        values = _axis_to_front(y, axis)
        values_shape = tuple(values.shape)
        values = values.reshape(values_shape[0], -1)
        intervals = torch.diff(x)
        second_derivatives = _not_a_knot_second_derivatives(intervals, values)

        self.axis = axis
        self._x, self._y = x, y
        self._values_shape, self._values = values_shape, values
        self._intervals = intervals
        self._second_derivatives = second_derivatives

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

    def __call__(self, x: Any) -> torch.Tensor:
        """Evaluate the cubic spline at the specified point(s)."""

        x = _as_float(x, like=self._values)
        x_i = self._x
        h = self._intervals
        y_i = self._values
        second = self._second_derivatives

        below = x < x_i[0]
        above = x > x_i[-1]
        # Filled queries must not evaluate an overflowing inactive polynomial:
        # masking its result afterwards can still contaminate the gradients.
        evaluated_x = x if self._extrapolate else torch.where(below | above, x_i[0], x)
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

        if self.bounds_error and bool(torch.any(below | above)):
            error = "A value in x_new is outside the interpolation range."
            raise ValueError(error)
        if not self._extrapolate:
            remaining_ndim = self._y.ndim - 1
            below = below.reshape((*below.shape, *((1,) * remaining_ndim)))
            above = above.reshape((*above.shape, *((1,) * remaining_ndim)))
            values = torch.where(
                below,
                torch.as_tensor(
                    self._fill_value_below, dtype=values.dtype, device=values.device
                ),
                values,
            )
            values = torch.where(
                above,
                torch.as_tensor(
                    self._fill_value_above, dtype=values.dtype, device=values.device
                ),
                values,
            )

        return _restore_axis(values, x.ndim, self.axis)

    @property
    def x(self) -> torch.Tensor:
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property; re-solves the spline."""

        self._set_data(value, self._y, reference=self._y)

    @property
    def y(self) -> torch.Tensor:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; re-solves the spline."""

        self._set_data(self._x, value, reference=self._x)

    @property
    def dtype(self) -> torch.dtype:
        """Getter for the data dtype."""

        return self._y.dtype


class PchipInterpolator:
    """Native PCHIP interpolant over *PyTorch* inputs."""

    def __init__(
        self,
        x: Any,
        y: Any,
        axis: int = 0,
        extrapolate: bool | Literal["periodic"] | None = None,
        *,
        dtype: Any = None,
    ) -> None:
        self._dtype_arg = dtype
        self.axis = axis
        self._extrapolate: bool | Literal["periodic"] | None = (
            True if extrapolate is None else extrapolate
        )
        self._set_data(x, y)

    def _set_data(
        self, x: Any, y: Any, *, reference: torch.Tensor | None = None
    ) -> None:
        """Validate and solve before replacing the interpolation data."""

        if reference is None and isinstance(x, torch.Tensor):
            reference = x
        y = _as_float(y, self._dtype_arg, like=reference)
        x = _as_float(x, self._dtype_arg, like=y)
        if y.ndim == 0:
            error = "`y` must have at least one dimension."
            raise ValueError(error)
        axis = int(self.axis % y.ndim)

        if x.ndim != 1:
            error = "`x` must be 1-dimensional."
            raise ValueError(error)
        if y.shape[axis] != x.shape[0]:
            error = "`x` and `y` must match in length along the interpolation axis."
            raise ValueError(error)
        if x.shape[0] < 2:
            error = "PCHIP interpolation requires at least 2 points."
            raise ValueError(error)
        if not bool(torch.all(torch.isfinite(x))) or not bool(
            torch.all(torch.isfinite(y))
        ):
            error = "`x` and `y` must contain only finite values."
            raise ValueError(error)
        if bool(torch.any(torch.diff(x) <= 0)):
            error = "`x` must be strictly increasing."
            raise ValueError(error)

        values = _axis_to_front(y, axis)
        values_shape = tuple(values.shape)
        values = values.reshape(values_shape[0], -1)
        intervals = torch.diff(x)
        derivatives, derivative_scales = self._solve(intervals, values)

        self.axis = axis
        self._x, self._y = x, y
        self._values_shape, self._values = values_shape, values
        self._intervals = intervals
        self._derivatives, self._derivative_scales = derivatives, derivative_scales

    @staticmethod
    def _solve(
        h: torch.Tensor, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the shape-preserving derivatives for the specified data."""

        # Scale each difference locally so opposite-sign ordinates cannot
        # overflow before division, while unrelated tiny intervals stay intact.
        difference_scale = torch.where(
            torch.maximum(torch.abs(values[1:]), torch.abs(values[:-1]))
            > (torch.finfo(values.dtype).max / 65536)
            * torch.clamp(torch.abs(h[:, None]), max=1),
            torch.full_like(values[:-1], 2.0**64),
            torch.ones_like(values[:-1]),
        )
        difference_scale = difference_scale.to(dtype=values.dtype)
        slopes = (values[1:] / difference_scale - values[:-1] / difference_scale) / h[
            :, None
        ]

        if values.shape[0] == 2:
            derivatives = torch.stack([slopes[0], slopes[0]])
            derivative_scales = torch.stack([difference_scale[0], difference_scale[0]])
            return derivatives, derivative_scales

        previous_original, following_original = slopes[:-1], slopes[1:]
        previous_scale, following_scale = difference_scale[:-1], difference_scale[1:]
        pair_scale = torch.maximum(previous_scale, following_scale)
        previous = previous_original * (previous_scale / pair_scale)
        following = following_original * (following_scale / pair_scale)
        use_zero = (
            (previous_original == 0)
            | (following_original == 0)
            | (torch.sign(previous_original) != torch.sign(following_original))
        )
        previous_smaller = torch.abs(previous) <= torch.abs(following)
        maximum = torch.maximum(torch.abs(previous), torch.abs(following))
        minimum = torch.minimum(torch.abs(previous), torch.abs(following))
        disparate = (
            minimum <= maximum * (torch.finfo(previous.dtype).eps / 16)
        ) & ~use_zero
        ratio_disparate = torch.where(disparate, minimum, 0) / torch.where(
            disparate, maximum, 1
        )
        small_original = torch.where(
            previous_smaller, previous_original, following_original
        )
        small_scale = torch.where(previous_smaller, previous_scale, following_scale)
        disparate_smaller = torch.where(disparate, small_original, 0)
        safe_previous = torch.where(
            use_zero | disparate, torch.ones_like(previous), previous
        )
        safe_following = torch.where(
            use_zero | disparate, torch.ones_like(following), following
        )
        w_1 = 2 * h[1:, None] + h[:-1, None]
        w_2 = h[1:, None] + 2 * h[:-1, None]
        # Anchor the harmonic mean at the smaller slope. Its ratio to the
        # larger slope may underflow harmlessly, but is never inverted.
        # Keep the original ordering: guarded disparate slopes are both one,
        # but their harmonic mean still needs the original interval weight.
        smaller = torch.where(previous_smaller, safe_previous, safe_following)
        larger = torch.where(previous_smaller, safe_following, safe_previous)
        # Normalize the ratio with a normal binary factor: JAX division
        # derivatives otherwise lose inverse squares, and MPS flushes factors
        # at the smallest normal exponent. Keep two bits of exponent headroom.
        raw_smaller = smaller
        _, exponent = torch.frexp(torch.abs(larger))
        limit = -int(np.log2(torch.finfo(larger.dtype).tiny)) - 2
        factor = torch.exp2(
            -torch.clamp(exponent, -limit, limit).to(dtype=larger.dtype)
        )
        ratio = (smaller * factor) / (larger * factor)
        denominator = torch.where(
            previous_smaller, w_1 + w_2 * ratio, w_2 + w_1 * ratio
        )
        disparate_denominator = torch.where(
            previous_smaller, w_1 + w_2 * ratio_disparate, w_2 + w_1 * ratio_disparate
        )
        disparate_harmonic = disparate_smaller * ((w_1 + w_2) / disparate_denominator)
        interior = torch.where(
            use_zero,
            torch.zeros_like(previous),
            torch.where(
                disparate, disparate_harmonic, raw_smaller * ((w_1 + w_2) / denominator)
            ),
        )
        interior_scale = torch.where(
            use_zero, 1, torch.where(disparate, small_scale, pair_scale)
        )

        def endpoint(
            h_0: torch.Tensor,
            h_1: torch.Tensor,
            slope_0: torch.Tensor,
            slope_1: torch.Tensor,
            scale_0: torch.Tensor,
            scale_1: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            scale = torch.maximum(scale_0, scale_1)
            normalized_0 = slope_0 * (scale_0 / scale)
            normalized_1 = slope_1 * (scale_1 / scale)
            derivative = normalized_0 + (h_0 / (h_0 + h_1)) * (
                normalized_0 - normalized_1
            )
            opposite = torch.sign(derivative) != torch.sign(slope_0)
            limited = (torch.sign(slope_0) != torch.sign(slope_1)) & (
                torch.abs(derivative) > 3 * torch.abs(normalized_0)
            )
            result = torch.where(
                opposite, 0, torch.where(limited, 3 * slope_0, derivative)
            )
            result_scale = torch.where(
                opposite, 1, torch.where(limited, scale_0, scale)
            )
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
        derivatives = torch.cat([first[None, ...], interior, last[None, ...]], dim=0)
        derivative_scales = torch.cat(
            [first_scale[None, ...], interior_scale, last_scale[None, ...]], dim=0
        )
        return derivatives, derivative_scales

    def __call__(
        self, x: Any, nu: int = 0, extrapolate: bool | Literal["periodic"] | None = None
    ) -> torch.Tensor:
        """Evaluate the piecewise cubic interpolant at the specified point(s)."""

        x = _as_float(x, like=self._values)
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
        scale = torch.maximum(
            torch.where(
                torch.maximum(torch.abs(y_left), torch.abs(y_right))
                > torch.finfo(y_i.dtype).max / 65536,
                torch.full_like(y_left, 2.0**64),
                torch.ones_like(y_left),
            ),
            torch.maximum(left_scale, right_scale),
        )
        y_left, y_right = y_left / scale, y_right / scale
        d_left, d_right = d_left * (left_scale / scale), d_right * (right_scale / scale)
        delta = y_right - y_left
        coefficient_1 = h_interval * d_left
        right_step = h_interval * d_right
        coefficient_3 = (coefficient_1 - delta) + (right_step - delta)
        coefficient_2 = delta - coefficient_1 - coefficient_3
        remaining = torch.where(
            scale > 1, (x_i[indices + 1] - x_flat)[:, None] / h_interval, 1 - t
        )
        right_coefficient_2 = -delta + right_step + coefficient_3

        if nu == 0:
            # Anchor at the nearer original ordinate: tiny knot values must
            # survive next to huge neighbours without subtractive cancellation.
            ordinary = torch.where(
                t <= 0.5,
                coefficient_3 * t**3 + coefficient_2 * t**2 + coefficient_1 * t,
                -coefficient_3 * remaining**3
                + right_coefficient_2 * remaining**2
                - h_interval * d_right * remaining,
            )
            u = torch.where(t <= 0.5, t, remaining) * 256
            first_coefficient = torch.where(
                t <= 0.5, coefficient_1, -h_interval * d_right
            )
            second_coefficient = torch.where(
                t <= 0.5, coefficient_2, right_coefficient_2
            )
            third_coefficient = torch.where(t <= 0.5, coefficient_3, -coefficient_3)
            horner = (
                (third_coefficient / 256**3 * u + second_coefficient / 256**2) * u
                + first_coefficient / 256
            ) * u
            values = torch.where(scale > 1, horner, ordinary)
        elif nu == 1:
            values = (
                torch.where(
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
                torch.where(
                    t <= 0.5,
                    6 * coefficient_3 * t + 2 * coefficient_2,
                    -6 * coefficient_3 * remaining + 2 * right_coefficient_2,
                )
                / h_interval**2
            )
        elif nu == 3:
            values = 6 * coefficient_3 / h_interval**3
        else:
            values = torch.zeros_like(coefficient_3)

        values = values * scale
        if nu == 0:
            values = values + torch.where(t <= 0.5, y_i[indices], y_i[indices + 1])
        values = values.reshape((*x.shape, *self._values_shape[1:]))
        values = _restore_axis(values, x.ndim, self.axis)

        if not extrapolate:
            outside = (x < x_i[0]) | (x > x_i[-1])
            remaining_ndim = self._y.ndim - 1
            outside = outside.reshape((*outside.shape, *((1,) * remaining_ndim)))
            outside = _restore_axis(outside, x.ndim, self.axis)
            values = torch.where(
                outside,
                torch.as_tensor(float("nan"), dtype=values.dtype, device=values.device),
                values,
            )

        return values

    @property
    def x(self) -> torch.Tensor:
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property; recomputes the derivatives."""

        self._set_data(value, self._y, reference=self._y)

    @property
    def y(self) -> torch.Tensor:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; recomputes the derivatives."""

        self._set_data(self._x, value, reference=self._x)

    @property
    def extrapolate(self) -> bool | Literal["periodic"] | None:
        """Getter and setter for extrapolation outside the interpolation range."""

        return self._extrapolate

    @extrapolate.setter
    def extrapolate(self, value: bool | Literal["periodic"] | None) -> None:
        """Set extrapolation; assigned *None* masks out-of-domain queries."""

        self._extrapolate = value

    @property
    def dtype(self) -> torch.dtype:
        """Getter for the data dtype."""

        return self._y.dtype


class LinearInterpolator:
    """Linear interpolant over *PyTorch* inputs; raises outside the range."""

    def __init__(
        self,
        x: Any,
        y: Any,
        dtype: Any = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        self._dtype_arg = dtype
        self._y = torch.atleast_1d(
            _as_float(y, dtype, like=x if isinstance(x, torch.Tensor) else None)
        )
        self._x = torch.atleast_1d(_as_float(x, dtype, like=self._y))
        validate_dimensions(self._x, self._y)

    def __call__(self, x: Any) -> torch.Tensor:
        """Evaluate the linear interpolant at the specified point(s)."""

        x = _as_float(x, like=self._y)
        _validate_range(x, self._x)

        if len(self._x) == 1:
            return torch.broadcast_to(
                self._y[0], (*x.shape, *self._y.shape[1:])
            ).clone()

        i = torch.clamp(torch.searchsorted(self._x, x) - 1, 0, len(self._x) - 2)
        t = (x - self._x[i]) / (self._x[i + 1] - self._x[i])
        if self._y.ndim > 1:
            t = t[..., None]

        return self._y[i] + (self._y[i + 1] - self._y[i]) * t

    @property
    def x(self) -> torch.Tensor:
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property."""

        x = torch.atleast_1d(_as_float(value, self._dtype_arg, like=self._y))
        validate_dimensions(x, self._y)
        self._x = x

    @property
    def y(self) -> torch.Tensor:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property."""

        y = torch.atleast_1d(_as_float(value, self._dtype_arg, like=self._x))
        validate_dimensions(self._x, y)
        self._x = _colocate(self._x, y)
        self._y = y

    @property
    def dtype(self) -> torch.dtype:
        """Getter for the data dtype."""

        return self._y.dtype


class NearestNeighbourInterpolator:
    """Nearest-neighbour interpolant over *PyTorch* inputs."""

    def __init__(
        self,
        x: Any,
        y: Any,
        dtype: Any = None,
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        self._dtype_arg = dtype
        self._y = _as_float(y, dtype, like=x if isinstance(x, torch.Tensor) else None)
        self._x = _as_float(x, dtype, like=self._y)
        validate_dimensions(self._x, self._y)

    def __call__(self, x: Any) -> torch.Tensor:
        """Evaluate the nearest-neighbour interpolant at the specified point(s)."""

        x = _as_float(x, like=self._y)
        _validate_range(x, self._x)

        right = torch.clamp(torch.searchsorted(self._x, x), 0, len(self._x) - 1)
        left = torch.clamp(right - 1, 0, len(self._x) - 1)
        choose_left = torch.abs(x - self._x[left]) <= torch.abs(self._x[right] - x)

        return self._y[torch.where(choose_left, left, right)]

    @property
    def x(self) -> torch.Tensor:
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property."""

        x = _as_float(value, self._dtype_arg, like=self._y)
        validate_dimensions(x, self._y)
        self._x = x

    @property
    def y(self) -> torch.Tensor:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property."""

        y = _as_float(value, self._dtype_arg, like=self._x)
        validate_dimensions(self._x, y)
        self._x = _colocate(self._x, y)
        self._y = y

    @property
    def dtype(self) -> torch.dtype:
        """Getter for the data dtype."""

        return self._y.dtype


class NullInterpolator:
    """Null interpolant over *PyTorch* inputs (exact matches within tolerance)."""

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
        self._y = torch.atleast_1d(
            _as_float(y, dtype, like=x if isinstance(x, torch.Tensor) else None)
        )
        self._x = torch.atleast_1d(_as_float(x, dtype, like=self._y))
        validate_dimensions(self._x, self._y)
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.default = default

    def __call__(self, x: Any) -> torch.Tensor:
        """Evaluate the null interpolant at the specified point(s)."""

        x = _as_float(x, like=self._y)
        _validate_range(x, self._x)

        right = torch.clamp(torch.searchsorted(self._x, x), 0, len(self._x) - 1)
        left = torch.clamp(right - 1, 0, len(self._x) - 1)
        distance_left = torch.abs(x - self._x[left])
        distance_right = torch.abs(self._x[right] - x)
        choose_left = distance_left <= distance_right
        nearest = torch.where(choose_left, left, right)
        distance = torch.where(choose_left, distance_left, distance_right)

        tolerance = self.absolute_tolerance + self.relative_tolerance * torch.abs(
            self._x[nearest]
        )
        matched = distance <= tolerance
        if self._y.ndim > 1:
            matched = matched[..., None]

        return torch.where(
            matched,
            self._y[nearest],
            torch.as_tensor(self.default, dtype=self._y.dtype, device=self._y.device),
        )

    @property
    def x(self) -> torch.Tensor:
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property."""

        x = torch.atleast_1d(_as_float(value, self._dtype_arg, like=self._y))
        validate_dimensions(x, self._y)
        self._x = x

    @property
    def y(self) -> torch.Tensor:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property."""

        y = torch.atleast_1d(_as_float(value, self._dtype_arg, like=self._x))
        validate_dimensions(self._x, y)
        self._x = _colocate(self._x, y)
        self._y = y

    @property
    def dtype(self) -> torch.dtype:
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
        self._y = _as_float(y, dtype, like=x if isinstance(x, torch.Tensor) else None)
        self._x = _as_float(x, dtype, like=self._y)
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
        unit = torch.ones((), dtype=y.dtype, device=y.device)
        self._y_scale = torch.where(
            torch.max(torch.abs(y)) > torch.finfo(y.dtype).max / 65536,
            unit * 65536,
            unit,
        )
        y = y / self._y_scale
        h = (x[-1] - x[0]) / (x.shape[0] - 1)
        self._x_p = torch.cat(
            [
                torch.stack([x[0] - 2 * h, x[0] - h]),
                x,
                torch.stack([x[-1] + h, x[-1] + 2 * h]),
            ]
        )

        coefficients = torch.as_tensor(
            SPRAGUE_C_COEFFICIENTS, dtype=y.dtype, device=y.device
        )
        if y.ndim == 2:
            coefficients = coefficients[..., None]
        windows = torch.stack([y[0:6], y[0:6], y[-6:], y[-6:]])
        # Preserve the common ordinate offset before weighted cancellation.
        large = self._y_scale > 1
        origins = torch.where(large, windows[:, 0], 0)
        slope = torch.where(large, (windows[:, -1] - windows[:, 0]) / 5, 0)
        ramp = torch.arange(6, dtype=y.dtype, device=y.device)
        positions = torch.as_tensor([-2, -1, 6, 7], dtype=y.dtype, device=y.device)
        if y.ndim == 2:
            ramp = ramp[:, None]
            positions = positions[:, None]
        residual = windows - origins[:, None] - slope[:, None] * ramp[None, ...]
        y_boundary = (
            origins + slope * positions + (coefficients * residual).sum(dim=1) / 209
        )
        self._y_p = torch.cat([y_boundary[:2], y, y_boundary[2:]], dim=0)

    def __call__(self, x: Any) -> torch.Tensor:
        """Evaluate the *Sprague (1880)* interpolant at the specified point(s)."""

        x = _as_float(x, like=self._y)
        _validate_range(torch.atleast_1d(x), self._x)

        x_flat = torch.atleast_1d(x).reshape(-1)
        large = self._y_scale > 1
        i = torch.searchsorted(self._x_p, x_flat) - 1
        i = torch.where(
            large,
            torch.clamp(
                torch.searchsorted(self._x_p, x_flat, right=True) - 1, 2, len(self._x)
            ),
            i,
        )
        t = (x_flat - self._x_p[i]) / (self._x_p[i + 1] - self._x_p[i])
        reverse = large & (t > 0.5)
        anchor_index = torch.where(reverse, i + 1, i)
        t = torch.where(reverse, 1 - t, t)

        windows = torch.stack([self._y_p[i + k] for k in (-2, -1, 0, 1, 2, 3)])
        reverse_b = reverse[..., None] if self._y.ndim == 2 else reverse
        windows = torch.where(
            reverse_b[None, ...], torch.flip(windows, dims=(0,)), windows
        )
        weights = torch.as_tensor(
            SPRAGUE_A_COEFFICIENTS, dtype=self._y.dtype, device=self._y.device
        )
        windows_shape = windows.shape
        large = self._y_scale > 1
        slope = torch.where(large, (windows[3] - windows[1]) / 2, 0)
        curvature = torch.where(
            large, (windows[3] - 2 * windows[2] + windows[1]) / 2, 0
        )
        offsets = torch.as_tensor(
            [-2, -1, 0, 1, 2, 3], dtype=windows.dtype, device=windows.device
        )
        offsets = offsets.reshape((6,) + (1,) * (windows.ndim - 1))
        centered = (
            windows
            - torch.where(large, windows[2], 0)[None, ...]
            - slope[None, ...] * offsets
            - curvature[None, ...] * offsets**2
        )
        a = (weights @ centered.reshape(windows_shape[0], -1) / 24).reshape(
            weights.shape[0], *windows_shape[1:]
        )

        a = torch.cat(
            [a[:1] + slope[None, ...], a[1:2] + curvature[None, ...], a[2:]], dim=0
        )

        powers = torch.as_tensor(
            [[1.0], [2.0], [3.0], [4.0], [5.0]],
            dtype=self._y.dtype,
            device=self._y.device,
        )
        basis = t**powers
        if self._y.ndim == 2:
            basis = basis[..., None]
        ordinary = self._y_p[i] + (a * basis).sum(dim=0)
        # Horner evaluation about the nearer original node keeps both tiny
        # knot values and reverse-mode query derivatives representable.
        t_b = (t[..., None] if self._y.ndim == 2 else t) * 256
        # Binary coordinate scaling keeps individual reverse-mode derivative
        # terms small until they have cancelled, then restores the query scale.
        coordinate_powers = torch.as_tensor(
            [256.0**k for k in range(1, 6)], dtype=a.dtype, device=a.device
        )
        coordinate_powers = coordinate_powers.reshape((5,) + (1,) * (a.ndim - 1))
        horner_a = a / coordinate_powers
        correction = horner_a[-1]
        for coefficient in reversed(horner_a[:-1]):
            correction = correction * t_b + coefficient
        correction = correction * t_b
        anchor = self._y[torch.clamp(anchor_index - 2, 0, len(self._y) - 1)]
        values = torch.where(
            large, anchor + correction * self._y_scale, ordinary * self._y_scale
        )
        return values.reshape((*x.shape, *self._y.shape[1:]))

    @property
    def x(self) -> torch.Tensor:
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property; rebuilds the boundary extension."""

        x = _as_float(value, self._dtype_arg, like=self._y)
        validate_dimensions(x, self._y)
        self._x = x
        self._solve()

    @property
    def y(self) -> torch.Tensor:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; rebuilds the boundary extension."""

        y = _as_float(value, self._dtype_arg, like=self._x)
        validate_dimensions(self._x, y)
        self._x = _colocate(self._x, y)
        self._y = y
        self._solve()

    @property
    def dtype(self) -> torch.dtype:
        """Getter for the data dtype."""

        return self._y.dtype


def _reflect_pad(y: torch.Tensor, window: int) -> torch.Tensor:
    """
    Reflect-pad the leading axis of ``y`` by ``window`` on each side, matching
    ``np.pad(mode="reflect")`` for any width.

    The integer-index gather keeps the padded values' autodiff graph with
    respect to ``y``.
    """

    indices = torch.as_tensor(
        reflect_indices(y.shape[0], window), device=y.device, dtype=torch.long
    )

    return y[indices]


class KernelInterpolator:
    """
    Native kernel-based (convolution) interpolant over *PyTorch* inputs.

    Evaluation preserves autodiff with respect to the dependent values (through
    the gathered padded values) and the evaluation points (through the smooth
    kernel weights). Uniform ``x`` spacing is assumed. Only reflect padding with
    a ``pad_width`` equal to ``window`` is supported; other ``padding_kwargs``
    raise.
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

        self._y = _as_float(y, dtype, like=x if isinstance(x, torch.Tensor) else None)
        self._x = _as_float(x, dtype, like=self._y)
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

    def __call__(self, x: Any) -> torch.Tensor:
        """Evaluate the interpolator at the specified point(s)."""

        x = _as_float(x, like=self._y)
        values = self._evaluate(torch.atleast_1d(x))

        return values[0] if x.ndim == 0 else values

    def _evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the interpolating convolution at the specified point(s)."""

        if bool(torch.any(x < self._x[0])) or bool(torch.any(x > self._x[-1])):
            error = "A value in x is outside the interpolation range."
            raise ValueError(error)

        w = int(self._window)
        interval = self._interval
        clip_l = self._x_p_min / interval
        clip_h = self._x_p_max / interval

        offsets = torch.arange(-w + 1, w + 1, device=x.device, dtype=x.dtype)
        windows = torch.floor(x / interval)[..., None] + offsets
        windows = torch.clamp(windows, clip_l, clip_h) - clip_l
        windows_i = torch.round(windows).to(torch.long)

        weights = self._kernel(
            x[..., None] / interval - windows_i.to(x.dtype) - self._x_p_min / interval,
            **self._kernel_kwargs,
        )
        if self._y.ndim > 1:
            weights = weights.reshape(weights.shape + (1,) * (self._y.ndim - 1))

        return torch.sum(self._y_p[windows_i] * weights, dim=x.ndim)

    @property
    def x(self) -> torch.Tensor:
        """Getter and setter for the independent :math:`x` variable."""

        return self._x

    @x.setter
    def x(self, value: Any) -> None:
        """Setter for the **self.x** property; rebuilds the padded values."""

        x = _as_float(value, self._dtype_arg, like=self._y)
        validate_dimensions(x, self._y)
        self._x = x
        self._build()

    @property
    def y(self) -> torch.Tensor:
        """Getter and setter for the dependent :math:`y` variable."""

        return self._y

    @y.setter
    def y(self, value: Any) -> None:
        """Setter for the **self.y** property; rebuilds the padded values."""

        y = _as_float(value, self._dtype_arg, like=self._x)
        validate_dimensions(self._x, y)
        self._x = _colocate(self._x, y)
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
    def dtype(self) -> torch.dtype:
        """Getter for the data dtype."""

        return self._y.dtype


def _safe_divide(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Divide ``a`` by ``b``, returning 0 where ``b`` is 0 (degenerate knots)."""

    b_safe = torch.where(b == 0, torch.ones_like(b), b)

    return torch.where(b == 0, torch.zeros_like(a + b), a / b_safe)


class Extrapolator:
    """
    Native extrapolator wrapping an interpolator over *PyTorch* inputs.

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
                torch.tensor([-torch.inf, torch.inf]),
                torch.tensor([-torch.inf, torch.inf]),
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

    def __call__(self, x: Any) -> torch.Tensor:
        """Evaluate the extrapolator at the specified point(s)."""

        xi = self._interpolator.x
        yi = self._interpolator.y
        x = _as_float(x, like=yi)

        return self._evaluate(x, xi, yi)

    def _evaluate(
        self, x: torch.Tensor, xi: torch.Tensor, yi: torch.Tensor
    ) -> torch.Tensor:
        """Perform the extrapolating evaluation at the specified point(s)."""

        input_rank = yi.ndim
        if input_rank == 1:
            yi = yi[..., None]

        below = x < xi[0]
        above = x > xi[-1]
        in_range = torch.logical_and(x >= xi[0], x <= xi[-1])

        y = torch.zeros_like(x[..., None] + yi[0])
        below_b = below[..., None]
        above_b = above[..., None]
        # Fixed overrides and inactive sides must not record infinite offsets:
        # their zero reverse-mode cotangents would otherwise multiply infinity.
        x_offset_low = torch.where(below & (self._left is None), x - xi[0], 0)[
            ..., None
        ]
        x_offset_high = torch.where(above & (self._right is None), x - xi[-1], 0)[
            ..., None
        ]

        if self._method == "linear":
            y = torch.where(
                below_b,
                yi[0] + x_offset_low * _safe_divide(yi[1] - yi[0], xi[1] - xi[0]),
                y,
            )
            y = torch.where(
                above_b,
                yi[-1] + x_offset_high * _safe_divide(yi[-1] - yi[-2], xi[-1] - xi[-2]),
                y,
            )
        elif self._method == "constant":
            y = torch.where(below_b, yi[0], y)
            y = torch.where(above_b, yi[-1], y)

        if self._left is not None:
            y = torch.where(
                below_b, torch.as_tensor(float(self._left), dtype=y.dtype), y
            )
        if self._right is not None:
            y = torch.where(
                above_b, torch.as_tensor(float(self._right), dtype=y.dtype), y
            )

        if bool(torch.any(in_range)):
            x_ravel = x.reshape(-1)
            in_range_ravel = in_range.reshape(-1)
            y_ravel = y.reshape(-1, yi.shape[1])

            interpolated = torch.atleast_1d(self._interpolator(x_ravel[in_range_ravel]))
            if interpolated.ndim == 1:
                interpolated = interpolated[..., None]

            dense_idx = torch.cumsum(in_range_ravel.to(torch.long), dim=0) - 1
            safe_idx = torch.clamp(dense_idx, 0, interpolated.shape[0] - 1)
            y_ravel = torch.where(
                in_range_ravel[..., None], interpolated[safe_idx], y_ravel
            )
            y = y_ravel.reshape((*x.shape, yi.shape[1]))

        if input_rank == 1:
            y = y[..., 0]

        return y
