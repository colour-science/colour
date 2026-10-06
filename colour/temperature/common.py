"""
Common Correlated Colour Temperature Utilities
==============================================

Define common utilities for correlated colour temperature :math:`T_{cp}`
computation methods:

-   :func:`colour.temperature.x0_CCT_grid`: Compute a per-sample initial
    guess for :func:`colour.temperature.solve_CCT_Newton` by
    nearest-neighbour lookup against a coarse grid sampled from a forward
    transform.
-   :func:`colour.temperature.solve_CCT_Newton`: Solve correlated colour
    temperature :math:`T_{cp}` from chromaticity coordinates given a
    forward transform using a vectorised *Gauss-Newton* iteration.
-   :func:`colour.temperature.solve_xy_Newton`: Solve *CIE xy* chromaticity
    coordinates from a target correlated colour temperature
    :math:`T_{cp}` given a forward 2-D-to-1-D transform using a
    Tikhonov-regularised vectorised *Gauss-Newton* iteration.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ArrayLike, Callable, NDArrayFloat

from colour.hints import cast
from colour.utilities import (
    array_namespace,
    as_float_array,
    tstack,
    usage_warning,
    xp_as_float_array,
    xp_broadcast_to,
    xp_linspace,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CCT_INVERSION_GRID_SAMPLES",
    "x0_CCT_grid",
    "solve_CCT_Newton",
    "solve_xy_Newton",
]


CCT_INVERSION_GRID_SAMPLES: int = 50
"""
Default number of samples used by :func:`x0_CCT_grid` balancing
basin-coverage against grid construction cost; the resulting spacing
(e.g. ~470 K over the :math:`T_{cp} \\in [1667, 25000] K` *Kang et al.
(2002)* domain) lands the per-pixel initial guess within the basin of
the true root either side of any piecewise discontinuity in the forward,
where the central-difference Jacobian otherwise stagnates.
"""


_JACOBIAN_FLOOR: float = 1e-30
"""*Tikhonov* floor on the squared Jacobian norm ``J . J`` used by
:func:`solve_CCT_Newton` and :func:`solve_xy_Newton` to damp the
*Gauss-Newton* update when the forward goes flat at a domain edge.
Numerical-stability constant; not a user knob."""


def x0_CCT_grid(
    forward: Callable[[NDArrayFloat], NDArrayFloat],
    target: ArrayLike,
    domain: tuple[float, float],
    samples: int = CCT_INVERSION_GRID_SAMPLES,
) -> NDArrayFloat:
    """
    Compute a per-sample initial guess for
    :func:`colour.temperature.solve_CCT_Newton` by nearest-neighbour lookup
    against a coarse linearly-spaced grid of correlated colour temperature
    :math:`T_{cp}` values mapped through the analytical forward.

    Parameters
    ----------
    forward
        Callable mapping a correlated colour temperature :math:`T_{cp}` array
        of shape ``(...,)`` to a chromaticity coordinates array of shape
        ``(..., 2)``.
    target
        Target chromaticity coordinates of shape ``(..., 2)``.
    domain
        Inclusive ``(low, high)`` bounds in kelvins of the linearly-spaced
        grid; should match the published valid domain of ``forward``.
    samples
        Number of grid samples; defaults to
        :attr:`CCT_INVERSION_GRID_SAMPLES`.

    Returns
    -------
    :class:`numpy.ndarray`
        Per-sample initial guess :math:`T_{cp}` of shape ``(...,)``.

    Examples
    --------
    >>> from colour.temperature import CCT_to_xy_Kang2002
    >>> x0_CCT_grid(
    ...     CCT_to_xy_Kang2002,
    ...     [0.31342600, 0.32359597],
    ...     (1667.0, 25000.0),
    ... )  # doctest: +ELLIPSIS
    np.float64(6428.836734...)
    """

    target = as_float_array(target)

    xp = array_namespace(target)

    # ``like=target`` creates the seed grid on the target's device; without
    # it ``xp.linspace`` defaults to the host device and mismatches a
    # device-resident ``target`` (e.g. *PyTorch* on *MPS*).
    grid_CCT = cast(
        "NDArrayFloat",
        xp_linspace(domain[0], domain[1], num=samples, xp=xp, like=target),
    )
    grid_target = forward(grid_CCT)
    distances_squared = xp.sum((target[..., None, :] - grid_target) ** 2, axis=-1)
    return grid_CCT[xp.argmin(distances_squared, axis=-1)]


def solve_CCT_Newton(
    forward: Callable[[NDArrayFloat], NDArrayFloat],
    target: ArrayLike,
    x0: ArrayLike = 6500,
    tolerance: float = 1e-10,
    newton_iterations: int = 30,
    backtrack_iterations: int = 20,
) -> NDArrayFloat:
    """
    Solve the correlated colour temperature :math:`T_{cp}` from the specified
    target chromaticity coordinates using a vectorised damped *Gauss-Newton*
    iteration on the given forward transform.

    Parameters
    ----------
    forward
        Callable mapping a correlated colour temperature :math:`T_{cp}` array
        of shape ``(...,)`` to a chromaticity coordinates array of shape
        ``(..., 2)``.
    target
        Target chromaticity coordinates of shape ``(..., 2)``.
    x0
        Initial guess for the correlated colour temperature :math:`T_{cp}` in
        kelvins. Scalar values broadcast to the leading shape of ``target``;
        an array-like of matching shape ``(...,)`` may be passed when a
        per-sample initial guess is required, for example a
        nearest-neighbour lookup against a coarse grid sampled from the
        analytical forward when the latter is non-monotonic outside its
        valid domain (see
        :func:`colour.temperature.xy_to_CCT_Kang2002`).
    tolerance
        Convergence tolerance on the maximum absolute Newton step.
    newton_iterations
        Maximum number of *Gauss-Newton* outer iterations. The default of
        30 covers the slow tail of the iteration when the
        central-difference Jacobian straddles a piecewise boundary in the
        forward (e.g. *Kang et al. (2002)* at :math:`T_{cp} = 4000 K`).
    backtrack_iterations
        Maximum number of step-halvings performed by the per-sample
        backtracking line search. The default of 20 reduces the step by a
        factor of :math:`2^{-20} \\approx 10^{-6}` in the worst case,
        which is below the convergence ``tolerance`` for any realistic
        :math:`T_{cp}`.

    Returns
    -------
    :class:`numpy.ndarray`
        Correlated colour temperature :math:`T_{cp}` of shape ``(...,)``.

    Notes
    -----
    -   *Gauss-Newton* on the residual :math:`r(T) = forward(T) - target`
        with central-difference Jacobian; the 1-D normal equations
        collapse to :math:`\\delta T = -(J \\cdot r) / (J \\cdot J)`.
    -   A per-sample backtracking line search halves the step until the
        squared residual decreases on every sample, guarding against
        overshoot on highly non-linear forwards (e.g.
        *Krystek (1985)*'s rational polynomial).
    -   A :func:`colour.utilities.usage_warning` is issued if the maximum
        absolute step has not dropped below ``tolerance`` within
        ``newton_iterations`` updates.

    Examples
    --------
    >>> from colour.temperature import CCT_to_uv_Krystek1985
    >>> solve_CCT_Newton(
    ...     CCT_to_uv_Krystek1985, [0.20047203, 0.31029290]
    ... )  # doctest: +ELLIPSIS
    np.float64(6504.389416...)
    """

    target = as_float_array(target)

    xp = array_namespace(target)

    # Carries ``target``'s namespace and device through the broadcast.
    CCT = xp.zeros_like(target[..., 0]) + xp_as_float_array(x0, xp=xp, like=target)

    residual = forward(CCT) - target
    objective = xp.sum(residual * residual, axis=-1)

    # ``tolerance`` bounds an absolute *Newton* step expressed in *Kelvin*: at
    # float32 the spacing near 6500 K is circa 5e-4, so the default 1e-10 could
    # never be attained and the iteration would always exhaust its budget and
    # warn. It is raised to the representable spacing of the working precision.
    tolerance = max(
        tolerance,
        float(xp.finfo(CCT.dtype).eps) * float(xp.max(xp.abs(CCT))),
    )

    converged = False
    for _iteration in range(newton_iterations):
        # Relative step ``1e-5 * |CCT| + 1e-6`` sits near
        # ``epsilon ** (1 / 3) ~= 6.06e-6`` for float64, the central-
        # difference truncation/roundoff balance derived in Dennis &
        # Schnabel (1983), *Numerical Methods for Unconstrained
        # Optimization and Nonlinear Equations*, Section 5.4. The
        # additive ``1e-6`` floor prevents zero-step at ``CCT == 0``.
        h = xp.abs(CCT) * 1e-5 + 1e-6
        jacobian = (forward(CCT + h) - forward(CCT - h)) / (2 * h[..., None])

        # 1-D *Gauss-Newton*: :math:`\\delta T = -(J \\cdot r) /
        # \\max(J \\cdot J, \\lambda)`. The ``\\lambda`` *Tikhonov* floor
        # only kicks in when the Jacobian collapses (flat ``forward`` at
        # a domain edge); without it ``step`` blows up to ``inf`` /
        # ``nan`` and the line search never recovers because
        # ``nan < objective`` is ``False``.
        numerator = xp.sum(jacobian * residual, axis=-1)
        denominator = xp.sum(jacobian * jacobian, axis=-1)
        step = -numerator / xp.where(
            denominator > _JACOBIAN_FLOOR, denominator, _JACOBIAN_FLOOR
        )

        # Backtracking line search. Halve per-sample until every sample's
        # squared residual decreases; guards against the local
        # linearisation overshooting on highly non-linear forwards
        # (e.g. *Krystek (1985)*'s rational polynomial) without
        # sacrificing the quadratic regime inside the trust region.
        # Runs the full iteration count rather than early-exiting on
        # ``xp.all(improved)`` so the inner loop stays free of
        # device-host syncs on ``jax`` / ``torch``; per-sample masking
        # naturally freezes the step once a sample improves.
        for _backtrack in range(backtrack_iterations):
            residual_trial = forward(CCT + step) - target
            objective_trial = xp.sum(residual_trial * residual_trial, axis=-1)
            improved = objective_trial < objective
            step = xp.where(improved, step, step * 0.5)

        CCT = CCT + step
        residual = forward(CCT) - target
        objective = xp.sum(residual * residual, axis=-1)

        # One device-host sync per outer iteration to enable early-exit.
        if bool(xp.max(xp.abs(step)) < tolerance):
            converged = True
            break

    if not converged:
        usage_warning(
            f'"Newton" iteration for "CCT" inversion did not converge to '
            f"tolerance {tolerance:.1e} within {newton_iterations} "
            "iterations."
        )

    return CCT


def solve_xy_Newton(
    forward: Callable[[NDArrayFloat], NDArrayFloat],
    target: ArrayLike,
    x0: ArrayLike = (0.31270, 0.32900),
    reference_xy: ArrayLike = (0.31270, 0.32900),
    reference_weight: float = 1e-6,
    tolerance: float = 1e-10,
    newton_iterations: int = 30,
    backtrack_iterations: int = 20,
) -> NDArrayFloat:
    """
    Solve the *CIE xy* chromaticity coordinates from the specified target
    correlated colour temperature :math:`T_{cp}` using a vectorised damped
    *Gauss-Newton* iteration on the given forward transform with *Tikhonov*
    regularisation toward a reference *CIE xy* anchor.

    Parameters
    ----------
    forward
        Callable mapping a *CIE xy* chromaticity coordinates array of shape
        ``(..., 2)`` to a correlated colour temperature :math:`T_{cp}`
        array of shape ``(...,)``.
    target
        Target correlated colour temperature :math:`T_{cp}` of shape
        ``(...,)``.
    x0
        Initial guess for the *CIE xy* chromaticity coordinates. Scalar
        ``(2,)`` values broadcast to the shape of ``target``; an array-like
        of shape ``target.shape + (2,)`` may be passed when a per-sample
        initial guess is required. Defaults to the *CIE Standard Illuminant
        D65* chromaticity coordinates.
    reference_xy
        Reference *CIE xy* chromaticity coordinates of shape ``(2,)`` or
        ``target.shape + (2,)`` toward which the iteration is biased to
        resolve the rank-deficiency of the inversion. The level set of
        ``forward`` at ``target`` is a curve in the *CIE xy* plane and the
        regularisation picks the point on that curve closest to
        ``reference_xy``. Defaults to the *CIE Standard Illuminant D65*
        chromaticity coordinates.
    reference_weight
        Weight of the *Tikhonov* regularisation term. Should be small
        enough to leave the data fit dominant; the default of ``1e-6``
        works for chromaticity coordinates in the unit interval and
        correlated colour temperatures in :math:`T_{cp} \\in [10^3,
        10^5] K`.
    tolerance
        Convergence tolerance on the maximum absolute Newton step.
    newton_iterations
        Maximum number of *Gauss-Newton* outer iterations.
    backtrack_iterations
        Maximum number of step-halvings performed by the per-sample
        backtracking line search. The default of 20 reduces the step by a
        factor of :math:`2^{-20} \\approx 10^{-6}` in the worst case,
        which is below the convergence ``tolerance`` for any realistic
        *CIE xy* coordinates.

    Returns
    -------
    :class:`numpy.ndarray`
        *CIE xy* chromaticity coordinates of shape ``target.shape + (2,)``.

    Notes
    -----
    -   Solves :math:`\\min_{xy}\\,(forward(xy) - target)^2 +
        \\lambda\\,\\|xy - reference\\|^2` by *Gauss-Newton* update with
        an analytical 2x2 *Hessian* inversion. The :math:`\\lambda I`
        term resolves the rank-1 deficiency of :math:`J^T J` and pulls
        the solution toward ``reference_xy`` along the level set.
    -   A per-sample backtracking line search halves the step until the
        augmented squared residual decreases on every sample, guarding
        against overshoot on highly non-linear forwards (e.g.
        *McCamy (1992)*'s rational ``n``-polynomial).
    -   A :func:`colour.utilities.usage_warning` is issued if the maximum
        absolute step has not dropped below ``tolerance`` within
        ``newton_iterations`` updates.

    Examples
    --------
    >>> from colour.temperature import xy_to_CCT_McCamy1992
    >>> solve_xy_Newton(xy_to_CCT_McCamy1992, 6500.0)  # doctest: +ELLIPSIS
    array([0.312791..., 0.329012...])
    """

    target = as_float_array(target)

    xp = array_namespace(target)

    x0_array = xp_as_float_array(x0, xp=xp, like=target)
    xy = xp_broadcast_to(x0_array, (*target.shape, x0_array.shape[-1]), xp=xp)
    reference_xy = xp_as_float_array(reference_xy, xp=xp, like=target)

    residual = forward(xy) - target
    anchor_residual = xy - reference_xy
    objective = residual * residual + reference_weight * xp.sum(
        anchor_residual * anchor_residual, axis=-1
    )

    # ``tolerance`` bounds an absolute *Newton* step on the chromaticity
    # coordinates; it is raised to the representable spacing of the working
    # precision so that a float32 iteration can converge rather than always
    # exhausting its budget. See :func:`solve_CCT_Newton`.
    tolerance = max(tolerance, float(xp.finfo(xy.dtype).eps))

    converged = False
    for _iteration in range(newton_iterations):
        x = xy[..., 0]
        y = xy[..., 1]
        # See ``solve_CCT_Newton`` for the central-difference step
        # rationale (Dennis & Schnabel 1983, Section 5.4).
        h_x = xp.abs(x) * 1e-5 + 1e-6
        h_y = xp.abs(y) * 1e-5 + 1e-6
        df_dx = (forward(tstack([x + h_x, y])) - forward(tstack([x - h_x, y]))) / (
            2 * h_x
        )
        df_dy = (forward(tstack([x, y + h_y])) - forward(tstack([x, y - h_y]))) / (
            2 * h_y
        )

        # *Tikhonov*-regularised 2-D *Gauss-Newton*. The naive determinant
        # :math:`(J_x^2 + \\lambda)(J_y^2 + \\lambda) - (J_x J_y)^2`
        # cancels to roundoff once :math:`\\|J\\|^2 \\gg \\lambda`; the
        # algebraic identity :math:`\\det H = \\lambda (\\|J\\|^2 +
        # \\lambda)` and the closed-form step below absorb the
        # cancellation analytically.
        anchor_x = x - reference_xy[..., 0]
        anchor_y = y - reference_xy[..., 1]
        cross = df_dy * anchor_x - df_dx * anchor_y
        denominator = df_dx * df_dx + df_dy * df_dy + reference_weight
        step_x = (
            -(df_dx * residual + df_dy * cross + reference_weight * anchor_x)
            / denominator
        )
        step_y = (
            -(df_dy * residual - df_dx * cross + reference_weight * anchor_y)
            / denominator
        )
        step = tstack([step_x, step_y])

        # Backtracking line search on the augmented squared residual.
        # Halve per-sample until every sample's augmented squared
        # residual decreases; guards against overshooting on highly
        # non-linear forwards (e.g. *McCamy (1992)*'s rational
        # ``n``-polynomial). Runs the full iteration count rather than
        # early-exiting on ``xp.all(improved)`` so the inner loop stays
        # free of device-host syncs on ``jax`` / ``torch``; per-sample
        # masking naturally freezes the step once a sample improves.
        for _backtrack in range(backtrack_iterations):
            xy_trial = xy + step
            residual_trial = forward(xy_trial) - target
            anchor_residual_trial = xy_trial - reference_xy
            objective_trial = (
                residual_trial * residual_trial
                + reference_weight
                * xp.sum(anchor_residual_trial * anchor_residual_trial, axis=-1)
            )
            improved = objective_trial < objective
            step = xp.where(improved[..., None], step, step * 0.5)

        xy = xy + step
        residual = forward(xy) - target
        anchor_residual = xy - reference_xy
        objective = residual * residual + reference_weight * xp.sum(
            anchor_residual * anchor_residual, axis=-1
        )

        # One device-host sync per outer iteration to enable early-exit.
        if bool(xp.max(xp.abs(step)) < tolerance):
            converged = True
            break

    if not converged:
        usage_warning(
            f'"Newton" iteration for "xy" inversion did not converge to '
            f"tolerance {tolerance:.1e} within {newton_iterations} "
            "iterations."
        )

    return xy
