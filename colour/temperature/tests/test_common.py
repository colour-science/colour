"""Define the unit tests for the :mod:`colour.temperature.common` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType, NDArrayFloat

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import solve_CCT_Newton, solve_xy_Newton, x0_CCT_grid
from colour.utilities import (
    ColourUsageWarning,
    tstack,
    xp_as_array,
    xp_assert_close,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestX0_CCT_grid",
    "TestSolve_CCT_Newton",
    "TestSolve_xy_Newton",
]


def _forward_linear(CCT: NDArrayFloat) -> NDArrayFloat:
    """
    Smooth, monotonic synthetic forward where *Newton* converges in a
    single iteration.

    .. math::

        f(T) = \\left(10^{-4}\\,T,\\; 10^{-4}\\,T + 0.1\\right)
    """

    return tstack([CCT * 1.0e-4, CCT * 1.0e-4 + 0.1])


def _forward_rational(CCT: NDArrayFloat) -> NDArrayFloat:
    """
    Highly non-linear synthetic forward with rational structure that
    forces the per-sample backtracking line search to engage from a
    sufficiently distant initial guess.

    .. math::

        f(T) = \\left(\\frac{1000}{T + 500},\\; \\frac{T}{T + 5000}\\right)
    """

    return tstack([1000.0 / (CCT + 500.0), CCT / (CCT + 5000.0)])


def _forward_xy_linear(xy: NDArrayFloat) -> NDArrayFloat:
    """
    Smooth, monotonic synthetic 2-D-to-1-D forward where *Newton* converges
    in a single iteration.

    .. math::

        f(x, y) = x + 2 y
    """

    return xy[..., 0] + 2.0 * xy[..., 1]


def _forward_xy_rational(xy: NDArrayFloat) -> NDArrayFloat:
    """
    Non-linear synthetic 2-D-to-1-D forward with rational structure that
    forces the per-sample backtracking line search to engage from a
    sufficiently distant initial guess.

    .. math::

        f(x, y) = \\frac{1000}{x + y + 0.5}
    """

    return 1000.0 / (xy[..., 0] + xy[..., 1] + 0.5)


class TestX0_CCT_grid:
    """
    Define :func:`colour.temperature.common.x0_CCT_grid` definition unit
    tests methods.
    """

    def test_x0_CCT_grid(self, xp: ModuleType) -> None:
        """Test :func:`colour.temperature.common.x0_CCT_grid` definition."""

        # 25 samples over [1000, 25000] yields a step of 1000 K so the
        # nearest grid sample to the forward image at ``T = 4321 K`` is
        # ``T = 4000 K``.
        target = _forward_linear(xp_as_array([4321.0], xp=xp))
        xp_assert_close(
            x0_CCT_grid(_forward_linear, target, (1000.0, 25000.0), samples=25),
            xp_as_array([4000.0], xp=xp),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # A denser grid lands the initial guess closer to the true root.
        coarse = x0_CCT_grid(_forward_linear, target, (1000.0, 25000.0), samples=25)
        dense = x0_CCT_grid(_forward_linear, target, (1000.0, 25000.0), samples=2401)
        assert abs(coarse.item() - 4321.0) > abs(dense.item() - 4321.0)

    def test_n_dimensional_x0_CCT_grid(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.common.x0_CCT_grid` definition
        n-dimensional arrays support.
        """

        target = _forward_linear(
            xp_as_array([[1500.0, 6500.0], [10000.0, 25000.0]], xp=xp)
        )
        x0 = x0_CCT_grid(_forward_linear, target, (1000.0, 25000.0))
        assert x0.shape == (2, 2)


class TestSolve_CCT_Newton:
    """
    Define :func:`colour.temperature.common.solve_CCT_Newton` definition
    unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1)
    def test_solve_CCT_Newton(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.common.solve_CCT_Newton` definition.
        """

        # Convergence on a smooth monotonic forward.
        for CCT_reference in (1500.0, 6500.0, 15000.0, 25000.0):
            target = _forward_linear(xp_as_array([CCT_reference], xp=xp))
            xp_assert_close(
                solve_CCT_Newton(_forward_linear, target),
                xp_as_array([CCT_reference], xp=xp),
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

        # Convergence on a highly non-linear rational forward; the
        # backtracking line search must engage to keep the iteration in
        # the trust region.
        for CCT_reference in (1500.0, 6500.0, 15000.0):
            target = _forward_rational(xp_as_array([CCT_reference], xp=xp))
            xp_assert_close(
                solve_CCT_Newton(_forward_rational, target),
                xp_as_array([CCT_reference], xp=xp),
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

        # Explicit per-sample initial guess.
        CCT_reference = xp_as_array([1500.0, 6500.0, 15000.0], xp=xp)
        target = _forward_rational(CCT_reference)
        xp_assert_close(
            solve_CCT_Newton(
                _forward_rational,
                target,
                x0=xp_as_array([2000.0, 7000.0, 14000.0], xp=xp),
            ),
            CCT_reference,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # ``tolerance``, ``newton_iterations`` and ``backtrack_iterations``
        # plumbing.
        CCT_reference = xp_as_array([6500.0], xp=xp)
        target = _forward_rational(CCT_reference)
        xp_assert_close(
            solve_CCT_Newton(
                _forward_rational,
                target,
                tolerance=1e-12,
                newton_iterations=50,
                backtrack_iterations=30,
            ),
            CCT_reference,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # ``ColourUsageWarning`` raised when the iteration budget is
        # exhausted before ``tolerance`` is met.
        with pytest.warns(ColourUsageWarning):
            solve_CCT_Newton(
                _forward_linear,
                _forward_linear(np.array(6500.0)),
                newton_iterations=0,
            )

    def test_n_dimensional_solve_CCT_Newton(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.common.solve_CCT_Newton` definition
        n-dimensional arrays support.
        """

        CCT_reference = xp_as_array([[1500.0, 6500.0], [15000.0, 25000.0]], xp=xp)
        target = _forward_linear(CCT_reference)
        xp_assert_close(
            solve_CCT_Newton(_forward_linear, target),
            CCT_reference,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestSolve_xy_Newton:
    """
    Define :func:`colour.temperature.common.solve_xy_Newton` definition
    unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1e-1)
    def test_solve_xy_Newton(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.common.solve_xy_Newton` definition.
        """

        # Convergence on a smooth linear forward; the regularisation picks
        # the unique level-set point closest to ``reference_xy``.
        for target_value in (0.5, 1.0, 1.5):
            target = xp_as_array([target_value], xp=xp)
            xp_assert_close(
                _forward_xy_linear(solve_xy_Newton(_forward_xy_linear, target)),
                target,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

        # Convergence on a non-linear rational forward; the backtracking
        # line search must engage to keep the iteration in the trust
        # region.
        for target_value in (500.0, 1000.0, 1500.0):
            target = xp_as_array([target_value], xp=xp)
            xp_assert_close(
                _forward_xy_rational(solve_xy_Newton(_forward_xy_rational, target)),
                target,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

        # Explicit per-sample ``x0`` and ``reference_xy``.
        target = xp_as_array([1.0, 1.5, 2.0], xp=xp)
        xp_assert_close(
            _forward_xy_linear(
                solve_xy_Newton(
                    _forward_xy_linear,
                    target,
                    x0=xp_as_array([[0.5, 0.5], [0.4, 0.6], [0.3, 0.7]], xp=xp),
                    reference_xy=xp_as_array([0.4, 0.4], xp=xp),
                )
            ),
            target,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # ``reference_weight``, ``tolerance``, ``newton_iterations`` and
        # ``backtrack_iterations`` plumbing.
        target = xp_as_array([1000.0], xp=xp)
        xp_assert_close(
            _forward_xy_rational(
                solve_xy_Newton(
                    _forward_xy_rational,
                    target,
                    reference_weight=1e-8,
                    tolerance=1e-12,
                    newton_iterations=50,
                    backtrack_iterations=30,
                )
            ),
            target,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # ``ColourUsageWarning`` raised when the iteration budget is
        # exhausted before ``tolerance`` is met.
        with pytest.warns(ColourUsageWarning):
            solve_xy_Newton(
                _forward_xy_linear,
                xp_as_array([0.5], xp=xp),
                newton_iterations=0,
            )

    def test_n_dimensional_solve_xy_Newton(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.common.solve_xy_Newton` definition
        n-dimensional arrays support.
        """

        target = xp_as_array([[0.5, 1.0], [1.5, 2.0]], xp=xp)
        xp_assert_close(
            _forward_xy_linear(solve_xy_Newton(_forward_xy_linear, target)),
            target,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
