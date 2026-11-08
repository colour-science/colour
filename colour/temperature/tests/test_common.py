"""Define the unit tests for the :mod:`colour.temperature.common` module."""

from __future__ import annotations

import typing
import warnings
from contextlib import nullcontext

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType, NDArrayFloat

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import (
    CCT_to_uv_Krystek1985,
    CCT_to_xy_Kang2002,
    solve_CCT_Newton,
    solve_xy_Newton,
    x0_CCT_grid,
    xy_to_CCT_McCamy1992,
)
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

        scalar = xp_as_array(6500.0, xp=xp)
        result = solve_CCT_Newton(_forward_linear, _forward_linear(scalar))
        assert result.shape == ()
        if xp.__name__ == "numpy":
            assert isinstance(result, np.floating)
        elif hasattr(scalar, "device"):
            assert result.device == scalar.device

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

        native_guess = xp_as_array(6500.0, xp=xp)
        with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
            result = solve_CCT_Newton(
                CCT_to_uv_Krystek1985, [0.2, 0.31], x0=native_guess
            )
        xp_assert_close(
            result,
            solve_CCT_Newton(CCT_to_uv_Krystek1985, [0.2, 0.31]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        if hasattr(native_guess, "device"):
            assert result.device == native_guess.device

    def test_n_dimensional_solve_CCT_Newton(
        self, xp: ModuleType, request: pytest.FixtureRequest
    ) -> None:
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

        result = solve_CCT_Newton(
            _forward_linear, xp_as_array(np.empty((2, 0, 2)), xp=xp)
        )
        assert result.shape == (2, 0)

        # Exact on-locus and off-locus stationary initial solutions need the
        # inverse sensitivity, including the residual-curvature Hessian term.
        def curved(CCT: NDArrayFloat) -> NDArrayFloat:
            z = (CCT - 6500) / 6500
            return tstack([z, z**2])

        targets = [[0.0, 0.0], [0.0, 0.1], [0.02, 0.1], [0.1, 0.01]]
        roots = []
        for a, b in targets:
            candidates = np.roots([2, 0, 1 - 2 * b, -a])
            roots.append(float(candidates[np.argmin(np.abs(candidates.imag))].real))
        expected = np.asarray(roots)

        def inverse(values: NDArrayFloat) -> NDArrayFloat:
            with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
                return (solve_CCT_Newton(curved, values) - 6500) / 6500

        xp_assert_close(
            inverse(xp_as_array(targets, xp=xp)),
            expected,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        if xp.__name__ != "numpy":
            autodiff = request.getfixturevalue("autodiff")
            _result, (gradient,), _inputs = autodiff(inverse, targets)
            hessian = 1 + 6 * expected**2 - 2 * np.asarray(targets)[:, 1]
            xp_assert_close(
                gradient,
                np.column_stack([1 / hessian, 2 * expected / hessian]),
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

        # Local derivatives must not probe unrelated Kang polynomial pieces
        # or the invalid lower-temperature range from a smooth valid point.
        for seed in (1800.0, 2000.0, 2330.0, 3500.0, 4500.0):
            native_target = CCT_to_xy_Kang2002(xp_as_array(seed, xp=xp))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ColourUsageWarning)
                if xp.__name__ == "numpy":
                    solve_CCT_Newton(CCT_to_xy_Kang2002, native_target, x0=seed)
                else:
                    autodiff(
                        lambda values, seed=seed: (
                            solve_CCT_Newton(CCT_to_xy_Kang2002, values, x0=seed) - seed
                        )
                        / seed,
                        native_target,
                    )
            assert not caught

        if xp.__name__ != "numpy":

            def parameter_inverse(parameter: NDArrayFloat) -> NDArrayFloat:
                def forward(CCT: NDArrayFloat) -> NDArrayFloat:
                    z = (CCT - 6500) / 6500
                    return tstack([parameter * z, z**2])

                return (
                    solve_CCT_Newton(
                        forward,
                        [0.1, 0.01],
                        x0=xp_as_array(7150.0, xp=xp, like=parameter),
                    )
                    - 6500
                ) / 6500

            _result, (gradient,), _inputs = autodiff(parameter_inverse, 1.0)
            xp_assert_close(gradient, -0.1 / 1.04, atol=TOLERANCE_ABSOLUTE_TESTS)

            # A captured parameter can affect a constant map without affecting
            # its coordinate derivatives; unused autograd inputs are zeros.
            autodiff(
                lambda parameter: solve_CCT_Newton(
                    lambda CCT: tstack(
                        [
                            xp.ones_like(CCT) * parameter,
                            xp.zeros_like(CCT),
                        ]
                    ),
                    [0.5, 0.0],
                    x0=xp_as_array(6500.0, xp=xp, like=parameter),
                ),
                0.5,
            )


class TestSolve_xy_Newton:
    """
    Define :func:`colour.temperature.common.solve_xy_Newton` definition
    unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1e-1)
    def test_solve_xy_Newton(
        self, xp: ModuleType, request: pytest.FixtureRequest
    ) -> None:
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

        for key in ("x0", "reference_xy"):
            native_anchor = xp_as_array([0.3127, 0.329], xp=xp)
            with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
                result = (
                    solve_xy_Newton(xy_to_CCT_McCamy1992, 6500.0, x0=native_anchor)
                    if key == "x0"
                    else solve_xy_Newton(
                        xy_to_CCT_McCamy1992, 6500.0, reference_xy=native_anchor
                    )
                )
            xp_assert_close(
                xy_to_CCT_McCamy1992(result), 6500.0, atol=TOLERANCE_ABSOLUTE_TESTS
            )
            if hasattr(native_anchor, "device"):
                assert result.device == native_anchor.device

        if xp.__name__ != "numpy":
            autodiff = request.getfixturevalue("autodiff")
            autodiff(
                lambda anchor: solve_xy_Newton(
                    _forward_xy_linear, 1.0, reference_xy=anchor
                ),
                [0.3, 0.3],
            )

    def test_n_dimensional_solve_xy_Newton(
        self, xp: ModuleType, request: pytest.FixtureRequest
    ) -> None:
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

        result = solve_xy_Newton(
            _forward_xy_linear, xp_as_array(np.empty((2, 0)), xp=xp)
        )
        assert result.shape == (2, 0, 2)

        def curved(xy: NDArrayFloat) -> NDArrayFloat:
            x, y = xy[..., 0], xy[..., 1]
            return x**2 + x * y + y**2

        # At (.3, .4), J=(1, 1.1). The second anchor makes the nonzero
        # residual -.1 exactly balance the regularisation gradient.
        targets = [0.37, 0.47, 0.5]
        anchors = [[0.3, 0.4], [0.2, 0.29], [0.2, 0.29]]

        def inverse(values: NDArrayFloat) -> NDArrayFloat:
            with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
                return solve_xy_Newton(
                    curved,
                    values,
                    x0=(0.3, 0.4),
                    reference_xy=anchors,
                    reference_weight=1.0,
                )

        result = inverse(xp_as_array(targets, xp=xp))
        xp_assert_close(
            result[:2], [[0.3, 0.4], [0.3, 0.4]], atol=TOLERANCE_ABSOLUTE_TESTS
        )
        if xp.__name__ != "numpy":
            autodiff = request.getfixturevalue("autodiff")
            jacobian = np.asarray([1.0, 1.1])
            for index in range(2):
                _result, (gradient,), _inputs = autodiff(
                    lambda values, index=index: inverse(values)[..., index], targets
                )
                for row, residual in enumerate((0.0, -0.1)):
                    hessian = (
                        np.outer(jacobian, jacobian)
                        + np.eye(2)
                        + residual * np.asarray([[2.0, 1.0], [1.0, 2.0]])
                    )
                    xp_assert_close(
                        gradient[row],
                        np.linalg.solve(hessian, jacobian)[index],
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )

        # A real temperature forward has J.T J many orders above the default
        # regularisation. Its rank-one Hessian must retain the small anchor
        # term without catastrophic determinant cancellation at native32.
        if xp.__name__ != "numpy":
            native_target = xy_to_CCT_McCamy1992(xp_as_array([0.3127, 0.329], xp=xp))
            n = (0.3127 - 0.3320) / (0.329 - 0.1858)
            derivative = -1347 * n**2 + 7050 * n - 6823.3
            jacobian = derivative * np.asarray(
                [1 / (0.329 - 0.1858), -n / (0.329 - 0.1858)]
            )
            sensitivity = jacobian / (np.sum(jacobian**2) + 1e-6)
            for index in range(2):
                _result, (gradient,), _inputs = autodiff(
                    lambda target, index=index: solve_xy_Newton(
                        xy_to_CCT_McCamy1992, target
                    )[..., index]
                    * 1e5,
                    native_target,
                )
                xp_assert_close(
                    gradient, sensitivity[index] * 1e5, atol=TOLERANCE_ABSOLUTE_TESTS
                )

        if xp.__name__ != "numpy":

            def parameter_inverse(parameter: NDArrayFloat) -> NDArrayFloat:
                return solve_xy_Newton(
                    lambda xy: parameter * curved(xy),
                    0.37,
                    x0=xp_as_array([0.3, 0.4], xp=xp, like=parameter),
                    reference_xy=[0.3, 0.4],
                    reference_weight=1.0,
                )

            for index in range(2):
                _result, (gradient,), _inputs = autodiff(
                    lambda parameter, index=index: parameter_inverse(parameter)[index],
                    1.0,
                )
                xp_assert_close(
                    gradient,
                    -0.37 * [1.0, 1.1][index] / 3.21,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
