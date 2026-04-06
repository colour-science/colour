"""Define the unit tests for the :mod:`colour.algebra.extrapolation` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.algebra import (
    CubicSplineInterpolator,
    Extrapolator,
    LinearInterpolator,
    PchipInterpolator,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_ndarray,
    ignore_numpy_errors,
    is_scipy_installed,
    xp_as_array,
    xp_assert_close,
    xp_assert_equal,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestExtrapolator",
]


class TestExtrapolator:
    """
    Define :class:`colour.algebra.extrapolation.Extrapolator` class unit
    tests methods.
    """

    def test_required_attributes(self) -> None:
        """Test the presence of required attributes."""

        required_attributes = ("interpolator",)

        for attribute in required_attributes:
            assert attribute in dir(Extrapolator)

    def test_required_methods(self) -> None:
        """Test the presence of required methods."""

        required_methods = ("__init__",)

        for method in required_methods:  # pragma: no cover
            assert method in dir(Extrapolator)

    def test_interpolator(self) -> None:
        """
        Test :attr:`colour.algebra.extrapolation.Extrapolator.interpolator`
        property.
        """

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([5, 6, 7]), np.array([5, 6, 7]))
        )
        assert isinstance(extrapolator.interpolator, LinearInterpolator)

    def test_method(self) -> None:
        """
        Test :attr:`colour.algebra.extrapolation.Extrapolator.method`
        property.
        """

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([5, 6, 7]), np.array([5, 6, 7]))
        )
        assert extrapolator.method == "linear"

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([5, 6, 7]), np.array([5, 6, 7])),
            method="Constant",
        )
        assert extrapolator.method == "constant"

    def test_left(self) -> None:
        """
        Test :attr:`colour.algebra.extrapolation.Extrapolator.left`
        property.
        """

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([5, 6, 7]), np.array([5, 6, 7])),
            left=0,
        )
        assert extrapolator.left == 0

    def test_right(self) -> None:
        """
        Test :attr:`colour.algebra.extrapolation.Extrapolator.right`
        property.
        """

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([5, 6, 7]), np.array([5, 6, 7])),
            right=0,
        )
        assert extrapolator.right == 0

    def test__call__(self, xp: ModuleType) -> None:
        """
        Test :meth:`colour.algebra.extrapolation.Extrapolator.__call__`
        method.
        """

        if not is_scipy_installed():  # pragma: no cover
            return

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([5, 6, 7]), np.array([5, 6, 7]))
        )
        xp_assert_equal(extrapolator(xp_as_array([4, 8], xp=xp)), (4, 8))
        xp_assert_equal(extrapolator(xp_as_array([4], xp=xp)), [4])

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([3, 4, 5]), np.array([1, 2, 3])),
            method="Constant",
        )
        xp_assert_equal(
            extrapolator(xp_as_array([0.1, 0.2, 8, 9], xp=xp)),
            (1, 1, 3, 3),
        )
        xp_assert_equal(extrapolator(xp_as_array([0.1], xp=xp)), [1.0])

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([3, 4, 5]), np.array([1, 2, 3])),
            method="Constant",
            left=0,
        )
        xp_assert_equal(
            extrapolator(xp_as_array([0.1, 0.2, 8, 9], xp=xp)),
            (0, 0, 3, 3),
        )
        xp_assert_equal(extrapolator(xp_as_array([0.1], xp=xp)), [0])

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([3, 4, 5]), np.array([1, 2, 3])),
            method="Constant",
            right=0,
        )
        xp_assert_equal(
            extrapolator(xp_as_array([0.1, 0.2, 8, 9], xp=xp)),
            (1, 1, 0, 0),
        )
        xp_assert_equal(extrapolator(xp_as_array([9], xp=xp)), [0])

        extrapolator = Extrapolator(
            CubicSplineInterpolator(np.array([3, 4, 5, 6]), np.array([1, 2, 3, 4]))
        )
        xp_assert_close(
            extrapolator(xp_as_array([0.1, 0.2, 8.0, 9.0], xp=xp)),
            (-1.9, -1.8, 6.0, 7.0),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            extrapolator(xp_as_array([9], xp=xp)),
            [7],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        extrapolator = Extrapolator(
            PchipInterpolator(np.array([3, 4, 5]), np.array([1, 2, 3]))
        )
        xp_assert_close(
            extrapolator(xp_as_array([0.1, 0.2, 8.0, 9.0], xp=xp)),
            (-1.9, -1.8, 6.0, 7.0),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        xp_assert_close(
            extrapolator(xp_as_array([9], xp=xp)),
            [7.0],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        x = np.array([3, 4, 5])
        y = np.array([1, 2, 3])
        x_e = xp_as_array([0.1, 0.2, 4.5, 8.0, 9.0], xp=xp)
        reference = as_ndarray(Extrapolator(LinearInterpolator(x, y))(x_e))
        xp_assert_close(
            as_ndarray(Extrapolator(LinearInterpolator(x, np.transpose([y, y])))(x_e)),
            np.transpose([reference, reference]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        reference = as_ndarray(
            Extrapolator(LinearInterpolator(x, y), method="Constant", left=0)(x_e)
        )
        xp_assert_close(
            as_ndarray(
                Extrapolator(
                    LinearInterpolator(x, np.transpose([y, y])),
                    method="Constant",
                    left=0,
                )(x_e)
            ),
            np.transpose([reference, reference]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan__call__(self) -> None:
        """
        Test :method:`colour.algebra.extrapolation.Extrapolator.__call__`
        method nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            extrapolator = Extrapolator(LinearInterpolator(case, case))
            extrapolator(case[0])
