"""Define the unit tests for the :mod:`colour.algebra.regression` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import numpy as np

from colour.algebra import least_square_mapping_MoorePenrose
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import xp_as_array, xp_assert_close

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLeastSquareMappingMoorePenrose",
]


class TestLeastSquareMappingMoorePenrose:
    """
    Define :func:`colour.algebra.regression.\
least_square_mapping_MoorePenrose` definition unit tests methods.
    """

    def test_least_square_mapping_MoorePenrose(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.algebra.regression.\
least_square_mapping_MoorePenrose` definition.
        """

        prng = np.random.RandomState(2)
        y_np = prng.random_sample((24, 3))
        x_np = y_np + (prng.random_sample((24, 3)) - 0.5) * 0.5

        xp_assert_close(
            least_square_mapping_MoorePenrose(
                xp_as_array(y_np, xp=xp), xp_as_array(x_np, xp=xp)
            ),
            [
                [1.05263767, 0.13780789, -0.22763399],
                [0.07395843, 1.02939945, -0.10601150],
                [0.05725508, -0.20526336, 1.10151945],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        y_np = prng.random_sample((4, 3, 2))
        x_np = y_np + (prng.random_sample((4, 3, 2)) - 0.5) * 0.5
        xp_assert_close(
            least_square_mapping_MoorePenrose(
                xp_as_array(y_np, xp=xp), xp_as_array(x_np, xp=xp)
            ),
            [
                [[1.07636527, -0.256201], [0.06625818, 0.80475283]],
                [[0.51513719, 0.52756206], [1.87771063, 0.13030182]],
                [[1.16325211, -0.29657976], [0.25479095, 0.92809262]],
                [[1.37286297, -0.49899538], [0.10981647, 0.68105929]],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
