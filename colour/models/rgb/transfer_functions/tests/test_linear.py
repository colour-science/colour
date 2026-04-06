"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.linear` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import linear_function
from colour.utilities import (
    as_ndarray,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestLinearFunction",
]


class TestLinearFunction:
    """
    Define :func:`colour.models.rgb.transfer_functions.linear.\
linear_function` definition unit tests methods.
    """

    def test_linear_function(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.linear.\
linear_function` definition.
        """

        assert as_ndarray(linear_function(xp_as_array(0.0, xp=xp))) == 0.0

        assert as_ndarray(linear_function(xp_as_array(0.18, xp=xp))) == 0.18

        assert as_ndarray(linear_function(xp_as_array(1.0, xp=xp))) == 1.0

    def test_n_dimensional_linear_function(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.linear.\
linear_function` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = as_ndarray(linear_function(xp_as_array(a, xp=xp)))

        a = xp.tile(xp_as_array(a, xp=xp), (6,))
        a_p = xp.tile(xp_as_array(a_p, xp=xp), (6,))
        xp_assert_close(linear_function(a), a_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(linear_function(a), a_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3, 1), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(linear_function(a), a_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_linear_function(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.linear.\
linear_function` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        linear_function(cases)
