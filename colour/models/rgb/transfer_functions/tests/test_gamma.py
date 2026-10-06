"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.gamma` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import gamma_function
from colour.utilities import (
    as_ndarray,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_assert_equal,
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
    "TestGammaFunction",
]


class TestGammaFunction:
    """
    Define :func:`colour.models.rgb.transfer_functions.gamma.gamma_function`
    definition unit tests methods.
    """

    def test_gamma_function(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.gamma.\
gamma_function` definition.
        """

        xp_assert_close(
            gamma_function(xp_as_array(0.0, xp=xp), 2.2),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            gamma_function(xp_as_array(0.18, xp=xp), 2.2),
            0.022993204992707,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            gamma_function(xp_as_array(0.022993204992707, xp=xp), 1.0 / 2.2),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            gamma_function(xp_as_array(-0.18, xp=xp), 2.0),
            0.0323999999999998,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_equal(gamma_function(xp_as_array(-0.18, xp=xp), 2.2), np.nan)

        xp_assert_close(
            gamma_function(xp_as_array(-0.18, xp=xp), 2.2, "Mirror"),
            -0.022993204992707,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            gamma_function(xp_as_array(-0.18, xp=xp), 2.2, "Preserve"),
            -0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            gamma_function(xp_as_array(-0.18, xp=xp), 2.2, "Clamp"),
            0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_equal(gamma_function(xp_as_array(-0.18, xp=xp), -2.2), np.nan)

        xp_assert_close(
            gamma_function(xp_as_array(0.0, xp=xp), -2.2, "Mirror"),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            gamma_function(xp_as_array(0.0, xp=xp), 2.2, "Preserve"),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            gamma_function(xp_as_array(0.0, xp=xp), 2.2, "Clamp"),
            0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_gamma_function(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.gamma.\
gamma_function` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = as_ndarray(gamma_function(xp_as_array(a, xp=xp), 2.2))

        a = xp.tile(xp_as_array(a, xp=xp), (6,))
        a_p = xp.tile(xp_as_array(a_p, xp=xp), (6,))
        xp_assert_close(gamma_function(a, 2.2), a_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(gamma_function(a, 2.2), a_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3, 1), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(gamma_function(a, 2.2), a_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        a = -0.18
        a_p = -0.022993204992707
        xp_assert_close(
            gamma_function(a, 2.2, "Mirror"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp.tile(xp_as_array(a, xp=xp), (6,))
        a_p = xp.tile(xp_as_array(a_p, xp=xp), (6,))
        xp_assert_close(
            gamma_function(a, 2.2, "Mirror"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            gamma_function(a, 2.2, "Mirror"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3, 1), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            gamma_function(a, 2.2, "Mirror"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = -0.18
        a_p = -0.18
        xp_assert_close(
            gamma_function(a, 2.2, "Preserve"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp.tile(xp_as_array(a, xp=xp), (6,))
        a_p = xp.tile(xp_as_array(a_p, xp=xp), (6,))
        xp_assert_close(
            gamma_function(a, 2.2, "Preserve"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            gamma_function(a, 2.2, "Preserve"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3, 1), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            gamma_function(a, 2.2, "Preserve"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = -0.18
        a_p = 0.0
        xp_assert_close(
            gamma_function(a, 2.2, "Clamp"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp.tile(xp_as_array(a, xp=xp), (6,))
        a_p = xp.tile(xp_as_array(a_p, xp=xp), (6,))
        xp_assert_close(
            gamma_function(a, 2.2, "Clamp"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            gamma_function(a, 2.2, "Clamp"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = xp_reshape(xp_as_array(a, xp=xp), (2, 3, 1), xp=xp)
        a_p = xp_reshape(xp_as_array(a_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            gamma_function(a, 2.2, "Clamp"),
            a_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_gamma_function(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.gamma.\
gamma_function` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        gamma_function(cases, cases)
