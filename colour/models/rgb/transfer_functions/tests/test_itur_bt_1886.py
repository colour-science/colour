"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.itur_bt_1886` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import eotf_BT1886, eotf_inverse_BT1886
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
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
    "TestEotf_inverse_BT1886",
    "TestEotf_BT1886",
]


class TestEotf_inverse_BT1886:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition unit tests methods.
    """

    def test_eotf_inverse_BT1886(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition.
        """

        xp_assert_close(
            eotf_inverse_BT1886(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT1886(xp_as_array(0.016317514686316, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT1886(xp_as_array(1.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_inverse_BT1886(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition n-dimensional arrays support.
        """

        L = 0.016317514686316
        V = as_ndarray(eotf_inverse_BT1886(xp_as_array(L, xp=xp)))

        L = xp.tile(xp_as_array(L, xp=xp), (6,))
        V = xp.tile(xp_as_array(V, xp=xp), (6,))
        xp_assert_close(eotf_inverse_BT1886(L), V, atol=TOLERANCE_ABSOLUTE_TESTS)

        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3), xp=xp)
        V = xp_reshape(xp_as_array(V, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_inverse_BT1886(L), V, atol=TOLERANCE_ABSOLUTE_TESTS)

        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3, 1), xp=xp)
        V = xp_reshape(xp_as_array(V, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_inverse_BT1886(L), V, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_inverse_BT1886(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition domain and range scale support.
        """

        L = 0.18
        V = as_ndarray(eotf_inverse_BT1886(xp_as_array(L, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_inverse_BT1886(xp_as_array(L * factor, xp=xp)),
                    V * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_BT1886(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition nan support.
        """

        eotf_inverse_BT1886(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_BT1886:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition unit tests methods.
    """

    def test_eotf_BT1886(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition.
        """

        xp_assert_close(
            eotf_BT1886(xp_as_array(0.0, xp=xp)), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xp_assert_close(
            eotf_BT1886(xp_as_array(0.18, xp=xp)),
            0.016317514686316,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT1886(xp_as_array(1.0, xp=xp)), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_eotf_BT1886(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition n-dimensional arrays support.
        """

        V = 0.18
        L = as_ndarray(eotf_BT1886(xp_as_array(V, xp=xp)))

        V = xp.tile(xp_as_array(V, xp=xp), (6,))
        L = xp.tile(xp_as_array(L, xp=xp), (6,))
        xp_assert_close(eotf_BT1886(V), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        V = xp_reshape(xp_as_array(V, xp=xp), (2, 3), xp=xp)
        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_BT1886(V), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        V = xp_reshape(xp_as_array(V, xp=xp), (2, 3, 1), xp=xp)
        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_BT1886(V), L, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_BT1886(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition domain and range scale support.
        """

        V = 0.016317514686316
        L = as_ndarray(eotf_BT1886(xp_as_array(V, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_BT1886(xp_as_array(V * factor, xp=xp)),
                    L * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_BT1886(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition nan support.
        """

        eotf_BT1886(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
