"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.itur_bt_601` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import oetf_BT601, oetf_inverse_BT601
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
    "TestOetf_BT601",
    "TestOetf_inverse_BT601",
]


class TestOetf_BT601:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_601.oetf_BT601`
    definition unit tests methods.
    """

    def test_oetf_BT601(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_BT601` definition.
        """

        xp_assert_close(
            oetf_BT601(xp_as_array(0.0, xp=xp)), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xp_assert_close(
            oetf_BT601(xp_as_array(0.015, xp=xp)),
            0.067500000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BT601(xp_as_array(0.18, xp=xp)),
            0.409007728864150,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BT601(xp_as_array(1.0, xp=xp)), 1.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_n_dimensional_oetf_BT601(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_BT601` definition n-dimensional arrays support.
        """

        L = 0.18
        E = as_ndarray(oetf_BT601(xp_as_array(L, xp=xp)))

        L = xp.tile(xp_as_array(L, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(oetf_BT601(L), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_BT601(L), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_BT601(L), E, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_BT601(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_BT601` definition domain and range scale support.
        """

        L = 0.18
        E = as_ndarray(oetf_BT601(xp_as_array(L, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_BT601(xp_as_array(L * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_BT601(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_BT601` definition nan support.
        """

        oetf_BT601(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_BT601:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_inverse_BT601` definition unit tests methods.
    """

    def test_oetf_inverse_BT601(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_inverse_BT601` definition.
        """

        xp_assert_close(
            oetf_inverse_BT601(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BT601(xp_as_array(0.067500000000000, xp=xp)),
            0.015,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BT601(xp_as_array(0.409007728864150, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BT601(xp_as_array(1.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_inverse_BT601(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_inverse_BT601` definition n-dimensional arrays support.
        """

        E = 0.409007728864150
        L = as_ndarray(oetf_inverse_BT601(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        L = xp.tile(xp_as_array(L, xp=xp), (6,))
        xp_assert_close(oetf_inverse_BT601(E), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_inverse_BT601(E), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_inverse_BT601(E), L, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_inverse_BT601(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_inverse_BT601` definition domain and range scale support.
        """

        E = 0.409007728864150
        L = as_ndarray(oetf_inverse_BT601(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_inverse_BT601(xp_as_array(E * factor, xp=xp)),
                    L * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_BT601(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_inverse_BT601` definition nan support.
        """

        oetf_inverse_BT601(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
