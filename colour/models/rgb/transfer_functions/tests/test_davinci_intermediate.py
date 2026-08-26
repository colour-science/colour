"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.\
davinci_intermediate` module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    oetf_DaVinciIntermediate,
    oetf_inverse_DaVinciIntermediate,
)
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
    "TestOetf_DaVinciIntermediate",
    "TestOetf_inverse_DaVinciIntermediate",
]


class TestOetf_DaVinciIntermediate:
    """
    Define :func:`colour.models.rgb.transfer_functions.davinci_intermediate.\
oetf_DaVinciIntermediate` definition unit tests methods.
    """

    def test_oetf_DaVinciIntermediate(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
davinci_intermediate.oetf_DaVinciIntermediate` definition.
        """

        xp_assert_close(
            oetf_DaVinciIntermediate(xp_as_array(-0.01, xp=xp)),
            -0.104442685500000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_DaVinciIntermediate(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_DaVinciIntermediate(xp_as_array(0.18, xp=xp)),
            0.336043272384855,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_DaVinciIntermediate(xp_as_array(1.0, xp=xp)),
            0.513837441116225,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_DaVinciIntermediate(xp_as_array(100.0, xp=xp)),
            0.999999987016872,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_DaVinciIntermediate(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
davinci_intermediate.oetf_DaVinciIntermediate` definition n-dimensional arrays
        support.
        """

        L = 0.18
        V = as_ndarray(oetf_DaVinciIntermediate(xp_as_array(L, xp=xp)))

        L = xp.tile(xp_as_array(L, xp=xp), (6,))
        V = xp.tile(xp_as_array(V, xp=xp), (6,))
        xp_assert_close(oetf_DaVinciIntermediate(L), V, atol=TOLERANCE_ABSOLUTE_TESTS)

        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3), xp=xp)
        V = xp_reshape(xp_as_array(V, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_DaVinciIntermediate(L), V, atol=TOLERANCE_ABSOLUTE_TESTS)

        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3, 1), xp=xp)
        V = xp_reshape(xp_as_array(V, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_DaVinciIntermediate(L), V, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_DaVinciIntermediate(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
davinci_intermediate.oetf_DaVinciIntermediate` definition domain and range
        scale support.
        """

        L = 0.18
        V = as_ndarray(oetf_DaVinciIntermediate(xp_as_array(L, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_DaVinciIntermediate(xp_as_array(L * factor, xp=xp)),
                    V * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_DaVinciIntermediate(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
davinci_intermediate.oetf_DaVinciIntermediate` definition nan support.
        """

        oetf_DaVinciIntermediate(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_DaVinciIntermediate:
    """
    Define :func:`colour.models.rgb.transfer_functions.\
davinci_intermediate.oetf_inverse_DaVinciIntermediate` definition unit tests
    methods.
    """

    def test_oetf_inverse_DaVinciIntermediate(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
davinci_intermediate.oetf_inverse_DaVinciIntermediate` definition.
        """

        xp_assert_close(
            oetf_inverse_DaVinciIntermediate(xp_as_array(-0.104442685500000, xp=xp)),
            -0.01,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_DaVinciIntermediate(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_DaVinciIntermediate(xp_as_array(0.336043272384855, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_DaVinciIntermediate(xp_as_array(0.513837441116225, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_DaVinciIntermediate(xp_as_array(0.999999987016872, xp=xp)),
            100.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_inverse_DaVinciIntermediate(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
davinci_intermediate.oetf_inverse_DaVinciIntermediate` definition n-dimensional
        arrays support.
        """

        V = 0.336043272384855
        L = as_ndarray(oetf_inverse_DaVinciIntermediate(xp_as_array(V, xp=xp)))

        V = xp.tile(xp_as_array(V, xp=xp), (6,))
        L = xp.tile(xp_as_array(L, xp=xp), (6,))
        xp_assert_close(
            oetf_inverse_DaVinciIntermediate(V),
            L,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        V = xp_reshape(xp_as_array(V, xp=xp), (2, 3), xp=xp)
        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            oetf_inverse_DaVinciIntermediate(V),
            L,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        V = xp_reshape(xp_as_array(V, xp=xp), (2, 3, 1), xp=xp)
        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            oetf_inverse_DaVinciIntermediate(V),
            L,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_oetf_inverse_DaVinciIntermediate(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
davinci_intermediate.oetf_inverse_DaVinciIntermediate` definition domain and
        range scale support.
        """

        V = 0.336043272384855
        L = as_ndarray(oetf_inverse_DaVinciIntermediate(xp_as_array(V, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_inverse_DaVinciIntermediate(xp_as_array(V * factor, xp=xp)),
                    L * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_DaVinciIntermediate(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
davinci_intermediate.oetf_inverse_DaVinciIntermediate` definition nan support.
        """

        oetf_inverse_DaVinciIntermediate(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )
