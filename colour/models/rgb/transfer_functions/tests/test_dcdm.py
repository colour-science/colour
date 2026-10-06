"""
Define the unit tests for the :mod:`colour.models.rgb.transfer_functions.dcdm`
module.
"""

from __future__ import annotations

import typing

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import eotf_DCDM, eotf_inverse_DCDM
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
    "TestEotf_inverse_DCDM",
    "TestEotf_DCDM",
]


class TestEotf_inverse_DCDM:
    """
    Define :func:`colour.models.rgb.transfer_functions.dcdm.eotf_inverse_DCDM`
    definition unit tests methods.
    """

    def test_eotf_inverse_DCDM(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
dcdm.eotf_inverse_DCDM` definition.
        """

        xp_assert_close(
            eotf_inverse_DCDM(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_DCDM(xp_as_array(0.18, xp=xp)),
            0.11281861,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_DCDM(xp_as_array(1.0, xp=xp)),
            0.21817973,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        assert (
            as_ndarray(eotf_inverse_DCDM(xp_as_array(0.18, xp=xp), out_int=True)) == 462
        )

    def test_n_dimensional_eotf_inverse_DCDM(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dcdm.\
eotf_inverse_DCDM` definition n-dimensional arrays support.
        """

        XYZ = 0.18
        XYZ_p = as_ndarray(eotf_inverse_DCDM(xp_as_array(XYZ, xp=xp)))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6,))
        XYZ_p = xp.tile(xp_as_array(XYZ_p, xp=xp), (6,))
        xp_assert_close(eotf_inverse_DCDM(XYZ), XYZ_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3), xp=xp)
        XYZ_p = xp_reshape(xp_as_array(XYZ_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_inverse_DCDM(XYZ), XYZ_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 1), xp=xp)
        XYZ_p = xp_reshape(xp_as_array(XYZ_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_inverse_DCDM(XYZ), XYZ_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_inverse_DCDM(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.\
dcdm.eotf_inverse_DCDM` definition domain and range scale support.
        """

        XYZ = 0.18
        XYZ_p = as_ndarray(eotf_inverse_DCDM(xp_as_array(XYZ, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_inverse_DCDM(xp_as_array(XYZ * factor, xp=xp)),
                    XYZ_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_DCDM(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dcdm.\
eotf_inverse_DCDM` definition nan support.
        """

        eotf_inverse_DCDM(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_DCDM:
    """
    Define :func:`colour.models.rgb.transfer_functions.dcdm.eotf_DCDM`
    definition unit tests methods.
    """

    def test_eotf_DCDM(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dcdm.eotf_DCDM`
        definition.
        """

        xp_assert_close(
            eotf_DCDM(xp_as_array(0.0, xp=xp)), 0.0, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xp_assert_close(
            eotf_DCDM(xp_as_array(0.11281861, xp=xp)),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_DCDM(xp_as_array(0.21817973, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_DCDM(xp_as_array(462, xp=xp), in_int=True),
            0.18,
            atol=TOLERANCE_ABSOLUTE_TESTS * 100,
        )

    def test_n_dimensional_eotf_DCDM(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dcdm.eotf_DCDM`
        definition n-dimensional arrays support.
        """

        XYZ_p = 0.11281861
        XYZ = as_ndarray(eotf_DCDM(xp_as_array(XYZ_p, xp=xp)))

        XYZ_p = xp.tile(xp_as_array(XYZ_p, xp=xp), (6,))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6,))
        xp_assert_close(eotf_DCDM(XYZ_p), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ_p = xp_reshape(xp_as_array(XYZ_p, xp=xp), (2, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_DCDM(XYZ_p), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ_p = xp_reshape(xp_as_array(XYZ_p, xp=xp), (2, 3, 1), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_DCDM(XYZ_p), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_DCDM(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dcdm.eotf_DCDM`
        definition domain and range scale support.
        """

        XYZ_p = 0.11281861
        XYZ = as_ndarray(eotf_DCDM(xp_as_array(XYZ_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_DCDM(xp_as_array(XYZ_p * factor, xp=xp)),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_DCDM(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.dcdm.eotf_DCDM`
        definition nan support.
        """

        eotf_DCDM(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
