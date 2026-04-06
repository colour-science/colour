"""
Define the unit tests for the
:mod:`colour.models.rgb.transfer_functions.itur_bt_2100` module.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models.rgb.transfer_functions import (
    eotf_BT2100_PQ,
    eotf_inverse_BT2100_PQ,
    oetf_BT2100_HLG,
    oetf_BT2100_PQ,
    oetf_inverse_BT2100_HLG,
    oetf_inverse_BT2100_PQ,
    ootf_BT2100_PQ,
    ootf_inverse_BT2100_PQ,
)
from colour.models.rgb.transfer_functions.itur_bt_2100 import (
    eotf_BT2100_HLG_1,
    eotf_BT2100_HLG_2,
    eotf_inverse_BT2100_HLG_1,
    eotf_inverse_BT2100_HLG_2,
    gamma_function_BT2100_HLG,
    ootf_BT2100_HLG,
    ootf_BT2100_HLG_1,
    ootf_BT2100_HLG_2,
    ootf_inverse_BT2100_HLG,
    ootf_inverse_BT2100_HLG_1,
    ootf_inverse_BT2100_HLG_2,
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
    "TestOetf_BT2100_PQ",
    "TestOetf_inverse_BT2100_PQ",
    "TestEotf_BT2100_PQ",
    "TestEotf_inverse_BT2100_PQ",
    "TestOotf_BT2100_PQ",
    "TestOotf_inverse_BT2100_PQ",
    "TestGamma_function_BT2100_HLG",
    "TestOetf_BT2100_HLG",
    "TestOetf_inverse_BT2100_HLG",
    "TestEotf_BT2100_HLG_1",
    "TestEotf_BT2100_HLG_2",
    "TestEotf_inverse_BT2100_HLG_1",
    "TestEotf_inverse_BT2100_HLG_2",
    "TestOotf_BT2100_HLG_1",
    "TestOotf_BT2100_HLG_2",
    "TestOotfBT2100HLG",
    "TestOotf_inverse_BT2100_HLG_1",
    "TestOotf_inverse_BT2100_HLG_2",
    "TestOotfInverseBT2100HLG",
]


class TestOetf_BT2100_PQ:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition unit tests methods.
    """

    def test_oetf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition.
        """

        xp_assert_close(
            oetf_BT2100_PQ(xp_as_array(0.0, xp=xp)),
            0.000000730955903,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BT2100_PQ(xp_as_array(0.1, xp=xp)),
            0.724769816665726,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BT2100_PQ(xp_as_array(1.0, xp=xp)),
            0.999999934308041,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition n-dimensional arrays support.
        """

        E = 0.1
        E_p = as_ndarray(oetf_BT2100_PQ(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(oetf_BT2100_PQ(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_BT2100_PQ(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_BT2100_PQ(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition domain and range scale support.
        """

        E = 0.1
        E_p = as_ndarray(oetf_BT2100_PQ(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_BT2100_PQ(xp_as_array(E * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_BT2100_PQ(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_PQ` definition nan support.
        """

        oetf_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_BT2100_PQ:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition unit tests methods.
    """

    def test_oetf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition.
        """

        xp_assert_close(
            oetf_inverse_BT2100_PQ(xp_as_array(0.000000730955903, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BT2100_PQ(xp_as_array(0.724769816665726, xp=xp)),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BT2100_PQ(xp_as_array(0.999999934308041, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition n-dimensional arrays support.
        """

        E_p = 0.724769816665726
        E = as_ndarray(oetf_inverse_BT2100_PQ(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(oetf_inverse_BT2100_PQ(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_inverse_BT2100_PQ(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_inverse_BT2100_PQ(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition domain and range scale support.
        """

        E_p = 0.724769816665726
        E = as_ndarray(oetf_inverse_BT2100_PQ(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_inverse_BT2100_PQ(xp_as_array(E_p * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_BT2100_PQ(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_PQ` definition nan support.
        """

        oetf_inverse_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_BT2100_PQ:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1e-1)
    def test_eotf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition.
        """

        xp_assert_close(
            eotf_BT2100_PQ(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_PQ(xp_as_array(0.724769816665726, xp=xp)),
            779.98836083408537,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_PQ(xp_as_array(1.0, xp=xp)),
            10000.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition n-dimensional arrays support.
        """

        E_p = 0.724769816665726
        F_D = as_ndarray(eotf_BT2100_PQ(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        xp_assert_close(eotf_BT2100_PQ(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_BT2100_PQ(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_BT2100_PQ(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition domain and range scale support.
        """

        E_p = 0.724769816665726
        F_D = as_ndarray(eotf_BT2100_PQ(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_BT2100_PQ(xp_as_array(E_p * factor, xp=xp)),
                    F_D * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_BT2100_PQ(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_PQ` definition nan support.
        """

        eotf_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_inverse_BT2100_PQ:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition unit tests methods.
    """

    def test_eotf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition.
        """

        xp_assert_close(
            eotf_inverse_BT2100_PQ(xp_as_array(0.0, xp=xp)),
            0.000000730955903,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_PQ(xp_as_array(779.98836083408537, xp=xp)),
            0.724769816665726,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_PQ(xp_as_array(10000.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition n-dimensional arrays support.
        """

        F_D = 779.98836083408537
        E_p = as_ndarray(eotf_inverse_BT2100_PQ(xp_as_array(F_D, xp=xp)))

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(eotf_inverse_BT2100_PQ(F_D), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_inverse_BT2100_PQ(F_D), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_inverse_BT2100_PQ(F_D), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition domain and range scale support.
        """

        F_D = 779.98836083408537
        E_p = as_ndarray(eotf_inverse_BT2100_PQ(xp_as_array(F_D, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_inverse_BT2100_PQ(xp_as_array(F_D * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_BT2100_PQ(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_PQ` definition nan support.
        """

        eotf_inverse_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_BT2100_PQ:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition unit tests methods.
    """

    def test_ootf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition.
        """

        xp_assert_close(
            ootf_BT2100_PQ(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_PQ(xp_as_array(0.1, xp=xp)),
            779.98836083411584,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_PQ(xp_as_array(1.0, xp=xp)),
            9999.993723673924300,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ootf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition n-dimensional arrays support.
        """

        E = 0.1
        F_D = as_ndarray(ootf_BT2100_PQ(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        xp_assert_close(ootf_BT2100_PQ(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        xp_assert_close(ootf_BT2100_PQ(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(ootf_BT2100_PQ(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_ootf_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition domain and range scale support.
        """

        E = 0.1
        F_D = as_ndarray(ootf_BT2100_PQ(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ootf_BT2100_PQ(xp_as_array(E * factor, xp=xp)),
                    F_D * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ootf_BT2100_PQ(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_PQ` definition nan support.
        """

        ootf_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_inverse_BT2100_PQ:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition unit tests methods.
    """

    def test_ootf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition.
        """

        xp_assert_close(
            ootf_inverse_BT2100_PQ(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_PQ(xp_as_array(779.98836083411584, xp=xp)),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_PQ(xp_as_array(9999.993723673924300, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ootf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition n-dimensional arrays support.
        """

        F_D = 779.98836083411584
        E = as_ndarray(ootf_inverse_BT2100_PQ(xp_as_array(F_D, xp=xp)))

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(ootf_inverse_BT2100_PQ(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(ootf_inverse_BT2100_PQ(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(ootf_inverse_BT2100_PQ(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_ootf_inverse_BT2100_PQ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition domain and range scale support.
        """

        F_D = 779.98836083411584
        E = as_ndarray(ootf_inverse_BT2100_PQ(xp_as_array(F_D, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ootf_inverse_BT2100_PQ(xp_as_array(F_D * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ootf_inverse_BT2100_PQ(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_PQ` definition nan support.
        """

        ootf_inverse_BT2100_PQ(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestGamma_function_BT2100_HLG:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
gamma_function_BT2100_HLG` definition unit tests methods.
    """

    def test_gamma_function_BT2100_HLG(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
gamma_function_BT2100_HLG` definition.
        """

        xp_assert_close(
            gamma_function_BT2100_HLG(1000.0),
            1.2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            gamma_function_BT2100_HLG(2000.0),
            1.326432598178872,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            gamma_function_BT2100_HLG(4000.0),
            1.452865196357744,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            gamma_function_BT2100_HLG(10000.0),
            1.619999999999999,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestOetf_BT2100_HLG:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition unit tests methods.
    """

    def test_oetf_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition.
        """

        xp_assert_close(
            oetf_BT2100_HLG(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BT2100_HLG(xp_as_array(0.18 / 12, xp=xp)),
            0.212132034355964,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_BT2100_HLG(xp_as_array(1.0, xp=xp)),
            0.999999995536569,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition n-dimensional arrays support.
        """

        E = 0.18 / 12
        E_p = as_ndarray(oetf_BT2100_HLG(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(oetf_BT2100_HLG(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_BT2100_HLG(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_BT2100_HLG(E), E_p, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition domain and range scale support.
        """

        E = 0.18 / 12
        E_p = as_ndarray(oetf_BT2100_HLG(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_BT2100_HLG(xp_as_array(E * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_BT2100_HLG(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_BT2100_HLG` definition nan support.
        """

        oetf_BT2100_HLG(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_inverse_BT2100_HLG:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition unit tests methods.
    """

    def test_oetf_inverse_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition.
        """

        xp_assert_close(
            oetf_inverse_BT2100_HLG(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BT2100_HLG(xp_as_array(0.212132034355964, xp=xp)),
            0.18 / 12,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            oetf_inverse_BT2100_HLG(xp_as_array(0.999999995536569, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_oetf_inverse_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        E = as_ndarray(oetf_inverse_BT2100_HLG(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(oetf_inverse_BT2100_HLG(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(oetf_inverse_BT2100_HLG(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(oetf_inverse_BT2100_HLG(E_p), E, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_oetf_inverse_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        E = as_ndarray(oetf_inverse_BT2100_HLG(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    oetf_inverse_BT2100_HLG(xp_as_array(E_p * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_oetf_inverse_BT2100_HLG(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
oetf_inverse_BT2100_HLG` definition nan support.
        """

        oetf_inverse_BT2100_HLG(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_BT2100_HLG_1:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition unit tests methods.
    """

    def test_eotf_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition.
        """

        xp_assert_close(
            eotf_BT2100_HLG_1(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_HLG_1(xp_as_array(0.212132034355964, xp=xp)),
            6.476039825649814,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_HLG_1(xp_as_array(1.0, xp=xp)),
            1000.000032321769100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_HLG_1(xp_as_array(0.212132034355964, xp=xp), 0.001, 10000, 1.4),
            27.96039175299561,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        F_D = as_ndarray(eotf_BT2100_HLG_1(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (6, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_as_array([0.25, 0.50, 0.75], xp=xp)
        F_D = np.array([12.49759413, 49.99037650, 158.94693786])
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6, 1))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_1(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        F_D = as_ndarray(eotf_BT2100_HLG_1(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_BT2100_HLG_1(xp_as_array(E_p * factor, xp=xp)),
                    F_D * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_BT2100_HLG_1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_1` definition nan support.
        """

        eotf_BT2100_HLG_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_BT2100_HLG_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition unit tests methods.
    """

    def test_eotf_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition.
        """

        xp_assert_close(
            eotf_BT2100_HLG_2(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_HLG_2(xp_as_array(0.212132034355964, xp=xp)),
            6.476039825649814,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_HLG_2(xp_as_array(1.0, xp=xp)),
            1000.000032321769100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_BT2100_HLG_2(xp_as_array(0.212132034355964, xp=xp), 0.001, 10000, 1.4),
            29.581261576946076,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition n-dimensional arrays support.
        """

        E_p = 0.212132034355964
        F_D = as_ndarray(eotf_BT2100_HLG_2(xp_as_array(E_p, xp=xp)))

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (6, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_as_array([0.25, 0.50, 0.75], xp=xp)
        F_D = np.array([12.49759413, 49.99037650, 158.94693786])
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6, 1))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(eotf_BT2100_HLG_2(E_p), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_eotf_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition domain and range scale support.
        """

        E_p = 0.212132034355964
        F_D = as_ndarray(eotf_BT2100_HLG_2(xp_as_array(E_p, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_BT2100_HLG_2(xp_as_array(E_p * factor, xp=xp)),
                    F_D * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_BT2100_HLG_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_BT2100_HLG_2` definition nan support.
        """

        eotf_BT2100_HLG_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_inverse_BT2100_HLG_1:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition unit tests methods.
    """

    def test_eotf_inverse_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition.
        """

        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(xp_as_array(6.476039825649814, xp=xp)),
            0.212132034355964,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(xp_as_array(1000.000032321769100, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(
                xp_as_array(27.96039175299561, xp=xp), 0.001, 10000, 1.4
            ),
            0.212132034355964,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_inverse_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition n-dimensional arrays support.
        """

        F_D = 6.476039825649814
        E_p = as_ndarray(eotf_inverse_BT2100_HLG_1(xp_as_array(F_D, xp=xp)))

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (6, 1), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_as_array([12.49759413, 49.99037650, 158.94693786], xp=xp)
        E_p = np.array([0.25, 0.50, 0.75])
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6, 1))
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_1(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_eotf_inverse_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition domain and range scale support.
        """

        F_D = 6.476039825649814
        E_p = as_ndarray(eotf_inverse_BT2100_HLG_1(xp_as_array(F_D, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_inverse_BT2100_HLG_1(xp_as_array(F_D * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_BT2100_HLG_1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_1` definition nan support.
        """

        eotf_inverse_BT2100_HLG_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_inverse_BT2100_HLG_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition unit tests methods.
    """

    def test_eotf_inverse_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition.
        """

        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(xp_as_array(6.476039825649814, xp=xp)),
            0.212132034355964,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(xp_as_array(1000.000032321769100, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(
                xp_as_array(29.581261576946076, xp=xp), 0.001, 10000, 1.4
            ),
            0.212132034355964,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_eotf_inverse_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition n-dimensional arrays support.
        """

        F_D = 6.476039825649814
        E_p = as_ndarray(eotf_inverse_BT2100_HLG_2(xp_as_array(F_D, xp=xp)))

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6,))
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (6, 1), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_as_array([12.49759413, 49.99037650, 158.94693786], xp=xp)
        E_p = np.array([0.25, 0.50, 0.75])
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        E_p = xp.tile(xp_as_array(E_p, xp=xp), (6, 1))
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        E_p = xp_reshape(xp_as_array(E_p, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            eotf_inverse_BT2100_HLG_2(F_D),
            E_p,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_eotf_inverse_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition domain and range scale support.
        """

        F_D = 6.476039825649814
        E_p = as_ndarray(eotf_inverse_BT2100_HLG_2(xp_as_array(F_D, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    eotf_inverse_BT2100_HLG_2(xp_as_array(F_D * factor, xp=xp)),
                    E_p * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_eotf_inverse_BT2100_HLG_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
eotf_inverse_BT2100_HLG_2` definition nan support.
        """

        eotf_inverse_BT2100_HLG_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_BT2100_HLG_1:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition unit tests methods.
    """

    def test_ootf_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition.
        """

        xp_assert_close(
            ootf_BT2100_HLG_1(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_HLG_1(xp_as_array(0.1, xp=xp)),
            63.095734448019336,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_HLG_1(xp_as_array(1.0, xp=xp)),
            1000.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_HLG_1(xp_as_array(0.1, xp=xp), 0.001, 10000, 1.4),
            398.108130742780300,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.array(
            [
                [45.884942278760597, 0.000000000000000, -45.884942278760597],
                [
                    -63.095734448019336,
                    -63.095734448019336,
                    -63.095734448019336,
                ],
                [63.095734448019336, 63.095734448019336, 63.095734448019336],
                [51.320396090100672, -51.320396090100672, 51.320396090100672],
            ],
        )
        xp_assert_close(
            ootf_BT2100_HLG_1(
                xp_as_array(
                    [
                        [0.1, 0.0, -0.1],
                        [-0.1, -0.1, -0.1],
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                    ],
                    xp=xp,
                )
            ),
            a,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ootf_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition n-dimensional arrays support.
        """

        E = 0.1
        F_D = as_ndarray(ootf_BT2100_HLG_1(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (6, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_as_array([0.25, 0.50, 0.75], xp=xp)
        F_D = np.array([213.01897444, 426.03794887, 639.05692331])
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp.tile(xp_as_array(E, xp=xp), (6, 1))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_1(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_ootf_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition domain and range scale support.
        """

        E = 0.1
        F_D = as_ndarray(ootf_BT2100_HLG_1(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ootf_BT2100_HLG_1(xp_as_array(E * factor, xp=xp)),
                    F_D * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ootf_BT2100_HLG_1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition nan support.
        """

        ootf_BT2100_HLG_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_BT2100_HLG_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_2` definition unit tests methods.
    """

    def test_ootf_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_2` definition.
        """

        xp_assert_close(
            ootf_BT2100_HLG_2(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_HLG_2(xp_as_array(0.1, xp=xp)),
            63.095734448019336,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_HLG_2(xp_as_array(1.0, xp=xp)),
            1000.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_BT2100_HLG_2(xp_as_array(0.1, xp=xp), 10000, 1.4),
            398.107170553497380,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.array(
            [
                [45.884942278760597, 0.000000000000000, -45.884942278760597],
                [
                    -63.095734448019336,
                    -63.095734448019336,
                    -63.095734448019336,
                ],
                [63.095734448019336, 63.095734448019336, 63.095734448019336],
                [51.320396090100672, -51.320396090100672, 51.320396090100672],
            ],
        )
        xp_assert_close(
            ootf_BT2100_HLG_2(
                xp_as_array(
                    [
                        [0.1, 0.0, -0.1],
                        [-0.1, -0.1, -0.1],
                        [0.1, 0.1, 0.1],
                        [0.1, -0.1, 0.1],
                    ],
                    xp=xp,
                )
            ),
            a,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ootf_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_2` definition n-dimensional arrays support.
        """

        E = 0.1
        F_D = as_ndarray(ootf_BT2100_HLG_2(xp_as_array(E, xp=xp)))

        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (6, 1), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_as_array([0.25, 0.50, 0.75], xp=xp)
        F_D = np.array([213.01897444, 426.03794887, 639.05692331])
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp.tile(xp_as_array(E, xp=xp), (6, 1))
        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 3), xp=xp)
        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(ootf_BT2100_HLG_2(E), F_D, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_ootf_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_2` definition domain and range scale support.
        """

        E = 0.1
        F_D = as_ndarray(ootf_BT2100_HLG_1(xp_as_array(E, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ootf_BT2100_HLG_1(xp_as_array(E * factor, xp=xp)),
                    F_D * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ootf_BT2100_HLG_1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG_1` definition nan support.
        """

        ootf_BT2100_HLG_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_inverse_BT2100_HLG_1:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition unit tests methods.
    """

    def test_ootf_inverse_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition.
        """

        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(xp_as_array(63.095734448019336, xp=xp)),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(xp_as_array(1000.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(
                xp_as_array(398.108130742780300, xp=xp), 0.001, 10000, 1.4
            ),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.array(
            [
                [45.884942278760597, 0.000000000000000, -45.884942278760597],
                [
                    -63.095734448019336,
                    -63.095734448019336,
                    -63.095734448019336,
                ],
                [63.095734448019336, 63.095734448019336, 63.095734448019336],
                [51.320396090100672, -51.320396090100672, 51.320396090100672],
            ]
        )
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(xp_as_array(a, xp=xp)),
            [
                [0.1, 0.0, -0.1],
                [-0.1, -0.1, -0.1],
                [0.1, 0.1, 0.1],
                [0.1, -0.1, 0.1],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ootf_inverse_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition n-dimensional arrays support.
        """

        F_D = 63.095734448019336
        E = as_ndarray(ootf_inverse_BT2100_HLG_1(xp_as_array(F_D, xp=xp)))

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (6, 1), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_as_array([213.01897444, 426.03794887, 639.05692331], xp=xp)
        E = np.array([0.25, 0.50, 0.75])
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        E = xp.tile(xp_as_array(E, xp=xp), (6, 1))
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_1(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_ootf_inverse_BT2100_HLG_1(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition domain and range scale support.
        """

        F_D = 63.095734448019336
        E = as_ndarray(ootf_inverse_BT2100_HLG_1(xp_as_array(F_D, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ootf_inverse_BT2100_HLG_1(xp_as_array(F_D * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ootf_inverse_BT2100_HLG_1(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_1` definition nan support.
        """

        ootf_inverse_BT2100_HLG_1(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotf_inverse_BT2100_HLG_2:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition unit tests methods.
    """

    def test_ootf_inverse_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition.
        """

        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(xp_as_array(0.0, xp=xp)),
            0.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(xp_as_array(63.095734448019336, xp=xp)),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(xp_as_array(1000.0, xp=xp)),
            1.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(
                xp_as_array(398.107170553497380, xp=xp), 10000, 1.4
            ),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        a = np.array(
            [
                [45.884942278760597, 0.000000000000000, -45.884942278760597],
                [
                    -63.095734448019336,
                    -63.095734448019336,
                    -63.095734448019336,
                ],
                [63.095734448019336, 63.095734448019336, 63.095734448019336],
                [51.320396090100672, -51.320396090100672, 51.320396090100672],
            ]
        )
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(xp_as_array(a, xp=xp)),
            [
                [0.1, 0.0, -0.1],
                [-0.1, -0.1, -0.1],
                [0.1, 0.1, 0.1],
                [0.1, -0.1, 0.1],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_ootf_inverse_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition n-dimensional arrays support.
        """

        F_D = 63.095734448019336
        E = as_ndarray(ootf_inverse_BT2100_HLG_2(xp_as_array(F_D, xp=xp)))

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6,))
        E = xp.tile(xp_as_array(E, xp=xp), (6,))
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (6, 1), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (6, 1), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_as_array([213.01897444, 426.03794887, 639.05692331], xp=xp)
        E = np.array([0.25, 0.50, 0.75])
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp.tile(xp_as_array(F_D, xp=xp), (6, 1))
        E = xp.tile(xp_as_array(E, xp=xp), (6, 1))
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        F_D = xp_reshape(xp_as_array(F_D, xp=xp), (2, 3, 3), xp=xp)
        E = xp_reshape(xp_as_array(E, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            ootf_inverse_BT2100_HLG_2(F_D), E, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_ootf_inverse_BT2100_HLG_2(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition domain and range scale support.
        """

        F_D = 63.095734448019336
        E = as_ndarray(ootf_inverse_BT2100_HLG_2(xp_as_array(F_D, xp=xp)))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    ootf_inverse_BT2100_HLG_2(xp_as_array(F_D * factor, xp=xp)),
                    E * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_ootf_inverse_BT2100_HLG_2(self) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG_2` definition nan support.
        """

        ootf_inverse_BT2100_HLG_2(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOotfBT2100HLG:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG` definition unit tests methods.
    """

    def test_ootf_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_BT2100_HLG` definition.
        """

        # Test default method (ITU-R BT.2100-2)
        xp_assert_close(
            ootf_BT2100_HLG(xp_as_array(0.1, xp=xp)),
            63.095734448019336,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test ITU-R BT.2100-1 method
        xp_assert_close(
            ootf_BT2100_HLG(xp_as_array(0.1, xp=xp), 0.01, method="ITU-R BT.2100-1"),
            63.105103490674857,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test with different L_W value
        xp_assert_close(
            ootf_BT2100_HLG(xp_as_array(0.1, xp=xp), L_W=2000),
            94.3186112317,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestOotfInverseBT2100HLG:
    """
    Define :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG` definition unit tests methods.
    """

    def test_ootf_inverse_BT2100_HLG(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.rgb.transfer_functions.itur_bt_2100.\
ootf_inverse_BT2100_HLG` definition.
        """

        # Test default method (ITU-R BT.2100-2)
        xp_assert_close(
            ootf_inverse_BT2100_HLG(xp_as_array(63.095734448019336, xp=xp)),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test ITU-R BT.2100-1 method
        xp_assert_close(
            ootf_inverse_BT2100_HLG(
                xp_as_array(63.105103490674857, xp=xp), 0.01, method="ITU-R BT.2100-1"
            ),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test with different L_W value
        xp_assert_close(
            ootf_inverse_BT2100_HLG(xp_as_array(94.3186112317, xp=xp), L_W=2000),
            0.1,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
