"""Define the unit tests for the :mod:`colour.temperature.ohno2013` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np
import pytest

from colour.colorimetry import MSDS_CMFS
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import (
    CCT_to_uv_Ohno2013,
    CCT_to_XYZ_Ohno2013,
    XYZ_to_CCT_Ohno2013,
    uv_to_CCT_Ohno2013,
)
from colour.temperature.ohno2013 import planckian_table
from colour.utilities import (
    as_ndarray,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlanckianTable",
    "TestUv_to_CCT_Ohno2013",
    "TestCCT_to_uv_Ohno2013",
    "Test_XYZ_to_CCT_Ohno2013",
    "Test_CCT_to_XYZ_Ohno2013",
]


class TestPlanckianTable:
    """
    Define :func:`colour.temperature.ohno2013.planckian_table` definition
    unit tests methods.
    """

    def test_planckian_table(self) -> None:
        """Test :func:`colour.temperature.ohno2013.planckian_table` definition."""

        xp_assert_close(
            planckian_table(
                MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
                5000,
                6000,
                1.01,
            ),
            [
                [5.00000000e03, 2.11424442e-01, 3.23115810e-01],
                [5.00100000e03, 2.11414166e-01, 3.23105716e-01],
                [5.05101000e03, 2.10906941e-01, 3.22603850e-01],
                [5.09965995e03, 2.10425840e-01, 3.22121155e-01],
                [5.14875592e03, 2.09952257e-01, 3.21639518e-01],
                [5.19830158e03, 2.09486095e-01, 3.21159015e-01],
                [5.24830059e03, 2.09027261e-01, 3.20679719e-01],
                [5.29875665e03, 2.08575658e-01, 3.20201701e-01],
                [5.34967349e03, 2.08131192e-01, 3.19725033e-01],
                [5.40105483e03, 2.07693769e-01, 3.19249784e-01],
                [5.45290444e03, 2.07263296e-01, 3.18776019e-01],
                [5.50522609e03, 2.06839680e-01, 3.18303806e-01],
                [5.55802360e03, 2.06422828e-01, 3.17833209e-01],
                [5.61130078e03, 2.06012650e-01, 3.17364290e-01],
                [5.66506148e03, 2.05609054e-01, 3.16897111e-01],
                [5.71930956e03, 2.05211949e-01, 3.16431730e-01],
                [5.77404891e03, 2.04821246e-01, 3.15968207e-01],
                [5.82928344e03, 2.04436856e-01, 3.15506598e-01],
                [5.88501707e03, 2.04058690e-01, 3.15046958e-01],
                [5.94125375e03, 2.03686660e-01, 3.14589340e-01],
                [5.99799745e03, 2.03320679e-01, 3.14133796e-01],
                [5.99900000e03, 2.03314296e-01, 3.14125803e-01],
                [6.00000000e03, 2.03307932e-01, 3.14117832e-01],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS * 10,
        )


class TestUv_to_CCT_Ohno2013:
    """
    Define :func:`colour.temperature.ohno2013.uv_to_CCT_Ohno2013` definition
    unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1)
    def test_uv_to_CCT_Ohno2013(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.ohno2013.uv_to_CCT_Ohno2013`
        definition.
        """

        CCT = np.linspace(1_000, 100_000, 3_000)
        D_uv = np.linspace(-0.01, 0.01, 10)

        CCT, D_uv = np.meshgrid(CCT, D_uv)
        table_r = np.transpose((np.ravel(CCT), np.ravel(D_uv)))
        table_t = as_ndarray(uv_to_CCT_Ohno2013(CCT_to_uv_Ohno2013(table_r)))

        xp_assert_close(
            table_t[1, :], table_r[1, :], atol=TOLERANCE_ABSOLUTE_TESTS * 10000000
        )

        xp_assert_close(
            uv_to_CCT_Ohno2013(xp_as_array([0.1978, 0.3122], xp=xp)),
            [6507.474788799616363, 0.003223346337596],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            uv_to_CCT_Ohno2013(xp_as_array([0.4328, 0.2883], xp=xp)),
            [1041.678320000468375, -0.067378053475797],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            uv_to_CCT_Ohno2013(xp_as_array([0.2927, 0.2722], xp=xp)),
            [2444.971818951082696, -0.084370641205118],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_uv_to_CCT_Ohno2013(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.ohno2013.uv_to_CCT_Ohno2013` definition
        n-dimensional arrays support.
        """

        uv = xp_as_array([0.1978, 0.3122], xp=xp)
        CCT_D_uv = as_ndarray(uv_to_CCT_Ohno2013(uv))

        uv = xp.tile(xp_as_array(uv, xp=xp), (6, 1))
        CCT_D_uv = xp.tile(xp_as_array(CCT_D_uv, xp=xp), (6, 1))
        xp_assert_close(uv_to_CCT_Ohno2013(uv), CCT_D_uv, atol=TOLERANCE_ABSOLUTE_TESTS)

        uv = xp_reshape(xp_as_array(uv, xp=xp), (2, 3, 2), xp=xp)
        CCT_D_uv = xp_reshape(xp_as_array(CCT_D_uv, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(uv_to_CCT_Ohno2013(uv), CCT_D_uv, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_uv_to_CCT_Ohno2013(self) -> None:
        """
        Test :func:`colour.temperature.ohno2013.uv_to_CCT_Ohno2013` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        uv_to_CCT_Ohno2013(cases)


class TestCCT_to_uv_Ohno2013:
    """
    Define :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013` definition
    unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1e-2)
    def test_CCT_to_uv_Ohno2013(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013`
        definition.
        """

        xp_assert_close(
            CCT_to_uv_Ohno2013(xp_as_array([6507.47380460, 0.00322335], xp=xp)),
            [0.19779997, 0.31219997],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_uv_Ohno2013(xp_as_array([1041.68315360, -0.06737802], xp=xp)),
            [0.43279885, 0.28830013],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CCT_to_uv_Ohno2013(xp_as_array([2452.15316417, -0.08437064], xp=xp)),
            [0.29247364, 0.27215157],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CCT_to_uv_Ohno2013(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013` definition
        n-dimensional arrays support.
        """

        CCT_D_uv = xp_as_array([6507.47380460, 0.00322335], xp=xp)
        uv = as_ndarray(CCT_to_uv_Ohno2013(CCT_D_uv))

        CCT_D_uv = xp.tile(xp_as_array(CCT_D_uv, xp=xp), (6, 1))
        uv = xp.tile(xp_as_array(uv, xp=xp), (6, 1))
        xp_assert_close(CCT_to_uv_Ohno2013(CCT_D_uv), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

        CCT_D_uv = xp_reshape(xp_as_array(CCT_D_uv, xp=xp), (2, 3, 2), xp=xp)
        uv = xp_reshape(xp_as_array(uv, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(CCT_to_uv_Ohno2013(CCT_D_uv), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_CCT_to_uv_Ohno2013(self) -> None:
        """
        Test :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        CCT_to_uv_Ohno2013(cases)


class Test_XYZ_to_CCT_Ohno2013:
    """
    Define :func:`colour.temperature.ohno2013.XYZ_to_CCT_Ohno2013` definition
    unit tests methods.
    """

    # NOTE: The iterative *CCT* search amplifies float32 rounding by a
    # *Metal* compiler version dependent factor: the observed *CCT* drift is
    # 0.16K on *macOS* 15 and 0.51K on *macOS* 26 for a ~6503K result.
    @pytest.mark.mps_tolerance_absolute(1)
    def test_XYZ_to_CCT_Ohno2013(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.ohno2013.XYZ_to_CCT_Ohno2013` definition.
        """

        xp_assert_close(
            XYZ_to_CCT_Ohno2013(xp_as_array([95.04, 100.00, 108.88], xp=xp)),
            [6503.30711709, 0.00321729],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_CCT_Ohno2013(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.ohno2013.XYZ_to_CCT_Ohno2013` definition
        n-dimensional arrays support.
        """

        XYZ = xp_as_array([95.04, 100.00, 108.88], xp=xp)
        CCT_D_uv = as_ndarray(XYZ_to_CCT_Ohno2013(XYZ))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        CCT_D_uv = xp.tile(xp_as_array(CCT_D_uv, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_CCT_Ohno2013(XYZ),
            CCT_D_uv,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        CCT_D_uv = xp_reshape(xp_as_array(CCT_D_uv, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(
            XYZ_to_CCT_Ohno2013(XYZ),
            CCT_D_uv,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_XYZ_to_CCT_Ohno2013(self) -> None:
        """
        Test :func:`colour.temperature.ohno2013.XYZ_to_CCT_Ohno2013` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_CCT_Ohno2013(cases)


class Test_CCT_to_XYZ_Ohno2013:
    """
    Define :func:`colour.temperature.ohno2013.CCT_to_XYZ_Ohno2013` definition
    unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1e-1)
    def test_CCT_to_XYZ_Ohno2013(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.ohno2013.CCT_to_XYZ_Ohno2013` definition.
        """

        xp_assert_close(
            CCT_to_XYZ_Ohno2013(xp_as_array([6503.30711709, 0.00321729], xp=xp)),
            xp_as_array([95.04, 100.00, 108.88], xp=xp) / 100,
            atol=TOLERANCE_ABSOLUTE_TESTS * 10,
        )

    @pytest.mark.mps_tolerance_absolute(1e-2)
    def test_n_dimensional_CCT_to_XYZ_Ohno2013(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.temperature.ohno2013.CCT_to_XYZ_Ohno2013` definition
        n-dimensional arrays support.
        """

        CCT_D_uv = xp_as_array([6503.30711709, 0.00321729], xp=xp)
        XYZ = as_ndarray(CCT_to_XYZ_Ohno2013(CCT_D_uv))

        CCT_D_uv = xp.tile(xp_as_array(CCT_D_uv, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            CCT_to_XYZ_Ohno2013(CCT_D_uv),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        CCT_D_uv = xp_reshape(xp_as_array(CCT_D_uv, xp=xp), (2, 3, 2), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            CCT_to_XYZ_Ohno2013(CCT_D_uv),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_CCT_to_uv_Ohno2013(self) -> None:
        """
        Test :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        CCT_to_uv_Ohno2013(cases)
