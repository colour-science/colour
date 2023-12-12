# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.temperature.ohno2013` module."""

import unittest
from itertools import product

import numpy as np

from colour.colorimetry import MSDS_CMFS
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import (
    CCT_to_uv_Ohno2013,
    CCT_to_XYZ_Ohno2013,
    XYZ_to_CCT_Ohno2013,
    uv_to_CCT_Ohno2013,
)
from colour.temperature.ohno2013 import (
    planckian_table,
)
from colour.utilities import ignore_numpy_errors

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


class TestPlanckianTable(unittest.TestCase):
    """
    Define :func:`colour.temperature.ohno2013.planckian_table` definition
    unit tests methods.
    """

    def test_planckian_table(self):
        """Test :func:`colour.temperature.ohno2013.planckian_table` definition."""

        np.testing.assert_allclose(
            planckian_table(
                MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
                5000,
                6000,
                1.01,
            ),
            np.array(
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
                ]
            ),
            atol=1e-6,
        )


class TestUv_to_CCT_Ohno2013(unittest.TestCase):
    """
    Define :func:`colour.temperature.ohno2013.uv_to_CCT_Ohno2013` definition
    unit tests methods.
    """

    def test_uv_to_CCT_Ohno2013(self):
        """
        Test :func:`colour.temperature.ohno2013.uv_to_CCT_Ohno2013`
        definition.
        """

        CCT = np.linspace(1_000, 100_000, 3_000)
        D_uv = np.linspace(-0.01, 0.01, 10)

        CCT, D_uv = np.meshgrid(CCT, D_uv)
        table_r = np.transpose((np.ravel(CCT), np.ravel(D_uv)))
        table_t = uv_to_CCT_Ohno2013(CCT_to_uv_Ohno2013(table_r))

        np.testing.assert_allclose(table_t[1, :], table_r[1, :], atol=1)

        np.testing.assert_allclose(
            uv_to_CCT_Ohno2013(np.array([0.1978, 0.3122])),
            np.array([6507.474788799616363, 0.003223346337596]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            uv_to_CCT_Ohno2013(np.array([0.4328, 0.2883])),
            np.array([1041.678320000468375, -0.067378053475797]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            uv_to_CCT_Ohno2013(np.array([0.2927, 0.2722])),
            np.array([2444.971818951082696, -0.084370641205118]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_uv_to_CCT_Ohno2013(self):
        """
        Test :func:`colour.temperature.ohno2013.uv_to_CCT_Ohno2013` definition
        n-dimensional arrays support.
        """

        uv = np.array([0.1978, 0.3122])
        CCT_D_uv = uv_to_CCT_Ohno2013(uv)

        uv = np.tile(uv, (6, 1))
        CCT_D_uv = np.tile(CCT_D_uv, (6, 1))
        np.testing.assert_allclose(
            uv_to_CCT_Ohno2013(uv), CCT_D_uv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        uv = np.reshape(uv, (2, 3, 2))
        CCT_D_uv = np.reshape(CCT_D_uv, (2, 3, 2))
        np.testing.assert_allclose(
            uv_to_CCT_Ohno2013(uv), CCT_D_uv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_uv_to_CCT_Ohno2013(self):
        """
        Test :func:`colour.temperature.ohno2013.uv_to_CCT_Ohno2013` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        uv_to_CCT_Ohno2013(cases)


class TestCCT_to_uv_Ohno2013(unittest.TestCase):
    """
    Define :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013` definition
    unit tests methods.
    """

    def test_CCT_to_uv_Ohno2013(self):
        """
        Test :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013`
        definition.
        """

        np.testing.assert_allclose(
            CCT_to_uv_Ohno2013(np.array([6507.47380460, 0.00322335])),
            np.array([0.19779997, 0.31219997]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CCT_to_uv_Ohno2013(np.array([1041.68315360, -0.06737802])),
            np.array([0.43279885, 0.28830013]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CCT_to_uv_Ohno2013(np.array([2452.15316417, -0.08437064])),
            np.array([0.29247364, 0.27215157]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CCT_to_uv_Ohno2013(self):
        """
        Test :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013` definition
        n-dimensional arrays support.
        """

        CCT_D_uv = np.array([6507.47380460, 0.00322335])
        uv = CCT_to_uv_Ohno2013(CCT_D_uv)

        CCT_D_uv = np.tile(CCT_D_uv, (6, 1))
        uv = np.tile(uv, (6, 1))
        np.testing.assert_allclose(
            CCT_to_uv_Ohno2013(CCT_D_uv), uv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        CCT_D_uv = np.reshape(CCT_D_uv, (2, 3, 2))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_allclose(
            CCT_to_uv_Ohno2013(CCT_D_uv), uv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_CCT_to_uv_Ohno2013(self):
        """
        Test :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        CCT_to_uv_Ohno2013(cases)


class Test_XYZ_to_CCT_Ohno2013(unittest.TestCase):
    """
    Define :func:`colour.temperature.ohno2013.XYZ_to_CCT_Ohno2013` definition
    unit tests methods.
    """

    def test_XYZ_to_CCT_Ohno2013(self):
        """
        Test :func:`colour.temperature.ohno2013.XYZ_to_CCT_Ohno2013` definition.
        """

        np.testing.assert_allclose(
            XYZ_to_CCT_Ohno2013(np.array([95.04, 100.00, 108.88])),
            np.array([6503.30711709, 0.00321729]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_CCT_Ohno2013(self):
        """
        Test :func:`colour.temperature.ohno2013.XYZ_to_CCT_Ohno2013` definition
        n-dimensional arrays support.
        """

        XYZ = np.array([95.04, 100.00, 108.88])
        CCT_D_uv = XYZ_to_CCT_Ohno2013(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        CCT_D_uv = np.tile(CCT_D_uv, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_CCT_Ohno2013(XYZ), CCT_D_uv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        CCT_D_uv = np.reshape(CCT_D_uv, (2, 3, 2))
        np.testing.assert_allclose(
            XYZ_to_CCT_Ohno2013(XYZ), CCT_D_uv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_XYZ_to_CCT_Ohno2013(self):
        """
        Test :func:`colour.temperature.ohno2013.XYZ_to_CCT_Ohno2013` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_CCT_Ohno2013(cases)


class Test_CCT_to_XYZ_Ohno2013(unittest.TestCase):
    """
    Define :func:`colour.temperature.ohno2013.CCT_to_XYZ_Ohno2013` definition
    unit tests methods.
    """

    def test_CCT_to_XYZ_Ohno2013(self):
        """
        Test :func:`colour.temperature.ohno2013.CCT_to_XYZ_Ohno2013` definition.
        """

        np.testing.assert_allclose(
            CCT_to_XYZ_Ohno2013(np.array([6503.30711709, 0.00321729])),
            np.array([95.04, 100.00, 108.88]) / 100,
            atol=1e-6,
        )

    def test_n_dimensional_CCT_to_XYZ_Ohno2013(self):
        """
        Test :func:`colour.temperature.ohno2013.CCT_to_XYZ_Ohno2013` definition
        n-dimensional arrays support.
        """

        CCT_D_uv = np.array([6503.30711709, 0.00321729])
        XYZ = CCT_to_XYZ_Ohno2013(CCT_D_uv)

        CCT_D_uv = np.tile(CCT_D_uv, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            CCT_to_XYZ_Ohno2013(CCT_D_uv), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        CCT_D_uv = np.reshape(CCT_D_uv, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            CCT_to_XYZ_Ohno2013(CCT_D_uv), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_CCT_to_uv_Ohno2013(self):
        """
        Test :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        CCT_to_uv_Ohno2013(cases)


if __name__ == "__main__":
    unittest.main()
