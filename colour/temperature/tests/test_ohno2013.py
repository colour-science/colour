# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.temperature.ohno2013` module."""

import numpy as np
import unittest
from itertools import product

from colour.colorimetry import MSDS_CMFS
from colour.temperature import CCT_to_uv_Ohno2013, uv_to_CCT_Ohno2013
from colour.temperature.ohno2013 import (
    CCT_to_XYZ_Ohno2013,
    XYZ_to_CCT_Ohno2013,
    planckian_table,
)
from colour.utilities import ignore_numpy_errors
from colour.utilities.array import as_float_array

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestPlanckianTable",
    "TestUv_to_CCT_Ohno2013",
    "TestCCT_to_uv_Ohno2013",
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
            rtol=0.000001,
            atol=0.000001,
        )


class TestXYZ_tofrom_CCT_Ohno2013(unittest.TestCase):
    """Define :func:`colour.temperature.ohno2013.XYZ_to_CCT_Ohno2013` definition
    unit tests methods.
    """

    def test_XYZ_to_CCT_Ohno2013(self):
        """Test the XYZ to CCT Ohno method conversion"""
        XYZ = [95, 100, 108]
        np.testing.assert_allclose(
            XYZ_to_CCT_Ohno2013(XYZ),
            np.array([6.45204726e03, 0.0033180561896173355]),
            rtol=1e-6,
        )

    def test_CCT_to_duv_Ohno2013(self):
        """Test the CCT to Duv Ohno method conversion"""
        CCT_duv = [5000, 0.002]
        np.testing.assert_allclose(
            CCT_to_XYZ_Ohno2013(CCT_duv),
            np.array([0.9707001144701166, 1.0, 0.8389509590168589]),
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
        duv = np.linspace(-0.01, 0.01, 10)

        CCT, duv = np.meshgrid(CCT, duv)
        exact_table = as_float_array((CCT.flatten(), duv.flatten())).T
        uv_table = CCT_to_uv_Ohno2013(exact_table)
        calc_cct = uv_to_CCT_Ohno2013(uv_table)

        np.testing.assert_allclose(calc_cct[1, :], exact_table[1, :], atol=1)

        np.testing.assert_array_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.1978, 0.3122])),
            np.array([6.50760081e03, 3.22333560e-03]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.4328, 0.2883])),
            np.array([1041.692826446935, -0.067378054025829637]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.2927, 0.2722])),
            np.array([2445.0416505823432, -0.084370640066503882]),
            decimal=7,
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
        np.testing.assert_array_almost_equal(
            uv_to_CCT_Ohno2013(uv), CCT_D_uv, decimal=7
        )

        uv = np.reshape(uv, (2, 3, 2))
        CCT_D_uv = np.reshape(CCT_D_uv, (2, 3, 2))
        np.testing.assert_array_almost_equal(
            uv_to_CCT_Ohno2013(uv), CCT_D_uv, decimal=7
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

        np.testing.assert_array_almost_equal(
            CCT_to_uv_Ohno2013(np.array([6507.47380460, 0.00322335])),
            np.array([0.19779997, 0.31219997]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            CCT_to_uv_Ohno2013(np.array([1041.68315360, -0.06737802])),
            np.array([0.43279885, 0.28830013]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            CCT_to_uv_Ohno2013(np.array([2452.15316417, -0.08437064])),
            np.array([0.29247364, 0.27215157]),
            decimal=7,
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
        np.testing.assert_array_almost_equal(
            CCT_to_uv_Ohno2013(CCT_D_uv), uv, decimal=7
        )

        CCT_D_uv = np.reshape(CCT_D_uv, (2, 3, 2))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_array_almost_equal(
            CCT_to_uv_Ohno2013(CCT_D_uv), uv, decimal=7
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
