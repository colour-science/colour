# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.temperature.ohno2013` module."""

import numpy as np
import unittest
from itertools import product

from colour.colorimetry import MSDS_CMFS
from colour.temperature import CCT_to_uv_Ohno2013, uv_to_CCT_Ohno2013
from colour.temperature.ohno2013 import planckian_table
from colour.utilities import ignore_numpy_errors

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
                np.array([0.1978, 0.3122]),
                MSDS_CMFS["CIE 1931 2 Degree Standard Observer"],
                5000,
                6000,
                10,
            ),
            np.array(
                [
                    [
                        5.00000000e03,
                        2.11424442e-01,
                        3.23115810e-01,
                        1.74579593e-02,
                    ],
                    [
                        5.11111111e03,
                        2.10314324e-01,
                        3.22008326e-01,
                        1.59000490e-02,
                    ],
                    [
                        5.22222222e03,
                        2.09265149e-01,
                        3.20929009e-01,
                        1.44099008e-02,
                    ],
                    [
                        5.33333333e03,
                        2.08272619e-01,
                        3.19877383e-01,
                        1.29852974e-02,
                    ],
                    [
                        5.44444444e03,
                        2.07332799e-01,
                        3.18852924e-01,
                        1.16247859e-02,
                    ],
                    [
                        5.55555556e03,
                        2.06442082e-01,
                        3.17855076e-01,
                        1.03278973e-02,
                    ],
                    [
                        5.66666667e03,
                        2.05597159e-01,
                        3.16883254e-01,
                        9.09552364e-03,
                    ],
                    [
                        5.77777778e03,
                        2.04794988e-01,
                        3.15936852e-01,
                        7.93056919e-03,
                    ],
                    [
                        5.88888889e03,
                        2.04032772e-01,
                        3.15015254e-01,
                        6.83908641e-03,
                    ],
                    [
                        6.00000000e03,
                        2.03307932e-01,
                        3.14117832e-01,
                        5.83227190e-03,
                    ],
                ]
            ),
            rtol=0.000001,
            atol=0.000001,
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

        np.testing.assert_array_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.1978, 0.3122])),
            np.array([6507.47380460, 0.00322335]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.4328, 0.2883])),
            np.array([1041.68315360, -0.06737802]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.2927, 0.2722]), iterations=4),
            np.array([2452.15316417, -0.08437064]),
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
