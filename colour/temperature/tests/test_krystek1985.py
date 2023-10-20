# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.temperature.krystek1985` module."""

import unittest
from itertools import product

import numpy as np

from colour.temperature import CCT_to_uv_Krystek1985, uv_to_CCT_Krystek1985
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestUv_to_CCT_Krystek1985",
]


class TestUv_to_CCT_Krystek1985(unittest.TestCase):
    """
    Define :func:`colour.temperature.krystek1985.uv_to_CCT_Krystek1985`
    definition unit tests methods.
    """

    def test_uv_to_CCT_Krystek1985(self):
        """
        Test :func:`colour.temperature.krystek1985.uv_to_CCT_Krystek1985`
        definition.
        """

        np.testing.assert_allclose(
            uv_to_CCT_Krystek1985(
                np.array([0.448087794140145, 0.354731965027727]),
                {"method": "Nelder-Mead"},
            ),
            1000,
            rtol=0.0000001,
            atol=0.0000001,
        )

        np.testing.assert_allclose(
            uv_to_CCT_Krystek1985(
                np.array([0.198152565091092, 0.307023596915037]),
                {"method": "Nelder-Mead"},
            ),
            7000,
            rtol=0.0000001,
            atol=0.0000001,
        )

        np.testing.assert_allclose(
            uv_to_CCT_Krystek1985(
                np.array([0.185675876767054, 0.282233658593898]),
                {"method": "Nelder-Mead"},
            ),
            15000,
            rtol=0.0000001,
            atol=0.0000001,
        )

    def test_n_dimensional_uv_to_CCT_Krystek1985(self):
        """
        Test :func:`colour.temperature.krystek1985.uv_to_CCT_Krystek1985`
        definition n-dimensional arrays support.
        """

        uv = np.array([0.198152565091092, 0.307023596915037])
        CCT = uv_to_CCT_Krystek1985(uv)

        uv = np.tile(uv, (6, 1))
        CCT = np.tile(CCT, 6)
        np.testing.assert_array_almost_equal(
            uv_to_CCT_Krystek1985(uv), CCT, decimal=7
        )

        uv = np.reshape(uv, (2, 3, 2))
        CCT = np.reshape(CCT, (2, 3))
        np.testing.assert_array_almost_equal(
            uv_to_CCT_Krystek1985(uv), CCT, decimal=7
        )

    @ignore_numpy_errors
    def test_nan_uv_to_CCT_Krystek1985(self):
        """
        Test :func:`colour.temperature.krystek1985.uv_to_CCT_Krystek1985`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        uv_to_CCT_Krystek1985(cases)


class TestCCT_to_uv_Krystek1985(unittest.TestCase):
    """
    Define :func:`colour.temperature.krystek1985.CCT_to_uv_Krystek1985`
    definition unit tests methods.
    """

    def test_CCT_to_uv_Krystek1985(self):
        """
        Test :func:`colour.temperature.krystek1985.CCT_to_uv_Krystek1985`
        definition.
        """

        np.testing.assert_array_almost_equal(
            CCT_to_uv_Krystek1985(1000),
            np.array([0.448087794140145, 0.354731965027727]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            CCT_to_uv_Krystek1985(7000),
            np.array([0.198152565091092, 0.307023596915037]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            CCT_to_uv_Krystek1985(15000),
            np.array([0.185675876767054, 0.282233658593898]),
            decimal=7,
        )

    def test_n_dimensional_CCT_to_uv_Krystek1985(self):
        """
        Test :func:`colour.temperature.krystek1985.CCT_to_uv_Krystek1985`
        definition n-dimensional arrays support.
        """

        CCT = 7000
        uv = CCT_to_uv_Krystek1985(CCT)

        CCT = np.tile(CCT, 6)
        uv = np.tile(uv, (6, 1))
        np.testing.assert_array_almost_equal(
            CCT_to_uv_Krystek1985(CCT), uv, decimal=7
        )

        CCT = np.reshape(CCT, (2, 3))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_array_almost_equal(
            CCT_to_uv_Krystek1985(CCT), uv, decimal=7
        )

    @ignore_numpy_errors
    def test_nan_CCT_to_uv_Krystek1985(self):
        """
        Test :func:`colour.temperature.krystek1985.CCT_to_uv_Krystek1985`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        CCT_to_uv_Krystek1985(cases)


if __name__ == "__main__":
    unittest.main()
