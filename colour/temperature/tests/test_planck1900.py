# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.temperature.planck1900` module."""

import unittest
from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import CCT_to_uv_Planck1900, uv_to_CCT_Planck1900
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestUv_to_CCT_Planck1900",
    "TestCCT_to_uv_Planck1900",
]


class TestUv_to_CCT_Planck1900(unittest.TestCase):
    """
    Define :func:`colour.temperature.planck1900.uv_to_CCT_Planck1900`
    definition unit tests methods.
    """

    def test_uv_to_CCT_Planck1900(self):
        """
        Test :func:`colour.temperature.planck1900.uv_to_CCT_Planck1900`
        definition.
        """

        np.testing.assert_allclose(
            uv_to_CCT_Planck1900(
                np.array([0.225109670227493, 0.334387366663923]),
                optimisation_kwargs={"method": "Nelder-Mead"},
            ),
            4000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            uv_to_CCT_Planck1900(
                np.array([0.198126929048352, 0.307025980523306]),
                optimisation_kwargs={"method": "Nelder-Mead"},
            ),
            7000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            uv_to_CCT_Planck1900(
                np.array([0.182932683590136, 0.274073232217536]),
                optimisation_kwargs={"method": "Nelder-Mead"},
            ),
            25000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_uv_to_CCT_Planck1900(self):
        """
        Test :func:`colour.temperature.planck1900.uv_to_CCT_Planck1900`
        definition n-dimensional arrays support.
        """

        uv = np.array([0.225109670227493, 0.334387366663923])
        CCT = uv_to_CCT_Planck1900(uv)

        uv = np.tile(uv, (6, 1))
        CCT = np.tile(CCT, 6)
        np.testing.assert_allclose(
            uv_to_CCT_Planck1900(uv), CCT, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        uv = np.reshape(uv, (2, 3, 2))
        CCT = np.reshape(CCT, (2, 3))
        np.testing.assert_allclose(
            uv_to_CCT_Planck1900(uv), CCT, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_uv_to_CCT_Planck1900(self):
        """
        Test :func:`colour.temperature.planck1900.uv_to_CCT_Planck1900`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        uv_to_CCT_Planck1900(cases)


class TestCCT_to_uv_Planck1900(unittest.TestCase):
    """
    Define :func:`colour.temperature.planck1900.CCT_to_uv_Planck1900` definition
    unit tests methods.
    """

    def test_CCT_to_uv_Planck1900(self):
        """
        Test :func:`colour.temperature.planck1900.CCT_to_uv_Planck1900`
        definition.
        """

        np.testing.assert_allclose(
            CCT_to_uv_Planck1900(4000),
            np.array([0.225109670227493, 0.334387366663923]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CCT_to_uv_Planck1900(7000),
            np.array([0.198126929048352, 0.307025980523306]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CCT_to_uv_Planck1900(25000),
            np.array([0.182932683590136, 0.274073232217536]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CCT_to_uv_Planck1900(self):
        """
        Test :func:`colour.temperature.planck1900.CCT_to_uv_Planck1900` definition
        n-dimensional arrays support.
        """

        CCT = 4000
        uv = CCT_to_uv_Planck1900(CCT)

        CCT = np.tile(CCT, 6)
        uv = np.tile(uv, (6, 1))
        np.testing.assert_allclose(
            CCT_to_uv_Planck1900(CCT), uv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        CCT = np.reshape(CCT, (2, 3))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_allclose(
            CCT_to_uv_Planck1900(CCT), uv, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_CCT_to_uv_Planck1900(self):
        """
        Test :func:`colour.temperature.planck1900.CCT_to_uv_Planck1900` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        CCT_to_uv_Planck1900(cases)


if __name__ == "__main__":
    unittest.main()
