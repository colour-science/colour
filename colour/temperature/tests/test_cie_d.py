# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.temperature.cie_d` module."""

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import CCT_to_xy_CIE_D, xy_to_CCT_CIE_D
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXy_to_CCT_CIE_D",
    "TestCCT_to_xy_CIE_D",
]


class TestXy_to_CCT_CIE_D:
    """
    Define :func:`colour.temperature.cie_d.xy_to_CCT_CIE_D` definition unit
    tests methods.
    """

    def test_xy_to_CCT_CIE_D(self):
        """Test :func:`colour.temperature.cie_d.xy_to_CCT_CIE_D` definition."""

        np.testing.assert_allclose(
            xy_to_CCT_CIE_D(
                np.array([0.382343625000000, 0.383766261015578]),
                {"method": "Nelder-Mead"},
            ),
            4000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_CCT_CIE_D(
                np.array([0.305357431486880, 0.321646345474552]),
                {"method": "Nelder-Mead"},
            ),
            7000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_CCT_CIE_D(
                np.array([0.24985367, 0.254799464210944]),
                {"method": "Nelder-Mead"},
            ),
            25000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xy_to_CCT_CIE_D(self):
        """
        Test :func:`colour.temperature.cie_d.xy_to_CCT_CIE_D` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.382343625000000, 0.383766261015578])
        CCT = xy_to_CCT_CIE_D(xy)

        xy = np.tile(xy, (6, 1))
        CCT = np.tile(CCT, 6)
        np.testing.assert_allclose(
            xy_to_CCT_CIE_D(xy), CCT, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        xy = np.reshape(xy, (2, 3, 2))
        CCT = np.reshape(CCT, (2, 3))
        np.testing.assert_allclose(
            xy_to_CCT_CIE_D(xy), CCT, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_xy_to_CCT_CIE_D(self):
        """
        Test :func:`colour.temperature.cie_d.xy_to_CCT_CIE_D` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_CCT_CIE_D(cases)


class TestCCT_to_xy_CIE_D:
    """
    Define :func:`colour.temperature.cie_d.CCT_to_xy_CIE_D` definition
    unit tests methods.
    """

    def test_CCT_to_xy_CIE_D(self):
        """Test :func:`colour.temperature.cie_d.CCT_to_xy_CIE_D` definition."""

        np.testing.assert_allclose(
            CCT_to_xy_CIE_D(4000),
            np.array([0.382343625000000, 0.383766261015578]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CCT_to_xy_CIE_D(7000),
            np.array([0.305357431486880, 0.321646345474552]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CCT_to_xy_CIE_D(25000),
            np.array([0.24985367, 0.254799464210944]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CCT_to_xy_CIE_D(self):
        """
        Test :func:`colour.temperature.cie_d.CCT_to_xy_CIE_D` definition
        n-dimensional arrays support.
        """

        CCT = 4000
        xy = CCT_to_xy_CIE_D(CCT)

        CCT = np.tile(CCT, 6)
        xy = np.tile(xy, (6, 1))
        np.testing.assert_allclose(
            CCT_to_xy_CIE_D(CCT), xy, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        CCT = np.reshape(CCT, (2, 3))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_allclose(
            CCT_to_xy_CIE_D(CCT), xy, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_CCT_to_xy_CIE_D(self):
        """
        Test :func:`colour.temperature.cie_d.CCT_to_xy_CIE_D` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        CCT_to_xy_CIE_D(cases)
