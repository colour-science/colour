# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.temperature.kang2002` module."""

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.temperature import CCT_to_xy_Kang2002, xy_to_CCT_Kang2002
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXy_to_CCT_Kang2002",
    "TestCCT_to_xy_Kang2002",
]


class TestXy_to_CCT_Kang2002:
    """
    Define :func:`colour.temperature.kang2002.xy_to_CCT_Kang2002`
    definition unit tests methods.
    """

    def test_xy_to_CCT_Kang2002(self):
        """
        Test :func:`colour.temperature.kang2002.xy_to_CCT_Kang2002`
        definition.
        """

        np.testing.assert_allclose(
            xy_to_CCT_Kang2002(
                np.array([0.380528282812500, 0.376733530961114]),
                {"method": "Nelder-Mead"},
            ),
            4000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_CCT_Kang2002(
                np.array([0.306374019533528, 0.316552869726577]),
                {"method": "Nelder-Mead"},
            ),
            7000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_CCT_Kang2002(
                np.array([0.252472994438400, 0.252254791243654]),
                {"method": "Nelder-Mead"},
            ),
            25000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xy_to_CCT_Kang2002(self):
        """
        Test :func:`colour.temperature.kang2002.xy_to_CCT_Kang2002`
        definition n-dimensional arrays support.
        """

        uv = np.array([0.380528282812500, 0.376733530961114])
        CCT = xy_to_CCT_Kang2002(uv)

        uv = np.tile(uv, (6, 1))
        CCT = np.tile(CCT, 6)
        np.testing.assert_allclose(
            xy_to_CCT_Kang2002(uv), CCT, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        uv = np.reshape(uv, (2, 3, 2))
        CCT = np.reshape(CCT, (2, 3))
        np.testing.assert_allclose(
            xy_to_CCT_Kang2002(uv), CCT, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_xy_to_CCT_Kang2002(self):
        """
        Test :func:`colour.temperature.kang2002.xy_to_CCT_Kang2002`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_CCT_Kang2002(cases)


class TestCCT_to_xy_Kang2002:
    """
    Define :func:`colour.temperature.kang2002.CCT_to_xy_Kang2002` definition
    unit tests methods.
    """

    def test_CCT_to_xy_Kang2002(self):
        """
        Test :func:`colour.temperature.kang2002.CCT_to_xy_Kang2002`
        definition.
        """

        np.testing.assert_allclose(
            CCT_to_xy_Kang2002(4000),
            np.array([0.380528282812500, 0.376733530961114]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CCT_to_xy_Kang2002(7000),
            np.array([0.306374019533528, 0.316552869726577]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            CCT_to_xy_Kang2002(25000),
            np.array([0.252472994438400, 0.252254791243654]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CCT_to_xy_Kang2002(self):
        """
        Test :func:`colour.temperature.kang2002.CCT_to_xy_Kang2002` definition
        n-dimensional arrays support.
        """

        CCT = 4000
        xy = CCT_to_xy_Kang2002(CCT)

        CCT = np.tile(CCT, 6)
        xy = np.tile(xy, (6, 1))
        np.testing.assert_allclose(
            CCT_to_xy_Kang2002(CCT), xy, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        CCT = np.reshape(CCT, (2, 3))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_allclose(
            CCT_to_xy_Kang2002(CCT), xy, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_CCT_to_xy_Kang2002(self):
        """
        Test :func:`colour.temperature.kang2002.CCT_to_xy_Kang2002` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        CCT_to_xy_Kang2002(cases)
