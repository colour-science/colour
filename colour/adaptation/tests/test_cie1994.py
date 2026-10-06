"""Define the unit tests for the :mod:`colour.adaptation.cie1994` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.adaptation import chromatic_adaptation_CIE1994
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
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
    "TestChromaticAdaptationCIE1994",
]


class TestChromaticAdaptationCIE1994:
    """
    Define :func:`colour.adaptation.cie1994.chromatic_adaptation_CIE1994`
    definition unit tests methods.
    """

    def test_chromatic_adaptation_CIE1994(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.cie1994.chromatic_adaptation_CIE1994`
        definition.
        """

        xp_assert_close(
            chromatic_adaptation_CIE1994(
                XYZ_1=xp_as_array([28.00, 21.26, 5.27], xp=xp),
                xy_o1=xp_as_array([0.44760, 0.40740], xp=xp),
                xy_o2=xp_as_array([0.31270, 0.32900], xp=xp),
                Y_o=20,
                E_o1=1000,
                E_o2=1000,
            ),
            [24.03379521, 21.15621214, 17.64301199],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_CIE1994(
                XYZ_1=xp_as_array([21.77, 19.18, 16.73], xp=xp),
                xy_o1=xp_as_array([0.31270, 0.32900], xp=xp),
                xy_o2=xp_as_array([0.31270, 0.32900], xp=xp),
                Y_o=50,
                E_o1=100,
                E_o2=1000,
            ),
            [21.12891746, 19.42980532, 19.49577765],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_CIE1994(
                XYZ_1=xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp) * 100,
                xy_o1=xp_as_array([0.31270, 0.32900], xp=xp),
                xy_o2=xp_as_array([0.37208, 0.37529], xp=xp),
                Y_o=20,
                E_o1=100,
                E_o2=1000,
            ),
            [9.14287406, 9.35843355, 15.95753504],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_chromatic_adaptation_CIE1994(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.cie1994.chromatic_adaptation_CIE1994`
        definition n-dimensional arrays support.
        """

        XYZ_1 = xp_as_array([28.00, 21.26, 5.27], xp=xp)
        xy_o1 = xp_as_array([0.44760, 0.40740], xp=xp)
        xy_o2 = xp_as_array([0.31270, 0.32900], xp=xp)
        Y_o = 20
        E_o1 = 1000
        E_o2 = 1000
        XYZ_2 = as_ndarray(
            chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)
        )

        XYZ_1 = xp.tile(xp_as_array(XYZ_1, xp=xp), (6, 1))
        XYZ_2 = xp.tile(xp_as_array(XYZ_2, xp=xp), (6, 1))
        xp_assert_close(
            chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2),
            XYZ_2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xy_o1 = xp.tile(xp_as_array(xy_o1, xp=xp), (6, 1))
        xy_o2 = xp.tile(xp_as_array(xy_o2, xp=xp), (6, 1))
        Y_o = xp.tile(xp_as_array(Y_o, xp=xp), (6,))
        E_o1 = xp.tile(xp_as_array(E_o1, xp=xp), (6,))
        E_o2 = xp.tile(xp_as_array(E_o2, xp=xp), (6,))
        xp_assert_close(
            chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2),
            XYZ_2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_1 = xp_reshape(xp_as_array(XYZ_1, xp=xp), (2, 3, 3), xp=xp)
        xy_o1 = xp_reshape(xp_as_array(xy_o1, xp=xp), (2, 3, 2), xp=xp)
        xy_o2 = xp_reshape(xp_as_array(xy_o2, xp=xp), (2, 3, 2), xp=xp)
        Y_o = xp_reshape(xp_as_array(Y_o, xp=xp), (2, 3), xp=xp)
        E_o1 = xp_reshape(xp_as_array(E_o1, xp=xp), (2, 3), xp=xp)
        E_o2 = xp_reshape(xp_as_array(E_o2, xp=xp), (2, 3), xp=xp)
        XYZ_2 = xp_reshape(xp_as_array(XYZ_2, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2),
            XYZ_2,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_chromatic_adaptation_CIE1994(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.cie1994.chromatic_adaptation_CIE1994`
        definition domain and range scale support.
        """

        XYZ_1 = xp_as_array([28.00, 21.26, 5.27], xp=xp)
        xy_o1 = xp_as_array([0.44760, 0.40740], xp=xp)
        xy_o2 = xp_as_array([0.31270, 0.32900], xp=xp)
        Y_o = 20
        E_o1 = 1000
        E_o2 = 1000
        XYZ_2 = as_ndarray(
            chromatic_adaptation_CIE1994(XYZ_1, xy_o1, xy_o2, Y_o, E_o1, E_o2)
        )

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    chromatic_adaptation_CIE1994(
                        XYZ_1 * factor, xy_o1, xy_o2, Y_o * factor, E_o1, E_o2
                    ),
                    XYZ_2 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_CIE1994(self) -> None:
        """
        Test :func:`colour.adaptation.cie1994.chromatic_adaptation_CIE1994`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        chromatic_adaptation_CIE1994(
            cases,
            cases[..., 0:2],
            cases[..., 0:2],
            cases[..., 0],
            cases[..., 0],
            cases[..., 0],
        )
