"""Define the unit tests for the :mod:`colour.adaptation.li2025` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.adaptation import chromatic_adaptation_Li2025
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
    "TestChromaticAdaptationLi2025",
]


class TestChromaticAdaptationLi2025:
    """
    Define :func:`colour.adaptation.li2025.chromatic_adaptation_Li2025`
    definition unit tests methods.
    """

    def test_chromatic_adaptation_Li2025(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.li2025.chromatic_adaptation_Li2025`
        definition.
        """

        xp_assert_close(
            chromatic_adaptation_Li2025(
                XYZ_s=xp_as_array([48.900, 43.620, 6.250], xp=xp),
                XYZ_ws=xp_as_array([109.850, 100, 35.585], xp=xp),
                XYZ_wd=xp_as_array([95.047, 100, 108.883], xp=xp),
                L_A=318.31,
                F_surround=1.0,
            ),
            [40.00725815, 43.70148954, 21.32902932],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_Li2025(
                XYZ_s=xp_as_array([52.034, 58.824, 23.703], xp=xp),
                XYZ_ws=xp_as_array([92.288, 100, 38.775], xp=xp),
                XYZ_wd=xp_as_array([105.432, 100, 137.392], xp=xp),
                L_A=318.31,
                F_surround=1.0,
            ),
            [59.99869086, 58.81067197, 83.41018242],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_Li2025(
                XYZ_s=xp_as_array([48.900, 43.620, 6.250], xp=xp),
                XYZ_ws=xp_as_array([109.850, 100, 35.585], xp=xp),
                XYZ_wd=xp_as_array([95.047, 100, 108.883], xp=xp),
                L_A=20.0,
                F_surround=1.0,
            ),
            [41.22388901, 43.69034082, 19.26604215],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_Li2025(
                XYZ_s=xp_as_array([48.900, 43.620, 6.250], xp=xp),
                XYZ_ws=xp_as_array([109.850, 100, 35.585], xp=xp),
                XYZ_wd=xp_as_array([95.047, 100, 108.883], xp=xp),
                L_A=318.31,
                F_surround=1.0,
                discount_illuminant=True,
            ),
            [39.95779686, 43.70194278, 21.41289865],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_chromatic_adaptation_Li2025(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.li2025.chromatic_adaptation_Li2025`
        definition n-dimensional arrays support.
        """

        XYZ_s = xp_as_array([48.900, 43.620, 6.250], xp=xp)
        XYZ_ws = xp_as_array([109.850, 100, 35.585], xp=xp)
        XYZ_wd = xp_as_array([95.047, 100, 108.883], xp=xp)
        L_A = 318.31
        F_surround = 1.0
        XYZ_d = as_ndarray(
            chromatic_adaptation_Li2025(XYZ_s, XYZ_ws, XYZ_wd, L_A, F_surround)
        )

        XYZ_s = xp.tile(xp_as_array(XYZ_s, xp=xp), (6, 1))
        XYZ_d = xp.tile(xp_as_array(XYZ_d, xp=xp), (6, 1))
        xp_assert_close(
            chromatic_adaptation_Li2025(XYZ_s, XYZ_ws, XYZ_wd, L_A, F_surround),
            XYZ_d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_ws = xp.tile(xp_as_array(XYZ_ws, xp=xp), (6, 1))
        XYZ_wd = xp.tile(xp_as_array(XYZ_wd, xp=xp), (6, 1))
        L_A = xp.tile(xp_as_array(L_A, xp=xp), (6,))
        F_surround = xp.tile(xp_as_array(F_surround, xp=xp), (6,))
        xp_assert_close(
            chromatic_adaptation_Li2025(XYZ_s, XYZ_ws, XYZ_wd, L_A, F_surround),
            XYZ_d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_s = xp_reshape(xp_as_array(XYZ_s, xp=xp), (2, 3, 3), xp=xp)
        XYZ_ws = xp_reshape(xp_as_array(XYZ_ws, xp=xp), (2, 3, 3), xp=xp)
        XYZ_wd = xp_reshape(xp_as_array(XYZ_wd, xp=xp), (2, 3, 3), xp=xp)
        L_A = xp_reshape(xp_as_array(L_A, xp=xp), (2, 3), xp=xp)
        F_surround = xp_reshape(xp_as_array(F_surround, xp=xp), (2, 3), xp=xp)
        XYZ_d = xp_reshape(xp_as_array(XYZ_d, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            chromatic_adaptation_Li2025(XYZ_s, XYZ_ws, XYZ_wd, L_A, F_surround),
            XYZ_d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_chromatic_adaptation_Li2025(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.li2025.chromatic_adaptation_Li2025`
        definition domain and range scale support.
        """

        XYZ_s = xp_as_array([48.900, 43.620, 6.250], xp=xp)
        XYZ_ws = xp_as_array([109.850, 100, 35.585], xp=xp)
        XYZ_wd = xp_as_array([95.047, 100, 108.883], xp=xp)
        L_A = 318.31
        F_surround = 1.0
        XYZ_d = as_ndarray(
            chromatic_adaptation_Li2025(XYZ_s, XYZ_ws, XYZ_wd, L_A, F_surround)
        )

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    chromatic_adaptation_Li2025(
                        XYZ_s * factor,
                        XYZ_ws * factor,
                        XYZ_wd * factor,
                        L_A,
                        F_surround,
                    ),
                    XYZ_d * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_Li2025(self) -> None:
        """
        Test :func:`colour.adaptation.li2025.chromatic_adaptation_Li2025`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        chromatic_adaptation_Li2025(cases, cases, cases, cases[0, 0], cases[0, 0])
