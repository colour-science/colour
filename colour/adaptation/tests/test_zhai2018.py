"""Define the unit tests for the :mod:`colour.adaptation.zhai2018` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.adaptation import chromatic_adaptation_Zhai2018
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
    "TestChromaticAdaptationZhai2018",
]


class TestChromaticAdaptationZhai2018:
    """
    Define :func:`colour.adaptation.zhai2018.chromatic_adaptation_Zhai2018`
    definition unit tests methods.
    """

    def test_chromatic_adaptation_Zhai2018(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.zhai2018.chromatic_adaptation_Zhai2018`
        definition.
        """

        xp_assert_close(
            chromatic_adaptation_Zhai2018(
                XYZ_b=xp_as_array([48.900, 43.620, 6.250], xp=xp),
                XYZ_wb=xp_as_array([109.850, 100, 35.585], xp=xp),
                XYZ_wd=xp_as_array([95.047, 100, 108.883], xp=xp),
                D_b=0.9407,
                D_d=0.9800,
                XYZ_wo=xp_as_array([100, 100, 100], xp=xp),
            ),
            [39.18561644, 42.15461798, 19.23672036],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_Zhai2018(
                XYZ_b=xp_as_array([48.900, 43.620, 6.250], xp=xp),
                XYZ_wb=xp_as_array([109.850, 100, 35.585], xp=xp),
                XYZ_wd=xp_as_array([95.047, 100, 108.883], xp=xp),
                D_b=0.9407,
                D_d=0.9800,
                XYZ_wo=xp_as_array([100, 100, 100], xp=xp),
                transform="CAT16",
            ),
            [40.37398343, 43.69426311, 20.51733764],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_Zhai2018(
                XYZ_b=xp_as_array([52.034, 58.824, 23.703], xp=xp),
                XYZ_wb=xp_as_array([92.288, 100, 38.775], xp=xp),
                XYZ_wd=xp_as_array([105.432, 100, 137.392], xp=xp),
                D_b=0.6709,
                D_d=0.5331,
                XYZ_wo=xp_as_array([97.079, 100, 141.798], xp=xp),
            ),
            [57.03242915, 58.93434364, 64.76261333],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_Zhai2018(
                XYZ_b=xp_as_array([52.034, 58.824, 23.703], xp=xp),
                XYZ_wb=xp_as_array([92.288, 100, 38.775], xp=xp),
                XYZ_wd=xp_as_array([105.432, 100, 137.392], xp=xp),
                D_b=0.6709,
                D_d=0.5331,
                XYZ_wo=xp_as_array([97.079, 100, 141.798], xp=xp),
                transform="CAT16",
            ),
            [56.77130011, 58.81317888, 64.66922808],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_Zhai2018(
                XYZ_b=xp_as_array([48.900, 43.620, 6.250], xp=xp),
                XYZ_wb=xp_as_array([109.850, 100, 35.585], xp=xp),
                XYZ_wd=xp_as_array([95.047, 100, 108.883], xp=xp),
            ),
            [38.72444735, 42.09232891, 20.05297620],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_chromatic_adaptation_Zhai2018(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.zhai2018.chromatic_adaptation_Zhai2018`
        definition n-dimensional arrays support.
        """

        XYZ_b = xp_as_array([48.900, 43.620, 6.250], xp=xp)
        XYZ_wb = xp_as_array([109.850, 100, 35.585], xp=xp)
        XYZ_wd = xp_as_array([95.047, 100, 108.883], xp=xp)
        D_b = 0.9407
        D_d = 0.9800
        XYZ_d = as_ndarray(
            chromatic_adaptation_Zhai2018(XYZ_b, XYZ_wb, XYZ_wd, D_b, D_d)
        )

        XYZ_b = xp.tile(xp_as_array(XYZ_b, xp=xp), (6, 1))
        XYZ_d = xp.tile(xp_as_array(XYZ_d, xp=xp), (6, 1))
        xp_assert_close(
            chromatic_adaptation_Zhai2018(XYZ_b, XYZ_wb, XYZ_wd, D_b, D_d),
            XYZ_d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_wb = xp.tile(xp_as_array(XYZ_wb, xp=xp), (6, 1))
        XYZ_wd = xp.tile(xp_as_array(XYZ_wd, xp=xp), (6, 1))
        D_b = xp.tile(xp_as_array(D_b, xp=xp), (6, 1))
        D_d = xp.tile(xp_as_array(D_d, xp=xp), (6, 1))
        xp_assert_close(
            chromatic_adaptation_Zhai2018(XYZ_b, XYZ_wb, XYZ_wd, D_b, D_d),
            XYZ_d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_b = xp_reshape(xp_as_array(XYZ_b, xp=xp), (2, 3, 3), xp=xp)
        XYZ_wb = xp_reshape(xp_as_array(XYZ_wb, xp=xp), (2, 3, 3), xp=xp)
        XYZ_wd = xp_reshape(xp_as_array(XYZ_wd, xp=xp), (2, 3, 3), xp=xp)
        D_b = xp_reshape(xp_as_array(D_b, xp=xp), (2, 3, 1), xp=xp)
        D_d = xp_reshape(xp_as_array(D_d, xp=xp), (2, 3, 1), xp=xp)
        XYZ_d = xp_reshape(xp_as_array(XYZ_d, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            chromatic_adaptation_Zhai2018(XYZ_b, XYZ_wb, XYZ_wd, D_b, D_d),
            XYZ_d,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_chromatic_adaptation_Zhai2018(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.zhai2018.chromatic_adaptation_Zhai2018`
        definition domain and range scale support.
        """

        XYZ_b = xp_as_array([48.900, 43.620, 6.250], xp=xp)
        XYZ_wb = xp_as_array([109.850, 100, 35.585], xp=xp)
        XYZ_wd = xp_as_array([95.047, 100, 108.883], xp=xp)
        XYZ_d = as_ndarray(chromatic_adaptation_Zhai2018(XYZ_b, XYZ_wb, XYZ_wd))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    chromatic_adaptation_Zhai2018(
                        XYZ_b * factor, XYZ_wb * factor, XYZ_wd * factor
                    ),
                    XYZ_d * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_Zhai2018(self) -> None:
        """
        Test :func:`colour.adaptation.zhai2018.chromatic_adaptation_Zhai2018`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        chromatic_adaptation_Zhai2018(
            cases, cases, cases, cases[0, 0], cases[0, 0], cases
        )
