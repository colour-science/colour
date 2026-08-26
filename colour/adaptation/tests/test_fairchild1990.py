"""Define the unit tests for the :mod:`colour.adaptation.fairchild1990` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import contextlib
from itertools import product

import numpy as np
from numpy.linalg import LinAlgError

from colour.adaptation import chromatic_adaptation_Fairchild1990
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
    "TestChromaticAdaptationFairchild1990",
]


class TestChromaticAdaptationFairchild1990:
    """
    Define :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition unit tests methods.
    """

    def test_chromatic_adaptation_Fairchild1990(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition.
        """

        xp_assert_close(
            chromatic_adaptation_Fairchild1990(
                xp_as_array([19.53, 23.07, 24.97], xp=xp),
                xp_as_array([111.15, 100.00, 35.20], xp=xp),
                xp_as_array([94.81, 100.00, 107.30], xp=xp),
                200,
            ),
            [23.32526349, 23.32455819, 76.11593750],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_Fairchild1990(
                xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp) * 100,
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp) * 100,
                xp_as_array([1.09846607, 1.00000000, 0.35582280], xp=xp) * 100,
                200,
            ),
            [19.28089326, 22.91583715, 3.42923503],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_Fairchild1990(
                xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp) * 100,
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp) * 100,
                xp_as_array([0.99144661, 1.00000000, 0.67315942], xp=xp) * 100,
                200,
            ),
            [6.35093475, 6.13061347, 17.36852430],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        # Test with discount_illuminant=True
        xp_assert_close(
            chromatic_adaptation_Fairchild1990(
                xp_as_array([19.53, 23.07, 24.97], xp=xp),
                xp_as_array([111.15, 100.00, 35.20], xp=xp),
                xp_as_array([94.81, 100.00, 107.30], xp=xp),
                200,
                discount_illuminant=True,
            ),
            [23.32526349, 23.32455819, 76.11593750],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_chromatic_adaptation_Fairchild1990(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition n-dimensional arrays support.
        """

        XYZ_1 = xp_as_array([19.53, 23.07, 24.97], xp=xp)
        XYZ_n = xp_as_array([111.15, 100.00, 35.20], xp=xp)
        XYZ_r = xp_as_array([94.81, 100.00, 107.30], xp=xp)
        Y_n = 200
        XYZ_c = as_ndarray(chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n))

        XYZ_1 = xp.tile(xp_as_array(XYZ_1, xp=xp), (6, 1))
        XYZ_c = xp.tile(xp_as_array(XYZ_c, xp=xp), (6, 1))
        xp_assert_close(
            chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n),
            XYZ_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_n = xp.tile(xp_as_array(XYZ_n, xp=xp), (6, 1))
        XYZ_r = xp.tile(xp_as_array(XYZ_r, xp=xp), (6, 1))
        Y_n = xp.tile(xp_as_array(Y_n, xp=xp), (6,))
        xp_assert_close(
            chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n),
            XYZ_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_1 = xp_reshape(xp_as_array(XYZ_1, xp=xp), (2, 3, 3), xp=xp)
        XYZ_n = xp_reshape(xp_as_array(XYZ_n, xp=xp), (2, 3, 3), xp=xp)
        XYZ_r = xp_reshape(xp_as_array(XYZ_r, xp=xp), (2, 3, 3), xp=xp)
        Y_n = xp_reshape(xp_as_array(Y_n, xp=xp), (2, 3), xp=xp)
        XYZ_c = xp_reshape(xp_as_array(XYZ_c, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n),
            XYZ_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_chromatic_adaptation_Fairchild1990(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition domain and range scale support.
        """

        XYZ_1 = xp_as_array([19.53, 23.07, 24.97], xp=xp)
        XYZ_n = xp_as_array([111.15, 100.00, 35.20], xp=xp)
        XYZ_r = xp_as_array([94.81, 100.00, 107.30], xp=xp)
        Y_n = 200
        XYZ_c = as_ndarray(chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    chromatic_adaptation_Fairchild1990(
                        XYZ_1 * factor, XYZ_n * factor, XYZ_r * factor, Y_n
                    ),
                    XYZ_c * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_Fairchild1990(self) -> None:
        """
        Test :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            XYZ_1 = case
            XYZ_n = case
            XYZ_r = case
            Y_n = case[0]
            with contextlib.suppress(LinAlgError):
                chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n)
