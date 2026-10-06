"""Define the unit tests for the :mod:`colour.appearance.llab` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product
from unittest import mock

import numpy as np

from colour.appearance import (
    VIEWING_CONDITIONS_LLAB,
    InductionFactors_LLAB,
    XYZ_to_LLAB,
    llab,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_float_array,
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
    "TestXYZ_to_LLAB",
]


class TestXYZ_to_LLAB:
    """
    Define :func:`colour.appearance.llab.XYZ_to_LLAB` definition unit tests
    methods.
    """

    def test_XYZ_to_LLAB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.llab.XYZ_to_LLAB` definition.

        Notes
        -----
        -   Reference values are taken from *Fairchild (2013)* Table 14.3 for
            the *LLAB(l:c)* colour appearance model. The
            ``MATRIX_RGB_TO_XYZ_LLAB`` constant is patched to its 4-decimal
            rounded form to match the precision of the published reference
            data.
        """

        with mock.patch(
            "colour.appearance.llab.MATRIX_RGB_TO_XYZ_LLAB",
            np.around(np.linalg.inv(llab.MATRIX_XYZ_TO_RGB_LLAB), decimals=4),
        ):
            surround = VIEWING_CONDITIONS_LLAB["ref_average_4_minus"]

            XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
            XYZ_0 = xp_as_array([95.05, 100.00, 108.88], xp=xp)
            Y_b = 20.0
            L = 318.31
            xp_assert_close(
                XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
                [37.37, 0.01, 229.5, 0, 0.02, np.nan, -0.01, -0.01],
                atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
            )

            XYZ = xp_as_array([57.06, 43.06, 31.96], xp=xp)
            L = 31.83
            xp_assert_close(
                XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
                [61.26, 30.51, 22.3, 0.5, 56.55, np.nan, 52.33, 21.43],
                atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
            )

            XYZ = xp_as_array([3.53, 6.56, 2.14], xp=xp)
            XYZ_0 = xp_as_array([109.85, 100.00, 35.58], xp=xp)
            L = 318.31
            xp_assert_close(
                XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
                [16.25, 30.43, 173.8, 1.87, 53.83, np.nan, -53.51, 5.83],
                atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
            )

            XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
            L = 31.83
            xp_assert_close(
                XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
                [39.82, 29.34, 271.9, 0.74, 54.59, np.nan, 1.76, -54.56],
                atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
            )

    def test_n_dimensional_XYZ_to_LLAB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.llab.XYZ_to_LLAB` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_0 = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        Y_b = 20.0
        L = 318.31
        surround = VIEWING_CONDITIONS_LLAB["ref_average_4_minus"]
        specification = XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround)

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        specification = np.tile(specification, (6, 1))
        xp_assert_close(
            XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_0 = xp.tile(xp_as_array(XYZ_0, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_0 = xp_reshape(xp_as_array(XYZ_0, xp=xp), (2, 3, 3), xp=xp)
        specification = xp_reshape(xp_as_array(specification, xp=xp), (2, 3, 8), xp=xp)
        xp_assert_close(
            XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_colourspace_conversion_matrices_precision(self) -> None:
        """
        Test for loss of precision in conversion between *LLAB(l:c)* colour
        appearance model *CIE XYZ* tristimulus values and normalised cone
        responses matrix.
        """

        start = np.array([1.0, 1.0, 1.0])
        result = np.array(start)
        for _ in range(100000):
            result = llab.MATRIX_RGB_TO_XYZ_LLAB @ result
            result = llab.MATRIX_XYZ_TO_RGB_LLAB @ result
        xp_assert_close(start, result, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_LLAB(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.llab.XYZ_to_LLAB` definition domain and
        range scale support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_0 = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        Y_b = 20.0
        L = 318.31
        surround = VIEWING_CONDITIONS_LLAB["ref_average_4_minus"]
        specification = XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround)

        d_r = (
            ("reference", 1, 1),
            ("1", 0.01, np.array([1, 1, 1 / 360, 1, 1, np.nan, 1, 1])),
            ("100", 1, np.array([1, 1, 100 / 360, 1, 1, np.nan, 1, 1])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_LLAB(
                        XYZ * xp_as_array(factor_a, xp=xp),
                        XYZ_0 * xp_as_array(factor_a, xp=xp),
                        Y_b,
                        L,
                        surround,
                    ),
                    as_float_array(specification) * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_LLAB(self) -> None:
        """
        Test :func:`colour.appearance.llab.XYZ_to_LLAB` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            XYZ = case
            XYZ_0 = case
            Y_b = case[0]
            L = case[0]
            surround = InductionFactors_LLAB(1, case[0], case[0], case[0])
            XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround)
