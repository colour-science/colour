"""Define the unit tests for the :mod:`colour.appearance.hunt` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

import contextlib
from itertools import product

import numpy as np

from colour.appearance import (
    VIEWING_CONDITIONS_HUNT,
    InductionFactors_Hunt,
    XYZ_to_Hunt,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_float_array,
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
    "TestXYZ_to_Hunt",
]


class TestXYZ_to_Hunt:
    """
    Define :func:`colour.appearance.hunt.XYZ_to_Hunt` definition unit tests
    methods.
    """

    def test_XYZ_to_Hunt(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.hunt.XYZ_to_Hunt` definition.

        Notes
        -----
        -   The test values have been generated from data of the following file
            by *Fairchild (2013)*:
            http://rit-mcsl.org/fairchild//files/AppModEx.xls
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        XYZ_b = XYZ_w * xp_as_array([1, 0.2, 1], xp=xp)
        L_A = 318.31
        surround = VIEWING_CONDITIONS_HUNT["Normal Scenes"]
        CCT_w = 6504.0
        xp_assert_close(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            [42.12, 0.16, 269.3, 0.03, 31.92, 0.16, np.nan, np.nan],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        XYZ = xp_as_array([57.06, 43.06, 31.96], xp=xp)
        L_A = 31.83
        xp_assert_close(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            [66.76, 63.89, 18.6, 153.36, 31.22, 58.28, np.nan, np.nan],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        XYZ = xp_as_array([3.53, 6.56, 2.14], xp=xp)
        XYZ_w = xp_as_array([109.85, 100.00, 35.58], xp=xp)
        L_A = 318.31
        CCT_w = 2856
        xp_assert_close(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            [19.56, 74.58, 178.3, 245.4, 18.9, 76.33, np.nan, np.nan],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        L_A = 31.83
        xp_assert_close(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            [40.27, 73.84, 262.8, 209.29, 22.15, 67.35, np.nan, np.nan],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

    def test_n_dimensional_XYZ_to_Hunt(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.hunt.XYZ_to_Hunt` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        XYZ_b = XYZ_w * xp_as_array([1, 0.2, 1], xp=xp)
        L_A = 318.31
        surround = VIEWING_CONDITIONS_HUNT["Normal Scenes"]
        CCT_w = 6504.0
        specification = XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w)

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        specification = xp_as_array(np.tile(as_ndarray(specification), (6, 1)), xp=xp)
        xp_assert_close(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        XYZ_b = xp.tile(xp_as_array(XYZ_b, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        XYZ_b = xp_reshape(xp_as_array(XYZ_b, xp=xp), (2, 3, 3), xp=xp)
        specification = xp_reshape(specification, (2, 3, 8), xp=xp)
        xp_assert_close(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_Hunt(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.hunt.XYZ_to_Hunt` definition domain
        and range scale support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        XYZ_b = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        surround = VIEWING_CONDITIONS_HUNT["Normal Scenes"]
        CCT_w = 6504.0
        specification = XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w)

        d_r = (
            ("reference", 1, 1),
            ("1", 0.01, np.array([1, 1, 1 / 360, 1, 1, 1, np.nan, np.nan])),
            ("100", 1, np.array([1, 1, 100 / 360, 1, 1, 1, np.nan, np.nan])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_Hunt(
                        XYZ * xp_as_array(factor_a, xp=xp),
                        XYZ_w * xp_as_array(factor_a, xp=xp),
                        XYZ_b * xp_as_array(factor_a, xp=xp),
                        L_A,
                        surround,
                        CCT_w=CCT_w,
                    ),
                    as_float_array(specification) * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_raise_exception_XYZ_to_Hunt(self) -> None:
        """
        Test :func:`colour.appearance.hunt.XYZ_to_Hunt` definition raised
        exception.
        """

        XYZ = np.array([19.01, 20.00, 21.78])
        XYZ_w = np.array([95.05, 100.00, 108.88])
        XYZ_b = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        surround = VIEWING_CONDITIONS_HUNT["Normal Scenes"]
        CCT_w = 6504.0
        S = S_w = 0.5

        with contextlib.suppress(ValueError):
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround)

        with contextlib.suppress(ValueError):
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w, S=S)

        with contextlib.suppress(ValueError):
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w, S_w=S_w)

    @ignore_numpy_errors
    def test_XYZ_p_XYZ_to_Hunt(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.hunt.XYZ_to_Hunt` definition *XYZ_p* and
        *p* argument handling, exercising the proximal-field adjusted
        reference white branch per *Hunt (1991b)* / *Fairchild (2013)*
        Equations 12.23-12.28.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        XYZ_b = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        XYZ_p = xp_as_array([50.00, 30.00, 80.00], xp=xp)
        L_A = 318.31
        surround = VIEWING_CONDITIONS_HUNT["Normal Scenes"]
        CCT_w = 6504.0

        xp_assert_close(
            XYZ_to_Hunt(
                XYZ,
                XYZ_w,
                XYZ_b,
                L_A,
                surround,
                XYZ_p=XYZ_p,
                p=0.5,
                CCT_w=CCT_w,
            ),
            [
                28.36030943086153,
                24.97959282007880,
                317.98269937454876,
                46.96217778613629,
                17.71048437831100,
                25.56679971175820,
                np.nan,
                np.nan,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Hunt(self) -> None:
        """
        Test :func:`colour.appearance.hunt.XYZ_to_Hunt` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        surround = InductionFactors_Hunt(cases[0, 0], cases[0, 0])
        XYZ_to_Hunt(
            cases,
            cases,
            cases,
            cases[0, 0],
            surround,
            cases[0, 0],
            CCT_w=cases,
        )
