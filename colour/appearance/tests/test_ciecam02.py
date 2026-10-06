"""Define the unit tests for the :mod:`colour.appearance.ciecam02` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np
import pytest

from colour.appearance import (
    VIEWING_CONDITIONS_CIECAM02,
    CAM_Specification_CIECAM02,
    CIECAM02_to_XYZ,
    InductionFactors_CIECAM02,
    XYZ_to_CIECAM02,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_float_array,
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    tsplit,
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
    "TestXYZ_to_CIECAM02",
    "TestCIECAM02_to_XYZ",
]


class TestXYZ_to_CIECAM02:
    """
    Define :func:`colour.appearance.ciecam02.XYZ_to_CIECAM02` definition unit
    tests methods.
    """

    def test_XYZ_to_CIECAM02(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.ciecam02.XYZ_to_CIECAM02` definition.

        Notes
        -----
        -   The test values have been generated from data of the following file
            by *Fairchild (2013)*:
            http://rit-mcsl.org/fairchild//files/AppModEx.xls
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = InductionFactors_CIECAM02(1, 0.69, 1)
        xp_assert_close(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            [41.73, 0.1, 219, 2.36, 195.37, 0.11, 278.1, np.nan],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        XYZ = xp_as_array([57.06, 43.06, 31.96], xp=xp)
        L_A = 31.83
        xp_assert_close(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            [65.96, 48.57, 19.6, 52.25, 152.67, 41.67, 399.6, np.nan],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        XYZ = xp_as_array([3.53, 6.56, 2.14], xp=xp)
        XYZ_w = xp_as_array([109.85, 100.00, 35.58], xp=xp)
        L_A = 318.31
        xp_assert_close(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            [21.79, 46.94, 177.1, 58.79, 141.17, 48.8, 220.4, np.nan],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        L_A = 31.83
        xp_assert_close(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            [42.53, 51.92, 248.9, 60.22, 122.83, 44.54, 305.8, np.nan],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        XYZ = xp_as_array([61.45276998, 7.00421901, 82.24067384], xp=xp)
        XYZ_w = xp_as_array([95.05, 100, 108.88], xp=xp)
        L_A = 4.074366543152521
        xp_assert_close(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            [
                21.72630603341673,
                411.5190338631848,
                349.12875710099053,
                227.15081998415593,
                57.657243286322725,
                297.49693233026602,
                375.5788601911363,
                np.nan,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

    def test_n_dimensional_XYZ_to_CIECAM02(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.ciecam02.XYZ_to_CIECAM02` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CIECAM02["Average"]
        specification = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True)

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        specification = np.tile(specification, (6, 1))
        xp_assert_close(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        specification = xp_reshape(xp_as_array(specification, xp=xp), (2, 3, 8), xp=xp)
        xp_assert_close(
            XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_CIECAM02(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.ciecam02.XYZ_to_CIECAM02` definition
        domain and range scale support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CIECAM02["Average"]
        specification = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True)

        d_r = (
            ("reference", 1, 1),
            (
                "1",
                0.01,
                np.array(
                    [
                        1 / 100,
                        1 / 100,
                        1 / 360,
                        1 / 100,
                        1 / 100,
                        1 / 100,
                        1 / 400,
                        np.nan,
                    ]
                ),
            ),
            (
                "100",
                1,
                np.array([1, 1, 100 / 360, 1, 1, 1, 100 / 400, np.nan]),
            ),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_CIECAM02(
                        XYZ * xp_as_array(factor_a, xp=xp),
                        XYZ_w * xp_as_array(factor_a, xp=xp),
                        L_A,
                        Y_b,
                        surround,
                        compute_H=True,
                    ),
                    as_float_array(specification) * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_CIECAM02(self) -> None:
        """
        Test :func:`colour.appearance.ciecam02.XYZ_to_CIECAM02` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        surround = InductionFactors_CIECAM02(cases[0, 0], cases[0, 0], cases[0, 0])
        XYZ_to_CIECAM02(
            cases, cases, cases[..., 0], cases[..., 0], surround, compute_H=True
        )


class TestCIECAM02_to_XYZ:
    """
    Define :func:`colour.appearance.ciecam02.CIECAM02_to_XYZ` definition unit
    tests methods.
    """

    def test_CIECAM02_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.appearance.ciecam02.CIECAM02_to_XYZ` definition."""

        specification = CAM_Specification_CIECAM02(
            41.73, 0.1, 219, 2.36, 195.37, 0.11, 278.1
        )
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = InductionFactors_CIECAM02(1, 0.69, 1)
        xp_assert_close(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            [19.01, 20.00, 21.78],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        specification = CAM_Specification_CIECAM02(
            65.96, 48.57, 19.6, 52.25, 152.67, 41.67, 399.6, np.nan
        )
        L_A = 31.83
        xp_assert_close(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            [57.06, 43.06, 31.96],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        specification = CAM_Specification_CIECAM02(
            21.79, 46.94, 177.1, 58.79, 141.17, 48.8, 220.4, np.nan
        )
        XYZ_w = xp_as_array([109.85, 100.00, 35.58], xp=xp)
        L_A = 318.31
        xp_assert_close(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            [3.53, 6.56, 2.14],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        specification = CAM_Specification_CIECAM02(
            42.53, 51.92, 248.9, 60.22, 122.83, 44.54, 305.8, np.nan
        )
        L_A = 31.83
        xp_assert_close(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            [19.01, 20.00, 21.78],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

        specification = CAM_Specification_CIECAM02(
            21.72630603341673,
            411.5190338631848,
            349.12875710099053,
            227.15081998415593,
            57.657243286322725,
            297.49693233026602,
            375.5788601911363,
            np.nan,
        )
        XYZ_w = xp_as_array([95.05, 100, 108.88], xp=xp)
        L_A = 4.074366543152521
        xp_assert_close(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            [61.45276998, 7.00421901, 82.24067384],
            atol=TOLERANCE_ABSOLUTE_TESTS * 500000,
        )

    def test_n_dimensional_CIECAM02_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.ciecam02.CIECAM02_to_XYZ` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CIECAM02["Average"]
        specification = XYZ_to_CIECAM02(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True)
        XYZ = as_ndarray(CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround))

        specification = CAM_Specification_CIECAM02(
            *np.transpose(np.tile(tsplit(specification), (6, 1))).tolist()
        )
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        xp_assert_close(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        specification = CAM_Specification_CIECAM02(
            *tsplit(np.reshape(specification, (2, 3, 8))).tolist()
        )
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_CIECAM02_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.ciecam02.CIECAM02_to_XYZ` definition
        domain and range scale support.
        """

        XYZ_i = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CIECAM02["Average"]
        specification = XYZ_to_CIECAM02(
            XYZ_i, XYZ_w, L_A, Y_b, surround, compute_H=True
        )
        XYZ = as_ndarray(CIECAM02_to_XYZ(specification, XYZ_w, L_A, Y_b, surround))

        d_r = (
            ("reference", 1, 1),
            (
                "1",
                np.array(
                    [
                        1 / 100,
                        1 / 100,
                        1 / 360,
                        1 / 100,
                        1 / 100,
                        1 / 100,
                        1 / 400,
                        np.nan,
                    ]
                ),
                0.01,
            ),
            (
                "100",
                np.array([1, 1, 100 / 360, 1, 1, 1, 100 / 400, np.nan]),
                1,
            ),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    CIECAM02_to_XYZ(
                        specification * xp_as_array(factor_a, xp=xp),
                        XYZ_w * factor_b,
                        L_A,
                        Y_b,
                        surround,
                    ),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_raise_exception_CIECAM02_to_XYZ(self) -> None:
        """
        Test :func:`colour.appearance.ciecam02.CIECAM02_to_XYZ` definition
        raised exception.
        """

        with pytest.raises(ValueError):
            CIECAM02_to_XYZ(
                CAM_Specification_CIECAM02(41.73109113251392, None, 219.04843265831178),
                np.array([95.05, 100.0, 108.88]),
                318.31,
                20.0,
                VIEWING_CONDITIONS_CIECAM02["Average"],
            )

    @ignore_numpy_errors
    def test_nan_CIECAM02_to_XYZ(self) -> None:
        """
        Test :func:`colour.appearance.ciecam02.CIECAM02_to_XYZ` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        surround = InductionFactors_CIECAM02(cases[0, 0], cases[0, 0], cases[0, 0])
        CIECAM02_to_XYZ(
            CAM_Specification_CIECAM02(
                cases[..., 0], cases[..., 0], cases[..., 0], M=50
            ),
            cases,
            cases[..., 0],
            cases[..., 0],
            surround,
        )
