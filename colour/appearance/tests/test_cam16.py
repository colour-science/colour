"""Define the unit tests for the :mod:`colour.appearance.cam16` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np
import pytest

from colour.appearance import (
    VIEWING_CONDITIONS_CAM16,
    CAM16_to_XYZ,
    CAM_Specification_CAM16,
    InductionFactors_CAM16,
    XYZ_to_CAM16,
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
    "TestXYZ_to_CAM16",
    "TestCAM16_to_XYZ",
]


class TestXYZ_to_CAM16:
    """
    Define :func:`colour.appearance.cam16.XYZ_to_CAM16` definition unit
    tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1e-1)
    def test_XYZ_to_CAM16(self, xp: ModuleType) -> None:
        """Test :func:`colour.appearance.cam16.XYZ_to_CAM16` definition."""

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CAM16["Average"]
        xp_assert_close(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            [
                41.73120791,
                0.10335574,
                217.06795977,
                2.34501507,
                195.37170899,
                0.10743677,
                275.59498615,
                np.nan,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_as_array([57.06, 43.06, 31.96], xp=xp)
        L_A = 31.83
        xp_assert_close(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            [
                65.42828069,
                49.67956420,
                17.48659243,
                52.94308868,
                152.06985268,
                42.62473321,
                398.03047943,
                np.nan,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_as_array([3.53, 6.56, 2.14], xp=xp)
        XYZ_w = xp_as_array([109.85, 100, 35.58], xp=xp)
        L_A = 318.31
        xp_assert_close(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            [
                21.36052893,
                50.99381895,
                178.86724266,
                61.57953092,
                139.78582768,
                53.00732582,
                223.01823806,
                np.nan,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        L_A = 318.31
        xp_assert_close(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            [
                41.36326063,
                52.81154022,
                258.88676291,
                53.12406914,
                194.52011798,
                54.89682038,
                311.24768647,
                np.nan,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_as_array([61.45276998, 7.00421901, 82.2406738], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 4.074366543152521
        xp_assert_close(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            [
                21.03801957,
                457.78881613,
                350.06445098,
                241.50642846,
                56.74143988,
                330.94646237,
                376.43915877,
                np.nan,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_CAM16(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.cam16.XYZ_to_CAM16` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CAM16["Average"]
        specification = XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True)

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        specification = np.tile(specification, (6, 1))
        xp_assert_close(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        specification = xp_reshape(xp_as_array(specification, xp=xp), (2, 3, 8), xp=xp)
        xp_assert_close(
            XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_CAM16(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.cam16.XYZ_to_CAM16` definition domain
        and range scale support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CAM16["Average"]
        specification = XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True)

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
                    XYZ_to_CAM16(
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
    def test_nan_XYZ_to_CAM16(self) -> None:
        """
        Test :func:`colour.appearance.cam16.XYZ_to_CAM16` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        surround = InductionFactors_CAM16(cases[0, 0], cases[0, 0], cases[0, 0])
        XYZ_to_CAM16(
            cases, cases, cases[..., 0], cases[..., 0], surround, compute_H=True
        )


class TestCAM16_to_XYZ:
    """
    Define :func:`colour.appearance.cam16.CAM16_to_XYZ` definition unit tests
    methods.
    """

    def test_CAM16_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.appearance.cam16.CAM16_to_XYZ` definition."""

        specification = CAM_Specification_CAM16(41.73120791, 0.10335574, 217.06795977)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CAM16["Average"]
        xp_assert_close(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            [19.01, 20.00, 21.78],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        specification = CAM_Specification_CAM16(65.42828069, 49.67956420, 17.48659243)
        L_A = 31.83
        xp_assert_close(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            [57.06, 43.06, 31.96],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        specification = CAM_Specification_CAM16(21.36052893, 50.99381895, 178.86724266)
        XYZ_w = xp_as_array([109.85, 100, 35.58], xp=xp)
        L_A = 318.31
        xp_assert_close(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            [3.53, 6.56, 2.14],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        specification = CAM_Specification_CAM16(41.36326063, 52.81154022, 258.88676291)
        L_A = 318.31
        xp_assert_close(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            [19.01, 20.00, 21.78],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        specification = CAM_Specification_CAM16(21.03801957, 457.78881613, 350.06445098)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 4.074366543152521
        xp_assert_close(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            [61.45276998, 7.00421901, 82.2406738],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CAM16_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.cam16.CAM16_to_XYZ` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CAM16["Average"]
        specification = XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True)
        XYZ = as_ndarray(CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround))

        specification = CAM_Specification_CAM16(
            *np.transpose(np.tile(tsplit(specification), (6, 1))).tolist()
        )
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        xp_assert_close(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        specification = CAM_Specification_CAM16(
            *tsplit(np.reshape(specification, (2, 3, 8))).tolist()
        )
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_CAM16_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.cam16.CAM16_to_XYZ` definition domain
        and range scale support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_CAM16["Average"]
        specification = XYZ_to_CAM16(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True)
        XYZ = as_ndarray(CAM16_to_XYZ(specification, XYZ_w, L_A, Y_b, surround))

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
                    CAM16_to_XYZ(
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
    def test_raise_exception_CAM16_to_XYZ(self) -> None:
        """
        Test :func:`colour.appearance.cam16.CAM16_to_XYZ` definition raised
        exception.
        """

        with pytest.raises(ValueError):
            CAM16_to_XYZ(
                CAM_Specification_CAM16(41.73120790512664, None, 217.067959767393),
                np.array([95.05, 100.0, 108.88]),
                318.31,
                20.0,
                VIEWING_CONDITIONS_CAM16["Average"],
            )

    @ignore_numpy_errors
    def test_nan_CAM16_to_XYZ(self) -> None:
        """
        Test :func:`colour.appearance.cam16.CAM16_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        surround = InductionFactors_CAM16(cases[0, 0], cases[0, 0], cases[0, 0])
        CAM16_to_XYZ(
            CAM_Specification_CAM16(cases[..., 0], cases[..., 0], cases[..., 0], M=50),
            cases,
            cases[..., 0],
            cases[..., 0],
            surround,
        )
