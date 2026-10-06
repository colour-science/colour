"""
Define the unit tests for the :mod:`colour.appearance.zcam` module.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import permutations

import numpy as np
import pytest

from colour.appearance import (
    VIEWING_CONDITIONS_ZCAM,
    CAM_Specification_ZCAM,
    InductionFactors_ZCAM,
    XYZ_to_ZCAM,
    ZCAM_to_XYZ,
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

__all__ = ["TestXYZ_to_ZCAM", "TestZCAM_to_XYZ"]


class TestXYZ_to_ZCAM:
    """
    Defines :func:`colour.appearance.zcam.XYZ_to_ZCAM` definition unit tests
    methods.
    """

    def test_XYZ_to_ZCAM(self, xp: ModuleType) -> None:
        """
        Tests :func:`colour.appearance.zcam.XYZ_to_ZCAM` definition.
        """

        XYZ = xp_as_array([185, 206, 163], xp=xp)
        XYZ_w = xp_as_array([256, 264, 202], xp=xp)
        L_a = 264
        Y_b = 100
        surround = VIEWING_CONDITIONS_ZCAM["Average"]
        xp_assert_close(
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround, compute_H=True),
            [
                92.2520,
                3.0216,
                196.3524,
                19.1314,
                321.3464,
                10.5252,
                237.6401,
                np.nan,
                34.7022,
                25.2994,
                91.6837,
            ],
            rtol=0.025,
            atol=TOLERANCE_ABSOLUTE_TESTS * 250000,
        )

        XYZ = xp_as_array([89, 96, 120], xp=xp)
        xp_assert_close(
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround, compute_H=True),
            [
                71.2071,
                6.8539,
                250.6422,
                32.7963,
                248.0394,
                23.8744,
                307.0595,
                np.nan,
                18.2796,
                40.4621,
                70.4026,
            ],
            rtol=0.025,
            atol=TOLERANCE_ABSOLUTE_TESTS * 250000,
        )

        # NOTE: Hue quadrature :math:`H_z` is significantly different for this
        # test, i.e., 47.748252 vs 43.8258.
        # NOTE: :math:`F_L` as reported in the supplemental document has the
        # same value as for :math:`L_a` = 264 instead of 150. The values seem
        # to be computed for :math:`L_a` = 264 and :math:`Y_b` = 100.
        XYZ = xp_as_array([79, 81, 62], xp=xp)
        # L_a = 150
        # Y_b = 60
        surround = VIEWING_CONDITIONS_ZCAM["Dim"]
        xp_assert_close(
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround, compute_H=True),
            [
                68.8890,
                0.9774,
                58.7532,
                12.5916,
                196.7686,
                2.7918,
                43.8258,
                np.nan,
                11.0371,
                44.4143,
                68.8737,
            ],
            rtol=0.025,
            atol=TOLERANCE_ABSOLUTE_TESTS * 40000000,
        )

        XYZ = xp_as_array([910, 1114, 500], xp=xp)
        XYZ_w = xp_as_array([2103, 2259, 1401], xp=xp)
        L_a = 359
        Y_b = 16
        surround = VIEWING_CONDITIONS_ZCAM["Dark"]
        xp_assert_close(
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround, compute_H=True),
            [
                82.6445,
                13.0838,
                123.9464,
                44.7277,
                114.7431,
                18.1655,
                178.6422,
                np.nan,
                34.4874,
                26.8778,
                78.2653,
            ],
            rtol=0.025,
            atol=TOLERANCE_ABSOLUTE_TESTS * 250000,
        )

        XYZ = xp_as_array([96, 67, 28], xp=xp)
        xp_assert_close(
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround, compute_H=True),
            [
                33.0139,
                19.4070,
                389.7720 % 360,
                86.1882,
                45.8363,
                26.9446,
                397.3301,
                np.nan,
                43.6447,
                47.9942,
                30.2593,
            ],
            rtol=0.025,
            atol=TOLERANCE_ABSOLUTE_TESTS * 250000,
        )

    def test_n_dimensional_XYZ_to_ZCAM(self, xp: ModuleType) -> None:
        """
        Tests :func:`colour.appearance.zcam.XYZ_to_ZCAM` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([185, 206, 163], xp=xp)
        XYZ_w = xp_as_array([256, 264, 202], xp=xp)
        L_a = 264
        Y_b = 100
        surround = VIEWING_CONDITIONS_ZCAM["Average"]
        specification = XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround, compute_H=True)

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        specification = np.tile(specification, (6, 1))
        xp_assert_close(
            as_ndarray(XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround, compute_H=True)),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        xp_assert_close(
            as_ndarray(XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround, compute_H=True)),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        specification = xp_reshape(xp_as_array(specification, xp=xp), (2, 3, 11), xp=xp)
        xp_assert_close(
            as_ndarray(XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround, compute_H=True)),
            as_ndarray(specification),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_ZCAM(self, xp: ModuleType) -> None:
        """
        Tests :func:`colour.appearance.zcam.XYZ_to_ZCAM` definition
        domain and range scale support.
        """

        XYZ = xp_as_array([185, 206, 163], xp=xp)
        XYZ_w = xp_as_array([256, 264, 202], xp=xp)
        L_a = 264
        Y_b = 100
        surround = VIEWING_CONDITIONS_ZCAM["Average"]
        specification = XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround, compute_H=True)

        d_r = (
            ("reference", 1, 1),
            ("1", 1, np.array([1, 1, 1 / 360, 1, 1, 1, 1 / 400, np.nan, 1, 1, 1])),
            (
                "100",
                100,
                np.array(
                    [
                        100,
                        100,
                        100 / 360,
                        100,
                        100,
                        100,
                        100 / 400,
                        np.nan,
                        100,
                        100,
                        100,
                    ]
                ),
            ),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    as_ndarray(
                        XYZ_to_ZCAM(
                            XYZ * xp_as_array(factor_a, xp=xp),
                            XYZ_w * xp_as_array(factor_a, xp=xp),
                            L_a,
                            Y_b,
                            surround,
                            compute_H=True,
                        )
                    ),
                    as_float_array(specification) * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_ZCAM(self) -> None:
        """
        Tests :func:`colour.appearance.zcam.XYZ_to_ZCAM` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_w = np.array(case)
            L_a = case[0]
            Y_b = 100
            surround = InductionFactors_ZCAM(case[0], case[0], case[0], case[0])
            XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround, compute_H=True)


class TestZCAM_to_XYZ:
    """
    Defines :func:`colour.appearance.zcam.ZCAM_to_XYZ` definition unit
    tests methods.
    """

    def test_ZCAM_to_XYZ(self, xp: ModuleType) -> None:
        """
        Tests :func:`colour.appearance.zcam.ZCAM_to_XYZ` definition.
        """

        specification = CAM_Specification_ZCAM(
            92.2520,
            3.0216,
            196.3524,
            19.1314,
            321.3464,
            10.5252,
            237.6401,
            np.nan,
            34.7022,
            25.2994,
            91.6837,
        )
        XYZ_w = xp_as_array([256, 264, 202], xp=xp)
        L_a = 264
        Y_b = 100
        surround = VIEWING_CONDITIONS_ZCAM["Average"]
        xp_assert_close(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            [185, 206, 163],
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
            rtol=0.01,
        )

        specification = CAM_Specification_ZCAM(
            71.2071,
            6.8539,
            250.6422,
            32.7963,
            248.0394,
            23.8744,
            307.0595,
            np.nan,
            18.2796,
            40.4621,
            70.4026,
        )
        xp_assert_close(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            [89, 96, 120],
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
            rtol=0.01,
        )

        specification = CAM_Specification_ZCAM(
            68.8890,
            0.9774,
            58.7532,
            12.5916,
            196.7686,
            2.7918,
            43.8258,
            np.nan,
            11.0371,
            44.4143,
            68.8737,
        )
        surround = VIEWING_CONDITIONS_ZCAM["Dim"]
        xp_assert_close(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            [79, 81, 62],
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
            rtol=0.01,
        )

        specification = CAM_Specification_ZCAM(
            82.6445,
            13.0838,
            123.9464,
            44.7277,
            114.7431,
            18.1655,
            178.6422,
            np.nan,
            34.4874,
            26.8778,
            78.2653,
        )
        XYZ_w = xp_as_array([2103, 2259, 1401], xp=xp)
        L_a = 359
        Y_b = 16
        surround = VIEWING_CONDITIONS_ZCAM["Dark"]
        xp_assert_close(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            [910, 1114, 500],
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
            rtol=0.01,
        )

        specification = CAM_Specification_ZCAM(
            33.0139,
            19.4070,
            389.7720 % 360,
            86.1882,
            45.8363,
            26.9446,
            397.3301,
            np.nan,
            43.6447,
            47.9942,
            30.2593,
        )
        xp_assert_close(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            [96, 67, 28],
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
            rtol=0.01,
        )

        # Test using C instead of M
        specification = CAM_Specification_ZCAM(
            J=82.61980483202505, C=13.194790413382647, h=123.77987744640157
        )
        XYZ_w = xp_as_array([2103, 2259, 1401], xp=xp)
        L_a = 359
        Y_b = 16
        surround = VIEWING_CONDITIONS_ZCAM["Dark"]
        xp_assert_close(
            ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround),
            [910, 1114, 500],
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
            rtol=0.01,
        )

    def test_n_dimensional_ZCAM_to_XYZ(self, xp: ModuleType) -> None:
        """
        Tests :func:`colour.appearance.zcam.ZCAM_to_XYZ` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([185, 206, 163], xp=xp)
        XYZ_w = xp_as_array([256, 264, 202], xp=xp)
        L_a = 264
        Y_b = 100
        surround = VIEWING_CONDITIONS_ZCAM["Average"]
        specification = XYZ_to_ZCAM(XYZ, XYZ_w, L_a, Y_b, surround, compute_H=True)
        XYZ = as_ndarray(ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround))

        specification = CAM_Specification_ZCAM(
            *np.transpose(np.tile(tsplit(specification), (6, 1))).tolist()
        )
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            as_ndarray(ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround)),
            as_ndarray(XYZ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        xp_assert_close(
            as_ndarray(ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround)),
            as_ndarray(XYZ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        specification = CAM_Specification_ZCAM(
            *tsplit(np.reshape(specification, (2, 3, 11))).tolist()
        )
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            as_ndarray(ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround)),
            as_ndarray(XYZ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_ZCAM_to_XYZ(self, xp: ModuleType) -> None:
        """
        Tests :func:`colour.appearance.zcam.ZCAM_to_XYZ` definition
        domain and range scale support.
        """

        XYZ_i = xp_as_array([185, 206, 163], xp=xp)
        XYZ_w = xp_as_array([256, 264, 202], xp=xp)
        L_a = 264
        Y_b = 100
        surround = VIEWING_CONDITIONS_ZCAM["Average"]
        specification = XYZ_to_ZCAM(XYZ_i, XYZ_w, L_a, Y_b, surround, compute_H=True)
        XYZ = as_ndarray(ZCAM_to_XYZ(specification, XYZ_w, L_a, Y_b, surround))

        d_r = (
            ("reference", 1, 1),
            ("1", np.array([1, 1, 1 / 360, 1, 1, 1, 1 / 400, np.nan, 1, 1, 1]), 1),
            (
                "100",
                np.array(
                    [
                        100,
                        100,
                        100 / 360,
                        100,
                        100,
                        100,
                        100 / 400,
                        np.nan,
                        100,
                        100,
                        100,
                    ]
                ),
                100,
            ),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    as_ndarray(
                        ZCAM_to_XYZ(
                            specification * xp_as_array(factor_a, xp=xp),
                            XYZ_w * factor_b,
                            L_a,
                            Y_b,
                            surround,
                        )
                    ),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_raise_exception_ZCAM_to_XYZ(self) -> None:
        """
        Tests :func:`colour.appearance.zcam.ZCAM_to_XYZ` definition
        raised exception.
        """

        with pytest.raises(ValueError):
            ZCAM_to_XYZ(
                CAM_Specification_ZCAM(41.73109113251392, None, 219.04843265831178),
                np.array([256, 264, 202]),
                318.31,
                20.0,
                VIEWING_CONDITIONS_ZCAM["Average"],
            )

    @ignore_numpy_errors
    def test_nan_ZCAM_to_XYZ(self) -> None:
        """
        Tests :func:`colour.appearance.zcam.ZCAM_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            J = case[0]
            C = case[0]
            h = case[0]
            XYZ_w = np.array(case)
            L_a = case[0]
            Y_b = 100
            surround = InductionFactors_ZCAM(case[0], case[0], case[0], case[0])
            ZCAM_to_XYZ(
                CAM_Specification_ZCAM(J, C, h, M=50), XYZ_w, L_a, Y_b, surround
            )
