"""
Define the unit tests for the :mod:`colour.appearance.scam` module.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np
import pytest

from colour.appearance import (
    CAM_Specification_sCAM,
    VIEWING_CONDITIONS_sCAM,
    XYZ_to_sCAM,
    sCAM_to_XYZ,
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

__author__ = "Colour Developers, UltraMo114(Molin Li)"
__copyright__ = "Copyright 2024 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = ["TestXYZ_to_sCAM", "TestsCAM_to_XYZ"]


class TestXYZ_to_sCAM:
    """
    Define :func:`colour.appearance.scam.XYZ_to_sCAM` definition unit
    tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(2e-1)
    def test_XYZ_to_sCAM(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.scam.XYZ_to_sCAM` definition.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_sCAM["Average"]
        xp_assert_close(
            XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            [
                49.97956680,
                0.01405311,
                328.27249244,
                195.23024234,
                0.00502448,
                363.60134377,
                np.nan,
                49.97957273,
                50.02042727,
                34.97343274,
                65.02656726,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_as_array([57.06, 43.06, 31.96], xp=xp)
        L_A = 31.83
        xp_assert_close(
            XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            [
                71.63079886,
                37.33838127,
                18.75135858,
                259.13174065,
                10.66713872,
                4.20415978,
                np.nan,
                96.50614225,
                3.49385775,
                28.37649889,
                71.62350111,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_as_array([3.53, 6.56, 2.14], xp=xp)
        XYZ_w = xp_as_array([109.85, 100, 35.58], xp=xp)
        L_A = 318.31
        xp_assert_close(
            XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            [
                29.61821869,
                25.97461207,
                178.56952253,
                115.69472052,
                10.76901611,
                227.46922207,
                np.nan,
                53.86353400,
                46.13646600,
                -0.97480767,
                100.97480767,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_sCAM(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.scam.XYZ_to_sCAM` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_sCAM["Average"]
        specification = XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True)

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        specification = np.tile(specification, (6, 1))
        xp_assert_close(
            XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        specification = xp_reshape(xp_as_array(specification, xp=xp), (2, 3, 11), xp=xp)
        xp_assert_close(
            XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_sCAM(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.scam.XYZ_to_sCAM` definition
        domain and range scale support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_sCAM["Average"]

        specification = XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True)

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
                        1 / 400,
                        1,
                        1 / 100,
                        1 / 100,
                        1 / 100,
                        1 / 100,
                    ]
                ),
            ),
            (
                "100",
                1,
                np.array([1, 1, 100 / 360, 1, 1, 100 / 400, 1, 1, 1, 1, 1]),
            ),
        )

        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_sCAM(
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
    def test_nan_XYZ_to_sCAM(self) -> None:
        """
        Test :func:`colour.appearance.scam.XYZ_to_sCAM` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        surround = VIEWING_CONDITIONS_sCAM["Average"]
        XYZ_to_sCAM(
            cases, cases, cases[..., 0], cases[..., 0], surround, compute_H=True
        )


class TestsCAM_to_XYZ:
    """
    Define :func:`colour.appearance.scam.sCAM_to_XYZ` definition unit
    tests methods.
    """

    def test_sCAM_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.scam.sCAM_to_XYZ` definition.
        """

        specification = CAM_Specification_sCAM(49.97956680, 0.01405311, 328.27249244)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_sCAM["Average"]
        xp_assert_close(
            sCAM_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            [19.01, 20.00, 21.78],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        specification = CAM_Specification_sCAM(71.63079886, 37.33838127, 18.75135858)
        L_A = 31.83
        xp_assert_close(
            sCAM_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            [57.06, 43.06, 31.96],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        specification = CAM_Specification_sCAM(29.61821869, 25.97461207, 178.56952253)
        XYZ_w = xp_as_array([109.85, 100, 35.58], xp=xp)
        L_A = 318.31
        xp_assert_close(
            sCAM_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            [3.53256359, 6.56009775, 2.15585716],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_sCAM_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.scam.sCAM_to_XYZ` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_sCAM["Average"]
        specification = XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True)
        XYZ = as_ndarray(sCAM_to_XYZ(specification, XYZ_w, L_A, Y_b, surround))

        specification = CAM_Specification_sCAM(
            *np.transpose(np.tile(tsplit(specification), (6, 1))).tolist()
        )
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            sCAM_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        xp_assert_close(
            sCAM_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        specification = CAM_Specification_sCAM(
            *tsplit(np.reshape(specification, (2, 3, 11))).tolist()
        )
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            sCAM_to_XYZ(specification, XYZ_w, L_A, Y_b, surround),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_sCAM_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.scam.sCAM_to_XYZ` definition
        domain and range scale support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_sCAM["Average"]
        specification = XYZ_to_sCAM(XYZ, XYZ_w, L_A, Y_b, surround, compute_H=True)
        XYZ = as_ndarray(sCAM_to_XYZ(specification, XYZ_w, L_A, Y_b, surround))

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
                        1 / 400,
                        1,
                        1 / 100,
                        1 / 100,
                        1 / 100,
                        1 / 100,
                    ]
                ),
                0.01,
            ),
            (
                "100",
                np.array([1, 1, 100 / 360, 1, 1, 100 / 400, 1, 1, 1, 1, 1]),
                1,
            ),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    sCAM_to_XYZ(
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
    def test_raise_exception_sCAM_to_XYZ(self) -> None:
        """
        Test :func:`colour.appearance.scam.sCAM_to_XYZ` definition
        raised exception.
        """
        XYZ_w = np.array([95.05, 100.00, 108.88])
        L_A = 318.31
        Y_b = 20
        surround = VIEWING_CONDITIONS_sCAM["Average"]

        with pytest.raises(ValueError):
            sCAM_to_XYZ(
                CAM_Specification_sCAM(J=None, C=20.0, h=210.0),
                XYZ_w,
                L_A,
                Y_b,
                surround,
            )

        with pytest.raises(ValueError):
            sCAM_to_XYZ(
                CAM_Specification_sCAM(J=40.0, C=20.0, h=None),
                XYZ_w,
                L_A,
                Y_b,
                surround,
            )

        with pytest.raises(ValueError):
            sCAM_to_XYZ(
                CAM_Specification_sCAM(J=40.0, C=None, h=210.0, M=None),
                XYZ_w,
                L_A,
                Y_b,
                surround,
            )

    @ignore_numpy_errors
    def test_nan_sCAM_to_XYZ(self) -> None:
        """
        Test :func:`colour.appearance.scam.sCAM_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        surround = VIEWING_CONDITIONS_sCAM["Average"]
        sCAM_to_XYZ(
            CAM_Specification_sCAM(cases[..., 0], cases[..., 0], cases[..., 0], M=50),
            cases,
            cases[..., 0],
            cases[..., 0],
            surround,
        )
