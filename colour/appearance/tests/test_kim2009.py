"""Define the unit tests for the :mod:`colour.appearance.kim2009` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np
import pytest

from colour.appearance import (
    MEDIA_PARAMETERS_KIM2009,
    VIEWING_CONDITIONS_KIM2009,
    CAM_Specification_Kim2009,
    InductionFactors_Kim2009,
    Kim2009_to_XYZ,
    MediaParameters_Kim2009,
    XYZ_to_Kim2009,
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
    "TestXYZ_to_Kim2009",
    "TestKim2009_to_XYZ",
]


class TestXYZ_to_Kim2009:
    """
    Define :func:`colour.appearance.kim2009.XYZ_to_Kim2009` definition unit
    tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(5e-2)
    def test_XYZ_to_Kim2009(self, xp: ModuleType) -> None:
        """Test :func:`colour.appearance.kim2009.XYZ_to_Kim2009` definition."""

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_a = 318.31
        media = MEDIA_PARAMETERS_KIM2009["CRT Displays"]
        surround = VIEWING_CONDITIONS_KIM2009["Average"]
        xp_assert_close(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround, compute_H=True),
            [
                28.86190898,
                0.55924559,
                219.04806678,
                9.38377973,
                52.71388839,
                0.46417384,
                278.06028246,
                np.nan,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_as_array([57.06, 43.06, 31.96], xp=xp)
        L_a = 31.83
        xp_assert_close(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround, compute_H=True),
            [
                70.15940419,
                57.89295872,
                21.27017200,
                61.23630434,
                128.14034598,
                48.05115573,
                1.41841443,
                np.nan,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_as_array([3.53, 6.56, 2.14], xp=xp)
        XYZ_w = xp_as_array([109.85, 100.00, 35.58], xp=xp)
        L_a = 318.31
        xp_assert_close(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround, compute_H=True),
            [
                -4.83430022,
                37.42013921,
                177.12166057,
                np.nan,
                -8.82944930,
                31.05871555,
                220.36270343,
                np.nan,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        L_a = 31.83
        xp_assert_close(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround, compute_H=True),
            [
                47.20460719,
                56.35723637,
                241.04877377,
                73.65830083,
                86.21530880,
                46.77650619,
                301.77516676,
                np.nan,
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_Kim2009(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.kim2009.XYZ_to_Kim2009` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_a = 318.31
        media = MEDIA_PARAMETERS_KIM2009["CRT Displays"]
        surround = VIEWING_CONDITIONS_KIM2009["Average"]
        specification = XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround, compute_H=True)

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        specification = np.tile(specification, (6, 1))
        xp_assert_close(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        specification = xp_reshape(xp_as_array(specification, xp=xp), (2, 3, 8), xp=xp)
        xp_assert_close(
            XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround, compute_H=True),
            specification,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_XYZ_to_Kim2009(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.kim2009.XYZ_to_Kim2009` definition
        domain and range scale support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_a = 318.31
        media = MEDIA_PARAMETERS_KIM2009["CRT Displays"]
        surround = VIEWING_CONDITIONS_KIM2009["Average"]
        specification = XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround, compute_H=True)

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
                    XYZ_to_Kim2009(
                        XYZ * xp_as_array(factor_a, xp=xp),
                        XYZ_w * xp_as_array(factor_a, xp=xp),
                        L_a,
                        media,
                        surround,
                        compute_H=True,
                    ),
                    as_float_array(specification) * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Kim2009(self) -> None:
        """
        Test :func:`colour.appearance.kim2009.XYZ_to_Kim2009` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        media = MediaParameters_Kim2009(cases[0, 0])
        surround = InductionFactors_Kim2009(cases[0, 0], cases[0, 0], cases[0, 0])
        XYZ_to_Kim2009(cases, cases, cases[0, 0], media, surround, compute_H=True)


class TestKim2009_to_XYZ:
    """
    Define :func:`colour.appearance.kim2009.Kim2009_to_XYZ` definition unit
    tests methods.
    """

    def test_Kim2009_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.appearance.kim2009.Kim2009_to_XYZ` definition."""

        specification = CAM_Specification_Kim2009(
            28.86190898,
            0.55924559,
            219.04806678,
            9.38377973,
            52.71388839,
            0.46417384,
            278.06028246,
            np.nan,
        )
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_a = 318.31
        media = MEDIA_PARAMETERS_KIM2009["CRT Displays"]
        surround = VIEWING_CONDITIONS_KIM2009["Average"]
        xp_assert_close(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            [19.01, 20.00, 21.78],
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

        specification = CAM_Specification_Kim2009(
            70.15940419,
            57.89295872,
            21.27017200,
            61.23630434,
            128.14034598,
            48.05115573,
            1.41841443,
            np.nan,
        )
        L_a = 31.83
        xp_assert_close(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            [57.06, 43.06, 31.96],
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

        specification = CAM_Specification_Kim2009(
            -4.83430022,
            37.42013921,
            177.12166057,
            np.nan,
            -8.82944930,
            31.05871555,
            220.36270343,
            np.nan,
        )
        XYZ_w = xp_as_array([109.85, 100.00, 35.58], xp=xp)
        L_a = 318.31
        xp_assert_close(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            [3.53, 6.56, 2.14],
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

        specification = CAM_Specification_Kim2009(
            47.20460719,
            56.35723637,
            241.04877377,
            73.65830083,
            86.21530880,
            46.77650619,
            301.77516676,
            np.nan,
        )
        L_a = 31.83
        xp_assert_close(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            [19.01, 20.00, 21.78],
            atol=TOLERANCE_ABSOLUTE_TESTS * 100000,
        )

    def test_n_dimensional_Kim2009_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.kim2009.Kim2009_to_XYZ` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_a = 318.31
        media = MEDIA_PARAMETERS_KIM2009["CRT Displays"]
        surround = VIEWING_CONDITIONS_KIM2009["Average"]
        specification = XYZ_to_Kim2009(XYZ, XYZ_w, L_a, media, surround, compute_H=True)
        XYZ = as_ndarray(Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround))

        specification = CAM_Specification_Kim2009(
            *np.transpose(np.tile(tsplit(specification), (6, 1))).tolist()
        )
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        xp_assert_close(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        specification = CAM_Specification_Kim2009(
            *tsplit(np.reshape(specification, (2, 3, 8))).tolist()
        )
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_domain_range_scale_Kim2009_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.kim2009.Kim2009_to_XYZ` definition
        domain and range scale support.
        """

        XYZ_i = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        XYZ_w = xp_as_array([95.05, 100.00, 108.88], xp=xp)
        L_a = 318.31
        media = MEDIA_PARAMETERS_KIM2009["CRT Displays"]
        surround = VIEWING_CONDITIONS_KIM2009["Average"]
        specification = XYZ_to_Kim2009(
            XYZ_i, XYZ_w, L_a, media, surround, compute_H=True
        )
        XYZ = as_ndarray(Kim2009_to_XYZ(specification, XYZ_w, L_a, media, surround))

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
                    Kim2009_to_XYZ(
                        specification * xp_as_array(factor_a, xp=xp),
                        XYZ_w * factor_b,
                        L_a,
                        media,
                        surround,
                    ),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_raise_exception_Kim2009_to_XYZ(self) -> None:
        """
        Test :func:`colour.appearance.kim2009.Kim2009_to_XYZ` definition
        raised exception.
        """

        with pytest.raises(ValueError):
            Kim2009_to_XYZ(
                CAM_Specification_Kim2009(41.73109113251392, None, 219.04843265831178),
                np.array([95.05, 100.0, 108.88]),
                318.31,
                20.0,  # pyright: ignore
                VIEWING_CONDITIONS_KIM2009["Average"],
            )

    @ignore_numpy_errors
    def test_nan_Kim2009_to_XYZ(self) -> None:
        """
        Test :func:`colour.appearance.kim2009.Kim2009_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        media = MediaParameters_Kim2009(cases[0, 0])
        surround = InductionFactors_Kim2009(cases[0, 0], cases[0, 0], cases[0, 0])
        Kim2009_to_XYZ(
            CAM_Specification_Kim2009(
                cases[..., 0], cases[..., 0], cases[..., 0], M=50
            ),
            cases,
            cases[0, 0],
            media,
            surround,
        )
