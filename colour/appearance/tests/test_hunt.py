"""Define the unit tests for the :mod:`colour.appearance.hunt` module."""

from __future__ import annotations

import typing
from contextlib import nullcontext

if typing.TYPE_CHECKING:
    import pytest
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
    array_namespace,
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

    def test_XYZ_to_Hunt(self, xp: ModuleType, request: pytest.FixtureRequest) -> None:
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
        # Near-achromatic opponent coordinates amplify float32 hue rounding.
        specification = as_ndarray(
            XYZ_to_Hunt(
                XYZ,
                XYZ_w,
                XYZ_b,
                L_A,
                surround,
                CCT_w=CCT_w,
            )
        )
        expected = np.array(
            [
                42.12,
                0.16,
                269.3,
                0.03,
                31.92,
                0.16,
                np.nan,
                np.nan,
            ]
        )
        xp_assert_close(
            specification[[0, 1, 3, 4, 5, 6, 7]],
            expected[[0, 1, 3, 4, 5, 6, 7]],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05),
        )
        hue_tolerance = 0.6 if as_ndarray(XYZ).dtype.itemsize == 4 else 0
        xp_assert_close(
            specification[[2]],
            expected[[2]],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05, hue_tolerance),
        )

        XYZ = xp_as_array([57.06, 43.06, 31.96], xp=xp)
        L_A = 31.83
        xp_assert_close(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            [66.76, 63.89, 18.6, 153.36, 31.22, 58.28, np.nan, np.nan],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05),
        )

        XYZ = xp_as_array([3.53, 6.56, 2.14], xp=xp)
        XYZ_w = xp_as_array([109.85, 100.00, 35.58], xp=xp)
        L_A = 318.31
        CCT_w = 2856
        xp_assert_close(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            [19.56, 74.58, 178.3, 245.4, 18.9, 76.33, np.nan, np.nan],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05),
        )

        XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
        L_A = 31.83
        xp_assert_close(
            XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, surround, CCT_w=CCT_w),
            [40.27, 73.84, 262.8, 209.29, 22.15, 67.35, np.nan, np.nan],
            atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05),
        )

        # A sole native secondary operand must determine device placement.
        luminance = xp_as_array(318.31, xp=xp)
        with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
            result = XYZ_to_Hunt(
                [30, 20, 10],
                [95.05, 100, 108.88],
                [19.01, 20, 21.78],
                luminance,
                CCT_w=6500,
            ).J
            expected = XYZ_to_Hunt(
                xp_as_array([30, 20, 10], xp=xp, like=luminance),
                xp_as_array([95.05, 100, 108.88], xp=xp, like=luminance),
                [19.01, 20, 21.78],
                318.31,
                CCT_w=6500,
            ).J
        assert result is not None
        assert expected is not None
        assert array_namespace(result).__name__ == array_namespace(luminance).__name__
        assert getattr(result, "device", None) == getattr(luminance, "device", None)
        xp_assert_close(result, expected, atol=TOLERANCE_ABSOLUTE_TESTS)

        # Optional viewing inputs choose the native device only when active.
        host_arguments: dict[str, typing.Any] = {
            "XYZ": [19.01, 20.0, 21.78],
            "XYZ_w": [95.05, 100.0, 108.88],
            "XYZ_b": [19.01, 20.0, 21.78],
            "L_A": 318.31,
            "L_AS": 20.0,
            "CCT_w": 6504.0,
            "XYZ_p": [50.0, 30.0, 80.0],
            "p": 0.2,
            "S": 20.0,
            "S_w": 100.0,
        }
        for scotopic_input in ("L_AS", "CCT_w"):
            active_arguments = dict(host_arguments)
            if scotopic_input == "CCT_w":
                active_arguments.pop("L_AS")
            expected = XYZ_to_Hunt(**active_arguments).M
            assert expected is not None
            for parameter in (scotopic_input, "XYZ_p", "p", "S", "S_w"):
                native = xp_as_array(active_arguments[parameter], xp=xp)
                arguments = {**active_arguments, parameter: native}
                actual = XYZ_to_Hunt(**arguments).M
                assert actual is not None
                assert array_namespace(actual) is array_namespace(native)
                assert getattr(actual, "device", None) == getattr(
                    native, "device", None
                )
                xp_assert_close(actual, expected, atol=TOLERANCE_ABSOLUTE_TESTS)
                if xp.__name__ != "numpy":
                    autodiff = request.getfixturevalue("autodiff")
                    autodiff(
                        lambda value,
                        parameter=parameter,
                        active_arguments=active_arguments: XYZ_to_Hunt(
                            **{**active_arguments, parameter: value}
                        ).M,
                        native,
                    )
        inactive_arguments = dict(host_arguments)
        inactive_arguments["CCT_w"] = xp_as_array(5000.0, xp=xp)
        actual = XYZ_to_Hunt(**inactive_arguments).M
        expected = XYZ_to_Hunt(**host_arguments).M
        assert actual is not None and expected is not None
        xp_assert_close(actual, expected, atol=TOLERANCE_ABSOLUTE_TESTS)
        inactive_arguments = {**host_arguments, "p": None}
        expected = XYZ_to_Hunt(**inactive_arguments).M
        inactive_arguments["XYZ_p"] = xp_as_array([60.0, 40.0, 90.0], xp=xp)
        actual = XYZ_to_Hunt(**inactive_arguments).M
        assert actual is not None and expected is not None
        xp_assert_close(actual, expected, atol=TOLERANCE_ABSOLUTE_TESTS)

        if xp.__name__ != "numpy":
            autodiff = request.getfixturevalue("autodiff")
            for correlate in ("J", "C", "h", "s", "Q", "M"):
                autodiff(
                    lambda stimulus, white, luminance, correlate=correlate: getattr(
                        XYZ_to_Hunt(
                            stimulus,
                            white,
                            [95.05, 100.0, 108.88],
                            luminance,
                            CCT_w=6504.0,
                        ),
                        correlate,
                    ),
                    xp_as_array([30.0, 20.0, 10.0], xp=xp),
                    xp_as_array([95.05, 100.0, 108.88], xp=xp),
                    xp_as_array(318.31, xp=xp),
                )

    def test_n_dimensional_XYZ_to_Hunt(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.appearance.hunt.XYZ_to_Hunt` definition
        n-dimensional support.
        """

        # Keep hue well conditioned when comparing scalar and batched kernels.
        XYZ = xp_as_array([57.06, 43.06, 31.96], xp=xp)
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
