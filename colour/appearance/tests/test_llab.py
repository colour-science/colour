"""Define the unit tests for the :mod:`colour.appearance.llab` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from contextlib import nullcontext
from itertools import product
from unittest import mock

import numpy as np
import pytest

from colour.appearance import (
    VIEWING_CONDITIONS_LLAB,
    InductionFactors_LLAB,
    XYZ_to_LLAB,
    llab,
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
    "TestXYZ_to_LLAB",
]


class TestXYZ_to_LLAB:
    """
    Define :func:`colour.appearance.llab.XYZ_to_LLAB` definition unit tests
    methods.
    """

    def test_XYZ_to_LLAB(self, xp: ModuleType, request: pytest.FixtureRequest) -> None:
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
            # Near-achromatic opponent coordinates amplify float32 hue rounding.
            specification = as_ndarray(
                XYZ_to_LLAB(
                    XYZ,
                    XYZ_0,
                    Y_b,
                    L,
                    surround,
                )
            )
            expected = np.array(
                [
                    37.37,
                    0.01,
                    229.5,
                    0,
                    0.02,
                    np.nan,
                    -0.01,
                    -0.01,
                ]
            )
            xp_assert_close(
                specification[[0, 1, 3, 4, 5, 6, 7]],
                expected[[0, 1, 3, 4, 5, 6, 7]],
                atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05),
            )
            hue_tolerance = 0.1 if as_ndarray(XYZ).dtype.itemsize == 4 else 0
            xp_assert_close(
                specification[[2]],
                expected[[2]],
                atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05, hue_tolerance),
            )

            XYZ = xp_as_array([57.06, 43.06, 31.96], xp=xp)
            L = 31.83
            xp_assert_close(
                XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
                [61.26, 30.51, 22.3, 0.5, 56.55, np.nan, 52.33, 21.43],
                atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05),
            )

            XYZ = xp_as_array([3.53, 6.56, 2.14], xp=xp)
            XYZ_0 = xp_as_array([109.85, 100.00, 35.58], xp=xp)
            L = 318.31
            xp_assert_close(
                XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
                [16.25, 30.43, 173.8, 1.87, 53.83, np.nan, -53.51, 5.83],
                atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05),
            )

            XYZ = xp_as_array([19.01, 20.00, 21.78], xp=xp)
            L = 31.83
            xp_assert_close(
                XYZ_to_LLAB(XYZ, XYZ_0, Y_b, L, surround),
                [39.82, 29.34, 271.9, 0.74, 54.59, np.nan, 1.76, -54.56],
                atol=max(TOLERANCE_ABSOLUTE_TESTS, 0.05),
            )

        # A native secondary operand must retain its backend, device and graph
        # while every other operand remains a host array.
        host_arguments: dict[str, typing.Any] = {
            "XYZ": [19.01, 20.0, 21.78],
            "XYZ_0": [109.85, 100.0, 35.58],
            "Y_b": 20.0,
            "L": 31.83,
        }
        expected = XYZ_to_LLAB(**host_arguments).M
        assert expected is not None
        for parameter in ("XYZ_0", "Y_b", "L"):
            native = xp_as_array(host_arguments[parameter], xp=xp)
            arguments = {**host_arguments, parameter: native}
            actual = XYZ_to_LLAB(**arguments).M
            assert actual is not None
            assert array_namespace(actual) is array_namespace(native)
            assert getattr(actual, "device", None) == getattr(native, "device", None)
            xp_assert_close(actual, expected, atol=TOLERANCE_ABSOLUTE_TESTS)
            if xp.__name__ != "numpy":
                autodiff = request.getfixturevalue("autodiff")
                autodiff(
                    lambda value, parameter=parameter: XYZ_to_LLAB(
                        **{**host_arguments, parameter: value}
                    ).M,
                    native,
                )

        if xp.__name__ != "numpy":
            # At the reference neutral, adaptation is identity and each final
            # Cartesian signal is 1.25 * S_M * S_C * F_C times its raw opponent
            # coordinate to first order. Hue itself has no derivative here.
            surround = VIEWING_CONDITIONS_LLAB["ref_average_4_minus"]
            luminance = 31.83
            lightness = 116 * 0.2 ** ((1 + np.sqrt(0.2)) / 3) - 16
            S_M = 0.7 + 0.02 * lightness - 0.0002 * lightness**2
            S_C = 1 + 0.47 * np.log10(luminance) - 0.057 * np.log10(luminance) ** 2
            slope = (1.25 * S_M * S_C * surround.F_C / 3) * 0.2 ** (-2 / 3)
            expected = slope * np.array(
                [[500 / 95.05, -500 / 100, 0], [0, 200 / 100, -200 / 108.88]]
            )
            native = xp_as_array([19.01, 20.0, 21.776], xp=xp)
            background = xp_as_array(20.0, xp=xp)
            luminance_native = xp_as_array(luminance, xp=xp)

            def cartesian(
                stimulus: typing.Any, background: typing.Any, luminance: typing.Any
            ) -> typing.Any:
                specification = XYZ_to_LLAB(
                    stimulus, [95.05, 100.0, 108.88], background, luminance, surround
                )
                return xp.stack([specification.a, specification.b])

            with xp.device("cpu") if xp.__name__ == "torch" else nullcontext():
                arguments = (native, background, luminance_native)
                if xp.__name__ == "torch":
                    jacobians = (xp.autograd.functional.jacobian(cartesian, arguments),)
                else:
                    jax = pytest.importorskip("jax")
                    jacobians = (
                        jax.jacfwd(cartesian, argnums=(0, 1, 2))(*arguments),
                        jax.jacrev(cartesian, argnums=(0, 1, 2))(*arguments),
                    )
                for jacobian in jacobians:
                    for actual, reference in zip(
                        jacobian, (expected, [0.0, 0.0], [0.0, 0.0]), strict=True
                    ):
                        assert getattr(actual, "device", None) == getattr(
                            native, "device", None
                        )
                        xp_assert_close(
                            actual, reference, atol=TOLERANCE_ABSOLUTE_TESTS
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
