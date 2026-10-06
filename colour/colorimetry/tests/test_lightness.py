"""Define the unit tests for the :mod:`colour.colorimetry.lightness` module."""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from colour.colorimetry import (
    intermediate_lightness_function_CIE1976,
    lightness_Abebe2017,
    lightness_CIE1976,
    lightness_Fairchild2010,
    lightness_Fairchild2011,
    lightness_Glasser1958,
    lightness_Wyszecki1963,
)
from colour.colorimetry.lightness import lightness
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
    "TestLightnessGlasser1958",
    "TestLightnessWyszecki1963",
    "TestIntermediateLightnessFunctionCIE1976",
    "TestLightnessCIE1976",
    "TestLightnessFairchild2010",
    "TestLightnessFairchild2011",
    "TestLightnessAbebe2017",
    "TestLightness",
]


class TestLightnessGlasser1958:
    """
    Define :func:`colour.colorimetry.lightness.lightness_Glasser1958`
    definition unit tests methods.
    """

    def test_lightness_Glasser1958(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Glasser1958`
        definition.
        """

        xp_assert_close(
            lightness_Glasser1958(xp_as_array([12.19722535], xp=xp)),
            39.83512646492521,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Glasser1958(xp_as_array([23.04276781], xp=xp)),
            53.585946877480623,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Glasser1958(xp_as_array([6.15720079], xp=xp)),
            27.972867038082629,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_lightness_Glasser1958(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Glasser1958`
        definition n-dimensional arrays support.
        """

        Y = 12.19722535
        L = lightness_Glasser1958(Y)

        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        L = xp.tile(xp_as_array(L, xp=xp), (6,))
        xp_assert_close(lightness_Glasser1958(Y), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3), xp=xp)
        xp_assert_close(lightness_Glasser1958(Y), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(lightness_Glasser1958(Y), L, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_lightness_Glasser1958(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.colorimetry.lightness.lightness_Glasser1958`
        definition domain and range scale support.
        """

        L = as_ndarray(lightness_Glasser1958(12.19722535))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    lightness_Glasser1958(12.19722535 * factor),
                    L * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_lightness_Glasser1958(self) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Glasser1958`
        definition nan support.
        """

        lightness_Glasser1958(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLightnessWyszecki1963:
    """
    Define :func:`colour.colorimetry.lightness.lightness_Wyszecki1963`
    definition unit tests methods.
    """

    def test_lightness_Wyszecki1963(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Wyszecki1963`
        definition.
        """

        xp_assert_close(
            lightness_Wyszecki1963(xp_as_array([12.19722535], xp=xp)),
            40.547574599570197,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Wyszecki1963(xp_as_array([23.04276781], xp=xp)),
            54.140714588256841,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Wyszecki1963(xp_as_array([6.15720079], xp=xp)),
            28.821339499883976,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_lightness_Wyszecki1963(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Wyszecki1963`
        definition n-dimensional arrays support.
        """

        Y = 12.19722535
        W = lightness_Wyszecki1963(Y)

        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        W = xp.tile(xp_as_array(W, xp=xp), (6,))
        xp_assert_close(lightness_Wyszecki1963(Y), W, atol=TOLERANCE_ABSOLUTE_TESTS)

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        W = xp_reshape(xp_as_array(W, xp=xp), (2, 3), xp=xp)
        xp_assert_close(lightness_Wyszecki1963(Y), W, atol=TOLERANCE_ABSOLUTE_TESTS)

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        W = xp_reshape(xp_as_array(W, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(lightness_Wyszecki1963(Y), W, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_lightness_Wyszecki1963(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.colorimetry.lightness.lightness_Wyszecki1963`
        definition domain and range scale support.
        """

        W = as_ndarray(lightness_Wyszecki1963(12.19722535))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    lightness_Wyszecki1963(12.19722535 * factor),
                    W * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_lightness_Wyszecki1963(self) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Wyszecki1963`
        definition nan support.
        """

        lightness_Wyszecki1963(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestIntermediateLightnessFunctionCIE1976:
    """
    Define :func:`colour.colorimetry.lightness.\
intermediate_lightness_function_CIE1976` definition unit tests methods.
    """

    def test_intermediate_lightness_function_CIE1976(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.lightness.\
intermediate_lightness_function_CIE1976` definition.
        """

        xp_assert_close(
            intermediate_lightness_function_CIE1976(xp_as_array([12.19722535], xp=xp)),
            0.495929964178047,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            intermediate_lightness_function_CIE1976(xp_as_array([23.04276781], xp=xp)),
            0.613072093530391,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            intermediate_lightness_function_CIE1976(xp_as_array([6.15720079], xp=xp)),
            0.394876333449113,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_intermediate_lightness_function_CIE1976(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.colorimetry.lightness.\
intermediate_lightness_function_CIE1976` definition n-dimensional arrays
        support.
        """

        Y = 12.19722535
        f_Y_Y_n = intermediate_lightness_function_CIE1976(Y)

        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        f_Y_Y_n = xp.tile(xp_as_array(f_Y_Y_n, xp=xp), (6,))
        xp_assert_close(
            intermediate_lightness_function_CIE1976(Y),
            f_Y_Y_n,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        f_Y_Y_n = xp_reshape(xp_as_array(f_Y_Y_n, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            intermediate_lightness_function_CIE1976(Y),
            f_Y_Y_n,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        f_Y_Y_n = xp_reshape(xp_as_array(f_Y_Y_n, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            intermediate_lightness_function_CIE1976(Y),
            f_Y_Y_n,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_intermediate_lightness_function_CIE1976(
        self,
        xp: ModuleType,  # noqa: ARG002
    ) -> None:
        """
        Test :func:`colour.colorimetry.lightness.\
intermediate_lightness_function_CIE1976` definition domain and range scale
        support.
        """

        f_Y_Y_n = as_ndarray(intermediate_lightness_function_CIE1976(12.19722535, 100))

        for scale in ("reference", "1", "100"):
            with domain_range_scale(scale):
                xp_assert_close(
                    intermediate_lightness_function_CIE1976(12.19722535, 100),
                    f_Y_Y_n,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_intermediate_lightness_function_CIE1976(self) -> None:
        """
        Test :func:`colour.colorimetry.lightness.\
intermediate_lightness_function_CIE1976` definition nan support.
        """

        intermediate_lightness_function_CIE1976(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLightnessCIE1976:
    """
    Define :func:`colour.colorimetry.lightness.lightness_CIE1976` definition
    unit tests methods.
    """

    def test_lightness_CIE1976(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_CIE1976`
        definition.
        """

        xp_assert_close(
            lightness_CIE1976(xp_as_array([12.19722535], xp=xp)),
            41.527875844653451,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_CIE1976(xp_as_array([23.04276781], xp=xp)),
            55.116362849525402,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_CIE1976(xp_as_array([6.15720079], xp=xp)),
            29.805654680097106,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_CIE1976(xp_as_array([12.19722535], xp=xp), 50),
            56.480581732417676,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_CIE1976(xp_as_array([12.19722535], xp=xp), 75),
            47.317620274162735,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_CIE1976(xp_as_array([12.19722535], xp=xp), 95),
            42.519930728120940,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_lightness_CIE1976(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_CIE1976`
        definition n-dimensional arrays support.
        """

        Y = 12.19722535
        L_star = lightness_CIE1976(Y)

        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        L_star = xp.tile(xp_as_array(L_star, xp=xp), (6,))
        xp_assert_close(lightness_CIE1976(Y), L_star, atol=TOLERANCE_ABSOLUTE_TESTS)

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        L_star = xp_reshape(xp_as_array(L_star, xp=xp), (2, 3), xp=xp)
        xp_assert_close(lightness_CIE1976(Y), L_star, atol=TOLERANCE_ABSOLUTE_TESTS)

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        L_star = xp_reshape(xp_as_array(L_star, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(lightness_CIE1976(Y), L_star, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_lightness_CIE1976(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.colorimetry.lightness.lightness_CIE1976`
        definition domain and range scale support.
        """

        L_star = as_ndarray(lightness_CIE1976(12.19722535, 100))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    lightness_CIE1976(12.19722535 * factor, 100 * factor),
                    L_star * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_lightness_CIE1976(self) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_CIE1976`
        definition nan support.
        """

        lightness_CIE1976(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLightnessFairchild2010:
    """
    Define :func:`colour.colorimetry.lightness.lightness_Fairchild2010`
    definition unit tests methods.
    """

    def test_lightness_Fairchild2010(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Fairchild2010`
        definition.
        """

        xp_assert_close(
            lightness_Fairchild2010(xp_as_array([12.19722535 / 100], xp=xp)),
            31.996390226262736,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Fairchild2010(xp_as_array([23.04276781 / 100], xp=xp)),
            60.203153682783302,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Fairchild2010(xp_as_array([6.15720079 / 100], xp=xp)),
            11.836517240976489,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Fairchild2010(xp_as_array([12.19722535 / 100], xp=xp), 2.75),
            24.424283249379986,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Fairchild2010(xp_as_array([1008.0], xp=xp)),
            100.019986327374240,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Fairchild2010(xp_as_array([100800.0], xp=xp)),
            100.019999997090270,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_lightness_Fairchild2010(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Fairchild2010`
        definition n-dimensional arrays support.
        """

        Y = 12.19722535 / 100
        L_hdr = lightness_Fairchild2010(Y)

        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        L_hdr = xp.tile(xp_as_array(L_hdr, xp=xp), (6,))
        xp_assert_close(
            lightness_Fairchild2010(Y), L_hdr, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        L_hdr = xp_reshape(xp_as_array(L_hdr, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            lightness_Fairchild2010(Y), L_hdr, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        L_hdr = xp_reshape(xp_as_array(L_hdr, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            lightness_Fairchild2010(Y), L_hdr, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_lightness_Fairchild2010(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.colorimetry.lightness.lightness_Fairchild2010`
        definition domain and range scale support.
        """

        L_hdr = as_ndarray(lightness_Fairchild2010(12.19722535 / 100))

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    lightness_Fairchild2010(12.19722535 / 100 * factor_a),
                    L_hdr * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_lightness_Fairchild2010(self) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Fairchild2010`
        definition nan support.
        """

        lightness_Fairchild2010(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLightnessFairchild2011:
    """
    Define :func:`colour.colorimetry.lightness.lightness_Fairchild2011`
    definition unit tests methods.
    """

    def test_lightness_Fairchild2011(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Fairchild2011`
        definition.
        """

        xp_assert_close(
            lightness_Fairchild2011(xp_as_array([12.19722535 / 100], xp=xp)),
            51.852958445912506,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Fairchild2011(xp_as_array([23.04276781 / 100], xp=xp)),
            65.275207956353853,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Fairchild2011(xp_as_array([6.15720079 / 100], xp=xp)),
            39.818935510715917,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Fairchild2011(xp_as_array([12.19722535 / 100], xp=xp), 2.75),
            0.13268968410139345,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Fairchild2011(xp_as_array([1008.0], xp=xp)),
            234.72925682,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Fairchild2011(xp_as_array([100800.0], xp=xp)),
            245.5705978,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_lightness_Fairchild2011(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Fairchild2011`
        definition n-dimensional arrays support.
        """

        Y = 12.19722535 / 100
        L_hdr = lightness_Fairchild2011(Y)

        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        L_hdr = xp.tile(xp_as_array(L_hdr, xp=xp), (6,))
        xp_assert_close(
            lightness_Fairchild2011(Y), L_hdr, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        L_hdr = xp_reshape(xp_as_array(L_hdr, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            lightness_Fairchild2011(Y), L_hdr, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        L_hdr = xp_reshape(xp_as_array(L_hdr, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            lightness_Fairchild2011(Y), L_hdr, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_lightness_Fairchild2011(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.colorimetry.lightness.lightness_Fairchild2011`
        definition domain and range scale support.
        """

        L_hdr = as_ndarray(lightness_Fairchild2011(12.19722535 / 100))

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    lightness_Fairchild2011(12.19722535 / 100 * factor_a),
                    L_hdr * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_lightness_Fairchild2011(self) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Fairchild2011`
        definition nan support.
        """

        lightness_Fairchild2011(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLightnessAbebe2017:
    """
    Define :func:`colour.colorimetry.lightness.lightness_Abebe2017`
    definition unit tests methods.
    """

    def test_lightness_Abebe2017(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Abebe2017`
        definition.
        """

        xp_assert_close(
            lightness_Abebe2017(xp_as_array([12.19722535], xp=xp)),
            0.486955571109229,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Abebe2017(xp_as_array([12.19722535], xp=xp), method="Stevens"),
            0.474544792145434,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Abebe2017(xp_as_array([12.19722535], xp=xp), 1000),
            0.286847428534793,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Abebe2017(xp_as_array([12.19722535], xp=xp), 4000),
            0.192145492588158,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            lightness_Abebe2017(
                xp_as_array([12.19722535], xp=xp), 4000, method="Stevens"
            ),
            0.170365211220992,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_lightness_Abebe2017(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Abebe2017`
        definition n-dimensional arrays support.
        """

        Y = 12.19722535
        L = lightness_Abebe2017(Y)

        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        L = xp.tile(xp_as_array(L, xp=xp), (6,))
        xp_assert_close(lightness_Abebe2017(Y), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3), xp=xp)
        xp_assert_close(lightness_Abebe2017(Y), L, atol=TOLERANCE_ABSOLUTE_TESTS)

        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(lightness_Abebe2017(Y), L, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_lightness_Abebe2017(self) -> None:
        """
        Test :func:`colour.colorimetry.lightness.lightness_Abebe2017`
        definition nan support.
        """

        cases = np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        lightness_Abebe2017(cases, cases)


class TestLightness:
    """
    Define :func:`colour.colorimetry.lightness.lightness` definition unit
    tests methods.
    """

    def test_domain_range_scale_lightness(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.colorimetry.lightness.lightness` definition domain
        and range scale support.
        """

        m = (
            "Glasser 1958",
            "Wyszecki 1963",
            "CIE 1976",
            "Fairchild 2010",
            "Fairchild 2011",
            "Abebe 2017",
        )
        v = [as_ndarray(lightness(12.19722535, method, Y_n=100)) for method in m]

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for method, value in zip(m, v, strict=True):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    xp_assert_close(
                        lightness(12.19722535 * factor, method, Y_n=100 * factor),
                        value * factor,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )
