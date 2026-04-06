"""Define the unit tests for the :mod:`colour.colorimetry.luminance` module."""

from __future__ import annotations

import typing

import numpy as np
import pytest

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from colour.colorimetry import (
    intermediate_luminance_function_CIE1976,
    luminance_Abebe2017,
    luminance_ASTMD1535,
    luminance_CIE1976,
    luminance_Fairchild2010,
    luminance_Fairchild2011,
    luminance_Newhall1943,
)
from colour.colorimetry.luminance import luminance
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
    "TestLuminanceNewhall1943",
    "TestLuminanceASTMD1535",
    "TestIntermediateLuminanceFunctionCIE1976",
    "TestLuminanceCIE1976",
    "TestLuminanceFairchild2010",
    "TestLuminanceFairchild2011",
    "TestLuminanceAbebe2017",
    "TestLuminance",
]


class TestLuminanceNewhall1943:
    """
    Define :func:`colour.colorimetry.luminance.luminance_Newhall1943`
    definition unit tests methods.
    """

    def test_luminance_Newhall1943(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_Newhall1943`
        definition.
        """

        xp_assert_close(
            luminance_Newhall1943(xp_as_array([4.08244375], xp=xp)),
            12.550078816731881,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Newhall1943(xp_as_array([5.39132685], xp=xp)),
            23.481252371310738,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Newhall1943(xp_as_array([2.97619312], xp=xp)),
            6.4514266875601924,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_luminance_Newhall1943(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_Newhall1943`
        definition n-dimensional arrays support.
        """

        V = 4.08244375
        Y = luminance_Newhall1943(V)

        V = xp.tile(xp_as_array(V, xp=xp), (6,))
        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        xp_assert_close(luminance_Newhall1943(V), Y, atol=TOLERANCE_ABSOLUTE_TESTS)

        V = xp_reshape(xp_as_array(V, xp=xp), (2, 3), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(luminance_Newhall1943(V), Y, atol=TOLERANCE_ABSOLUTE_TESTS)

        V = xp_reshape(xp_as_array(V, xp=xp), (2, 3, 1), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(luminance_Newhall1943(V), Y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_luminance_Newhall1943(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.colorimetry.luminance.luminance_Newhall1943`
        definition domain and range scale support.
        """

        Y = as_ndarray(luminance_Newhall1943(4.08244375))

        d_r = (("reference", 1, 1), ("1", 0.1, 0.01), ("100", 10, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    luminance_Newhall1943(4.08244375 * factor_a),
                    Y * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_luminance_Newhall1943(self) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_Newhall1943`
        definition nan support.
        """

        luminance_Newhall1943(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLuminanceASTMD1535:
    """
    Define :func:`colour.colorimetry.luminance.luminance_ASTMD1535`
    definition unit tests methods.
    """

    def test_luminance_ASTMD1535(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_ASTMD1535`
        definition.
        """

        xp_assert_close(
            luminance_ASTMD1535(xp_as_array([4.08244375], xp=xp)),
            12.236342675366036,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_ASTMD1535(xp_as_array([5.39132685], xp=xp)),
            22.893999867280378,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_ASTMD1535(xp_as_array([2.97619312], xp=xp)),
            6.2902253509053132,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_luminance_ASTMD1535(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_ASTMD1535`
        definition n-dimensional arrays support.
        """

        V = 4.08244375
        Y = luminance_ASTMD1535(V)

        V = xp.tile(xp_as_array(V, xp=xp), (6,))
        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        xp_assert_close(luminance_ASTMD1535(V), Y, atol=TOLERANCE_ABSOLUTE_TESTS)

        V = xp_reshape(xp_as_array(V, xp=xp), (2, 3), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(luminance_ASTMD1535(V), Y, atol=TOLERANCE_ABSOLUTE_TESTS)

        V = xp_reshape(xp_as_array(V, xp=xp), (2, 3, 1), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(luminance_ASTMD1535(V), Y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_luminance_ASTMD1535(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.colorimetry.luminance.luminance_ASTMD1535`
        definition domain and range scale support.
        """

        Y = as_ndarray(luminance_ASTMD1535(4.08244375))

        d_r = (("reference", 1, 1), ("1", 0.1, 0.01), ("100", 10, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    luminance_ASTMD1535(4.08244375 * factor_a),
                    Y * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_luminance_ASTMD1535(self) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_ASTMD1535`
        definition nan support.
        """

        luminance_ASTMD1535(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestIntermediateLuminanceFunctionCIE1976:
    """
    Define :func:`colour.colorimetry.luminance.\
intermediate_luminance_function_CIE1976` definition unit tests methods.
    """

    def test_intermediate_luminance_function_CIE1976(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.luminance.\
intermediate_luminance_function_CIE1976` definition.
        """

        xp_assert_close(
            intermediate_luminance_function_CIE1976(
                xp_as_array([0.495929964178047], xp=xp)
            ),
            12.197225350000002,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            intermediate_luminance_function_CIE1976(
                xp_as_array([0.613072093530391], xp=xp)
            ),
            23.042767810000004,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            intermediate_luminance_function_CIE1976(
                xp_as_array([0.394876333449113], xp=xp)
            ),
            6.157200790000001,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_intermediate_luminance_function_CIE1976(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.colorimetry.luminance.\
intermediate_luminance_function_CIE1976` definition n-dimensional arrays
        support.
        """

        f_Y_Y_n = 0.495929964178047
        Y = intermediate_luminance_function_CIE1976(f_Y_Y_n)

        f_Y_Y_n = xp.tile(xp_as_array(f_Y_Y_n, xp=xp), (6,))
        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        xp_assert_close(
            intermediate_luminance_function_CIE1976(f_Y_Y_n),
            Y,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        f_Y_Y_n = xp_reshape(xp_as_array(f_Y_Y_n, xp=xp), (2, 3), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            intermediate_luminance_function_CIE1976(f_Y_Y_n),
            Y,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        f_Y_Y_n = xp_reshape(xp_as_array(f_Y_Y_n, xp=xp), (2, 3, 1), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            intermediate_luminance_function_CIE1976(f_Y_Y_n),
            Y,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_intermediate_luminance_function_CIE1976(
        self,
        xp: ModuleType,  # noqa: ARG002
    ) -> None:
        """
        Test :func:`colour.colorimetry.luminance.\
intermediate_luminance_function_CIE1976` definition domain and range scale
        support.
        """

        Y = as_ndarray(intermediate_luminance_function_CIE1976(41.527875844653451, 100))

        for scale in ("reference", "1", "100"):
            with domain_range_scale(scale):
                xp_assert_close(
                    intermediate_luminance_function_CIE1976(41.527875844653451, 100),
                    Y,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_intermediate_luminance_function_CIE1976(self) -> None:
        """
        Test :func:`colour.colorimetry.luminance.\
intermediate_luminance_function_CIE1976` definition nan support.
        """

        intermediate_luminance_function_CIE1976(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLuminanceCIE1976:
    """
    Define :func:`colour.colorimetry.luminance.luminance_CIE1976` definition
    unit tests methods.
    """

    def test_luminance_CIE1976(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_CIE1976`
        definition.
        """

        xp_assert_close(
            luminance_CIE1976(xp_as_array([41.527875844653451], xp=xp)),
            12.197225350000002,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_CIE1976(xp_as_array([55.116362849525402], xp=xp)),
            23.042767810000004,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_CIE1976(xp_as_array([29.805654680097106], xp=xp)),
            6.157200790000001,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_CIE1976(xp_as_array([56.480581732417676], xp=xp), 50),
            12.197225349999998,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_CIE1976(xp_as_array([47.317620274162735], xp=xp), 75),
            12.197225350000002,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_CIE1976(xp_as_array([42.519930728120940], xp=xp), 95),
            12.197225350000005,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_luminance_CIE1976(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_CIE1976`
        definition n-dimensional arrays support.
        """

        L_star = 41.527875844653451
        Y = luminance_CIE1976(L_star)

        L_star = xp.tile(xp_as_array(L_star, xp=xp), (6,))
        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        xp_assert_close(luminance_CIE1976(L_star), Y, atol=TOLERANCE_ABSOLUTE_TESTS)

        L_star = xp_reshape(xp_as_array(L_star, xp=xp), (2, 3), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(luminance_CIE1976(L_star), Y, atol=TOLERANCE_ABSOLUTE_TESTS)

        L_star = xp_reshape(xp_as_array(L_star, xp=xp), (2, 3, 1), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(luminance_CIE1976(L_star), Y, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_luminance_CIE1976(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.colorimetry.luminance.luminance_CIE1976`
        definition domain and range scale support.
        """

        Y = as_ndarray(luminance_CIE1976(41.527875844653451, 100))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    luminance_CIE1976(41.527875844653451 * factor, 100 * factor),
                    Y * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_luminance_CIE1976(self) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_CIE1976`
        definition nan support.
        """

        luminance_CIE1976(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLuminanceFairchild2010:
    """
    Define :func:`colour.colorimetry.luminance.luminance_Fairchild2010`
    definition unit tests methods.
    """

    @pytest.mark.mps_xfail("MPS float32 precision divergence")
    def test_luminance_Fairchild2010(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_Fairchild2010`
        definition.
        """

        xp_assert_close(
            luminance_Fairchild2010(xp_as_array([31.996390226262736], xp=xp)),
            0.12197225350000002,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Fairchild2010(xp_as_array([60.203153682783302], xp=xp)),
            0.23042767809999998,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Fairchild2010(xp_as_array([11.836517240976489], xp=xp)),
            0.06157200790000001,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Fairchild2010(xp_as_array([24.424283249379986], xp=xp), 2.75),
            0.12197225350000002,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Fairchild2010(xp_as_array([100.019986327374240], xp=xp)),
            1008.00000024,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Fairchild2010(xp_as_array([100.019999997090270], xp=xp)),
            100799.92312466,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_luminance_Fairchild2010(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_Fairchild2010`
        definition n-dimensional arrays support.
        """

        L_hdr = 31.996390226262736
        Y = luminance_Fairchild2010(L_hdr)

        L_hdr = xp.tile(xp_as_array(L_hdr, xp=xp), (6,))
        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        xp_assert_close(
            luminance_Fairchild2010(L_hdr), Y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        L_hdr = xp_reshape(xp_as_array(L_hdr, xp=xp), (2, 3), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            luminance_Fairchild2010(L_hdr), Y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        L_hdr = xp_reshape(xp_as_array(L_hdr, xp=xp), (2, 3, 1), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            luminance_Fairchild2010(L_hdr), Y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_luminance_Fairchild2010(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.colorimetry.luminance.luminance_Fairchild2010`
        definition domain and range scale support.
        """

        Y = as_ndarray(luminance_Fairchild2010(31.996390226262736))

        d_r = (("reference", 1, 1), ("1", 0.01, 1), ("100", 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    luminance_Fairchild2010(31.996390226262736 * factor_a),
                    Y * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_luminance_Fairchild2010(self) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_Fairchild2010`
        definition nan support.
        """

        luminance_Fairchild2010(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLuminanceFairchild2011:
    """
    Define :func:`colour.colorimetry.luminance.luminance_Fairchild2011`
    definition unit tests methods.
    """

    @pytest.mark.mps_tolerance_absolute(1e-1)
    def test_luminance_Fairchild2011(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_Fairchild2011`
        definition.
        """

        xp_assert_close(
            luminance_Fairchild2011(xp_as_array([51.852958445912506], xp=xp)),
            0.12197225350000007,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Fairchild2011(xp_as_array([65.275207956353853], xp=xp)),
            0.23042767809999998,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Fairchild2011(xp_as_array([39.818935510715917], xp=xp)),
            0.061572007900000038,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Fairchild2011(xp_as_array([0.13268968410139345], xp=xp), 2.75),
            0.12197225350000002,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Fairchild2011(xp_as_array([234.72925681957565], xp=xp)),
            1008.00000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Fairchild2011(xp_as_array([245.57059778237573], xp=xp)),
            100800.00000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_luminance_Fairchild2011(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_Fairchild2011`
        definition n-dimensional arrays support.
        """

        L_hdr = 51.852958445912506
        Y = luminance_Fairchild2011(L_hdr)

        L_hdr = xp.tile(xp_as_array(L_hdr, xp=xp), (6,))
        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        xp_assert_close(
            luminance_Fairchild2011(L_hdr), Y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        L_hdr = xp_reshape(xp_as_array(L_hdr, xp=xp), (2, 3), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            luminance_Fairchild2011(L_hdr), Y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        L_hdr = xp_reshape(xp_as_array(L_hdr, xp=xp), (2, 3, 1), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(
            luminance_Fairchild2011(L_hdr), Y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_luminance_Fairchild2011(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.colorimetry.luminance.luminance_Fairchild2011`
        definition domain and range scale support.
        """

        Y = as_ndarray(luminance_Fairchild2011(26.459509817572265))

        d_r = (("reference", 1, 1), ("1", 0.01, 1), ("100", 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    luminance_Fairchild2011(26.459509817572265 * factor_a),
                    Y * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_luminance_Fairchild2011(self) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_Fairchild2011`
        definition nan support.
        """

        luminance_Fairchild2011(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestLuminanceAbebe2017:
    """
    Define :func:`colour.colorimetry.luminance.luminance_Abebe2017`
    definition unit tests methods.
    """

    def test_luminance_Abebe2017(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_Abebe2017`
        definition.
        """

        xp_assert_close(
            luminance_Abebe2017(xp_as_array([0.486955571109229], xp=xp)),
            12.197225350000004,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Abebe2017(
                xp_as_array([0.474544792145434], xp=xp), method="Stevens"
            ),
            12.197225350000025,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Abebe2017(xp_as_array([0.286847428534793], xp=xp), 1000),
            12.197225350000046,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Abebe2017(xp_as_array([0.192145492588158], xp=xp), 4000),
            12.197225350000121,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            luminance_Abebe2017(
                xp_as_array([0.170365211220992], xp=xp), 4000, method="Stevens"
            ),
            12.197225349999933,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_luminance_Abebe2017(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_Abebe2017`
        definition n-dimensional arrays support.
        """

        L = 0.486955571109229
        Y = luminance_Abebe2017(L)

        L = xp.tile(xp_as_array(L, xp=xp), (6,))
        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        xp_assert_close(luminance_Abebe2017(L), Y, atol=TOLERANCE_ABSOLUTE_TESTS)

        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        xp_assert_close(luminance_Abebe2017(L), Y, atol=TOLERANCE_ABSOLUTE_TESTS)

        L = xp_reshape(xp_as_array(L, xp=xp), (2, 3, 1), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3, 1), xp=xp)
        xp_assert_close(luminance_Abebe2017(L), Y, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_luminance_Abebe2017(self) -> None:
        """
        Test :func:`colour.colorimetry.luminance.luminance_Abebe2017`
        definition nan support.
        """

        cases = np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        luminance_Abebe2017(cases, cases)


class TestLuminance:
    """
    Define :func:`colour.colorimetry.luminance.luminance` definition unit
    tests methods.
    """

    def test_domain_range_scale_luminance(self, xp: ModuleType) -> None:  # noqa: ARG002
        """
        Test :func:`colour.colorimetry.luminance.luminance` definition
        domain and range scale support.
        """

        m = (
            "Newhall 1943",
            "ASTM D1535",
            "CIE 1976",
            "Fairchild 2010",
            "Fairchild 2011",
            "Abebe 2017",
        )
        v = [as_ndarray(luminance(41.527875844653451, method, Y_n=100)) for method in m]

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for method, value in zip(m, v, strict=True):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    xp_assert_close(
                        luminance(
                            41.527875844653451 * factor,
                            method,
                            Y_n=100 * factor,
                        ),
                        value * factor,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )
