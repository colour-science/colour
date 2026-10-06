"""Define the unit tests for the :mod:`colour.colorimetry.yellowness` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.colorimetry import (
    yellowness_ASTMD1925,
    yellowness_ASTME313,
    yellowness_ASTME313_alternative,
)
from colour.colorimetry.yellowness import YELLOWNESS_COEFFICIENTS_ASTME313, yellowness
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
    "TestYellownessASTMD1925",
    "TestYellownessASTM313Alternative",
    "TestYellownessASTM313",
    "TestYellowness",
]


class TestYellownessASTMD1925:
    """
    Define :func:`colour.colorimetry.yellowness.yellowness_ASTMD1925`
    definition unit tests methods.
    """

    def test_yellowness_ASTMD1925(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.yellowness.yellowness_ASTMD1925`
        definition.
        """

        xp_assert_close(
            yellowness_ASTMD1925(
                xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
            ),
            10.299999999999997,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            yellowness_ASTMD1925(
                xp_as_array([105.00000000, 100.00000000, 95.00000000], xp=xp)
            ),
            33.700000000000003,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            yellowness_ASTMD1925(
                xp_as_array([100.00000000, 100.00000000, 100.00000000], xp=xp)
            ),
            22.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_yellowness_ASTMD1925(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.yellowness.yellowness_ASTMD1925`
        definition n_dimensional arrays support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
        YI = as_ndarray(yellowness_ASTMD1925(XYZ))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        YI = xp.tile(xp_as_array(YI, xp=xp), (6,))
        xp_assert_close(
            yellowness_ASTMD1925(XYZ),
            YI,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        YI = xp_reshape(xp_as_array(YI, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            yellowness_ASTMD1925(XYZ),
            YI,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_yellowness_ASTMD1925(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.yellowness.yellowness_ASTMD1925`
        definition domain and range scale support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
        YI = as_ndarray(yellowness_ASTMD1925(XYZ))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    yellowness_ASTMD1925(XYZ * factor),
                    YI * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_yellowness_ASTMD1925(self) -> None:
        """
        Test :func:`colour.colorimetry.yellowness.yellowness_ASTMD1925`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        yellowness_ASTMD1925(cases)


class TestYellownessASTM313Alternative:
    """
    Define :func:`colour.colorimetry.yellowness.\
yellowness_ASTME313_alternative` definition unit tests methods.
    """

    def test_yellowness_ASTME313_alternative(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.yellowness.\
yellowness_ASTME313_alternative` definition.
        """

        xp_assert_close(
            yellowness_ASTME313_alternative(
                xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
            ),
            11.065000000000003,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            yellowness_ASTME313_alternative(
                xp_as_array([105.00000000, 100.00000000, 95.00000000], xp=xp)
            ),
            19.534999999999989,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            yellowness_ASTME313_alternative(
                xp_as_array([100.00000000, 100.00000000, 100.00000000], xp=xp)
            ),
            15.300000000000002,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_yellowness_ASTME313_alternative(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.colorimetry.yellowness.\
yellowness_ASTME313_alternative` definition n_dimensional arrays support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
        YI = as_ndarray(yellowness_ASTME313_alternative(XYZ))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        YI = xp.tile(xp_as_array(YI, xp=xp), (6,))
        xp_assert_close(
            yellowness_ASTME313_alternative(XYZ),
            YI,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        YI = xp_reshape(xp_as_array(YI, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            yellowness_ASTME313_alternative(XYZ),
            YI,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_yellowness_ASTME313_alternative(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.colorimetry.yellowness.\
yellowness_ASTME313_alternative` definition domain and range scale support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
        YI = as_ndarray(yellowness_ASTME313_alternative(XYZ))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    yellowness_ASTME313_alternative(XYZ * factor),
                    YI * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_yellowness_ASTME313_alternative(self) -> None:
        """
        Test :func:`colour.colorimetry.yellowness.\
yellowness_ASTME313_alternative` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        yellowness_ASTME313_alternative(cases)


class TestYellownessASTM313:
    """
    Define :func:`colour.colorimetry.yellowness.yellowness_ASTME313`
    definition unit tests methods.
    """

    def test_yellowness_ASTME313(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.yellowness.yellowness_ASTME313`
        definition.
        """

        xp_assert_close(
            yellowness_ASTME313(
                xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
            ),
            4.340000000000003,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            yellowness_ASTME313(
                xp_as_array([105.00000000, 100.00000000, 95.00000000], xp=xp)
            ),
            28.660000000000011,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            yellowness_ASTME313(
                xp_as_array([100.00000000, 100.00000000, 100.00000000], xp=xp)
            ),
            16.500000000000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            yellowness_ASTME313(
                xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp),
                YELLOWNESS_COEFFICIENTS_ASTME313["CIE 1931 2 Degree Standard Observer"][
                    "C"
                ],
            ),
            10.089500000000001,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_yellowness_ASTME313(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.yellowness.yellowness_ASTME313`
        definition n_dimensional arrays support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
        YI = as_ndarray(yellowness_ASTME313(XYZ))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        YI = xp.tile(xp_as_array(YI, xp=xp), (6,))
        xp_assert_close(
            yellowness_ASTME313(XYZ),
            YI,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        YI = xp_reshape(xp_as_array(YI, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            yellowness_ASTME313(XYZ),
            YI,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_yellowness_ASTME313(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.yellowness.yellowness_ASTME313`
        definition domain and range scale support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
        YI = as_ndarray(yellowness_ASTME313(XYZ))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    yellowness_ASTME313(XYZ * factor),
                    YI * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_yellowness_ASTME313(self) -> None:
        """
        Test :func:`colour.colorimetry.yellowness.yellowness_ASTME313`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        yellowness_ASTME313(cases)


class TestYellowness:
    """
    Define :func:`colour.colorimetry.yellowness.yellowness` definition unit
    tests methods.
    """

    def test_domain_range_scale_yellowness(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.yellowness.yellowness` definition
        domain and range scale support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)

        m = ("ASTM D1925", "ASTM E313")
        v = [as_ndarray(yellowness(XYZ, method)) for method in m]

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for method, value in zip(m, v, strict=True):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    xp_assert_close(
                        yellowness(XYZ * factor, method),
                        value * factor,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )
