"""Define the unit tests for the :mod:`colour.colorimetry.whiteness` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.colorimetry import (
    whiteness_ASTME313,
    whiteness_Berger1959,
    whiteness_CIE2004,
    whiteness_Ganz1979,
    whiteness_Stensby1968,
    whiteness_Taube1960,
)
from colour.colorimetry.whiteness import whiteness
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
    "TestWhitenessBerger1959",
    "TestWhitenessTaube1960",
    "TestWhitenessStensby1968",
    "TestWhitenessASTM313",
    "TestWhitenessGanz1979",
    "TestWhitenessCIE2004",
    "TestWhiteness",
]


class TestWhitenessBerger1959:
    """
    Define :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
    definition unit tests methods.
    """

    def test_whiteness_Berger1959(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
        definition.
        """

        xp_assert_close(
            whiteness_Berger1959(
                xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp),
                xp_as_array([94.80966767, 100.00000000, 107.30513595], xp=xp),
            ),
            30.36380179,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            whiteness_Berger1959(
                xp_as_array([105.00000000, 100.00000000, 95.00000000], xp=xp),
                xp_as_array([94.80966767, 100.00000000, 107.30513595], xp=xp),
            ),
            5.530469280673941,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            whiteness_Berger1959(
                xp_as_array([100.00000000, 100.00000000, 100.00000000], xp=xp),
                xp_as_array([100.00000000, 100.00000000, 100.00000000], xp=xp),
            ),
            33.300000000000011,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_whiteness_Berger1959(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
        definition n_dimensional arrays support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
        XYZ_0 = xp_as_array([94.80966767, 100.00000000, 107.30513595], xp=xp)
        W = as_ndarray(whiteness_Berger1959(XYZ, XYZ_0))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        XYZ_0 = xp.tile(xp_as_array(XYZ_0, xp=xp), (6, 1))
        W = xp.tile(xp_as_array(W, xp=xp), (6,))
        xp_assert_close(
            whiteness_Berger1959(XYZ, XYZ_0),
            W,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_0 = xp_reshape(xp_as_array(XYZ_0, xp=xp), (2, 3, 3), xp=xp)
        W = xp_reshape(xp_as_array(W, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            whiteness_Berger1959(XYZ, XYZ_0),
            W,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_whiteness_Berger1959(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
        definition domain and range scale support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
        XYZ_0 = xp_as_array([94.80966767, 100.00000000, 107.30513595], xp=xp)
        W = as_ndarray(whiteness_Berger1959(XYZ, XYZ_0))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    whiteness_Berger1959(XYZ * factor, XYZ_0 * factor),
                    W * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_whiteness_Berger1959(self) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        whiteness_Berger1959(cases, cases)


class TestWhitenessTaube1960:
    """
    Define :func:`colour.colorimetry.whiteness.whiteness_Taube1960`
    definition unit tests methods.
    """

    def test_whiteness_Taube1960(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Taube1960`
        definition.
        """

        xp_assert_close(
            whiteness_Taube1960(
                xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp),
                xp_as_array([94.80966767, 100.00000000, 107.30513595], xp=xp),
            ),
            91.407173833416152,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            whiteness_Taube1960(
                xp_as_array([105.00000000, 100.00000000, 95.00000000], xp=xp),
                xp_as_array([94.80966767, 100.00000000, 107.30513595], xp=xp),
            ),
            54.130300134995593,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            whiteness_Taube1960(
                xp_as_array([100.00000000, 100.00000000, 100.00000000], xp=xp),
                xp_as_array([100.00000000, 100.00000000, 100.00000000], xp=xp),
            ),
            100.0,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_whiteness_Taube1960(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Taube1960`
        definition n_dimensional arrays support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
        XYZ_0 = xp_as_array([94.80966767, 100.00000000, 107.30513595], xp=xp)
        WI = as_ndarray(whiteness_Taube1960(XYZ, XYZ_0))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        XYZ_0 = xp.tile(xp_as_array(XYZ_0, xp=xp), (6, 1))
        WI = xp.tile(xp_as_array(WI, xp=xp), (6,))
        xp_assert_close(
            whiteness_Taube1960(XYZ, XYZ_0),
            WI,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_0 = xp_reshape(xp_as_array(XYZ_0, xp=xp), (2, 3, 3), xp=xp)
        WI = xp_reshape(xp_as_array(WI, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            whiteness_Taube1960(XYZ, XYZ_0),
            WI,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_whiteness_Taube1960(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Taube1960`
        definition domain and range scale support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
        XYZ_0 = xp_as_array([94.80966767, 100.00000000, 107.30513595], xp=xp)
        WI = as_ndarray(whiteness_Taube1960(XYZ, XYZ_0))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    whiteness_Taube1960(XYZ * factor, XYZ_0 * factor),
                    WI * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_whiteness_Berger1959(self) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Berger1959`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        whiteness_Berger1959(cases, cases)


class TestWhitenessStensby1968:
    """
    Define :func:`colour.colorimetry.whiteness.whiteness_Stensby1968`
    definition unit tests methods.
    """

    def test_whiteness_Stensby1968(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Stensby1968`
        definition.
        """

        xp_assert_close(
            whiteness_Stensby1968(
                xp_as_array([100.00000000, -2.46875131, -16.72486654], xp=xp)
            ),
            142.76834569,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            whiteness_Stensby1968(
                xp_as_array([100.00000000, 14.40943727, -9.61394885], xp=xp)
            ),
            172.07015836,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            whiteness_Stensby1968(xp_as_array([1, 1, 1], xp=xp)),
            1.00000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_whiteness_Stensby1968(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Stensby1968`
        definition n_dimensional arrays support.
        """

        Lab = xp_as_array([100.00000000, -2.46875131, -16.72486654], xp=xp)
        WI = as_ndarray(whiteness_Stensby1968(Lab))

        Lab = xp.tile(xp_as_array(Lab, xp=xp), (6, 1))
        WI = xp.tile(xp_as_array(WI, xp=xp), (6,))
        xp_assert_close(
            whiteness_Stensby1968(Lab),
            WI,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Lab = xp_reshape(xp_as_array(Lab, xp=xp), (2, 3, 3), xp=xp)
        WI = xp_reshape(xp_as_array(WI, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            whiteness_Stensby1968(Lab),
            WI,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_whiteness_Stensby1968(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Stensby1968`
        definition domain and range scale support.
        """

        Lab = xp_as_array([100.00000000, -2.46875131, -16.72486654], xp=xp)
        WI = as_ndarray(whiteness_Stensby1968(Lab))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    whiteness_Stensby1968(Lab * factor),
                    WI * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_whiteness_Stensby1968(self) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Stensby1968`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        whiteness_Stensby1968(cases)


class TestWhitenessASTM313:
    """
    Define :func:`colour.colorimetry.whiteness.whiteness_ASTME313`
    definition unit tests methods.
    """

    def test_whiteness_ASTME313(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_ASTME313`
        definition.
        """

        xp_assert_close(
            whiteness_ASTME313(
                xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
            ),
            55.740000000000009,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            whiteness_ASTME313(
                xp_as_array([105.00000000, 100.00000000, 95.00000000], xp=xp)
            ),
            21.860000000000014,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            whiteness_ASTME313(
                xp_as_array([100.00000000, 100.00000000, 100.00000000], xp=xp)
            ),
            38.800000000000011,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_whiteness_ASTME313(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_ASTME313`
        definition n_dimensional arrays support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
        WI = as_ndarray(whiteness_ASTME313(XYZ))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        WI = xp.tile(xp_as_array(WI, xp=xp), (6,))
        xp_assert_close(
            whiteness_ASTME313(XYZ),
            WI,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        WI = xp_reshape(xp_as_array(WI, xp=xp), (2, 3), xp=xp)
        xp_assert_close(
            whiteness_ASTME313(XYZ),
            WI,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_whiteness_ASTME313(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_ASTME313`
        definition domain and range scale support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
        WI = as_ndarray(whiteness_ASTME313(XYZ))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    whiteness_ASTME313(XYZ * factor),
                    WI * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_whiteness_ASTME313(self) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_ASTME313`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        whiteness_ASTME313(cases)


class TestWhitenessGanz1979:
    """
    Define :func:`colour.colorimetry.whiteness.whiteness_Ganz1979`
    definition unit tests methods.
    """

    def test_whiteness_Ganz1979(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Ganz1979`
        definition.
        """

        xp_assert_close(
            whiteness_Ganz1979(xp_as_array([0.3139, 0.3311], xp=xp), 100),
            [99.33176520, 1.76108290],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            whiteness_Ganz1979(xp_as_array([0.3500, 0.3334], xp=xp), 100),
            [23.38525400, -32.66182560],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            whiteness_Ganz1979(xp_as_array([0.3334, 0.3334], xp=xp), 100),
            [54.39939920, -16.04152380],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_whiteness_Ganz1979(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Ganz1979`
        definition n_dimensional arrays support.
        """

        xy = xp_as_array([0.3167, 0.3334], xp=xp)
        Y = 100
        WT = as_ndarray(whiteness_Ganz1979(xy, Y))

        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        WT = xp.tile(xp_as_array(WT, xp=xp), (6, 1))
        xp_assert_close(
            whiteness_Ganz1979(xy, Y),
            WT,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        xp_assert_close(
            whiteness_Ganz1979(xy, Y),
            WT,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        WT = xp_reshape(xp_as_array(WT, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(
            whiteness_Ganz1979(xy, Y),
            WT,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_whiteness_Ganz1979(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Ganz1979`
        definition domain and range scale support.
        """

        xy = xp_as_array([0.3167, 0.3334], xp=xp)
        Y = 100
        WT = as_ndarray(whiteness_Ganz1979(xy, Y))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    whiteness_Ganz1979(xy, Y * factor),
                    WT * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_whiteness_Ganz1979(self) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_Ganz1979`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        whiteness_Ganz1979(cases[..., 0:2], cases[..., 0])


class TestWhitenessCIE2004:
    """
    Define :func:`colour.colorimetry.whiteness.whiteness_CIE2004`
    definition unit tests methods.
    """

    def test_whiteness_CIE2004(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_CIE2004`
        definition.
        """

        xp_assert_close(
            whiteness_CIE2004(
                xp_as_array([0.3139, 0.3311], xp=xp),
                100,
                xp_as_array([0.3139, 0.3311], xp=xp),
            ),
            [100.00000000, 0.00000000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            whiteness_CIE2004(
                xp_as_array([0.3500, 0.3334], xp=xp),
                100,
                xp_as_array([0.3139, 0.3311], xp=xp),
            ),
            [67.21000000, -34.60500000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            whiteness_CIE2004(
                xp_as_array([0.3334, 0.3334], xp=xp),
                100,
                xp_as_array([0.3139, 0.3311], xp=xp),
            ),
            [80.49000000, -18.00500000],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_whiteness_CIE2004(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_CIE2004`
        definition n_dimensional arrays support.
        """

        xy = xp_as_array([0.3167, 0.3334], xp=xp)
        Y = 100
        xy_n = xp_as_array([0.3139, 0.3311], xp=xp)
        WT = as_ndarray(whiteness_CIE2004(xy, Y, xy_n))

        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        WT = xp.tile(xp_as_array(WT, xp=xp), (6, 1))
        xp_assert_close(
            whiteness_CIE2004(xy, Y, xy_n),
            WT,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Y = xp.tile(xp_as_array(Y, xp=xp), (6,))
        xy_n = xp.tile(xp_as_array(xy_n, xp=xp), (6, 1))
        xp_assert_close(
            whiteness_CIE2004(xy, Y, xy_n),
            WT,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        Y = xp_reshape(xp_as_array(Y, xp=xp), (2, 3), xp=xp)
        xy_n = xp_reshape(xp_as_array(xy_n, xp=xp), (2, 3, 2), xp=xp)
        WT = xp_reshape(xp_as_array(WT, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(
            whiteness_CIE2004(xy, Y, xy_n),
            WT,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_whiteness_CIE2004(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_CIE2004`
        definition domain and range scale support.
        """

        xy = xp_as_array([0.3167, 0.3334], xp=xp)
        Y = 100
        xy_n = xp_as_array([0.3139, 0.3311], xp=xp)
        WT = as_ndarray(whiteness_CIE2004(xy, Y, xy_n))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    whiteness_CIE2004(xy, Y * factor, xy_n),
                    WT * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_whiteness_CIE2004(self) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness_CIE2004`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        whiteness_CIE2004(cases[..., 0:2], cases[..., 0], cases[..., 0:2])


class TestWhiteness:
    """
    Define :func:`colour.colorimetry.whiteness.whiteness` definition unit
    tests methods.
    """

    def test_whiteness(self, xp: ModuleType) -> None:
        """Test :func:`colour.colorimetry.whiteness.whiteness` definition."""

        # NOTE: Sample ``Y`` is deliberately different from whitepoint ``Y`` to
        # ensure the dispatcher forwards the sample tristimulus ``Y`` to the
        # "Ganz 1979" and "CIE 2004" methods rather than the whitepoint one.
        XYZ = xp_as_array([95.00000000, 80.00000000, 105.00000000], xp=xp)
        XYZ_0 = xp_as_array([94.80966767, 100.00000000, 107.30513595], xp=xp)

        expected = {
            "Berger 1959": 23.70380179,
            "Taube 1960": 151.40717383,
            "Stensby 1968": 238.89308915,
            "ASTM E313": 115.74,
            "Ganz 1979": np.array([199.63460714, -57.62080357]),
            "CIE 2004": np.array([136.61314286, -54.90142858]),
        }

        for method, value in expected.items():
            xp_assert_close(
                whiteness(XYZ, XYZ_0, method),
                value,
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    def test_domain_range_scale_whiteness(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.colorimetry.whiteness.whiteness` definition domain
        and range scale support.
        """

        XYZ = xp_as_array([95.00000000, 100.00000000, 105.00000000], xp=xp)
        XYZ_0 = xp_as_array([94.80966767, 100.00000000, 107.30513595], xp=xp)

        m = (
            "Berger 1959",
            "Taube 1960",
            "Stensby 1968",
            "ASTM E313",
            "Ganz 1979",
            "CIE 2004",
        )
        v = [as_ndarray(whiteness(XYZ, XYZ_0, method)) for method in m]

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for method, value in zip(m, v, strict=True):
            for scale, factor in d_r:
                with domain_range_scale(scale):
                    xp_assert_close(
                        whiteness(XYZ * factor, XYZ_0 * factor, method),
                        value * factor,
                        atol=TOLERANCE_ABSOLUTE_TESTS,
                    )
