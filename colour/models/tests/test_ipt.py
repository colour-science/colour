"""Define the unit tests for the :mod:`colour.models.ipt` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import IPT_hue_angle, IPT_to_XYZ, XYZ_to_IPT
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
    "TestXYZ_to_IPT",
    "TestIPT_to_XYZ",
    "TestIPTHueAngle",
]


class TestXYZ_to_IPT:
    """Define :func:`colour.models.ipt.XYZ_to_IPT` definition unit tests methods."""

    def test_XYZ_to_IPT(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.ipt.XYZ_to_IPT` definition."""

        xp_assert_close(
            XYZ_to_IPT(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            [0.38426191, 0.38487306, 0.18886838],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_IPT(xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            [0.49437481, -0.19251742, 0.18080304],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_IPT(xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp)),
            [0.35167774, -0.07525627, -0.30921279],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_IPT(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.ipt.XYZ_to_IPT` definition n-dimensional
        support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        IPT = as_ndarray(XYZ_to_IPT(XYZ))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        IPT = xp.tile(xp_as_array(IPT, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_IPT(XYZ), IPT, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        IPT = xp_reshape(xp_as_array(IPT, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_IPT(XYZ), IPT, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_IPT(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.ipt.XYZ_to_IPT` definition domain and
        range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        IPT = as_ndarray(XYZ_to_IPT(XYZ))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_IPT(XYZ * factor),
                    IPT * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_IPT(self) -> None:
        """Test :func:`colour.models.ipt.XYZ_to_IPT` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_IPT(cases)


class TestIPT_to_XYZ:
    """
    Define :func:`colour.models.ipt.IPT_to_XYZ` definition unit tests
    methods.
    """

    def test_IPT_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.ipt.IPT_to_XYZ` definition."""

        xp_assert_close(
            IPT_to_XYZ(xp_as_array([0.38426191, 0.38487306, 0.18886838], xp=xp)),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            IPT_to_XYZ(xp_as_array([0.49437481, -0.19251742, 0.18080304], xp=xp)),
            [0.14222010, 0.23042768, 0.10495772],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            IPT_to_XYZ(xp_as_array([0.35167774, -0.07525627, -0.30921279], xp=xp)),
            [0.07818780, 0.06157201, 0.28099326],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_IPT_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.ipt.IPT_to_XYZ` definition n-dimensional
        support.
        """

        IPT = xp_as_array([0.38426191, 0.38487306, 0.18886838], xp=xp)
        XYZ = as_ndarray(IPT_to_XYZ(IPT))

        IPT = xp.tile(xp_as_array(IPT, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(IPT_to_XYZ(IPT), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        IPT = xp_reshape(xp_as_array(IPT, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(IPT_to_XYZ(IPT), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_IPT_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.ipt.IPT_to_XYZ` definition domain and
        range scale support.
        """

        IPT = xp_as_array([0.38426191, 0.38487306, 0.18886838], xp=xp)
        XYZ = as_ndarray(IPT_to_XYZ(IPT))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    IPT_to_XYZ(IPT * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_IPT_to_XYZ(self) -> None:
        """Test :func:`colour.models.ipt.IPT_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        IPT_to_XYZ(cases)


class TestIPTHueAngle:
    """
    Define :func:`colour.models.ipt.IPT_hue_angle` definition unit tests
    methods.
    """

    def test_IPT_hue_angle(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.ipt.IPT_hue_angle` definition."""

        xp_assert_close(
            IPT_hue_angle(xp_as_array([0.38426191, 0.38487306, 0.18886838], xp=xp)),
            26.138526939899490,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            IPT_hue_angle(xp_as_array([0.49437481, -0.19251742, 0.18080304], xp=xp)),
            136.797287973958500,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            IPT_hue_angle(xp_as_array([0.35167774, -0.07525627, -0.30921279], xp=xp)),
            256.321284526533300,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_IPT_hue_angle(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.ipt.IPT_hue_angle` definition n-dimensional
        support.
        """

        IPT = xp_as_array([0.38426191, 0.38487306, 0.18886838], xp=xp)
        hue = as_ndarray(IPT_hue_angle(IPT))

        IPT = xp.tile(xp_as_array(IPT, xp=xp), (6, 1))
        hue = xp.tile(xp_as_array(hue, xp=xp), (6,))
        xp_assert_close(IPT_hue_angle(IPT), hue, atol=TOLERANCE_ABSOLUTE_TESTS)

        IPT = xp_reshape(xp_as_array(IPT, xp=xp), (2, 3, 3), xp=xp)
        hue = xp_reshape(xp_as_array(hue, xp=xp), (2, 3), xp=xp)
        xp_assert_close(IPT_hue_angle(IPT), hue, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_IPT_hue_angle(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.ipt.IPT_hue_angle` definition domain and
        range scale support.
        """

        IPT = xp_as_array([0.38426191, 0.38487306, 0.18886838], xp=xp)
        hue = as_ndarray(IPT_hue_angle(IPT))

        d_r = (("reference", 1, 1), ("1", 1, 1 / 360), ("100", 100, 1 / 3.6))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    IPT_hue_angle(IPT * xp_as_array(factor_a, xp=xp)),
                    hue * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_IPT_hue_angle(self) -> None:
        """Test :func:`colour.models.ipt.IPT_hue_angle` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        IPT_hue_angle(cases)
