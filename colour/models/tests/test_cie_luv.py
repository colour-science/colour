"""Define the unit tests for the :mod:`colour.models.cie_luv` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    CIE1976UCS_to_XYZ,
    Luv_to_uv,
    Luv_to_XYZ,
    Luv_uv_to_xy,
    XYZ_to_CIE1976UCS,
    XYZ_to_Luv,
    uv_to_Luv,
    xy_to_Luv_uv,
)
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
    "TestXYZ_to_Luv",
    "TestLuv_to_XYZ",
    "TestLuv_to_uv",
    "Testuv_to_Luv",
    "TestLuv_uv_to_xy",
    "TestXy_to_Luv_uv",
    "TestXYZ_to_CIE1976UCS",
    "TestCIE1976UCS_to_XYZ",
]


class TestXYZ_to_Luv:
    """
    Define :func:`colour.models.cie_luv.XYZ_to_Luv` definition unit tests
    methods.
    """

    def test_XYZ_to_Luv(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_luv.XYZ_to_Luv` definition."""

        xp_assert_close(
            XYZ_to_Luv(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            [41.52787529, 96.83626054, 17.75210149],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Luv(xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            [55.11636304, -37.59308176, 44.13768458],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Luv(xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp)),
            [29.80565520, -10.96316802, -65.06751860],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Luv(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_as_array([0.44757, 0.40745], xp=xp),
            ),
            [41.52787529, 65.45180940, -12.46626977],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Luv(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_as_array([0.34570, 0.35850], xp=xp),
            ),
            [41.52787529, 90.70925962, 7.08455273],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Luv(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_as_array([0.34570, 0.35850, 1.00000], xp=xp),
            ),
            [41.52787529, 90.70925962, 7.08455273],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_Luv(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.XYZ_to_Luv` definition n-dimensional
        support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        Luv = as_ndarray(XYZ_to_Luv(XYZ, illuminant))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        Luv = xp.tile(xp_as_array(Luv, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_Luv(XYZ, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS)

        illuminant = xp.tile(xp_as_array(illuminant, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_Luv(XYZ, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        illuminant = xp_reshape(xp_as_array(illuminant, xp=xp), (2, 3, 2), xp=xp)
        Luv = xp_reshape(xp_as_array(Luv, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_Luv(XYZ, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_Luv(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.XYZ_to_Luv` definition
        domain and range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        Luv = as_ndarray(XYZ_to_Luv(XYZ, illuminant))

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_Luv(XYZ * xp_as_array(factor_a, xp=xp), illuminant),
                    Luv * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Luv(self) -> None:
        """Test :func:`colour.models.cie_luv.XYZ_to_Luv` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Luv(cases, cases[..., 0:2])


class TestLuv_to_XYZ:
    """
    Define :func:`colour.models.cie_luv.Luv_to_XYZ` definition unit tests
    methods.
    """

    def test_Luv_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_luv.Luv_to_XYZ` definition."""

        xp_assert_close(
            Luv_to_XYZ(xp_as_array([41.52787529, 96.83626054, 17.75210149], xp=xp)),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Luv_to_XYZ(xp_as_array([55.11636304, -37.59308176, 44.13768458], xp=xp)),
            [0.14222010, 0.23042768, 0.10495772],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Luv_to_XYZ(xp_as_array([29.80565520, -10.96316802, -65.06751860], xp=xp)),
            [0.07818780, 0.06157201, 0.28099326],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Luv_to_XYZ(
                xp_as_array([41.52787529, 65.45180940, -12.46626977], xp=xp),
                xp_as_array([0.44757, 0.40745], xp=xp),
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Luv_to_XYZ(
                xp_as_array([41.52787529, 90.70925962, 7.08455273], xp=xp),
                xp_as_array([0.34570, 0.35850], xp=xp),
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Luv_to_XYZ(
                xp_as_array([41.52787529, 90.70925962, 7.08455273], xp=xp),
                xp_as_array([0.34570, 0.35850, 1.00000], xp=xp),
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Luv_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.Luv_to_XYZ` definition n-dimensional
        support.
        """

        Luv = xp_as_array([41.52787529, 96.83626054, 17.75210149], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        XYZ = as_ndarray(Luv_to_XYZ(Luv, illuminant))

        Luv = xp.tile(xp_as_array(Luv, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(Luv_to_XYZ(Luv, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        illuminant = xp.tile(xp_as_array(illuminant, xp=xp), (6, 1))
        xp_assert_close(Luv_to_XYZ(Luv, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        Luv = xp_reshape(xp_as_array(Luv, xp=xp), (2, 3, 3), xp=xp)
        illuminant = xp_reshape(xp_as_array(illuminant, xp=xp), (2, 3, 2), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(Luv_to_XYZ(Luv, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_Luv_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.Luv_to_XYZ` definition
        domain and range scale support.
        """

        Luv = xp_as_array([41.52787529, 96.83626054, 17.75210149], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        XYZ = as_ndarray(Luv_to_XYZ(Luv, illuminant))

        d_r = (("reference", 1, 1), ("1", 0.01, 1), ("100", 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    Luv_to_XYZ(Luv * xp_as_array(factor_a, xp=xp), illuminant),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Luv_to_XYZ(self) -> None:
        """Test :func:`colour.models.cie_luv.Luv_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Luv_to_XYZ(cases, cases[..., 0:2])


class TestLuv_to_uv:
    """
    Define :func:`colour.models.cie_luv.Luv_to_uv` definition unit tests
    methods.
    """

    def test_Luv_to_uv(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_luv.Luv_to_uv` definition."""

        xp_assert_close(
            Luv_to_uv(xp_as_array([41.52787529, 96.83626054, 17.75210149], xp=xp)),
            [0.37720213, 0.50120264],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Luv_to_uv(xp_as_array([55.11636304, -37.59308176, 44.13768458], xp=xp)),
            [0.14536327, 0.52992069],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Luv_to_uv(xp_as_array([29.80565520, -10.96316802, -65.06751860], xp=xp)),
            [0.16953603, 0.30039234],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Luv_to_uv(
                xp_as_array([41.52787529, 65.45180940, -12.46626977], xp=xp),
                xp_as_array([0.44757, 0.40745], xp=xp),
            ),
            [0.37720213, 0.50120264],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Luv_to_uv(
                xp_as_array([41.52787529, 90.70925962, 7.08455273], xp=xp),
                xp_as_array([0.34570, 0.35850], xp=xp),
            ),
            [0.37720213, 0.50120264],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Luv_to_uv(
                xp_as_array([41.52787529, 90.70925962, 7.08455273], xp=xp),
                xp_as_array([0.34570, 0.35850, 1.00000], xp=xp),
            ),
            [0.37720213, 0.50120264],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Luv_to_uv(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.Luv_to_uv` definition n-dimensional
        support.
        """

        Luv = xp_as_array([41.52787529, 96.83626054, 17.75210149], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        uv = as_ndarray(Luv_to_uv(Luv, illuminant))

        Luv = xp.tile(xp_as_array(Luv, xp=xp), (6, 1))
        uv = xp.tile(xp_as_array(uv, xp=xp), (6, 1))
        xp_assert_close(Luv_to_uv(Luv, illuminant), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

        illuminant = xp.tile(xp_as_array(illuminant, xp=xp), (6, 1))
        xp_assert_close(Luv_to_uv(Luv, illuminant), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

        Luv = xp_reshape(xp_as_array(Luv, xp=xp), (2, 3, 3), xp=xp)
        illuminant = xp_reshape(xp_as_array(illuminant, xp=xp), (2, 3, 2), xp=xp)
        uv = xp_reshape(xp_as_array(uv, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(Luv_to_uv(Luv, illuminant), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_Luv_to_uv(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.Luv_to_uv` definition
        domain and range scale support.
        """

        Luv = xp_as_array([41.52787529, 96.83626054, 17.75210149], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        uv = as_ndarray(Luv_to_uv(Luv, illuminant))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    Luv_to_uv(Luv * factor, illuminant),
                    uv,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Luv_to_uv(self) -> None:
        """Test :func:`colour.models.cie_luv.Luv_to_uv` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Luv_to_uv(cases, cases[..., 0:2])


class Testuv_to_Luv:
    """
    Define :func:`colour.models.cie_luv.uv_to_Luv` definition unit tests
    methods.
    """

    def test_uv_to_Luv(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_luv.uv_to_Luv` definition."""

        xp_assert_close(
            uv_to_Luv(xp_as_array([0.37720213, 0.50120264], xp=xp)),
            [100.00000000, 233.18376036, 42.74743858],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            uv_to_Luv(xp_as_array([0.14536327, 0.52992069], xp=xp)),
            [100.00000000, -68.20675764, 80.08090358],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            uv_to_Luv(xp_as_array([0.16953603, 0.30039234], xp=xp)),
            [100.00000000, -36.78216964, -218.3059514],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            uv_to_Luv(
                xp_as_array([0.37720213, 0.50120264], xp=xp),
                xp_as_array([0.44757, 0.40745], xp=xp),
            ),
            [100.00000000, 157.60933976, -30.01903705],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            uv_to_Luv(
                xp_as_array([0.37720213, 0.50120264], xp=xp),
                xp_as_array([0.34570, 0.35850], xp=xp),
            ),
            [100.00000000, 218.42981284, 17.05975609],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            uv_to_Luv(
                xp_as_array([0.37720213, 0.50120264], xp=xp),
                xp_as_array([0.34570, 0.35850, 1.00000], xp=xp),
            ),
            [100.00000000, 218.42981284, 17.05975609],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            uv_to_Luv(xp_as_array([0.37720213, 0.50120264], xp=xp), L=41.5278752),
            [41.52787529, 96.83626054, 17.75210149],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_uv_to_Luv(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.uv_to_Luv` definition n-dimensional
        support.
        """

        uv = xp_as_array([0.37720213, 0.50120264], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        Luv = as_ndarray(uv_to_Luv(uv, illuminant))

        uv = xp.tile(xp_as_array(uv, xp=xp), (6, 1))
        Luv = xp.tile(xp_as_array(Luv, xp=xp), (6, 1))
        xp_assert_close(uv_to_Luv(uv, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS)

        illuminant = xp.tile(xp_as_array(illuminant, xp=xp), (6, 1))
        xp_assert_close(uv_to_Luv(uv, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS)

        uv = xp_reshape(xp_as_array(uv, xp=xp), (2, 3, 2), xp=xp)
        illuminant = xp_reshape(xp_as_array(illuminant, xp=xp), (2, 3, 2), xp=xp)
        Luv = xp_reshape(xp_as_array(Luv, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(uv_to_Luv(uv, illuminant), Luv, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_uv_to_Luv(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.uv_to_Luv` definition
        domain and range scale support.
        """

        uv = xp_as_array([0.37720213, 0.50120264], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        L = 100
        Luv = as_ndarray(uv_to_Luv(uv, illuminant, L))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    uv_to_Luv(uv, illuminant, L * factor),
                    Luv * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_uv_to_Luv(self) -> None:
        """Test :func:`colour.models.cie_luv.uv_to_Luv` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        uv_to_Luv(cases, cases[..., 0:2])


class TestLuv_uv_to_xy:
    """
    Define :func:`colour.models.cie_luv.Luv_uv_to_xy` definition unit tests
    methods.
    """

    def test_Luv_uv_to_xy(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_luv.Luv_uv_to_xy` definition."""

        xp_assert_close(
            Luv_uv_to_xy(xp_as_array([0.37720213, 0.50120264], xp=xp)),
            [0.54369558, 0.32107944],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Luv_uv_to_xy(xp_as_array([0.14536327, 0.52992069], xp=xp)),
            [0.29777734, 0.48246445],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Luv_uv_to_xy(xp_as_array([0.16953603, 0.30039234], xp=xp)),
            [0.18582824, 0.14633764],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Luv_uv_to_xy(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.Luv_uv_to_xy` definition
        n-dimensional arrays support.
        """

        uv = xp_as_array([0.37720213, 0.50120264], xp=xp)
        xy = as_ndarray(Luv_uv_to_xy(uv))

        uv = xp.tile(xp_as_array(uv, xp=xp), (6, 1))
        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        xp_assert_close(Luv_uv_to_xy(uv), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

        uv = xp_reshape(xp_as_array(uv, xp=xp), (2, 3, 2), xp=xp)
        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(Luv_uv_to_xy(uv), xy, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_Luv_uv_to_xy(self) -> None:
        """
        Test :func:`colour.models.cie_luv.Luv_uv_to_xy` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        Luv_uv_to_xy(cases)


class TestXy_to_Luv_uv:
    """
    Define :func:`colour.models.cie_luv.xy_to_Luv_uv` definition unit tests
    methods.
    """

    def test_xy_to_Luv_uv(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_luv.xy_to_Luv_uv` definition."""

        xp_assert_close(
            xy_to_Luv_uv(xp_as_array([0.54369558, 0.32107944], xp=xp)),
            [0.37720213, 0.50120264],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_Luv_uv(xp_as_array([0.29777734, 0.48246445], xp=xp)),
            [0.14536327, 0.52992069],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            xy_to_Luv_uv(xp_as_array([0.18582824, 0.14633764], xp=xp)),
            [0.16953603, 0.30039234],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xy_to_Luv_uv(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.xy_to_Luv_uv` definition
        n-dimensional arrays support.
        """

        xy = xp_as_array([0.54369558, 0.32107944], xp=xp)
        uv = as_ndarray(xy_to_Luv_uv(xy))

        xy = xp.tile(xp_as_array(xy, xp=xp), (6, 1))
        uv = xp.tile(xp_as_array(uv, xp=xp), (6, 1))
        xp_assert_close(xy_to_Luv_uv(xy), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = xp_reshape(xp_as_array(xy, xp=xp), (2, 3, 2), xp=xp)
        uv = xp_reshape(xp_as_array(uv, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(xy_to_Luv_uv(xy), uv, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_xy_to_Luv_uv(self) -> None:
        """
        Test :func:`colour.models.cie_luv.xy_to_Luv_uv` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_Luv_uv(cases)


class TestXYZ_to_CIE1976UCS:
    """
    Define :func:`colour.models.cie_luv.XYZ_to_CIE1976UCS` definition unit tests
    methods.
    """

    def test_XYZ_to_CIE1976UCS(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_luv.XYZ_to_CIE1976UCS` definition."""

        xp_assert_close(
            XYZ_to_CIE1976UCS(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            [0.37720213, 0.50120264, 41.52787529],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_CIE1976UCS(xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            [0.14536327, 0.52992069, 55.11636304],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_CIE1976UCS(xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp)),
            [0.16953603, 0.30039234, 29.80565520],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_CIE1976UCS(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_as_array([0.44757, 0.40745], xp=xp),
            ),
            [0.37720213, 0.50120264, 41.52787529],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_CIE1976UCS(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_as_array([0.34570, 0.35850], xp=xp),
            ),
            [0.37720213, 0.50120264, 41.52787529],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_CIE1976UCS(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_as_array([0.34570, 0.35850, 1.00000], xp=xp),
            ),
            [0.37720213, 0.50120264, 41.52787529],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_CIE1976UCS(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.XYZ_to_CIE1976UCS` definition n-dimensional
        support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        Luv = as_ndarray(XYZ_to_CIE1976UCS(XYZ, illuminant))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        Luv = xp.tile(xp_as_array(Luv, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_CIE1976UCS(XYZ, illuminant),
            Luv,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        illuminant = xp.tile(xp_as_array(illuminant, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_CIE1976UCS(XYZ, illuminant),
            Luv,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        illuminant = xp_reshape(xp_as_array(illuminant, xp=xp), (2, 3, 2), xp=xp)
        Luv = xp_reshape(xp_as_array(Luv, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            XYZ_to_CIE1976UCS(XYZ, illuminant),
            Luv,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_CIE1976UCS(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.XYZ_to_CIE1976UCS` definition
        domain and range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        uvL = as_ndarray(XYZ_to_CIE1976UCS(XYZ, illuminant))

        d_r = (("reference", 1, 1), ("1", 1, np.array([1, 1, 0.01])), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_CIE1976UCS(XYZ * xp_as_array(factor_a, xp=xp), illuminant),
                    uvL * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_CIE1976UCS(self) -> None:
        """
        Test :func:`colour.models.cie_luv.XYZ_to_CIE1976UCS` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_CIE1976UCS(cases, cases[..., 0:2])


class TestCIE1976UCS_to_XYZ:
    """
    Define :func:`colour.models.cie_luv.CIE1976UCS_to_XYZ` definition unit tests
    methods.
    """

    def test_CIE1976UCS_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_luv.CIE1976UCS_to_XYZ` definition."""

        xp_assert_close(
            CIE1976UCS_to_XYZ(
                xp_as_array([0.37720213, 0.50120264, 41.52787529], xp=xp)
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CIE1976UCS_to_XYZ(
                xp_as_array([0.14536327, 0.52992069, 55.11636304], xp=xp)
            ),
            [0.14222010, 0.23042768, 0.10495772],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CIE1976UCS_to_XYZ(
                xp_as_array([0.16953603, 0.30039234, 29.80565520], xp=xp)
            ),
            [0.07818780, 0.06157201, 0.28099326],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CIE1976UCS_to_XYZ(
                xp_as_array([0.37720213, 0.50120264, 41.52787529], xp=xp),
                xp_as_array([0.44757, 0.40745], xp=xp),
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CIE1976UCS_to_XYZ(
                xp_as_array([0.37720213, 0.50120264, 41.52787529], xp=xp),
                xp_as_array([0.34570, 0.35850], xp=xp),
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            CIE1976UCS_to_XYZ(
                xp_as_array([0.37720213, 0.50120264, 41.52787529], xp=xp),
                xp_as_array([0.34570, 0.35850, 1.00000], xp=xp),
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_CIE1976UCS_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.CIE1976UCS_to_XYZ` definition n-dimensional
        support.
        """

        Luv = xp_as_array([0.37720213, 0.50120264, 41.52787529], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        XYZ = as_ndarray(CIE1976UCS_to_XYZ(Luv, illuminant))

        Luv = xp.tile(xp_as_array(Luv, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            CIE1976UCS_to_XYZ(Luv, illuminant),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        illuminant = xp.tile(xp_as_array(illuminant, xp=xp), (6, 1))
        xp_assert_close(
            CIE1976UCS_to_XYZ(Luv, illuminant),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Luv = xp_reshape(xp_as_array(Luv, xp=xp), (2, 3, 3), xp=xp)
        illuminant = xp_reshape(xp_as_array(illuminant, xp=xp), (2, 3, 2), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            CIE1976UCS_to_XYZ(Luv, illuminant),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_CIE1976UCS_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_luv.CIE1976UCS_to_XYZ` definition
        domain and range scale support.
        """

        uvL = xp_as_array([0.37720213, 0.50120264, 41.52787529], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        XYZ = as_ndarray(CIE1976UCS_to_XYZ(uvL, illuminant))

        d_r = (("reference", 1, 1), ("1", np.array([1, 1, 0.01]), 1), ("100", 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    CIE1976UCS_to_XYZ(uvL * xp_as_array(factor_a, xp=xp), illuminant),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_CIE1976UCS_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.cie_luv.CIE1976UCS_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        CIE1976UCS_to_XYZ(cases, cases[..., 0:2])
