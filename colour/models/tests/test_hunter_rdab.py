"""Define the unit tests for the :mod:`colour.models.hunter_rdab` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.colorimetry import TVS_ILLUMINANTS_HUNTERLAB
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import Hunter_Rdab_to_XYZ, XYZ_to_Hunter_Rdab
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
    "TestXYZ_to_Hunter_Rdab",
    "TestHunter_Rdab_to_XYZ",
]


class TestXYZ_to_Hunter_Rdab:
    """
    Define :func:`colour.models.hunter_rdab.XYZ_to_Hunter_Rdab` definition
    unit tests methods.
    """

    def test_XYZ_to_Hunter_Rdab(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.hunter_rdab.XYZ_to_Hunter_Rdab` definition."""

        xp_assert_close(
            XYZ_to_Hunter_Rdab(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100
            ),
            [12.19722500, 57.12537874, 17.46241341],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Hunter_Rdab(
                xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp) * 100
            ),
            [23.04276800, -32.40057474, 20.96542183],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Hunter_Rdab(
                xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp) * 100
            ),
            [6.15720100, 18.13400284, -67.14408607],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        h_i = TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]
        A = h_i["A"]
        xp_assert_close(
            XYZ_to_Hunter_Rdab(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
                A.XYZ_n,
                A.K_ab,
            ),
            [12.19722500, 42.53572838, -3.00653110],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        D65 = h_i["D65"]
        xp_assert_close(
            XYZ_to_Hunter_Rdab(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
                D65.XYZ_n,
                D65.K_ab,
            ),
            [12.19722500, 57.12537874, 17.46241341],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Hunter_Rdab(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
                D65.XYZ_n,
                K_ab=None,
            ),
            [12.19722500, 57.11906384, 17.45962317],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_Hunter_Rdab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hunter_rdab.XYZ_to_Hunter_Rdab` definition
        n-dimensional support.
        """

        h_i = TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]
        D65 = h_i["D65"]

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        R_d_ab = as_ndarray(XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        R_d_ab = xp.tile(xp_as_array(R_d_ab, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab),
            R_d_ab,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_n = xp.tile(xp_as_array(XYZ_n, xp=xp), (6, 1))
        K_ab = xp.tile(xp_as_array(K_ab, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab),
            R_d_ab,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_n = xp_reshape(xp_as_array(XYZ_n, xp=xp), (2, 3, 3), xp=xp)
        K_ab = xp_reshape(xp_as_array(K_ab, xp=xp), (2, 3, 2), xp=xp)
        R_d_ab = xp_reshape(xp_as_array(R_d_ab, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab),
            R_d_ab,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_Hunter_Rdab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hunter_lab.XYZ_to_Hunter_Rdab` definition
        domain and range scale support.
        """

        h_i = TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]
        D65 = h_i["D65"]

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        R_d_ab = as_ndarray(XYZ_to_Hunter_Rdab(XYZ, XYZ_n, K_ab))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_Hunter_Rdab(XYZ * factor, XYZ_n * factor, K_ab),
                    R_d_ab * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Hunter_Rdab(self) -> None:
        """
        Test :func:`colour.models.hunter_rdab.XYZ_to_Hunter_Rdab` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Hunter_Rdab(cases, cases, cases[..., 0:2])


class TestHunter_Rdab_to_XYZ:
    """
    Define :func:`colour.models.hunter_rdab.Hunter_Rdab_to_XYZ` definition
    unit tests methods.
    """

    def test_Hunter_Rdab_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.hunter_rdab.Hunter_Rdab_to_XYZ` definition."""

        xp_assert_close(
            Hunter_Rdab_to_XYZ(
                xp_as_array([12.19722500, 57.12537874, 17.46241341], xp=xp)
            ),
            xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Hunter_Rdab_to_XYZ(
                xp_as_array([23.04276800, -32.40057474, 20.96542183], xp=xp)
            ),
            xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Hunter_Rdab_to_XYZ(
                xp_as_array([6.15720100, 18.13400284, -67.14408607], xp=xp)
            ),
            xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        h_i = TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]
        A = h_i["A"]
        xp_assert_close(
            Hunter_Rdab_to_XYZ(
                xp_as_array([12.19722500, 42.53572838, -3.00653110], xp=xp),
                A.XYZ_n,
                A.K_ab,
            ),
            xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        D65 = h_i["D65"]
        xp_assert_close(
            Hunter_Rdab_to_XYZ(
                xp_as_array([12.19722500, 57.12537874, 17.46241341], xp=xp),
                D65.XYZ_n,
                D65.K_ab,
            ),
            xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Hunter_Rdab_to_XYZ(
                xp_as_array([12.19722500, 57.11906384, 17.45962317], xp=xp),
                D65.XYZ_n,
                K_ab=None,
            ),
            xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Hunter_Rdab_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hunter_rdab.Hunter_Rdab_to_XYZ` definition
        n-dimensional support.
        """

        h_i = TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]
        D65 = h_i["D65"]

        R_d_ab = xp_as_array([12.19722500, 57.12537874, 17.46241341], xp=xp)
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        XYZ = as_ndarray(Hunter_Rdab_to_XYZ(R_d_ab, XYZ_n, K_ab))

        R_d_ab = xp.tile(xp_as_array(R_d_ab, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            Hunter_Rdab_to_XYZ(R_d_ab, XYZ_n, K_ab),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        K_ab = xp.tile(xp_as_array(K_ab, xp=xp), (6, 1))
        XYZ_n = xp.tile(xp_as_array(XYZ_n, xp=xp), (6, 1))
        xp_assert_close(
            Hunter_Rdab_to_XYZ(R_d_ab, XYZ_n, K_ab),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        R_d_ab = xp_reshape(xp_as_array(R_d_ab, xp=xp), (2, 3, 3), xp=xp)
        XYZ_n = xp_reshape(xp_as_array(XYZ_n, xp=xp), (2, 3, 3), xp=xp)
        K_ab = xp_reshape(xp_as_array(K_ab, xp=xp), (2, 3, 2), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            Hunter_Rdab_to_XYZ(R_d_ab, XYZ_n, K_ab),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_Hunter_Rdab_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hunter_lab.Hunter_Rdab_to_XYZ` definition
        domain and range scale support.
        """

        h_i = TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]
        D65 = h_i["D65"]

        R_d_ab = xp_as_array([12.19722500, 57.12537874, 17.46241341], xp=xp)
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        XYZ = as_ndarray(Hunter_Rdab_to_XYZ(R_d_ab, XYZ_n, K_ab))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    Hunter_Rdab_to_XYZ(R_d_ab * factor, XYZ_n * factor, K_ab),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Hunter_Rdab_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.hunter_rdab.Hunter_Rdab_to_XYZ` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Hunter_Rdab_to_XYZ(cases, cases, cases[..., 0:2])
