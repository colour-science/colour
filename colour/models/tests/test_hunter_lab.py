"""Define the unit tests for the :mod:`colour.models.hunter_lab` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.colorimetry import TVS_ILLUMINANTS_HUNTERLAB
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    Hunter_Lab_to_XYZ,
    XYZ_to_Hunter_Lab,
    XYZ_to_K_ab_HunterLab1966,
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
    "TestXYZ_to_K_ab_HunterLab1966",
    "TestXYZ_to_Hunter_Lab",
    "TestHunter_Lab_to_XYZ",
]


class TestXYZ_to_K_ab_HunterLab1966:
    """
    Define :func:`colour.models.hunter_lab.XYZ_to_K_ab_HunterLab1966`
    definition unit tests methods.
    """

    def test_XYZ_to_K_ab_HunterLab1966(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hunter_lab.XYZ_to_K_ab_HunterLab1966`
        definition.
        """

        xp_assert_close(
            XYZ_to_K_ab_HunterLab1966(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100
            ),
            [80.32152090, 14.59816495],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_K_ab_HunterLab1966(
                xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp) * 100
            ),
            [66.65154834, 20.86664881],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_K_ab_HunterLab1966(
                xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp) * 100
            ),
            [49.41960269, 34.14235426],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_K_ab_HunterLab1966(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hunter_lab.XYZ_to_K_ab_HunterLab1966`
        definition n-dimensional support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100
        K_ab = as_ndarray(XYZ_to_K_ab_HunterLab1966(XYZ))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        K_ab = xp.tile(xp_as_array(K_ab, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_K_ab_HunterLab1966(XYZ),
            K_ab,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        K_ab = xp_reshape(xp_as_array(K_ab, xp=xp), (2, 3, 2), xp=xp)
        xp_assert_close(
            XYZ_to_K_ab_HunterLab1966(XYZ),
            K_ab,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_XYZ_to_K_ab_HunterLab1966(self) -> None:
        """
        Test :func:`colour.models.hunter_lab.XYZ_to_K_ab_HunterLab1966`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_K_ab_HunterLab1966(cases)


class TestXYZ_to_Hunter_Lab:
    """
    Define :func:`colour.models.hunter_lab.XYZ_to_Hunter_Lab` definition unit
    tests methods.
    """

    def test_XYZ_to_Hunter_Lab(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.hunter_lab.XYZ_to_Hunter_Lab` definition."""

        xp_assert_close(
            XYZ_to_Hunter_Lab(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100
            ),
            [34.92452577, 47.06189858, 14.38615107],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Hunter_Lab(
                xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp) * 100
            ),
            [48.00288325, -28.98551622, 18.75564181],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Hunter_Lab(
                xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp) * 100
            ),
            [24.81370791, 14.38300039, -53.25539126],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        h_i = TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]
        A = h_i["A"]
        xp_assert_close(
            XYZ_to_Hunter_Lab(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
                A.XYZ_n,
                A.K_ab,
            ),
            [34.92452577, 35.04243086, -2.47688619],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        D65 = h_i["D65"]
        xp_assert_close(
            XYZ_to_Hunter_Lab(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
                D65.XYZ_n,
                D65.K_ab,
            ),
            [34.92452577, 47.06189858, 14.38615107],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Hunter_Lab(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
                D65.XYZ_n,
                K_ab=None,
            ),
            [34.92452577, 47.05669614, 14.38385238],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_Hunter_Lab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hunter_lab.XYZ_to_Hunter_Lab` definition
        n-dimensional support.
        """

        h_i = TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]
        D65 = h_i["D65"]

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        Lab = as_ndarray(XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        Lab = xp.tile(xp_as_array(Lab, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab),
            Lab,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_n = xp.tile(xp_as_array(XYZ_n, xp=xp), (6, 1))
        K_ab = xp.tile(xp_as_array(K_ab, xp=xp), (6, 1))
        xp_assert_close(
            XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab),
            Lab,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_n = xp_reshape(xp_as_array(XYZ_n, xp=xp), (2, 3, 3), xp=xp)
        K_ab = xp_reshape(xp_as_array(K_ab, xp=xp), (2, 3, 2), xp=xp)
        Lab = xp_reshape(xp_as_array(Lab, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab),
            Lab,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_XYZ_to_Hunter_Lab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hunter_lab.XYZ_to_Hunter_Lab` definition
        domain and range scale support.
        """

        h_i = TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]
        D65 = h_i["D65"]

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        Lab = as_ndarray(XYZ_to_Hunter_Lab(XYZ, XYZ_n, K_ab))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_Hunter_Lab(XYZ * factor, XYZ_n * factor, K_ab),
                    Lab * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Hunter_Lab(self) -> None:
        """
        Test :func:`colour.models.hunter_lab.XYZ_to_Hunter_Lab` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Hunter_Lab(cases, cases, cases[..., 0:2])


class TestHunter_Lab_to_XYZ:
    """
    Define :func:`colour.models.hunter_lab.Hunter_Lab_to_XYZ` definition unit
    tests methods.
    """

    def test_Hunter_Lab_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.hunter_lab.Hunter_Lab_to_XYZ` definition."""

        xp_assert_close(
            Hunter_Lab_to_XYZ(
                xp_as_array([34.92452577, 47.06189858, 14.38615107], xp=xp)
            ),
            [20.65400800, 12.19722500, 5.13695200],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Hunter_Lab_to_XYZ(
                xp_as_array([48.00288325, -28.98551622, 18.75564181], xp=xp)
            ),
            [14.22201000, 23.04276800, 10.49577200],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Hunter_Lab_to_XYZ(
                xp_as_array([24.81370791, 14.38300039, -53.25539126], xp=xp)
            ),
            [7.81878000, 6.15720100, 28.09932601],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        h_i = TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]
        A = h_i["A"]
        xp_assert_close(
            Hunter_Lab_to_XYZ(
                xp_as_array([34.92452577, 35.04243086, -2.47688619], xp=xp),
                A.XYZ_n,
                A.K_ab,
            ),
            [20.65400800, 12.19722500, 5.13695200],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        D65 = h_i["D65"]
        xp_assert_close(
            Hunter_Lab_to_XYZ(
                xp_as_array([34.92452577, 47.06189858, 14.38615107], xp=xp),
                D65.XYZ_n,
                D65.K_ab,
            ),
            [20.65400800, 12.19722500, 5.13695200],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Hunter_Lab_to_XYZ(
                xp_as_array([34.92452577, 47.05669614, 14.38385238], xp=xp),
                D65.XYZ_n,
                K_ab=None,
            ),
            [20.65400800, 12.19722500, 5.13695200],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Hunter_Lab_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hunter_lab.Hunter_Lab_to_XYZ` definition
        n-dimensional support.
        """

        h_i = TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]
        D65 = h_i["D65"]

        Lab = xp_as_array([34.92452577, 47.06189858, 14.38615107], xp=xp)
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        XYZ = as_ndarray(Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab))

        Lab = xp.tile(xp_as_array(Lab, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        K_ab = xp.tile(xp_as_array(K_ab, xp=xp), (6, 1))
        XYZ_n = xp.tile(xp_as_array(XYZ_n, xp=xp), (6, 1))
        xp_assert_close(
            Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        Lab = xp_reshape(xp_as_array(Lab, xp=xp), (2, 3, 3), xp=xp)
        XYZ_n = xp_reshape(xp_as_array(XYZ_n, xp=xp), (2, 3, 3), xp=xp)
        K_ab = xp_reshape(xp_as_array(K_ab, xp=xp), (2, 3, 2), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_Hunter_Lab_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.hunter_lab.Hunter_Lab_to_XYZ` definition
        domain and range scale support.
        """

        h_i = TVS_ILLUMINANTS_HUNTERLAB["CIE 1931 2 Degree Standard Observer"]
        D65 = h_i["D65"]

        Lab = xp_as_array([34.92452577, 47.06189858, 14.38615107], xp=xp)
        XYZ_n = D65.XYZ_n
        K_ab = D65.K_ab
        XYZ = as_ndarray(Hunter_Lab_to_XYZ(Lab, XYZ_n, K_ab))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    Hunter_Lab_to_XYZ(Lab * factor, XYZ_n * factor, K_ab),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Hunter_Lab_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.hunter_lab.Hunter_Lab_to_XYZ` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Hunter_Lab_to_XYZ(cases, cases, cases[..., 0:2])
