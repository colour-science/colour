"""Define the unit tests for the :mod:`colour.adaptation.vonkries` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.adaptation import (
    chromatic_adaptation_VonKries,
    matrix_chromatic_adaptation_VonKries,
)
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
    "TestMatrixChromaticAdaptationVonKries",
    "TestChromaticAdaptationVonKries",
]


class TestMatrixChromaticAdaptationVonKries:
    """
    Define :func:`colour.adaptation.vonkries.\
matrix_chromatic_adaptation_VonKries` definition unit tests methods.
    """

    def test_matrix_chromatic_adaptation_VonKries(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.vonkries.\
matrix_chromatic_adaptation_VonKries` definition.
        """

        xp_assert_close(
            matrix_chromatic_adaptation_VonKries(
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
            ),
            [
                [1.04257389, 0.03089108, -0.05281257],
                [0.02219345, 1.00185663, -0.02107375],
                [-0.00116488, -0.00342053, 0.76178907],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_chromatic_adaptation_VonKries(
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                xp_as_array([1.09846607, 1.00000000, 0.35582280], xp=xp),
            ),
            [
                [1.17159793, 0.16088780, -0.16158366],
                [0.11462057, 0.96182051, -0.06497572],
                [-0.00413024, -0.00912739, 0.33871096],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_chromatic_adaptation_VonKries(
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                xp_as_array([0.99144661, 1.00000000, 0.67315942], xp=xp),
            ),
            np.linalg.inv(
                as_ndarray(
                    matrix_chromatic_adaptation_VonKries(
                        xp_as_array([0.99144661, 1.00000000, 0.67315942], xp=xp),
                        xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                    )
                )
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_chromatic_adaptation_VonKries(
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                transform="XYZ Scaling",
            ),
            [
                [1.01456117, 0.00000000, 0.00000000],
                [0.00000000, 1.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.75763163],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_chromatic_adaptation_VonKries(
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                transform="Bradford",
            ),
            [
                [1.04792979, 0.02294687, -0.05019227],
                [0.02962781, 0.99043443, -0.01707380],
                [-0.00924304, 0.01505519, 0.75187428],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            matrix_chromatic_adaptation_VonKries(
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                transform="Von Kries",
            ),
            [
                [1.01611856, 0.05535971, -0.05219186],
                [0.00608087, 0.99555604, -0.00122642],
                [0.00000000, 0.00000000, 0.75763163],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_matrix_chromatic_adaptation_VonKries(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.vonkries.\
matrix_chromatic_adaptation_VonKries` definition n-dimensional arrays support.
        """

        XYZ_w = xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp)
        XYZ_wr = xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp)
        M = as_ndarray(matrix_chromatic_adaptation_VonKries(XYZ_w, XYZ_wr))

        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        XYZ_wr = xp.tile(xp_as_array(XYZ_wr, xp=xp), (6, 1))
        M = xp_reshape(xp.tile(xp_as_array(M, xp=xp), (6, 1)), (6, 3, 3), xp=xp)
        xp_assert_close(
            matrix_chromatic_adaptation_VonKries(XYZ_w, XYZ_wr),
            M,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        XYZ_wr = xp_reshape(xp_as_array(XYZ_wr, xp=xp), (2, 3, 3), xp=xp)
        M = xp_reshape(xp_as_array(M, xp=xp), (2, 3, 3, 3), xp=xp)
        xp_assert_close(
            matrix_chromatic_adaptation_VonKries(XYZ_w, XYZ_wr),
            M,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_matrix_chromatic_adaptation_VonKries(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.vonkries.\
matrix_chromatic_adaptation_VonKries` definition domain and range scale
        support.
        """

        XYZ_w = xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp)
        XYZ_wr = xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp)
        M = as_ndarray(matrix_chromatic_adaptation_VonKries(XYZ_w, XYZ_wr))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    matrix_chromatic_adaptation_VonKries(
                        XYZ_w * factor, XYZ_wr * factor
                    ),
                    M,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_matrix_chromatic_adaptation_VonKries(self) -> None:
        """
        Test :func:`colour.adaptation.vonkries.\
matrix_chromatic_adaptation_VonKries` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        matrix_chromatic_adaptation_VonKries(cases, cases)


class TestChromaticAdaptationVonKries:
    """
    Define :func:`colour.adaptation.vonkries.chromatic_adaptation_VonKries`
    definition unit tests methods.
    """

    def test_chromatic_adaptation_VonKries(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.vonkries.chromatic_adaptation_VonKries`
        definition.
        """

        xp_assert_close(
            chromatic_adaptation_VonKries(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
            ),
            [0.21638819, 0.12570000, 0.03847494],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_VonKries(
                xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp),
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                xp_as_array([1.09846607, 1.00000000, 0.35582280], xp=xp),
            ),
            [0.18673833, 0.23111171, 0.03285972],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_VonKries(
                xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp),
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                xp_as_array([0.99144661, 1.00000000, 0.67315942], xp=xp),
            ),
            [0.06385467, 0.05509729, 0.17506386],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_VonKries(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                transform="XYZ Scaling",
            ),
            [0.20954755, 0.12197225, 0.03891917],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_VonKries(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                transform="Bradford",
            ),
            [0.21666003, 0.12604777, 0.03855068],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            chromatic_adaptation_VonKries(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                transform="Von Kries",
            ),
            [0.21394049, 0.12262315, 0.03891917],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_chromatic_adaptation_VonKries(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.vonkries.chromatic_adaptation_VonKries`
        definition n-dimensional arrays support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        XYZ_w = xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp)
        XYZ_wr = xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp)
        XYZ_a = as_ndarray(chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        XYZ_w = xp.tile(xp_as_array(XYZ_w, xp=xp), (6, 1))
        XYZ_wr = xp.tile(xp_as_array(XYZ_wr, xp=xp), (6, 1))
        XYZ_a = xp.tile(xp_as_array(XYZ_a, xp=xp), (6, 1))
        xp_assert_close(
            chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr),
            XYZ_a,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_w = xp_reshape(xp_as_array(XYZ_w, xp=xp), (2, 3, 3), xp=xp)
        XYZ_wr = xp_reshape(xp_as_array(XYZ_wr, xp=xp), (2, 3, 3), xp=xp)
        XYZ_a = xp_reshape(xp_as_array(XYZ_a, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr),
            XYZ_a,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_chromatic_adaptation_VonKries(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.vonkries.chromatic_adaptation_VonKries`
        definition domain and range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        XYZ_w = xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp)
        XYZ_wr = xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp)
        XYZ_a = as_ndarray(chromatic_adaptation_VonKries(XYZ, XYZ_w, XYZ_wr))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    chromatic_adaptation_VonKries(
                        XYZ * factor, XYZ_w * factor, XYZ_wr * factor
                    ),
                    XYZ_a * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_VonKries(self) -> None:
        """
        Test :func:`colour.adaptation.vonkries.chromatic_adaptation_VonKries`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        chromatic_adaptation_VonKries(cases, cases, cases)
