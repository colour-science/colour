"""
Define the unit tests for the :mod:`colour.adaptation.fairchild2020` module.
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.adaptation import (
    chromatic_adaptation_vK20,
    matrix_chromatic_adaptation_vk20,
)
from colour.adaptation.fairchild2020 import CONDITIONS_DEGREE_OF_ADAPTATION_VK20
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
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestMatrixChromaticAdaptationVonKries",
    "TestChromaticAdaptationVonKries",
]


class TestMatrixChromaticAdaptationVonKries:
    """
    Define :func:`colour.adaptation.fairchild2020.\
matrix_chromatic_adaptation_vk20` definition unit tests methods.
    """

    def test_matrix_chromatic_adaptation_vk20(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.fairchild2020.\
matrix_chromatic_adaptation_vk20` definition.
        """

        xp_assert_close(
            as_ndarray(
                matrix_chromatic_adaptation_vk20(
                    xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                    xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                )
            ),
            [
                [1.02791390, 0.02913712, -0.02279407],
                [0.02070284, 0.99005317, -0.00921435],
                [-0.00063759, -0.00115773, 0.91296320],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            as_ndarray(
                matrix_chromatic_adaptation_vk20(
                    xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                    xp_as_array([1.09846607, 1.00000000, 0.35582280], xp=xp),
                )
            ),
            [
                [0.94760338, -0.05816939, 0.06647414],
                [-0.04151006, 1.02361127, 0.02667016],
                [0.00163074, 0.00391656, 1.29341031],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            as_ndarray(
                matrix_chromatic_adaptation_vk20(
                    xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                    xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                    transform="XYZ Scaling",
                )
            ),
            [
                [1.03217229, 0.00000000, 0.00000000],
                [0.00000000, 1.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.91134516],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            as_ndarray(
                matrix_chromatic_adaptation_vk20(
                    xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                    xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                    transform="Bradford",
                )
            ),
            [
                [1.03672305, 0.01955802, -0.02193210],
                [0.02763218, 0.98222961, -0.00824197],
                [-0.00295083, 0.00406903, 0.91024305],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            as_ndarray(
                matrix_chromatic_adaptation_vk20(
                    xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                    xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                    transform="Von Kries",
                    coefficients=CONDITIONS_DEGREE_OF_ADAPTATION_VK20[
                        "Simple Von Kries"
                    ],
                )
            ),
            [
                [0.98446157, -0.05474538, 0.06773143],
                [-0.00601339, 1.00479590, 0.00121235],
                [0.00000000, 0.00000000, 1.31990977],
            ],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_matrix_chromatic_adaptation_vk20(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.fairchild2020.\
matrix_chromatic_adaptation_vk20` definition n-dimensional arrays support.
        """

        XYZ_p = xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp)
        XYZ_n = xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp)
        M = as_ndarray(matrix_chromatic_adaptation_vk20(XYZ_p, XYZ_n))

        XYZ_p = xp.tile(xp_as_array(XYZ_p, xp=xp), (6, 1))
        XYZ_n = xp.tile(xp_as_array(XYZ_n, xp=xp), (6, 1))
        M = xp_reshape(xp.tile(xp_as_array(M, xp=xp), (6, 1)), (6, 3, 3), xp=xp)
        xp_assert_close(
            as_ndarray(matrix_chromatic_adaptation_vk20(XYZ_p, XYZ_n)),
            as_ndarray(M),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ_p = xp_reshape(xp_as_array(XYZ_p, xp=xp), (2, 3, 3), xp=xp)
        XYZ_n = xp_reshape(xp_as_array(XYZ_n, xp=xp), (2, 3, 3), xp=xp)
        M = xp_reshape(xp_as_array(M, xp=xp), (2, 3, 3, 3), xp=xp)
        xp_assert_close(
            as_ndarray(matrix_chromatic_adaptation_vk20(XYZ_p, XYZ_n)),
            as_ndarray(M),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_matrix_chromatic_adaptation_vk20(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.adaptation.fairchild2020.\
matrix_chromatic_adaptation_vk20` definition domain and range scale
        support.
        """

        XYZ_p = xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp)
        XYZ_n = xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp)
        XYZ_r = xp_as_array([0.97941176, 1.00000000, 1.73235294], xp=xp)
        M = as_ndarray(matrix_chromatic_adaptation_vk20(XYZ_p, XYZ_n))

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    as_ndarray(
                        matrix_chromatic_adaptation_vk20(
                            XYZ_p * factor, XYZ_n * factor, XYZ_r * factor
                        )
                    ),
                    M,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_matrix_chromatic_adaptation_vk20(self) -> None:
        """
        Test :func:`colour.adaptation.fairchild2020.\
matrix_chromatic_adaptation_vk20` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        matrix_chromatic_adaptation_vk20(cases, cases)


class TestChromaticAdaptationVonKries:
    """
    Define :func:`colour.adaptation.fairchild2020.chromatic_adaptation_vK20`
    definition unit tests methods.
    """

    def test_chromatic_adaptation_vK20(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.fairchild2020.chromatic_adaptation_vK20`
        definition.
        """

        xp_assert_close(
            as_ndarray(
                chromatic_adaptation_vK20(
                    xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                    xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                    xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                )
            ),
            [0.21468842, 0.12456164, 0.04662558],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            as_ndarray(
                chromatic_adaptation_vK20(
                    xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp),
                    xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                    xp_as_array([1.09846607, 1.00000000, 0.35582280], xp=xp),
                )
            ),
            [0.12834138, 0.23276404, 0.13688781],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            as_ndarray(
                chromatic_adaptation_vK20(
                    xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp),
                    xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                    xp_as_array([0.99144661, 1.00000000, 0.67315942], xp=xp),
                )
            ),
            [0.07908008, 0.06167829, 0.28354175],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            as_ndarray(
                chromatic_adaptation_vK20(
                    xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                    xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                    xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                    transform="XYZ Scaling",
                )
            ),
            [0.21318495, 0.12197225, 0.04681536],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            as_ndarray(
                chromatic_adaptation_vK20(
                    xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                    xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                    xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                    transform="Bradford",
                )
            ),
            [0.21538376, 0.12508852, 0.04664559],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            as_ndarray(
                chromatic_adaptation_vK20(
                    xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                    xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp),
                    xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp),
                    transform="Von Kries",
                    coefficients=CONDITIONS_DEGREE_OF_ADAPTATION_VK20[
                        "Simple Von Kries"
                    ],
                )
            ),
            [0.20013269, 0.12137749, 0.06780313],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_chromatic_adaptation_vK20(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.fairchild2020.chromatic_adaptation_vK20`
        definition n-dimensional arrays support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        XYZ_p = xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp)
        XYZ_n = xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp)
        XYZ_a = as_ndarray(chromatic_adaptation_vK20(XYZ, XYZ_p, XYZ_n))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        XYZ_p = xp.tile(xp_as_array(XYZ_p, xp=xp), (6, 1))
        XYZ_n = xp.tile(xp_as_array(XYZ_n, xp=xp), (6, 1))
        XYZ_a = xp.tile(xp_as_array(XYZ_a, xp=xp), (6, 1))
        xp_assert_close(
            as_ndarray(chromatic_adaptation_vK20(XYZ, XYZ_p, XYZ_n)),
            as_ndarray(XYZ_a),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        XYZ_p = xp_reshape(xp_as_array(XYZ_p, xp=xp), (2, 3, 3), xp=xp)
        XYZ_n = xp_reshape(xp_as_array(XYZ_n, xp=xp), (2, 3, 3), xp=xp)
        XYZ_a = xp_reshape(xp_as_array(XYZ_a, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            as_ndarray(chromatic_adaptation_vK20(XYZ, XYZ_p, XYZ_n)),
            as_ndarray(XYZ_a),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_domain_range_scale_chromatic_adaptation_vK20(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.adaptation.fairchild2020.chromatic_adaptation_vK20`
        definition domain and range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        XYZ_p = xp_as_array([0.95045593, 1.00000000, 1.08905775], xp=xp)
        XYZ_n = xp_as_array([0.96429568, 1.00000000, 0.82510460], xp=xp)
        XYZ_r = xp_as_array([0.97941176, 1.00000000, 1.73235294], xp=xp)
        XYZ_a = as_ndarray(chromatic_adaptation_vK20(XYZ, XYZ_p, XYZ_n))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    as_ndarray(
                        chromatic_adaptation_vK20(
                            XYZ * factor,
                            XYZ_p * factor,
                            XYZ_n * factor,
                            XYZ_r * factor,
                        )
                    ),
                    XYZ_a * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_vK20(self) -> None:
        """
        Test :func:`colour.adaptation.fairchild2020.chromatic_adaptation_vK20`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        chromatic_adaptation_vK20(cases, cases, cases)
