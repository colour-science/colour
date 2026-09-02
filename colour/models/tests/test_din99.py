"""Define the unit tests for the :mod:`colour.models.din99` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import DIN99_to_Lab, DIN99_to_XYZ, Lab_to_DIN99, XYZ_to_DIN99
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
    "TestLab_to_DIN99",
    "TestDIN99_to_Lab",
    "TestXYZ_to_DIN99",
    "TestDIN99_to_XYZ",
]


class TestLab_to_DIN99:
    """
    Define :func:`colour.models.din99.Lab_to_DIN99` definition unit tests
    methods.
    """

    def test_Lab_to_DIN99(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.din99.Lab_to_DIN99` definition."""

        xp_assert_close(
            Lab_to_DIN99(xp_as_array([41.52787529, 52.63858304, 26.92317922], xp=xp)),
            [53.22821988, 28.41634656, 3.89839552],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Lab_to_DIN99(xp_as_array([55.11636304, -41.08791787, 30.91825778], xp=xp)),
            [66.08943912, -17.35290106, 16.09690691],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Lab_to_DIN99(xp_as_array([29.80565520, 20.01830466, -48.34913874], xp=xp)),
            [40.71533366, 3.48714163, -21.45321411],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Lab_to_DIN99(
                xp_as_array([41.52787529, 52.63858304, 26.92317922], xp=xp),
                method="DIN99b",
            ),
            [45.58303137, 34.71824493, 17.61622149],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Lab_to_DIN99(
                xp_as_array([41.52787529, 52.63858304, 26.92317922], xp=xp),
                method="DIN99c",
            ),
            [45.40284208, 32.75074741, 15.74603532],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            Lab_to_DIN99(
                xp_as_array([41.52787529, 52.63858304, 26.92317922], xp=xp),
                method="DIN99d",
            ),
            [45.31204747, 31.42106716, 14.17004652],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Lab_to_DIN99(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.din99.Lab_to_DIN99` definition n-dimensional
        support.
        """

        Lab = xp_as_array([41.52787529, 52.63858304, 26.92317922], xp=xp)
        Lab_99 = as_ndarray(Lab_to_DIN99(Lab))

        Lab = xp.tile(xp_as_array(Lab, xp=xp), (6, 1))
        Lab_99 = xp.tile(xp_as_array(Lab_99, xp=xp), (6, 1))
        xp_assert_close(Lab_to_DIN99(Lab), Lab_99, atol=TOLERANCE_ABSOLUTE_TESTS)

        Lab = xp_reshape(xp_as_array(Lab, xp=xp), (2, 3, 3), xp=xp)
        Lab_99 = xp_reshape(xp_as_array(Lab_99, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(Lab_to_DIN99(Lab), Lab_99, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_Lab_to_DIN99(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.din99.Lab_to_DIN99` definition domain and
        range scale support.
        """

        Lab = xp_as_array([41.52787529, 52.63858304, 26.92317922], xp=xp)
        Lab_99 = as_ndarray(Lab_to_DIN99(Lab))
        Lab_99_b = as_ndarray(Lab_to_DIN99(Lab, method="DIN99b"))
        Lab_99_c = as_ndarray(Lab_to_DIN99(Lab, method="DIN99c"))
        Lab_99_d = as_ndarray(Lab_to_DIN99(Lab, method="DIN99d"))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    Lab_to_DIN99(Lab * factor),
                    Lab_99 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                xp_assert_close(
                    Lab_to_DIN99((Lab * factor), method="DIN99b"),
                    Lab_99_b * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                xp_assert_close(
                    Lab_to_DIN99((Lab * factor), method="DIN99c"),
                    Lab_99_c * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                xp_assert_close(
                    Lab_to_DIN99((Lab * factor), method="DIN99d"),
                    Lab_99_d * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Lab_to_DIN99(self) -> None:
        """Test :func:`colour.models.din99.Lab_to_DIN99` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Lab_to_DIN99(cases)
        Lab_to_DIN99(cases, method="DIN99b")
        Lab_to_DIN99(cases, method="DIN99c")
        Lab_to_DIN99(cases, method="DIN99d")


class TestDIN99_to_Lab:
    """
    Define :func:`colour.models.din99.DIN99_to_Lab` definition unit tests
    methods.
    """

    def test_DIN99_to_Lab(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.din99.DIN99_to_Lab` definition."""

        xp_assert_close(
            DIN99_to_Lab(xp_as_array([53.22821988, 28.41634656, 3.89839552], xp=xp)),
            [41.52787529, 52.63858304, 26.92317922],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            DIN99_to_Lab(xp_as_array([66.08943912, -17.35290106, 16.09690691], xp=xp)),
            [55.11636304, -41.08791787, 30.91825778],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            DIN99_to_Lab(xp_as_array([40.71533366, 3.48714163, -21.45321411], xp=xp)),
            [29.80565520, 20.01830466, -48.34913874],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            DIN99_to_Lab(
                xp_as_array([45.58303137, 34.71824493, 17.61622149], xp=xp),
                method="DIN99b",
            ),
            [41.52787529, 52.63858304, 26.92317922],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            DIN99_to_Lab(
                xp_as_array([45.40284208, 32.75074741, 15.74603532], xp=xp),
                method="DIN99c",
            ),
            [41.52787529, 52.63858304, 26.92317922],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            DIN99_to_Lab(
                xp_as_array([45.31204747, 31.42106716, 14.17004652], xp=xp),
                method="DIN99d",
            ),
            [41.52787529, 52.63858304, 26.92317922],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_DIN99_to_Lab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.din99.DIN99_to_Lab` definition n-dimensional
        support.
        """

        Lab_99 = xp_as_array([53.22821988, 28.41634656, 3.89839552], xp=xp)
        Lab = as_ndarray(DIN99_to_Lab(Lab_99))

        Lab_99 = xp.tile(xp_as_array(Lab_99, xp=xp), (6, 1))
        Lab = xp.tile(xp_as_array(Lab, xp=xp), (6, 1))
        xp_assert_close(DIN99_to_Lab(Lab_99), Lab, atol=TOLERANCE_ABSOLUTE_TESTS)

        Lab_99 = xp_reshape(xp_as_array(Lab_99, xp=xp), (2, 3, 3), xp=xp)
        Lab = xp_reshape(xp_as_array(Lab, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(DIN99_to_Lab(Lab_99), Lab, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_DIN99_to_Lab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.din99.DIN99_to_Lab` definition domain and
        range scale support.
        """

        Lab_99 = xp_as_array([53.22821988, 28.41634656, 3.89839552], xp=xp)
        Lab = as_ndarray(DIN99_to_Lab(Lab_99))
        Lab_b = as_ndarray(DIN99_to_Lab(Lab_99, method="DIN99b"))
        Lab_c = as_ndarray(DIN99_to_Lab(Lab_99, method="DIN99c"))
        Lab_d = as_ndarray(DIN99_to_Lab(Lab_99, method="DIN99d"))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    DIN99_to_Lab(Lab_99 * factor),
                    Lab * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                xp_assert_close(
                    DIN99_to_Lab((Lab_99 * factor), method="DIN99b"),
                    Lab_b * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                xp_assert_close(
                    DIN99_to_Lab((Lab_99 * factor), method="DIN99c"),
                    Lab_c * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                xp_assert_close(
                    DIN99_to_Lab((Lab_99 * factor), method="DIN99d"),
                    Lab_d * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_DIN99_to_Lab(self) -> None:
        """Test :func:`colour.models.din99.DIN99_to_Lab` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        DIN99_to_Lab(cases)
        DIN99_to_Lab(cases, method="DIN99b")
        DIN99_to_Lab(cases, method="DIN99c")
        DIN99_to_Lab(cases, method="DIN99d")


class TestXYZ_to_DIN99:
    """
    Define :func:`colour.models.din99.XYZ_to_DIN99` definition unit tests
    methods.
    """

    def test_XYZ_to_DIN99(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.din99.XYZ_to_DIN99` definition."""

        xp_assert_close(
            XYZ_to_DIN99(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            [53.22821988, 28.41634656, 3.89839552],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_DIN99(xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            [66.08943912, -17.35290106, 16.09690691],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_DIN99(xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp)),
            [40.71533366, 3.48714163, -21.45321411],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_DIN99(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp),
                method="DIN99b",
            ),
            [45.58303137, 34.71824493, 17.61622149],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_DIN99(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.din99.XYZ_to_DIN99` definition n-dimensional
        support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        Lab_99 = as_ndarray(XYZ_to_DIN99(XYZ))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        Lab_99 = xp.tile(xp_as_array(Lab_99, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_DIN99(XYZ), Lab_99, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        Lab_99 = xp_reshape(xp_as_array(Lab_99, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_DIN99(XYZ), Lab_99, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_DIN99(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.din99.XYZ_to_DIN99` definition domain and
        range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        Lab_99 = as_ndarray(XYZ_to_DIN99(XYZ))

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_DIN99(XYZ * xp_as_array(factor_a, xp=xp)),
                    Lab_99 * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_DIN99(self) -> None:
        """Test :func:`colour.models.din99.XYZ_to_DIN99` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_DIN99(cases)


class TestDIN99_to_XYZ:
    """
    Define :func:`colour.models.din99.DIN99_to_XYZ` definition unit tests
    methods.
    """

    def test_DIN99_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.din99.DIN99_to_XYZ` definition."""

        xp_assert_close(
            DIN99_to_XYZ(xp_as_array([53.22821988, 28.41634656, 3.89839552], xp=xp)),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            DIN99_to_XYZ(xp_as_array([66.08943912, -17.35290106, 16.09690691], xp=xp)),
            [0.14222010, 0.23042768, 0.10495772],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            DIN99_to_XYZ(xp_as_array([40.71533366, 3.48714163, -21.45321411], xp=xp)),
            [0.07818780, 0.06157201, 0.28099326],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            DIN99_to_XYZ(
                xp_as_array([45.58303137, 34.71824493, 17.61622149], xp=xp),
                method="DIN99b",
            ),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_DIN99_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.din99.DIN99_to_XYZ` definition n-dimensional
        support.
        """

        Lab_99 = xp_as_array([53.22821988, 28.41634656, 3.89839552], xp=xp)
        XYZ = as_ndarray(DIN99_to_XYZ(Lab_99))

        Lab_99 = xp.tile(xp_as_array(Lab_99, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(DIN99_to_XYZ(Lab_99), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        Lab_99 = xp_reshape(xp_as_array(Lab_99, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(DIN99_to_XYZ(Lab_99), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_DIN99_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.din99.DIN99_to_XYZ` definition domain and
        range scale support.
        """

        Lab_99 = xp_as_array([53.22821988, 28.41634656, 3.89839552], xp=xp)
        XYZ = as_ndarray(DIN99_to_XYZ(Lab_99))

        d_r = (("reference", 1, 1), ("1", 0.01, 1), ("100", 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    DIN99_to_XYZ(Lab_99 * xp_as_array(factor_a, xp=xp)),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_DIN99_to_XYZ(self) -> None:
        """Test :func:`colour.models.din99.DIN99_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        DIN99_to_XYZ(cases)
