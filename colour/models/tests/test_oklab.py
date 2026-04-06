"""Define the unit tests for the :mod:`colour.models.oklab` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import Oklab_to_XYZ, XYZ_to_Oklab
from colour.utilities import (
    as_ndarray,
    domain_range_scale,
    ignore_numpy_errors,
    xp_as_array,
    xp_assert_close,
    xp_assert_equal,
    xp_reshape,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_Oklab",
    "TestOklab_to_XYZ",
]


class TestXYZ_to_Oklab:
    """
    Define :func:`colour.models.oklab.TestXYZ_to_Oklab` definition unit
    tests methods.
    """

    def test_XYZ_to_Oklab(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.oklab.XYZ_to_Oklab` definition."""

        xp_assert_close(
            XYZ_to_Oklab(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)),
            [0.51634019, 0.15469500, 0.06289579],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Oklab(xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp)),
            [0.59910746, -0.11139207, 0.07508465],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_Oklab(xp_as_array([0.96907232, 1.00000000, 1.12179215], xp=xp)),
            [1.00121561, 0.00899591, -0.00535107],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_Oklab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.oklab.XYZ_to_Oklab` definition
        n-dimensional support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        Oklab = as_ndarray(XYZ_to_Oklab(XYZ))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        Oklab = xp.tile(xp_as_array(Oklab, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_Oklab(XYZ), Oklab, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        Oklab = xp_reshape(xp_as_array(Oklab, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_Oklab(XYZ), Oklab, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_Oklab(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.oklab.XYZ_to_Oklab` definition domain and
        range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp)
        Oklab = as_ndarray(XYZ_to_Oklab(XYZ))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_Oklab(XYZ * factor),
                    Oklab * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Oklab(self) -> None:
        """
        Test :func:`colour.models.oklab.XYZ_to_Oklab` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Oklab(cases)


class TestOklab_to_XYZ:
    """
    Define :func:`colour.models.oklab.Oklab_to_XYZ` definition unit tests
    methods.
    """

    def test_Oklab_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.oklab.Oklab_to_XYZ` definition."""

        xp_assert_close(
            Oklab_to_XYZ(xp_as_array([0.51634019, 0.15469500, 0.06289579], xp=xp)),
            [0.20654008, 0.12197225, 0.05136952],
            atol=TOLERANCE_ABSOLUTE_TESTS * 10,
        )

        xp_assert_close(
            Oklab_to_XYZ(xp_as_array([0.59910746, -0.11139207, 0.07508465], xp=xp)),
            [0.14222010, 0.23042768, 0.10495772],
            atol=TOLERANCE_ABSOLUTE_TESTS * 10,
        )

        xp_assert_close(
            Oklab_to_XYZ(xp_as_array([1.00121561, 0.00899591, -0.00535107], xp=xp)),
            [0.96907232, 1.00000000, 1.12179215],
            atol=TOLERANCE_ABSOLUTE_TESTS * 10,
        )

    def test_n_dimensional_Oklab_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.oklab.Oklab_to_XYZ` definition
        n-dimensional support.
        """

        Oklab = xp_as_array([0.51634019, 0.15469500, 0.06289579], xp=xp)
        XYZ = as_ndarray(Oklab_to_XYZ(Oklab))

        Oklab = xp.tile(xp_as_array(Oklab, xp=xp), (6, 1))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(Oklab_to_XYZ(Oklab), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        Oklab = xp_reshape(xp_as_array(Oklab, xp=xp), (2, 3, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(Oklab_to_XYZ(Oklab), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_Oklab_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.oklab.Oklab_to_XYZ` definition domain and
        range scale support.
        """

        Oklab = xp_as_array([0.51634019, 0.15469500, 0.06289579], xp=xp)
        XYZ = as_ndarray(Oklab_to_XYZ(Oklab))

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_equal(
                    Oklab_to_XYZ(Oklab * factor),
                    XYZ * factor,
                )

    @ignore_numpy_errors
    def test_nan_Oklab_to_XYZ(self) -> None:
        """
        Test :func:`colour.models.oklab.Oklab_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Oklab_to_XYZ(cases)
