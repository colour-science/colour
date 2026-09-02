"""Define the unit tests for the :mod:`colour.models.cie_uvw` module."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import UVW_to_XYZ, XYZ_to_UVW
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
    "TestXYZ_to_UVW",
    "TestUVW_to_XYZ",
]


class TestXYZ_to_UVW:
    """
    Define :func:`colour.models.cie_uvw.XYZ_to_UVW` definition unit tests
    methods.
    """

    def test_XYZ_to_UVW(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_uvw.XYZ_to_UVW` definition."""

        xp_assert_close(
            XYZ_to_UVW(xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100),
            [94.55035725, 11.55536523, 40.54757405],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_UVW(xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp) * 100),
            [-36.92762376, 28.90425105, 54.14071478],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_UVW(xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp) * 100),
            [-10.60111550, -41.94580000, 28.82134002],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_UVW(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
                xp_as_array([0.44757, 0.40745], xp=xp),
            ),
            [63.90676310, -8.11466183, 40.54757405],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_UVW(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
                xp_as_array([0.34570, 0.35850], xp=xp),
            ),
            [88.56798946, 4.61154385, 40.54757405],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            XYZ_to_UVW(
                xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
                xp_as_array([0.34570, 0.35850, 1.00000], xp=xp),
            ),
            [88.56798946, 4.61154385, 40.54757405],
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_UVW(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_uvw.XYZ_to_UVW` definition n-dimensional
        support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        UVW = as_ndarray(XYZ_to_UVW(XYZ, illuminant))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        UVW = xp.tile(xp_as_array(UVW, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_UVW(XYZ, illuminant), UVW, atol=TOLERANCE_ABSOLUTE_TESTS)

        illuminant = xp.tile(xp_as_array(illuminant, xp=xp), (6, 1))
        xp_assert_close(XYZ_to_UVW(XYZ, illuminant), UVW, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        illuminant = xp_reshape(xp_as_array(illuminant, xp=xp), (2, 3, 2), xp=xp)
        UVW = xp_reshape(xp_as_array(UVW, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(XYZ_to_UVW(XYZ, illuminant), UVW, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_XYZ_to_UVW(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_uvw.XYZ_to_UVW` definition domain and
        range scale support.
        """

        XYZ = xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        UVW = as_ndarray(XYZ_to_UVW(XYZ, illuminant))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    XYZ_to_UVW(XYZ * factor, illuminant),
                    UVW * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_UVW(self) -> None:
        """Test :func:`colour.models.cie_uvw.XYZ_to_UVW` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_UVW(cases, cases[..., 0:2])


class TestUVW_to_XYZ:
    """
    Define :func:`colour.models.cie_uvw.UVW_to_XYZ` definition unit tests
    methods.
    """

    def test_UVW_to_XYZ(self, xp: ModuleType) -> None:
        """Test :func:`colour.models.cie_uvw.UVW_to_XYZ` definition."""

        xp_assert_close(
            UVW_to_XYZ(xp_as_array([94.55035725, 11.55536523, 40.54757405], xp=xp)),
            xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UVW_to_XYZ(xp_as_array([-36.92762376, 28.90425105, 54.14071478], xp=xp)),
            xp_as_array([0.14222010, 0.23042768, 0.10495772], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UVW_to_XYZ(xp_as_array([-10.60111550, -41.94580000, 28.82134002], xp=xp)),
            xp_as_array([0.07818780, 0.06157201, 0.28099326], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UVW_to_XYZ(
                xp_as_array([63.90676310, -8.11466183, 40.54757405], xp=xp),
                xp_as_array([0.44757, 0.40745], xp=xp),
            ),
            xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UVW_to_XYZ(
                xp_as_array([88.56798946, 4.61154385, 40.54757405], xp=xp),
                xp_as_array([0.34570, 0.35850], xp=xp),
            ),
            xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        xp_assert_close(
            UVW_to_XYZ(
                xp_as_array([88.56798946, 4.61154385, 40.54757405], xp=xp),
                xp_as_array([0.34570, 0.35850, 1.00000], xp=xp),
            ),
            xp_as_array([0.20654008, 0.12197225, 0.05136952], xp=xp) * 100,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_UVW_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_uvw.UVW_to_XYZ` definition n-dimensional
        support.
        """

        UVW = xp_as_array([94.55035725, 11.55536523, 40.54757405], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        XYZ = as_ndarray(UVW_to_XYZ(UVW, illuminant))

        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        UVW = xp.tile(xp_as_array(UVW, xp=xp), (6, 1))
        xp_assert_close(UVW_to_XYZ(UVW, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        illuminant = xp.tile(xp_as_array(illuminant, xp=xp), (6, 1))
        xp_assert_close(UVW_to_XYZ(UVW, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        illuminant = xp_reshape(xp_as_array(illuminant, xp=xp), (2, 3, 2), xp=xp)
        UVW = xp_reshape(xp_as_array(UVW, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(UVW_to_XYZ(UVW, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS)

    def test_domain_range_scale_UVW_to_XYZ(self, xp: ModuleType) -> None:
        """
        Test :func:`colour.models.cie_uvw.UVW_to_XYZ` definition domain and
        range scale support.
        """

        UVW = xp_as_array([94.55035725, 11.55536523, 40.54757405], xp=xp)
        illuminant = xp_as_array([0.31270, 0.32900], xp=xp)
        XYZ = as_ndarray(UVW_to_XYZ(UVW, illuminant))

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                xp_assert_close(
                    UVW_to_XYZ(UVW * factor, illuminant),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_UVW_to_XYZ(self) -> None:
        """Test :func:`colour.models.cie_uvw.UVW_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        UVW_to_XYZ(cases, cases[..., 0:2])
