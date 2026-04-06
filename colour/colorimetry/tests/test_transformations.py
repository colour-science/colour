"""
Define the unit tests for the :mod:`colour.colorimetry.transformations`
module.
"""

from __future__ import annotations

import typing

import numpy as np

if typing.TYPE_CHECKING:
    from colour.hints import ModuleType

from colour.colorimetry import (
    MSDS_CMFS,
    LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs,
    LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs,
    RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs,
    RGB_10_degree_cmfs_to_LMS_10_degree_cmfs,
    RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import (
    as_ndarray,
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
    "TestRGB_2_degree_cmfs_to_XYZ_2_degree_cmfs",
    "TestRGB_10_degree_cmfs_to_XYZ_10_degree_cmfs",
    "TestRGB_10_degree_cmfs_to_LMS_10_degree_cmfs",
    "TestLMS_2_degree_cmfs_to_XYZ_2_degree_cmfs",
    "TestLMS_10_degree_cmfs_to_XYZ_10_degree_cmfs",
]


class TestRGB_2_degree_cmfs_to_XYZ_2_degree_cmfs:
    """
    Define :func:`colour.colorimetry.transformations.\
RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition unit tests methods.
    """

    def test_RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(self) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition.
        """

        cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        xp_assert_close(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(435),
            cmfs[435],
            atol=TOLERANCE_ABSOLUTE_TESTS * 25000,
        )

        xp_assert_close(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(545),
            cmfs[545],
            atol=TOLERANCE_ABSOLUTE_TESTS * 25000,
        )

        xp_assert_close(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700),
            cmfs[700],
            atol=TOLERANCE_ABSOLUTE_TESTS * 25000,
        )

    def test_n_dimensional_RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition n-dimensional arrays
        support.
        """

        wl = 700
        XYZ = as_ndarray(RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl))

        wl = xp.tile(xp_as_array(wl, xp=xp), (6,))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3, 1), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 1, 3), xp=xp)
        xp_assert_close(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(self) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition nan support.
        """

        RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestRGB_10_degree_cmfs_to_XYZ_10_degree_cmfs:
    """
    Define :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition unit tests methods.
    """

    def test_RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(self) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition.
        """

        cmfs = MSDS_CMFS["CIE 1964 10 Degree Standard Observer"]
        xp_assert_close(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(435),
            cmfs[435],
            atol=TOLERANCE_ABSOLUTE_TESTS * 250000,
        )

        xp_assert_close(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(545),
            cmfs[545],
            atol=TOLERANCE_ABSOLUTE_TESTS * 250000,
        )

        xp_assert_close(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700),
            cmfs[700],
            atol=TOLERANCE_ABSOLUTE_TESTS * 250000,
        )

    def test_n_dimensional_RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition n-dimensional arrays
        support.
        """

        wl = 700
        XYZ = as_ndarray(RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl))

        wl = xp.tile(xp_as_array(wl, xp=xp), (6,))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3, 1), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 1, 3), xp=xp)
        xp_assert_close(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(self) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition nan support.
        """

        RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestRGB_10_degree_cmfs_to_LMS_10_degree_cmfs:
    """
    Define :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_LMS_10_degree_cmfs` definition unit tests methods.
    """

    def test_RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(self) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_LMS_10_degree_cmfs` definition.
        """

        cmfs = MSDS_CMFS["Stockman & Sharpe 10 Degree Cone Fundamentals"]
        xp_assert_close(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(435),
            cmfs[435],
            atol=TOLERANCE_ABSOLUTE_TESTS * 25000,
        )

        xp_assert_close(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(545),
            cmfs[545],
            atol=TOLERANCE_ABSOLUTE_TESTS * 25000,
        )

        xp_assert_close(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700),
            cmfs[700],
            atol=TOLERANCE_ABSOLUTE_TESTS * 25000,
        )

    def test_n_dimensional_RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_LMS_10_degree_cmfs` definition n-dimensional arrays
        support.
        """

        wl = 700
        LMS = as_ndarray(RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(wl))

        wl = xp.tile(xp_as_array(wl, xp=xp), (6,))
        LMS = xp.tile(xp_as_array(LMS, xp=xp), (6, 1))
        xp_assert_close(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(wl),
            LMS,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3), xp=xp)
        LMS = xp_reshape(xp_as_array(LMS, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(wl),
            LMS,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3, 1), xp=xp)
        LMS = xp_reshape(xp_as_array(LMS, xp=xp), (2, 3, 1, 3), xp=xp)
        xp_assert_close(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(wl),
            LMS,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(self) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_LMS_10_degree_cmfs` definition nan support.
        """

        RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLMS_2_degree_cmfs_to_XYZ_2_degree_cmfs:
    """
    Define :func:`colour.colorimetry.transformations.\
LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition unit tests methods.
    """

    def test_LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(self) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition.
        """

        cmfs = MSDS_CMFS["CIE 2015 2 Degree Standard Observer"]
        xp_assert_close(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(435),
            cmfs[435],
            atol=TOLERANCE_ABSOLUTE_TESTS * 1500,
        )

        xp_assert_close(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(545),
            cmfs[545],
            atol=TOLERANCE_ABSOLUTE_TESTS * 1500,
        )

        xp_assert_close(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700),
            cmfs[700],
            atol=TOLERANCE_ABSOLUTE_TESTS * 1500,
        )

    def test_n_dimensional_LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition n-dimensional arrays
        support.
        """

        wl = 700
        XYZ = as_ndarray(LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl))

        wl = xp.tile(xp_as_array(wl, xp=xp), (6,))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3, 1), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 1, 3), xp=xp)
        xp_assert_close(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(self) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition nan support.
        """

        LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLMS_10_degree_cmfs_to_XYZ_10_degree_cmfs:
    """
    Define :func:`colour.colorimetry.transformations.\
LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition unit tests methods.
    """

    def test_LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(self) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition.
        """

        cmfs = MSDS_CMFS["CIE 2015 10 Degree Standard Observer"]
        xp_assert_close(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(435),
            cmfs[435],
            atol=TOLERANCE_ABSOLUTE_TESTS * 1500,
        )

        xp_assert_close(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(545),
            cmfs[545],
            atol=TOLERANCE_ABSOLUTE_TESTS * 1500,
        )

        xp_assert_close(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700),
            cmfs[700],
            atol=TOLERANCE_ABSOLUTE_TESTS * 1500,
        )

    def test_n_dimensional_LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(
        self, xp: ModuleType
    ) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition n-dimensional arrays
        support.
        """

        wl = 700
        XYZ = as_ndarray(LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl))

        wl = xp.tile(xp_as_array(wl, xp=xp), (6,))
        XYZ = xp.tile(xp_as_array(XYZ, xp=xp), (6, 1))
        xp_assert_close(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 3), xp=xp)
        xp_assert_close(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = xp_reshape(xp_as_array(wl, xp=xp), (2, 3, 1), xp=xp)
        XYZ = xp_reshape(xp_as_array(XYZ, xp=xp), (2, 3, 1, 3), xp=xp)
        xp_assert_close(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(self) -> None:
        """
        Test :func:`colour.colorimetry.transformations.\
LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition nan support.
        """

        LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )
