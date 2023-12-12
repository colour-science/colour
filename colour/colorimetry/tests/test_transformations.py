"""
Define the unit tests for the :mod:`colour.colorimetry.transformations`
module.
"""

import unittest

import numpy as np

from colour.colorimetry import (
    MSDS_CMFS,
    LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs,
    LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs,
    RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs,
    RGB_10_degree_cmfs_to_LMS_10_degree_cmfs,
    RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import ignore_numpy_errors

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


class TestRGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.transformations.\
RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition unit tests methods.
    """

    def test_RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition.
        """

        cmfs = MSDS_CMFS["CIE 1931 2 Degree Standard Observer"]
        np.testing.assert_allclose(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(435), cmfs[435], atol=0.0025
        )

        np.testing.assert_allclose(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(545), cmfs[545], atol=0.0025
        )

        np.testing.assert_allclose(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(700), cmfs[700], atol=0.0025
        )

    def test_n_dimensional_RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition n-dimensional arrays
        support.
        """

        wl = 700
        XYZ = RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl)

        wl = np.tile(wl, 6)
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = np.reshape(wl, (2, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = np.reshape(wl, (2, 3, 1))
        XYZ = np.reshape(XYZ, (2, 3, 1, 3))
        np.testing.assert_allclose(
            RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition nan support.
        """

        RGB_2_degree_cmfs_to_XYZ_2_degree_cmfs(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestRGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition unit tests methods.
    """

    def test_RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition.
        """

        cmfs = MSDS_CMFS["CIE 1964 10 Degree Standard Observer"]
        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(435),
            cmfs[435],
            atol=0.025,
        )

        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(545),
            cmfs[545],
            atol=0.025,
        )

        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(700),
            cmfs[700],
            atol=0.025,
        )

    def test_n_dimensional_RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition n-dimensional arrays
        support.
        """

        wl = 700
        XYZ = RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl)

        wl = np.tile(wl, 6)
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = np.reshape(wl, (2, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = np.reshape(wl, (2, 3, 1))
        XYZ = np.reshape(XYZ, (2, 3, 1, 3))
        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition nan support.
        """

        RGB_10_degree_cmfs_to_XYZ_10_degree_cmfs(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestRGB_10_degree_cmfs_to_LMS_10_degree_cmfs(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_LMS_10_degree_cmfs` definition unit tests methods.
    """

    def test_RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_LMS_10_degree_cmfs` definition.
        """

        cmfs = MSDS_CMFS["Stockman & Sharpe 10 Degree Cone Fundamentals"]
        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(435),
            cmfs[435],
            atol=0.0025,
        )

        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(545),
            cmfs[545],
            atol=0.0025,
        )

        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(700),
            cmfs[700],
            atol=0.0025,
        )

    def test_n_dimensional_RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_LMS_10_degree_cmfs` definition n-dimensional arrays
        support.
        """

        wl = 700
        LMS = RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(wl)

        wl = np.tile(wl, 6)
        LMS = np.tile(LMS, (6, 1))
        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(wl),
            LMS,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = np.reshape(wl, (2, 3))
        LMS = np.reshape(LMS, (2, 3, 3))
        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(wl),
            LMS,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = np.reshape(wl, (2, 3, 1))
        LMS = np.reshape(LMS, (2, 3, 1, 3))
        np.testing.assert_allclose(
            RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(wl),
            LMS,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
RGB_10_degree_cmfs_to_LMS_10_degree_cmfs` definition nan support.
        """

        RGB_10_degree_cmfs_to_LMS_10_degree_cmfs(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.transformations.\
LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition unit tests methods.
    """

    def test_LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition.
        """

        cmfs = MSDS_CMFS["CIE 2015 2 Degree Standard Observer"]
        np.testing.assert_allclose(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(435),
            cmfs[435],
            atol=0.00015,
        )

        np.testing.assert_allclose(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(545),
            cmfs[545],
            atol=0.00015,
        )

        np.testing.assert_allclose(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(700),
            cmfs[700],
            atol=0.00015,
        )

    def test_n_dimensional_LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition n-dimensional arrays
        support.
        """

        wl = 700
        XYZ = LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl)

        wl = np.tile(wl, 6)
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = np.reshape(wl, (2, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = np.reshape(wl, (2, 3, 1))
        XYZ = np.reshape(XYZ, (2, 3, 1, 3))
        np.testing.assert_allclose(
            LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs` definition nan support.
        """

        LMS_2_degree_cmfs_to_XYZ_2_degree_cmfs(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestLMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(unittest.TestCase):
    """
    Define :func:`colour.colorimetry.transformations.\
LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition unit tests methods.
    """

    def test_LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition.
        """

        cmfs = MSDS_CMFS["CIE 2015 10 Degree Standard Observer"]
        np.testing.assert_allclose(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(435),
            cmfs[435],
            atol=0.00015,
        )

        np.testing.assert_allclose(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(545),
            cmfs[545],
            atol=0.00015,
        )

        np.testing.assert_allclose(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(700),
            cmfs[700],
            atol=0.00015,
        )

    def test_n_dimensional_LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition n-dimensional arrays
        support.
        """

        wl = 700
        XYZ = LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl)

        wl = np.tile(wl, 6)
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = np.reshape(wl, (2, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        wl = np.reshape(wl, (2, 3, 1))
        XYZ = np.reshape(XYZ, (2, 3, 1, 3))
        np.testing.assert_allclose(
            LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(wl),
            XYZ,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(self):
        """
        Test :func:`colour.colorimetry.transformations.\
LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs` definition nan support.
        """

        LMS_10_degree_cmfs_to_XYZ_10_degree_cmfs(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


if __name__ == "__main__":
    unittest.main()
