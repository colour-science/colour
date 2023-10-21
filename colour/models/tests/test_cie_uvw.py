# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.cie_uvw` module."""

import unittest
from itertools import product

import numpy as np

from colour.models import UVW_to_XYZ, XYZ_to_UVW
from colour.utilities import domain_range_scale, ignore_numpy_errors

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


class TestXYZ_to_UVW(unittest.TestCase):
    """
    Define :func:`colour.models.cie_uvw.XYZ_to_UVW` definition unit tests
    methods.
    """

    def test_XYZ_to_UVW(self):
        """Test :func:`colour.models.cie_uvw.XYZ_to_UVW` definition."""

        np.testing.assert_array_almost_equal(
            XYZ_to_UVW(np.array([0.20654008, 0.12197225, 0.05136952]) * 100),
            np.array([94.55035725, 11.55536523, 40.54757405]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            XYZ_to_UVW(np.array([0.14222010, 0.23042768, 0.10495772]) * 100),
            np.array([-36.92762376, 28.90425105, 54.14071478]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            XYZ_to_UVW(np.array([0.07818780, 0.06157201, 0.28099326]) * 100),
            np.array([-10.60111550, -41.94580000, 28.82134002]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            XYZ_to_UVW(
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
                np.array([0.44757, 0.40745]),
            ),
            np.array([63.90676310, -8.11466183, 40.54757405]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            XYZ_to_UVW(
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
                np.array([0.34570, 0.35850]),
            ),
            np.array([88.56798946, 4.61154385, 40.54757405]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            XYZ_to_UVW(
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
                np.array([0.34570, 0.35850, 1.00000]),
            ),
            np.array([88.56798946, 4.61154385, 40.54757405]),
            decimal=7,
        )

    def test_n_dimensional_XYZ_to_UVW(self):
        """
        Test :func:`colour.models.cie_uvw.XYZ_to_UVW` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
        illuminant = np.array([0.31270, 0.32900])
        UVW = XYZ_to_UVW(XYZ, illuminant)

        XYZ = np.tile(XYZ, (6, 1))
        UVW = np.tile(UVW, (6, 1))
        np.testing.assert_array_almost_equal(
            XYZ_to_UVW(XYZ, illuminant), UVW, decimal=7
        )

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_array_almost_equal(
            XYZ_to_UVW(XYZ, illuminant), UVW, decimal=7
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        UVW = np.reshape(UVW, (2, 3, 3))
        np.testing.assert_array_almost_equal(
            XYZ_to_UVW(XYZ, illuminant), UVW, decimal=7
        )

    def test_domain_range_scale_XYZ_to_UVW(self):
        """
        Test :func:`colour.models.cie_uvw.XYZ_to_UVW` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
        illuminant = np.array([0.31270, 0.32900])
        UVW = XYZ_to_UVW(XYZ, illuminant)

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    XYZ_to_UVW(XYZ * factor, illuminant),
                    UVW * factor,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_UVW(self):
        """Test :func:`colour.models.cie_uvw.XYZ_to_UVW` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_UVW(cases, cases[..., 0:2])


class TestUVW_to_XYZ(unittest.TestCase):
    """
    Define :func:`colour.models.cie_uvw.UVW_to_XYZ` definition unit tests
    methods.
    """

    def test_UVW_to_XYZ(self):
        """Test :func:`colour.models.cie_uvw.UVW_to_XYZ` definition."""

        np.testing.assert_array_almost_equal(
            UVW_to_XYZ(np.array([94.55035725, 11.55536523, 40.54757405])),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            UVW_to_XYZ(np.array([-36.92762376, 28.90425105, 54.14071478])),
            np.array([0.14222010, 0.23042768, 0.10495772]) * 100,
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            UVW_to_XYZ(np.array([-10.60111550, -41.94580000, 28.82134002])),
            np.array([0.07818780, 0.06157201, 0.28099326]) * 100,
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            UVW_to_XYZ(
                np.array([63.90676310, -8.11466183, 40.54757405]),
                np.array([0.44757, 0.40745]),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            UVW_to_XYZ(
                np.array([88.56798946, 4.61154385, 40.54757405]),
                np.array([0.34570, 0.35850]),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            UVW_to_XYZ(
                np.array([88.56798946, 4.61154385, 40.54757405]),
                np.array([0.34570, 0.35850, 1.00000]),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            decimal=7,
        )

    def test_n_dimensional_UVW_to_XYZ(self):
        """
        Test :func:`colour.models.cie_uvw.UVW_to_XYZ` definition n-dimensional
        support.
        """

        UVW = np.array([94.55035725, 11.55536523, 40.54757405])
        illuminant = np.array([0.31270, 0.32900])
        XYZ = UVW_to_XYZ(UVW, illuminant)

        XYZ = np.tile(XYZ, (6, 1))
        UVW = np.tile(UVW, (6, 1))
        np.testing.assert_array_almost_equal(
            UVW_to_XYZ(UVW, illuminant), XYZ, decimal=7
        )

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_array_almost_equal(
            UVW_to_XYZ(UVW, illuminant), XYZ, decimal=7
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        UVW = np.reshape(UVW, (2, 3, 3))
        np.testing.assert_array_almost_equal(
            UVW_to_XYZ(UVW, illuminant), XYZ, decimal=7
        )

    def test_domain_range_scale_UVW_to_XYZ(self):
        """
        Test :func:`colour.models.cie_uvw.UVW_to_XYZ` definition domain and
        range scale support.
        """

        UVW = np.array([94.55035725, 11.55536523, 40.54757405])
        illuminant = np.array([0.31270, 0.32900])
        XYZ = UVW_to_XYZ(UVW, illuminant)

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    UVW_to_XYZ(UVW * factor, illuminant),
                    XYZ * factor,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_nan_UVW_to_XYZ(self):
        """Test :func:`colour.models.cie_uvw.UVW_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        UVW_to_XYZ(cases, cases[..., 0:2])


if __name__ == "__main__":
    unittest.main()
