# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.hunter_rdab` module."""

from itertools import product
import numpy as np
import unittest

from colour.models import XYZ_to_ICaCb, ICaCb_to_XYZ
from colour.utilities import ignore_numpy_errors, domain_range_scale

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_ICaCb",
    "TestICaCb_to_XYZ",
]


class TestXYZ_to_ICaCb(unittest.TestCase):
    """
    Define :func:`colour.models.icacb.XYZ_to_ICaCb` definition unit tests
    methods.
    """

    def test_XYZ_to_ICaCb(self):
        """Test :func:`colour.models.icacb.XYZ_to_ICaCb` definition."""

        np.testing.assert_almost_equal(
            XYZ_to_ICaCb(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.06875297, 0.05753352, 0.02081548]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            XYZ_to_ICaCb(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.08666353, -0.02479011, 0.03099396]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            XYZ_to_ICaCb(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.05102472, -0.00965461, -0.05150706]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            XYZ_to_ICaCb(np.array([0.00000000, 0.00000000, 1.00000000])),
            np.array([1702.0656419, 14738.00583456, 1239.66837927]),
            decimal=7,
        )

    def test_n_dimensional_XYZ_to_ICaCb(self):
        """
        Test :func:`colour.models.icacb.XYZ_to_ICaCb` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        ICaCb = XYZ_to_ICaCb(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        ICaCb = np.tile(ICaCb, (6, 1))
        np.testing.assert_almost_equal(XYZ_to_ICaCb(XYZ), ICaCb, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        ICaCb = np.reshape(ICaCb, (2, 3, 3))
        np.testing.assert_almost_equal(XYZ_to_ICaCb(XYZ), ICaCb, decimal=7)

    def test_domain_range_scale_XYZ_to_ICaCb(self):
        """
        Test :func:`colour.models.icacb.XYZ_to_ICaCb` definition domain and
        range scale support.
        """

        XYZ = np.array([0.07818780, 0.06157201, 0.28099326])
        ICaCb = XYZ_to_ICaCb(XYZ)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_ICaCb(XYZ * factor), ICaCb * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_ICaCb(self):
        """Test :func:`colour.models.cie_lab.XYZ_to_Lab` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_ICaCb(cases)


class TestICaCb_to_XYZ(unittest.TestCase):
    """Test :func:`colour.models.icacb.ICaCb_to_XYZ` definition."""

    def test_XYZ_to_ICaCb(self):
        """Test :func:`colour.models.icacb.ICaCb_to_XYZ` definition."""

        np.testing.assert_almost_equal(
            ICaCb_to_XYZ(np.array([0.06875297, 0.05753352, 0.02081548])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            ICaCb_to_XYZ(np.array([0.08666353, -0.02479011, 0.03099396])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            ICaCb_to_XYZ(np.array([0.05102472, -0.00965461, -0.05150706])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            ICaCb_to_XYZ(
                np.array([1702.0656419, 14738.00583456, 1239.66837927])
            ),
            np.array([0.00000000, 0.00000000, 1.00000000]),
            decimal=7,
        )

    def test_n_dimensional_ICaCb_to_XYZ(self):
        """
        Test :func:`colour.models.icacb.ICaCb_to_XYZ` definition
        n-dimensional support.
        """

        ICaCb = np.array([0.06875297, 0.05753352, 0.02081548])
        XYZ = ICaCb_to_XYZ(ICaCb)

        ICaCb = np.tile(ICaCb, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(ICaCb_to_XYZ(ICaCb), XYZ, decimal=7)

        ICaCb = np.reshape(ICaCb, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(ICaCb_to_XYZ(ICaCb), XYZ, decimal=7)

    def test_domain_range_scale_ICaCb_to_XYZ(self):
        """
        Test :func:`colour.models.icacb.ICaCb_to_XYZ` definition domain and
        range scale support.
        """

        ICaCb = np.array([0.06875297, 0.05753352, 0.02081548])
        XYZ = ICaCb_to_XYZ(ICaCb)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    ICaCb_to_XYZ(ICaCb * factor), XYZ * factor, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_ICaCb_to_XYZ(self):
        """Test :func:`colour.models.cie_lab.XYZ_to_Lab` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        ICaCb_to_XYZ(cases)


if __name__ == "__main__":
    unittest.main()
