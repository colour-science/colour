# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.munish2021` module."""

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_IPT_Munish2021, IPT_Munish2021_to_XYZ
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_IPT_Munish2021",
    "TestIPT_Munish2021_to_XYZ",
]


class TestXYZ_to_IPT_Munish2021(unittest.TestCase):
    """
    Define :func:`colour.models.munish2021.XYZ_to_IPT_Munish2021` definition
    unit tests methods.
    """

    def test_XYZ_to_IPT_Munish2021(self):
        """
        Test :func:`colour.models.munish2021.XYZ_to_IPT_Munish2021`
        definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_IPT_Munish2021(
                np.array([0.20654008, 0.12197225, 0.05136952])
            ),
            np.array([0.42248243, 0.29105140, 0.20410663]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            XYZ_to_IPT_Munish2021(
                np.array([0.14222010, 0.23042768, 0.10495772])
            ),
            np.array([0.54745257, -0.22795249, 0.10109646]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            XYZ_to_IPT_Munish2021(
                np.array([0.07818780, 0.06157201, 0.28099326])
            ),
            np.array([0.32151337, 0.06071424, -0.27388774]),
            decimal=7,
        )

    def test_n_dimensional_XYZ_to_IPT_Munish2021(self):
        """
        Test :func:`colour.models.munish2021.XYZ_to_IPT_Munish2021` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        IPT = XYZ_to_IPT_Munish2021(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        IPT = np.tile(IPT, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_IPT_Munish2021(XYZ), IPT, decimal=7
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        IPT = np.reshape(IPT, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_IPT_Munish2021(XYZ), IPT, decimal=7
        )

    def test_domain_range_scale_XYZ_to_IPT_Munish2021(self):
        """
        Test :func:`colour.models.munish2021.XYZ_to_IPT_Munish2021` definition
        domain and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        IPT = XYZ_to_IPT_Munish2021(XYZ)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_IPT_Munish2021(XYZ * factor),
                    IPT * factor,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_IPT_Munish2021(self):
        """
        Test :func:`colour.models.munish2021.XYZ_to_IPT_Munish2021` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_to_IPT_Munish2021(XYZ)


class TestIPT_Munish2021_to_XYZ(unittest.TestCase):
    """
    Define :func:`colour.models.munish2021.IPT_Munish2021_to_XYZ` definition
    unit tests methods.
    """

    def test_IPT_Munish2021_to_XYZ(self):
        """
        Test :func:`colour.models.munish2021.IPT_Munish2021_to_XYZ`
        definition.
        """

        np.testing.assert_almost_equal(
            IPT_Munish2021_to_XYZ(
                np.array([0.42248243, 0.29105140, 0.20410663])
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            IPT_Munish2021_to_XYZ(
                np.array([0.54745257, -0.22795249, 0.10109646])
            ),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            decimal=7,
        )

        np.testing.assert_almost_equal(
            IPT_Munish2021_to_XYZ(
                np.array([0.32151337, 0.06071424, -0.27388774])
            ),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            decimal=7,
        )

    def test_n_dimensional_IPT_Munish2021_to_XYZ(self):
        """
        Test :func:`colour.models.munish2021.IPT_Munish2021_to_XYZ` definition
        n-dimensional support.
        """

        IPT = np.array([0.42248243, 0.29105140, 0.20410663])
        XYZ = IPT_Munish2021_to_XYZ(IPT)

        IPT = np.tile(IPT, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            IPT_Munish2021_to_XYZ(IPT), XYZ, decimal=7
        )

        IPT = np.reshape(IPT, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            IPT_Munish2021_to_XYZ(IPT), XYZ, decimal=7
        )

    def test_domain_range_scale_IPT_Munish2021_to_XYZ(self):
        """
        Test :func:`colour.models.munish2021.IPT_Munish2021_to_XYZ` definition
        domain and range scale support.
        """

        IPT = np.array([0.42248243, 0.29105140, 0.20410663])
        XYZ = IPT_Munish2021_to_XYZ(IPT)

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    IPT_Munish2021_to_XYZ(IPT * factor),
                    XYZ * factor,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_nan_IPT_Munish2021_to_XYZ(self):
        """
        Test :func:`colour.models.munish2021.IPT_Munish2021_to_XYZ` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            IPT = np.array(case)
            IPT_Munish2021_to_XYZ(IPT)


if __name__ == "__main__":
    unittest.main()
