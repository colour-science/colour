# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.jzazbz` module."""

import unittest
from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    Izazbz_to_XYZ,
    Jzazbz_to_XYZ,
    XYZ_to_Izazbz,
    XYZ_to_Jzazbz,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_Izazbz",
    "TestIzazbz_to_XYZ",
    "TestXYZ_to_Jzazbz",
    "TestJzazbz_to_XYZ",
]


class TestXYZ_to_Izazbz(unittest.TestCase):
    """
    Define :func:`colour.models.jzazbz.TestXYZ_to_Izazbz` definition unit
    tests methods.
    """

    def test_XYZ_to_Izazbz(self):
        """Test :func:`colour.models.jzazbz.XYZ_to_Izazbz` definition."""

        np.testing.assert_allclose(
            XYZ_to_Izazbz(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.01207793, 0.00924302, 0.00526007]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Izazbz(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.01397346, -0.00608426, 0.00534077]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Izazbz(np.array([0.96907232, 1.00000000, 1.12179215])),
            np.array([0.03927203, 0.00064174, -0.00052906]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Izazbz(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                method="Safdar 2021",
            ),
            np.array([0.01049146, 0.00924302, 0.00526007]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_array_equal(
            XYZ_to_Izazbz(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                method="Safdar 2021",
            ),
            XYZ_to_Izazbz(
                np.array([0.20654008, 0.12197225, 0.05136952]), method="ZCAM"
            ),
        )

    def test_n_dimensional_XYZ_to_Izazbz(self):
        """
        Test :func:`colour.models.jzazbz.XYZ_to_Izazbz` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Izazbz = XYZ_to_Izazbz(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        Izazbz = np.tile(Izazbz, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_Izazbz(XYZ), Izazbz, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        Izazbz = np.reshape(Izazbz, (2, 3, 3))
        np.testing.assert_allclose(
            XYZ_to_Izazbz(XYZ), Izazbz, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_Izazbz(self):
        """
        Test :func:`colour.models.jzazbz.XYZ_to_Izazbz` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Izazbz = XYZ_to_Izazbz(XYZ)

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_Izazbz(XYZ * factor),
                    Izazbz * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Izazbz(self):
        """
        Test :func:`colour.models.jzazbz.XYZ_to_Izazbz` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Izazbz(cases)


class TestIzazbz_to_XYZ(unittest.TestCase):
    """
    Define :func:`colour.models.jzazbz.Izazbz_to_XYZ` definition unit tests
    methods.
    """

    def test_Izazbz_to_XYZ(self):
        """Test :func:`colour.models.jzazbz.Izazbz_to_XYZ` definition."""

        np.testing.assert_allclose(
            Izazbz_to_XYZ(np.array([0.01207793, 0.00924302, 0.00526007])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Izazbz_to_XYZ(np.array([0.01397346, -0.00608426, 0.00534077])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Izazbz_to_XYZ(np.array([0.03927203, 0.00064174, -0.00052906])),
            np.array([0.96907232, 1.00000000, 1.12179215]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Izazbz_to_XYZ(np.array([0.03927203, 0.00064174, -0.00052906])),
            np.array([0.96907232, 1.00000000, 1.12179215]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Izazbz_to_XYZ(
                np.array([0.01049146, 0.00924302, 0.00526007]),
                method="Safdar 2021",
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_array_equal(
            Izazbz_to_XYZ(
                np.array([0.01049146, 0.00924302, 0.00526007]),
                method="Safdar 2021",
            ),
            Izazbz_to_XYZ(
                np.array([0.01049146, 0.00924302, 0.00526007]), method="ZCAM"
            ),
        )

    def test_n_dimensional_Izazbz_to_XYZ(self):
        """
        Test :func:`colour.models.jzazbz.Izazbz_to_XYZ` definition
        n-dimensional support.
        """

        Izazbz = np.array([0.01207793, 0.00924302, 0.00526007])
        XYZ = Izazbz_to_XYZ(Izazbz)

        Izazbz = np.tile(Izazbz, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            Izazbz_to_XYZ(Izazbz), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Izazbz = np.reshape(Izazbz, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            Izazbz_to_XYZ(Izazbz), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_Izazbz_to_XYZ(self):
        """
        Test :func:`colour.models.jzazbz.Izazbz_to_XYZ` definition domain and
        range scale support.
        """

        Izazbz = np.array([0.01207793, 0.00924302, 0.00526007])
        XYZ = Izazbz_to_XYZ(Izazbz)

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    Izazbz_to_XYZ(Izazbz * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Izazbz_to_XYZ(self):
        """
        Test :func:`colour.models.jzazbz.Izazbz_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Izazbz_to_XYZ(cases)


class TestXYZ_to_Jzazbz(unittest.TestCase):
    """
    Define :func:`colour.models.jzazbz.TestXYZ_to_Jzazbz` definition unit
    tests methods.
    """

    def test_XYZ_to_Jzazbz(self):
        """Test :func:`colour.models.jzazbz.XYZ_to_Jzazbz` definition."""

        np.testing.assert_allclose(
            XYZ_to_Jzazbz(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.00535048, 0.00924302, 0.00526007]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Jzazbz(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.00619681, -0.00608426, 0.00534077]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Jzazbz(np.array([0.96907232, 1.00000000, 1.12179215])),
            np.array([0.01766826, 0.00064174, -0.00052906]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_Jzazbz(self):
        """
        Test :func:`colour.models.jzazbz.XYZ_to_Jzazbz` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Jzazbz = XYZ_to_Jzazbz(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        Jzazbz = np.tile(Jzazbz, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_Jzazbz(XYZ), Jzazbz, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        Jzazbz = np.reshape(Jzazbz, (2, 3, 3))
        np.testing.assert_allclose(
            XYZ_to_Jzazbz(XYZ), Jzazbz, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_Jzazbz(self):
        """
        Test :func:`colour.models.jzazbz.XYZ_to_Jzazbz` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Jzazbz = XYZ_to_Jzazbz(XYZ)

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_equal(
                    XYZ_to_Jzazbz(XYZ * factor), Jzazbz * factor
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Jzazbz(self):
        """
        Test :func:`colour.models.jzazbz.XYZ_to_Jzazbz` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Jzazbz(cases)


class TestJzazbz_to_XYZ(unittest.TestCase):
    """
    Define :func:`colour.models.jzazbz.Jzazbz_to_XYZ` definition unit tests
    methods.
    """

    def test_Jzazbz_to_XYZ(self):
        """Test :func:`colour.models.jzazbz.Jzazbz_to_XYZ` definition."""

        np.testing.assert_allclose(
            Jzazbz_to_XYZ(np.array([0.00535048, 0.00924302, 0.00526007])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=1e-6,
        )

        np.testing.assert_allclose(
            Jzazbz_to_XYZ(np.array([0.00619681, -0.00608426, 0.00534077])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=1e-6,
        )

        np.testing.assert_allclose(
            Jzazbz_to_XYZ(np.array([0.01766826, 0.00064174, -0.00052906])),
            np.array([0.96907232, 1.00000000, 1.12179215]),
            atol=1e-6,
        )

    def test_n_dimensional_Jzazbz_to_XYZ(self):
        """
        Test :func:`colour.models.jzazbz.Jzazbz_to_XYZ` definition
        n-dimensional support.
        """

        Jzazbz = np.array([0.00535048, 0.00924302, 0.00526007])
        XYZ = Jzazbz_to_XYZ(Jzazbz)

        Jzazbz = np.tile(Jzazbz, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            Jzazbz_to_XYZ(Jzazbz), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Jzazbz = np.reshape(Jzazbz, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            Jzazbz_to_XYZ(Jzazbz), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_Jzazbz_to_XYZ(self):
        """
        Test :func:`colour.models.jzazbz.Jzazbz_to_XYZ` definition domain and
        range scale support.
        """

        Jzazbz = np.array([0.00535048, 0.00924302, 0.00526007])
        XYZ = Jzazbz_to_XYZ(Jzazbz)

        d_r = (("reference", 1), ("1", 1), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    Jzazbz_to_XYZ(Jzazbz * factor),
                    XYZ * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Jzazbz_to_XYZ(self):
        """
        Test :func:`colour.models.jzazbz.Jzazbz_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Jzazbz_to_XYZ(cases)


if __name__ == "__main__":
    unittest.main()
