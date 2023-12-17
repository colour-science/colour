# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.din99` module."""

import unittest
from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    DIN99_to_Lab,
    DIN99_to_XYZ,
    Lab_to_DIN99,
    XYZ_to_DIN99,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

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


class TestLab_to_DIN99(unittest.TestCase):
    """
    Define :func:`colour.models.din99.Lab_to_DIN99` definition unit tests
    methods.
    """

    def test_Lab_to_DIN99(self):
        """Test :func:`colour.models.din99.Lab_to_DIN99` definition."""

        np.testing.assert_allclose(
            Lab_to_DIN99(np.array([41.52787529, 52.63858304, 26.92317922])),
            np.array([53.22821988, 28.41634656, 3.89839552]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Lab_to_DIN99(np.array([55.11636304, -41.08791787, 30.91825778])),
            np.array([66.08943912, -17.35290106, 16.09690691]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Lab_to_DIN99(np.array([29.80565520, 20.01830466, -48.34913874])),
            np.array([40.71533366, 3.48714163, -21.45321411]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Lab_to_DIN99(
                np.array([41.52787529, 52.63858304, 26.92317922]),
                method="DIN99b",
            ),
            np.array([45.58303137, 34.71824493, 17.61622149]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Lab_to_DIN99(
                np.array([41.52787529, 52.63858304, 26.92317922]),
                method="DIN99c",
            ),
            np.array([45.40284208, 32.75074741, 15.74603532]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Lab_to_DIN99(
                np.array([41.52787529, 52.63858304, 26.92317922]),
                method="DIN99d",
            ),
            np.array([45.31204747, 31.42106716, 14.17004652]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Lab_to_DIN99(self):
        """
        Test :func:`colour.models.din99.Lab_to_DIN99` definition n-dimensional
        support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        Lab_99 = Lab_to_DIN99(Lab)

        Lab = np.tile(Lab, (6, 1))
        Lab_99 = np.tile(Lab_99, (6, 1))
        np.testing.assert_allclose(
            Lab_to_DIN99(Lab), Lab_99, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Lab = np.reshape(Lab, (2, 3, 3))
        Lab_99 = np.reshape(Lab_99, (2, 3, 3))
        np.testing.assert_allclose(
            Lab_to_DIN99(Lab), Lab_99, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_Lab_to_DIN99(self):
        """
        Test :func:`colour.models.din99.Lab_to_DIN99` definition domain and
        range scale support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        Lab_99 = Lab_to_DIN99(Lab)
        Lab_99_b = Lab_to_DIN99(Lab, method="DIN99b")
        Lab_99_c = Lab_to_DIN99(Lab, method="DIN99c")
        Lab_99_d = Lab_to_DIN99(Lab, method="DIN99d")

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    Lab_to_DIN99(Lab * factor),
                    Lab_99 * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                np.testing.assert_allclose(
                    Lab_to_DIN99((Lab * factor), method="DIN99b"),
                    Lab_99_b * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                np.testing.assert_allclose(
                    Lab_to_DIN99((Lab * factor), method="DIN99c"),
                    Lab_99_c * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                np.testing.assert_allclose(
                    Lab_to_DIN99((Lab * factor), method="DIN99d"),
                    Lab_99_d * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Lab_to_DIN99(self):
        """Test :func:`colour.models.din99.Lab_to_DIN99` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Lab_to_DIN99(cases)
        Lab_to_DIN99(cases, method="DIN99b")
        Lab_to_DIN99(cases, method="DIN99c")
        Lab_to_DIN99(cases, method="DIN99d")


class TestDIN99_to_Lab(unittest.TestCase):
    """
    Define :func:`colour.models.din99.DIN99_to_Lab` definition unit tests
    methods.
    """

    def test_DIN99_to_Lab(self):
        """Test :func:`colour.models.din99.DIN99_to_Lab` definition."""

        np.testing.assert_allclose(
            DIN99_to_Lab(np.array([53.22821988, 28.41634656, 3.89839552])),
            np.array([41.52787529, 52.63858304, 26.92317922]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            DIN99_to_Lab(np.array([66.08943912, -17.35290106, 16.09690691])),
            np.array([55.11636304, -41.08791787, 30.91825778]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            DIN99_to_Lab(np.array([40.71533366, 3.48714163, -21.45321411])),
            np.array([29.80565520, 20.01830466, -48.34913874]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            DIN99_to_Lab(
                np.array([45.58303137, 34.71824493, 17.61622149]),
                method="DIN99b",
            ),
            np.array([41.52787529, 52.63858304, 26.92317922]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            DIN99_to_Lab(
                np.array([45.40284208, 32.75074741, 15.74603532]),
                method="DIN99c",
            ),
            np.array([41.52787529, 52.63858304, 26.92317922]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            DIN99_to_Lab(
                np.array([45.31204747, 31.42106716, 14.17004652]),
                method="DIN99d",
            ),
            np.array([41.52787529, 52.63858304, 26.92317922]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_DIN99_to_Lab(self):
        """
        Test :func:`colour.models.din99.DIN99_to_Lab` definition n-dimensional
        support.
        """

        Lab_99 = np.array([53.22821988, 28.41634656, 3.89839552])
        Lab = DIN99_to_Lab(Lab_99)

        Lab_99 = np.tile(Lab_99, (6, 1))
        Lab = np.tile(Lab, (6, 1))
        np.testing.assert_allclose(
            DIN99_to_Lab(Lab_99), Lab, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Lab_99 = np.reshape(Lab_99, (2, 3, 3))
        Lab = np.reshape(Lab, (2, 3, 3))
        np.testing.assert_allclose(
            DIN99_to_Lab(Lab_99), Lab, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_DIN99_to_Lab(self):
        """
        Test :func:`colour.models.din99.DIN99_to_Lab` definition domain and
        range scale support.
        """

        Lab_99 = np.array([53.22821988, 28.41634656, 3.89839552])
        Lab = DIN99_to_Lab(Lab_99)
        Lab_b = DIN99_to_Lab(Lab_99, method="DIN99b")
        Lab_c = DIN99_to_Lab(Lab_99, method="DIN99c")
        Lab_d = DIN99_to_Lab(Lab_99, method="DIN99d")

        d_r = (("reference", 1), ("1", 0.01), ("100", 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    DIN99_to_Lab(Lab_99 * factor),
                    Lab * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                np.testing.assert_allclose(
                    DIN99_to_Lab((Lab_99 * factor), method="DIN99b"),
                    Lab_b * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                np.testing.assert_allclose(
                    DIN99_to_Lab((Lab_99 * factor), method="DIN99c"),
                    Lab_c * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )
                np.testing.assert_allclose(
                    DIN99_to_Lab((Lab_99 * factor), method="DIN99d"),
                    Lab_d * factor,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_DIN99_to_Lab(self):
        """Test :func:`colour.models.din99.DIN99_to_Lab` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        DIN99_to_Lab(cases)
        DIN99_to_Lab(cases, method="DIN99b")
        DIN99_to_Lab(cases, method="DIN99c")
        DIN99_to_Lab(cases, method="DIN99d")


class TestXYZ_to_DIN99(unittest.TestCase):
    """
    Define :func:`colour.models.din99.XYZ_to_DIN99` definition unit tests
    methods.
    """

    def test_XYZ_to_DIN99(self):
        """Test :func:`colour.models.din99.XYZ_to_DIN99` definition."""

        np.testing.assert_allclose(
            XYZ_to_DIN99(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([53.22821988, 28.41634656, 3.89839552]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_DIN99(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([66.08943912, -17.35290106, 16.09690691]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_DIN99(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([40.71533366, 3.48714163, -21.45321411]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_DIN99(
                np.array([0.20654008, 0.12197225, 0.05136952]), method="DIN99b"
            ),
            np.array([45.58303137, 34.71824493, 17.61622149]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_DIN99(self):
        """
        Test :func:`colour.models.din99.XYZ_to_DIN99` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Lab_99 = XYZ_to_DIN99(XYZ)

        XYZ = np.tile(XYZ, (6, 1))
        Lab_99 = np.tile(Lab_99, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_DIN99(XYZ), Lab_99, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        Lab_99 = np.reshape(Lab_99, (2, 3, 3))
        np.testing.assert_allclose(
            XYZ_to_DIN99(XYZ), Lab_99, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_DIN99(self):
        """
        Test :func:`colour.models.din99.XYZ_to_DIN99` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Lab_99 = XYZ_to_DIN99(XYZ)

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_DIN99(XYZ * factor_a),
                    Lab_99 * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_DIN99(self):
        """Test :func:`colour.models.din99.XYZ_to_DIN99` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_DIN99(cases)


class TestDIN99_to_XYZ(unittest.TestCase):
    """
    Define :func:`colour.models.din99.DIN99_to_XYZ` definition unit tests
    methods.
    """

    def test_DIN99_to_XYZ(self):
        """Test :func:`colour.models.din99.DIN99_to_XYZ` definition."""

        np.testing.assert_allclose(
            DIN99_to_XYZ(np.array([53.22821988, 28.41634656, 3.89839552])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            DIN99_to_XYZ(np.array([66.08943912, -17.35290106, 16.09690691])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            DIN99_to_XYZ(np.array([40.71533366, 3.48714163, -21.45321411])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            DIN99_to_XYZ(
                np.array([45.58303137, 34.71824493, 17.61622149]),
                method="DIN99b",
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_DIN99_to_XYZ(self):
        """
        Test :func:`colour.models.din99.DIN99_to_XYZ` definition n-dimensional
        support.
        """

        Lab_99 = np.array([53.22821988, 28.41634656, 3.89839552])
        XYZ = DIN99_to_XYZ(Lab_99)

        Lab_99 = np.tile(Lab_99, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            DIN99_to_XYZ(Lab_99), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Lab_99 = np.reshape(Lab_99, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            DIN99_to_XYZ(Lab_99), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_DIN99_to_XYZ(self):
        """
        Test :func:`colour.models.din99.DIN99_to_XYZ` definition domain and
        range scale support.
        """

        Lab_99 = np.array([53.22821988, 28.41634656, 3.89839552])
        XYZ = DIN99_to_XYZ(Lab_99)

        d_r = (("reference", 1, 1), ("1", 0.01, 1), ("100", 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    DIN99_to_XYZ(Lab_99 * factor_a),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_DIN99_to_XYZ(self):
        """Test :func:`colour.models.din99.DIN99_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        DIN99_to_XYZ(cases)


if __name__ == "__main__":
    unittest.main()
