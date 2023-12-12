# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.cie_lab` module."""

import unittest
from itertools import product

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import Lab_to_LCHab, Lab_to_XYZ, LCHab_to_Lab, XYZ_to_Lab
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestXYZ_to_Lab",
    "TestLab_to_XYZ",
    "TestLab_to_LCHab",
    "TestLCHab_to_Lab",
]


class TestXYZ_to_Lab(unittest.TestCase):
    """
    Define :func:`colour.models.cie_lab.XYZ_to_Lab` definition unit tests
    methods.
    """

    def test_XYZ_to_Lab(self):
        """Test :func:`colour.models.cie_lab.XYZ_to_Lab` definition."""

        np.testing.assert_allclose(
            XYZ_to_Lab(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([41.52787529, 52.63858304, 26.92317922]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Lab(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([55.11636304, -41.08791787, 30.91825778]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Lab(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([29.80565520, 20.01830466, -48.34913874]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Lab(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.44757, 0.40745]),
            ),
            np.array([41.52787529, 38.48089305, -5.73295122]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Lab(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.34570, 0.35850]),
            ),
            np.array([41.52787529, 51.19354174, 19.91843098]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            XYZ_to_Lab(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.34570, 0.35850, 1.00000]),
            ),
            np.array([41.52787529, 51.19354174, 19.91843098]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_XYZ_to_Lab(self):
        """
        Test :func:`colour.models.cie_lab.XYZ_to_Lab` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
        Lab = XYZ_to_Lab(XYZ, illuminant)

        XYZ = np.tile(XYZ, (6, 1))
        Lab = np.tile(Lab, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_Lab(XYZ, illuminant), Lab, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_allclose(
            XYZ_to_Lab(XYZ, illuminant), Lab, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        Lab = np.reshape(Lab, (2, 3, 3))
        np.testing.assert_allclose(
            XYZ_to_Lab(XYZ, illuminant), Lab, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_XYZ_to_Lab(self):
        """
        Test :func:`colour.models.cie_lab.XYZ_to_Lab` definition
        domain and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
        Lab = XYZ_to_Lab(XYZ, illuminant)

        d_r = (("reference", 1, 1), ("1", 1, 0.01), ("100", 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    XYZ_to_Lab(XYZ * factor_a, illuminant),
                    Lab * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Lab(self):
        """Test :func:`colour.models.cie_lab.XYZ_to_Lab` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Lab(cases, cases[..., 0:2])


class TestLab_to_XYZ(unittest.TestCase):
    """
    Define :func:`colour.models.cie_lab.Lab_to_XYZ` definition unit tests
    methods.
    """

    def test_Lab_to_XYZ(self):
        """Test :func:`colour.models.cie_lab.Lab_to_XYZ` definition."""

        np.testing.assert_allclose(
            Lab_to_XYZ(np.array([41.52787529, 52.63858304, 26.92317922])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Lab_to_XYZ(np.array([55.11636304, -41.08791787, 30.91825778])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Lab_to_XYZ(np.array([29.80565520, 20.01830466, -48.34913874])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Lab_to_XYZ(
                np.array([41.52787529, 38.48089305, -5.73295122]),
                np.array([0.44757, 0.40745]),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Lab_to_XYZ(
                np.array([41.52787529, 51.19354174, 19.91843098]),
                np.array([0.34570, 0.35850]),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Lab_to_XYZ(
                np.array([41.52787529, 51.19354174, 19.91843098]),
                np.array([0.34570, 0.35850, 1.00000]),
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Lab_to_XYZ(self):
        """
        Test :func:`colour.models.cie_lab.Lab_to_XYZ` definition n-dimensional
        support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        illuminant = np.array([0.31270, 0.32900])
        XYZ = Lab_to_XYZ(Lab, illuminant)

        Lab = np.tile(Lab, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            Lab_to_XYZ(Lab, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_allclose(
            Lab_to_XYZ(Lab, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Lab = np.reshape(Lab, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            Lab_to_XYZ(Lab, illuminant), XYZ, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_Lab_to_XYZ(self):
        """
        Test :func:`colour.models.cie_lab.Lab_to_XYZ` definition
        domain and range scale support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        illuminant = np.array([0.31270, 0.32900])
        XYZ = Lab_to_XYZ(Lab, illuminant)

        d_r = (("reference", 1, 1), ("1", 0.01, 1), ("100", 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    Lab_to_XYZ(Lab * factor_a, illuminant),
                    XYZ * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Lab_to_XYZ(self):
        """Test :func:`colour.models.cie_lab.Lab_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Lab_to_XYZ(cases, cases[..., 0:2])


class TestLab_to_LCHab(unittest.TestCase):
    """
    Define :func:`colour.models.cie_lab.Lab_to_LCHab` definition unit tests
    methods.
    """

    def test_Lab_to_LCHab(self):
        """Test :func:`colour.models.cie_lab.Lab_to_LCHab` definition."""

        np.testing.assert_allclose(
            Lab_to_LCHab(np.array([41.52787529, 52.63858304, 26.92317922])),
            np.array([41.52787529, 59.12425901, 27.08848784]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Lab_to_LCHab(np.array([55.11636304, -41.08791787, 30.91825778])),
            np.array([55.11636304, 51.42135412, 143.03889556]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            Lab_to_LCHab(np.array([29.80565520, 20.01830466, -48.34913874])),
            np.array([29.80565520, 52.32945383, 292.49133666]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_Lab_to_LCHab(self):
        """
        Test :func:`colour.models.cie_lab.Lab_to_LCHab` definition
        n-dimensional arrays support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        LCHab = Lab_to_LCHab(Lab)

        Lab = np.tile(Lab, (6, 1))
        LCHab = np.tile(LCHab, (6, 1))
        np.testing.assert_allclose(
            Lab_to_LCHab(Lab), LCHab, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        Lab = np.reshape(Lab, (2, 3, 3))
        LCHab = np.reshape(LCHab, (2, 3, 3))
        np.testing.assert_allclose(
            Lab_to_LCHab(Lab), LCHab, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_Lab_to_LCHab(self):
        """
        Test :func:`colour.models.cie_lab.Lab_to_LCHab` definition domain and
        range scale support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        LCHab = Lab_to_LCHab(Lab)

        d_r = (
            ("reference", 1, 1),
            ("1", 0.01, np.array([0.01, 0.01, 1 / 360])),
            ("100", 1, np.array([1, 1, 1 / 3.6])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    Lab_to_LCHab(Lab * factor_a),
                    LCHab * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_Lab_to_LCHab(self):
        """
        Test :func:`colour.models.cie_lab.Lab_to_LCHab` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Lab_to_LCHab(cases)


class TestLCHab_to_Lab(unittest.TestCase):
    """
    Define :func:`colour.models.cie_lab.LCHab_to_Lab` definition unit tests
    methods.
    """

    def test_LCHab_to_Lab(self):
        """Test :func:`colour.models.cie_lab.LCHab_to_Lab` definition."""

        np.testing.assert_allclose(
            LCHab_to_Lab(np.array([41.52787529, 59.12425901, 27.08848784])),
            np.array([41.52787529, 52.63858304, 26.92317922]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            LCHab_to_Lab(np.array([55.11636304, 51.42135412, 143.03889556])),
            np.array([55.11636304, -41.08791787, 30.91825778]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            LCHab_to_Lab(np.array([29.80565520, 52.32945383, 292.49133666])),
            np.array([29.80565520, 20.01830466, -48.34913874]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_LCHab_to_Lab(self):
        """
        Test :func:`colour.models.cie_lab.LCHab_to_Lab` definition
        n-dimensional arrays support.
        """

        LCHab = np.array([41.52787529, 59.12425901, 27.08848784])
        Lab = LCHab_to_Lab(LCHab)

        LCHab = np.tile(LCHab, (6, 1))
        Lab = np.tile(Lab, (6, 1))
        np.testing.assert_allclose(
            LCHab_to_Lab(LCHab), Lab, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        LCHab = np.reshape(LCHab, (2, 3, 3))
        Lab = np.reshape(Lab, (2, 3, 3))
        np.testing.assert_allclose(
            LCHab_to_Lab(LCHab), Lab, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_LCHab_to_Lab(self):
        """
        Test :func:`colour.models.cie_lab.LCHab_to_Lab` definition domain and
        range scale support.
        """

        LCHab = np.array([41.52787529, 59.12425901, 27.08848784])
        Lab = LCHab_to_Lab(LCHab)

        d_r = (
            ("reference", 1, 1),
            ("1", np.array([0.01, 0.01, 1 / 360]), 0.01),
            ("100", np.array([1, 1, 1 / 3.6]), 1),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    LCHab_to_Lab(LCHab * factor_a),
                    Lab * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_LCHab_to_Lab(self):
        """
        Test :func:`colour.models.cie_lab.LCHab_to_Lab` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        LCHab_to_Lab(cases)


if __name__ == "__main__":
    unittest.main()
