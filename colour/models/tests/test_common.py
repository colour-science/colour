# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.common` module."""

import numpy as np
import unittest
from itertools import product

from colour.models import Iab_to_XYZ, Jab_to_JCh, JCh_to_Jab, XYZ_to_Iab
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestJab_to_JCh",
    "TestJCh_to_Jab",
    "TestXYZ_to_Iab",
    "TestIab_to_XYZ",
]


class TestJab_to_JCh(unittest.TestCase):
    """
    Define :func:`colour.models.common.Jab_to_JCh` definition unit tests
    methods.
    """

    def test_Jab_to_JCh(self):
        """Test :func:`colour.models.common.Jab_to_JCh` definition."""

        np.testing.assert_array_almost_equal(
            Jab_to_JCh(np.array([41.52787529, 52.63858304, 26.92317922])),
            np.array([41.52787529, 59.12425901, 27.08848784]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            Jab_to_JCh(np.array([55.11636304, -41.08791787, 30.91825778])),
            np.array([55.11636304, 51.42135412, 143.03889556]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            Jab_to_JCh(np.array([29.80565520, 20.01830466, -48.34913874])),
            np.array([29.80565520, 52.32945383, 292.49133666]),
            decimal=7,
        )

    def test_n_dimensional_Jab_to_JCh(self):
        """
        Test :func:`colour.models.common.Jab_to_JCh` definition n-dimensional
        arrays support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        LCHab = Jab_to_JCh(Lab)

        Lab = np.tile(Lab, (6, 1))
        LCHab = np.tile(LCHab, (6, 1))
        np.testing.assert_array_almost_equal(Jab_to_JCh(Lab), LCHab, decimal=7)

        Lab = np.reshape(Lab, (2, 3, 3))
        LCHab = np.reshape(LCHab, (2, 3, 3))
        np.testing.assert_array_almost_equal(Jab_to_JCh(Lab), LCHab, decimal=7)

    def test_domain_range_scale_Jab_to_JCh(self):
        """
        Test :func:`colour.models.common.Jab_to_JCh` definition domain and
        range scale support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        LCHab = Jab_to_JCh(Lab)

        d_r = (
            ("reference", 1, 1),
            ("1", 0.01, np.array([0.01, 0.01, 1 / 360])),
            ("100", 1, np.array([1, 1, 1 / 3.6])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    Jab_to_JCh(Lab * factor_a), LCHab * factor_b, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_Jab_to_JCh(self):
        """Test :func:`colour.models.common.Jab_to_JCh` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Jab_to_JCh(cases)


class TestJCh_to_Jab(unittest.TestCase):
    """
    Define :func:`colour.models.common.JCh_to_Jab` definition unit tests
    methods.
    """

    def test_JCh_to_Jab(self):
        """Test :func:`colour.models.common.JCh_to_Jab` definition."""

        np.testing.assert_array_almost_equal(
            JCh_to_Jab(np.array([41.52787529, 59.12425901, 27.08848784])),
            np.array([41.52787529, 52.63858304, 26.92317922]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            JCh_to_Jab(np.array([55.11636304, 51.42135412, 143.03889556])),
            np.array([55.11636304, -41.08791787, 30.91825778]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            JCh_to_Jab(np.array([29.80565520, 52.32945383, 292.49133666])),
            np.array([29.80565520, 20.01830466, -48.34913874]),
            decimal=7,
        )

    def test_n_dimensional_JCh_to_Jab(self):
        """
        Test :func:`colour.models.common.JCh_to_Jab` definition n-dimensional
        arrays support.
        """

        LCHab = np.array([41.52787529, 59.12425901, 27.08848784])
        Lab = JCh_to_Jab(LCHab)

        LCHab = np.tile(LCHab, (6, 1))
        Lab = np.tile(Lab, (6, 1))
        np.testing.assert_array_almost_equal(JCh_to_Jab(LCHab), Lab, decimal=7)

        LCHab = np.reshape(LCHab, (2, 3, 3))
        Lab = np.reshape(Lab, (2, 3, 3))
        np.testing.assert_array_almost_equal(JCh_to_Jab(LCHab), Lab, decimal=7)

    def test_domain_range_scale_JCh_to_Jab(self):
        """
        Test :func:`colour.models.common.JCh_to_Jab` definition domain and
        range scale support.
        """

        LCHab = np.array([41.52787529, 59.12425901, 27.08848784])
        Lab = JCh_to_Jab(LCHab)

        d_r = (
            ("reference", 1, 1),
            ("1", np.array([0.01, 0.01, 1 / 360]), 0.01),
            ("100", np.array([1, 1, 1 / 3.6]), 1),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    JCh_to_Jab(LCHab * factor_a), Lab * factor_b, decimal=7
                )

    @ignore_numpy_errors
    def test_nan_JCh_to_Jab(self):
        """Test :func:`colour.models.common.JCh_to_Jab` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        JCh_to_Jab(cases)


class TestXYZ_to_Iab(unittest.TestCase):
    """Define :func:`colour.models.common.XYZ_to_Iab` definition unit tests methods."""

    def setUp(self):
        """Initialise the common tests attributes."""

        self.LMS_to_LMS_p = lambda x: x**0.43
        self.M_XYZ_to_LMS = np.array(
            [
                [0.4002, 0.7075, -0.0807],
                [-0.2280, 1.1500, 0.0612],
                [0.0000, 0.0000, 0.9184],
            ]
        )
        self.M_LMS_p_to_Iab = np.array(
            [
                [0.4000, 0.4000, 0.2000],
                [4.4550, -4.8510, 0.3960],
                [0.8056, 0.3572, -1.1628],
            ]
        )

    def test_XYZ_to_Iab(self):
        """Test :func:`colour.models.common.XYZ_to_Iab` definition."""

        np.testing.assert_array_almost_equal(
            XYZ_to_Iab(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                self.LMS_to_LMS_p,
                self.M_XYZ_to_LMS,
                self.M_LMS_p_to_Iab,
            ),
            np.array([0.38426191, 0.38487306, 0.18886838]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            XYZ_to_Iab(
                np.array([0.14222010, 0.23042768, 0.10495772]),
                self.LMS_to_LMS_p,
                self.M_XYZ_to_LMS,
                self.M_LMS_p_to_Iab,
            ),
            np.array([0.49437481, -0.19251742, 0.18080304]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            XYZ_to_Iab(
                np.array([0.07818780, 0.06157201, 0.28099326]),
                self.LMS_to_LMS_p,
                self.M_XYZ_to_LMS,
                self.M_LMS_p_to_Iab,
            ),
            np.array([0.35167774, -0.07525627, -0.30921279]),
            decimal=7,
        )

    def test_n_dimensional_XYZ_to_Iab(self):
        """
        Test :func:`colour.models.common.XYZ_to_Iab` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Iab = XYZ_to_Iab(
            XYZ, self.LMS_to_LMS_p, self.M_XYZ_to_LMS, self.M_LMS_p_to_Iab
        )

        XYZ = np.tile(XYZ, (6, 1))
        Iab = np.tile(Iab, (6, 1))
        np.testing.assert_array_almost_equal(
            XYZ_to_Iab(
                XYZ, self.LMS_to_LMS_p, self.M_XYZ_to_LMS, self.M_LMS_p_to_Iab
            ),
            Iab,
            decimal=7,
        )

        XYZ = np.reshape(XYZ, (2, 3, 3))
        Iab = np.reshape(Iab, (2, 3, 3))
        np.testing.assert_array_almost_equal(
            XYZ_to_Iab(
                XYZ, self.LMS_to_LMS_p, self.M_XYZ_to_LMS, self.M_LMS_p_to_Iab
            ),
            Iab,
            decimal=7,
        )

    def test_domain_range_scale_XYZ_to_Iab(self):
        """
        Test :func:`colour.models.common.XYZ_to_Iab` definition domain and
        range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        Iab = XYZ_to_Iab(
            XYZ, self.LMS_to_LMS_p, self.M_XYZ_to_LMS, self.M_LMS_p_to_Iab
        )

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    XYZ_to_Iab(
                        XYZ * factor,
                        self.LMS_to_LMS_p,
                        self.M_XYZ_to_LMS,
                        self.M_LMS_p_to_Iab,
                    ),
                    Iab * factor,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_nan_XYZ_to_Iab(self):
        """Test :func:`colour.models.common.XYZ_to_Iab` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        XYZ_to_Iab(
            cases, self.LMS_to_LMS_p, self.M_XYZ_to_LMS, self.M_LMS_p_to_Iab
        )


class TestIab_to_XYZ(unittest.TestCase):
    """
    Define :func:`colour.models.common.Iab_to_XYZ` definition unit tests
    methods.
    """

    def setUp(self):
        """Initialise the common tests attributes."""

        self.LMS_p_to_LMS = lambda x: x ** (1 / 0.43)
        self.M_Iab_to_LMS_p = np.linalg.inv(
            np.array(
                [
                    [0.4000, 0.4000, 0.2000],
                    [4.4550, -4.8510, 0.3960],
                    [0.8056, 0.3572, -1.1628],
                ]
            )
        )
        self.M_LMS_to_XYZ = np.linalg.inv(
            np.array(
                [
                    [0.4002, 0.7075, -0.0807],
                    [-0.2280, 1.1500, 0.0612],
                    [0.0000, 0.0000, 0.9184],
                ]
            )
        )

    def test_Iab_to_XYZ(self):
        """Test :func:`colour.models.common.Iab_to_XYZ` definition."""

        np.testing.assert_array_almost_equal(
            Iab_to_XYZ(
                np.array([0.38426191, 0.38487306, 0.18886838]),
                self.LMS_p_to_LMS,
                self.M_Iab_to_LMS_p,
                self.M_LMS_to_XYZ,
            ),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            Iab_to_XYZ(
                np.array([0.49437481, -0.19251742, 0.18080304]),
                self.LMS_p_to_LMS,
                self.M_Iab_to_LMS_p,
                self.M_LMS_to_XYZ,
            ),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            decimal=7,
        )

        np.testing.assert_array_almost_equal(
            Iab_to_XYZ(
                np.array([0.35167774, -0.07525627, -0.30921279]),
                self.LMS_p_to_LMS,
                self.M_Iab_to_LMS_p,
                self.M_LMS_to_XYZ,
            ),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            decimal=7,
        )

    def test_n_dimensional_Iab_to_XYZ(self):
        """
        Test :func:`colour.models.common.Iab_to_XYZ` definition n-dimensional
        support.
        """

        Iab = np.array([0.38426191, 0.38487306, 0.18886838])
        XYZ = Iab_to_XYZ(
            Iab, self.LMS_p_to_LMS, self.M_Iab_to_LMS_p, self.M_LMS_to_XYZ
        )

        Iab = np.tile(Iab, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_array_almost_equal(
            Iab_to_XYZ(
                Iab, self.LMS_p_to_LMS, self.M_Iab_to_LMS_p, self.M_LMS_to_XYZ
            ),
            XYZ,
            decimal=7,
        )

        Iab = np.reshape(Iab, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_array_almost_equal(
            Iab_to_XYZ(
                Iab, self.LMS_p_to_LMS, self.M_Iab_to_LMS_p, self.M_LMS_to_XYZ
            ),
            XYZ,
            decimal=7,
        )

    def test_domain_range_scale_Iab_to_XYZ(self):
        """
        Test :func:`colour.models.common.Iab_to_XYZ` definition domain and
        range scale support.
        """

        Iab = np.array([0.38426191, 0.38487306, 0.18886838])
        XYZ = Iab_to_XYZ(
            Iab, self.LMS_p_to_LMS, self.M_Iab_to_LMS_p, self.M_LMS_to_XYZ
        )

        d_r = (("reference", 1), ("1", 1), ("100", 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_array_almost_equal(
                    Iab_to_XYZ(
                        Iab * factor,
                        self.LMS_p_to_LMS,
                        self.M_Iab_to_LMS_p,
                        self.M_LMS_to_XYZ,
                    ),
                    XYZ * factor,
                    decimal=7,
                )

    @ignore_numpy_errors
    def test_nan_Iab_to_XYZ(self):
        """Test :func:`colour.models.common.Iab_to_XYZ` definition nan support."""

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        Iab_to_XYZ(
            cases, self.LMS_p_to_LMS, self.M_Iab_to_LMS_p, self.M_LMS_to_XYZ
        )


if __name__ == "__main__":
    unittest.main()
