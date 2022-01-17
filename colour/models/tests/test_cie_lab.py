# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.cie_lab` module.
"""

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_Lab, Lab_to_XYZ, Lab_to_LCHab, LCHab_to_Lab
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_Lab',
    'TestLab_to_XYZ',
    'TestLab_to_LCHab',
    'TestLCHab_to_Lab',
]


class TestXYZ_to_Lab(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_lab.XYZ_to_Lab` definition unit tests
    methods.
    """

    def test_XYZ_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.XYZ_to_Lab` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([41.52787529, 52.63858304, 26.92317922]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([55.11636304, -41.08791787, 30.91825778]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([29.80565520, 20.01830466, -48.34913874]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.44757, 0.40745])),
            np.array([41.52787529, 38.48089305, -5.73295122]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.34570, 0.35850])),
            np.array([41.52787529, 51.19354174, 19.91843098]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.34570, 0.35850, 1.00000])),
            np.array([41.52787529, 51.19354174, 19.91843098]),
            decimal=7)

    def test_n_dimensional_XYZ_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.XYZ_to_Lab` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
        Lab = XYZ_to_Lab(XYZ, illuminant)

        XYZ = np.tile(XYZ, (6, 1))
        Lab = np.tile(Lab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Lab(XYZ, illuminant), Lab, decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Lab(XYZ, illuminant), Lab, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        Lab = np.reshape(Lab, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_Lab(XYZ, illuminant), Lab, decimal=7)

    def test_domain_range_scale_XYZ_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.XYZ_to_Lab` definition
        domain and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
        Lab = XYZ_to_Lab(XYZ, illuminant)

        d_r = (('reference', 1, 1), ('1', 1, 0.01), ('100', 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_Lab(XYZ * factor_a, illuminant),
                    Lab * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.XYZ_to_Lab` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            illuminant = np.array(case[0:2])
            XYZ_to_Lab(XYZ, illuminant)


class TestLab_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_lab.Lab_to_XYZ` definition unit tests
    methods.
    """

    def test_Lab_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([41.52787529, 52.63858304, 26.92317922])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([55.11636304, -41.08791787, 30.91825778])),
            np.array([0.14222010, 0.23042768, 0.10495772]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([29.80565520, 20.01830466, -48.34913874])),
            np.array([0.07818780, 0.06157201, 0.28099326]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(
                np.array([41.52787529, 38.48089305, -5.73295122]),
                np.array([0.44757, 0.40745])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(
                np.array([41.52787529, 51.19354174, 19.91843098]),
                np.array([0.34570, 0.35850])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(
                np.array([41.52787529, 51.19354174, 19.91843098]),
                np.array([0.34570, 0.35850, 1.00000])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

    def test_n_dimensional_Lab_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_XYZ` definition n-dimensional
        support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        illuminant = np.array([0.31270, 0.32900])
        XYZ = Lab_to_XYZ(Lab, illuminant)

        Lab = np.tile(Lab, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            Lab_to_XYZ(Lab, illuminant), XYZ, decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            Lab_to_XYZ(Lab, illuminant), XYZ, decimal=7)

        Lab = np.reshape(Lab, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            Lab_to_XYZ(Lab, illuminant), XYZ, decimal=7)

    def test_domain_range_scale_Lab_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_XYZ` definition
        domain and range scale support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        illuminant = np.array([0.31270, 0.32900])
        XYZ = Lab_to_XYZ(Lab, illuminant)

        d_r = (('reference', 1, 1), ('1', 0.01, 1), ('100', 1, 100))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    Lab_to_XYZ(Lab * factor_a, illuminant),
                    XYZ * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_Lab_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_XYZ` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab = np.array(case)
            illuminant = np.array(case[0:2])
            Lab_to_XYZ(Lab, illuminant)


class TestLab_to_LCHab(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_lab.Lab_to_LCHab` definition unit tests
    methods.
    """

    def test_Lab_to_LCHab(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_LCHab` definition.
        """

        np.testing.assert_almost_equal(
            Lab_to_LCHab(np.array([41.52787529, 52.63858304, 26.92317922])),
            np.array([41.52787529, 59.12425901, 27.08848784]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_LCHab(np.array([55.11636304, -41.08791787, 30.91825778])),
            np.array([55.11636304, 51.42135412, 143.03889556]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_LCHab(np.array([29.80565520, 20.01830466, -48.34913874])),
            np.array([29.80565520, 52.32945383, 292.49133666]),
            decimal=7)

    def test_n_dimensional_Lab_to_LCHab(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_LCHab` definition
        n-dimensional arrays support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        LCHab = Lab_to_LCHab(Lab)

        Lab = np.tile(Lab, (6, 1))
        LCHab = np.tile(LCHab, (6, 1))
        np.testing.assert_almost_equal(Lab_to_LCHab(Lab), LCHab, decimal=7)

        Lab = np.reshape(Lab, (2, 3, 3))
        LCHab = np.reshape(LCHab, (2, 3, 3))
        np.testing.assert_almost_equal(Lab_to_LCHab(Lab), LCHab, decimal=7)

    def test_domain_range_scale_Lab_to_LCHab(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_LCHab` definition domain and
        range scale support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        LCHab = Lab_to_LCHab(Lab)

        d_r = (
            ('reference', 1, 1),
            ('1', 0.01, np.array([0.01, 0.01, 1 / 360])),
            ('100', 1, np.array([1, 1, 1 / 3.6])),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    Lab_to_LCHab(Lab * factor_a), LCHab * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_Lab_to_LCHab(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_LCHab` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab = np.array(case)
            Lab_to_LCHab(Lab)


class TestLCHab_to_Lab(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_lab.LCHab_to_Lab` definition unit tests
    methods.
    """

    def test_LCHab_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.LCHab_to_Lab` definition.
        """

        np.testing.assert_almost_equal(
            LCHab_to_Lab(np.array([41.52787529, 59.12425901, 27.08848784])),
            np.array([41.52787529, 52.63858304, 26.92317922]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHab_to_Lab(np.array([55.11636304, 51.42135412, 143.03889556])),
            np.array([55.11636304, -41.08791787, 30.91825778]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHab_to_Lab(np.array([29.80565520, 52.32945383, 292.49133666])),
            np.array([29.80565520, 20.01830466, -48.34913874]),
            decimal=7)

    def test_n_dimensional_LCHab_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.LCHab_to_Lab` definition
        n-dimensional arrays support.
        """

        LCHab = np.array([41.52787529, 59.12425901, 27.08848784])
        Lab = LCHab_to_Lab(LCHab)

        LCHab = np.tile(LCHab, (6, 1))
        Lab = np.tile(Lab, (6, 1))
        np.testing.assert_almost_equal(LCHab_to_Lab(LCHab), Lab, decimal=7)

        LCHab = np.reshape(LCHab, (2, 3, 3))
        Lab = np.reshape(Lab, (2, 3, 3))
        np.testing.assert_almost_equal(LCHab_to_Lab(LCHab), Lab, decimal=7)

    def test_domain_range_scale_LCHab_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.LCHab_to_Lab` definition domain and
        range scale support.
        """

        LCHab = np.array([41.52787529, 59.12425901, 27.08848784])
        Lab = LCHab_to_Lab(LCHab)

        d_r = (
            ('reference', 1, 1),
            ('1', np.array([0.01, 0.01, 1 / 360]), 0.01),
            ('100', np.array([1, 1, 1 / 3.6]), 1),
        )
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    LCHab_to_Lab(LCHab * factor_a), Lab * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_LCHab_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.LCHab_to_Lab` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            LCHab = np.array(case)
            LCHab_to_Lab(LCHab)


if __name__ == '__main__':
    unittest.main()
