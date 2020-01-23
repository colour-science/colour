# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.hdr_cie_lab` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_hdr_CIELab, hdr_CIELab_to_XYZ
from colour.models.hdr_cie_lab import exponent_hdr_CIELab
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestExponent_hdr_CIELab', 'TestXYZ_to_hdr_CIELab', 'TestHdr_CIELab_to_XYZ'
]


class TestExponent_hdr_CIELab(unittest.TestCase):
    """
    Defines :func:`colour.models.hdr_cie_lab.exponent_hdr_CIELab`
    definition unit tests methods.
    """

    def test_exponent_hdr_CIELab(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.exponent_hdr_CIELab`
        definition.
        """

        self.assertAlmostEqual(
            exponent_hdr_CIELab(0.2, 100), 0.473851073746817, places=7)

        self.assertAlmostEqual(
            exponent_hdr_CIELab(0.4, 100), 0.656101486726362, places=7)

        self.assertAlmostEqual(
            exponent_hdr_CIELab(0.4, 100, method='Fairchild 2010'),
            1.326014370643925,
            places=7)

        self.assertAlmostEqual(
            exponent_hdr_CIELab(0.2, 1000), 0.710776610620225, places=7)

    def test_n_dimensional_exponent_hdr_CIELab(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.exponent_hdr_CIELab`
        definition n-dimensional arrays support.
        """

        Y_s = 0.2
        Y_abs = 100
        epsilon = exponent_hdr_CIELab(Y_s, Y_abs)

        Y_s = np.tile(Y_s, 6)
        Y_abs = np.tile(Y_abs, 6)
        epsilon = np.tile(epsilon, 6)
        np.testing.assert_almost_equal(
            exponent_hdr_CIELab(Y_s, Y_abs), epsilon, decimal=7)

        Y_s = np.reshape(Y_s, (2, 3))
        Y_abs = np.reshape(Y_abs, (2, 3))
        epsilon = np.reshape(epsilon, (2, 3))
        np.testing.assert_almost_equal(
            exponent_hdr_CIELab(Y_s, Y_abs), epsilon, decimal=7)

        Y_s = np.reshape(Y_s, (2, 3, 1))
        Y_abs = np.reshape(Y_abs, (2, 3, 1))
        epsilon = np.reshape(epsilon, (2, 3, 1))
        np.testing.assert_almost_equal(
            exponent_hdr_CIELab(Y_s, Y_abs), epsilon, decimal=7)

    def test_domain_range_scale_exponent_hdr_CIELab(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.exponent_hdr_CIELab` definition
        domain and range scale support.
        """

        Y_s = 0.2
        Y_abs = 100
        epsilon = exponent_hdr_CIELab(Y_s, Y_abs)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    exponent_hdr_CIELab(Y_s * factor, Y_abs),
                    epsilon,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_exponent_hdr_CIELab(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.exponent_hdr_CIELab`
        definition nan support.
        """

        cases = np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        exponent_hdr_CIELab(cases, cases)


class TestXYZ_to_hdr_CIELab(unittest.TestCase):
    """
    Defines :func:`colour.models.hdr_cie_lab.XYZ_to_hdr_CIELab` definition unit
    tests methods.
    """

    def test_XYZ_to_hdr_CIELab(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.XYZ_to_hdr_CIELab` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_hdr_CIELab(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([51.87002062, 60.47633850, 32.14551912]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_hdr_CIELab(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.44757, 0.40745])),
            np.array([51.87002062, 44.49667330, -6.69619196]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_hdr_CIELab(
                np.array([0.20654008, 0.12197225, 0.05136952]),
                np.array([0.44757, 0.40745]),
                method='Fairchild 2010'),
            np.array([31.99621114, 95.08564341, -14.14047055]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_hdr_CIELab(
                np.array([0.20654008, 0.12197225, 0.05136952]), Y_s=0.5),
            np.array([23.10388654, 59.31425004, 23.69960142]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_hdr_CIELab(
                np.array([0.20654008, 0.12197225, 0.05136952]), Y_abs=1000),
            np.array([29.77261805, 62.58315675, 27.31232673]),
            decimal=7)

    def test_n_dimensional_XYZ_to_hdr_CIELab(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.XYZ_to_hdr_CIELab` definition
        n-dimensional support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
        Y_s = 0.2
        Y_abs = 100
        Lab_hdr = XYZ_to_hdr_CIELab(XYZ, illuminant, Y_s, Y_abs)

        XYZ = np.tile(XYZ, (6, 1))
        Lab_hdr = np.tile(Lab_hdr, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_hdr_CIELab(XYZ, illuminant, Y_s, Y_abs), Lab_hdr, decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        Y_s = np.tile(Y_s, 6)
        Y_abs = np.tile(Y_abs, 6)
        np.testing.assert_almost_equal(
            XYZ_to_hdr_CIELab(XYZ, illuminant, Y_s, Y_abs), Lab_hdr, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        Y_s = np.reshape(Y_s, (2, 3))
        Y_abs = np.reshape(Y_abs, (2, 3))
        Lab_hdr = np.reshape(Lab_hdr, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_hdr_CIELab(XYZ, illuminant, Y_s, Y_abs), Lab_hdr, decimal=7)

    def test_domain_range_scale_XYZ_to_hdr_CIELab(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.XYZ_to_hdr_CIELab` definition
        domain and range scale support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952])
        illuminant = np.array([0.31270, 0.32900])
        Y_s = 0.2
        Y_abs = 100
        Lab_hdr = XYZ_to_hdr_CIELab(XYZ, illuminant, Y_s, Y_abs)

        d_r = (('reference', 1, 1), (1, 1, 0.01), (100, 100, 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_hdr_CIELab(XYZ * factor_a, illuminant,
                                      Y_s * factor_a, Y_abs),
                    Lab_hdr * factor_b,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_hdr_CIELab(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.XYZ_to_hdr_CIELab` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            illuminant = np.array(case[0:2])
            Y_s = case[0]
            Y_abs = case[0]
            XYZ_to_hdr_CIELab(XYZ, illuminant, Y_s, Y_abs)


class TestHdr_CIELab_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.hdr_cie_lab.hdr_CIELab_to_XYZ` definition unit
    tests methods.
    """

    def test_hdr_CIELab_to_XYZ(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.hdr_CIELab_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            hdr_CIELab_to_XYZ(
                np.array([51.87002062, 60.47633850, 32.14551912])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            hdr_CIELab_to_XYZ(
                np.array([51.87002062, 44.49667330, -6.69619196]),
                np.array([0.44757, 0.40745])),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            hdr_CIELab_to_XYZ(
                np.array([31.99621114, 95.08564341, -14.14047055]),
                np.array([0.44757, 0.40745]),
                method='Fairchild 2010'),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            hdr_CIELab_to_XYZ(
                np.array([23.10388654, 59.31425004, 23.69960142]), Y_s=0.5),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

        np.testing.assert_almost_equal(
            hdr_CIELab_to_XYZ(
                np.array([29.77261805, 62.58315675, 27.31232673]), Y_abs=1000),
            np.array([0.20654008, 0.12197225, 0.05136952]),
            decimal=7)

    def test_n_dimensional_hdr_CIELab_to_XYZ(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.hdr_CIELab_to_XYZ` definition
        n-dimensional support.
        """

        Lab_hdr = np.array([51.87002062, 60.47633850, 32.14551912])
        illuminant = np.array([0.31270, 0.32900])
        Y_s = 0.2
        Y_abs = 100
        XYZ = hdr_CIELab_to_XYZ(Lab_hdr, illuminant, Y_s, Y_abs)

        Lab_hdr = np.tile(Lab_hdr, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            hdr_CIELab_to_XYZ(Lab_hdr, illuminant, Y_s, Y_abs), XYZ, decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        Y_s = np.tile(Y_s, 6)
        Y_abs = np.tile(Y_abs, 6)
        np.testing.assert_almost_equal(
            hdr_CIELab_to_XYZ(Lab_hdr, illuminant, Y_s, Y_abs), XYZ, decimal=7)

        Lab_hdr = np.reshape(Lab_hdr, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        Y_s = np.reshape(Y_s, (2, 3))
        Y_abs = np.reshape(Y_abs, (2, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            hdr_CIELab_to_XYZ(Lab_hdr, illuminant, Y_s, Y_abs), XYZ, decimal=7)

    def test_domain_range_scale_hdr_CIELab_to_XYZ(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.hdr_CIELab_to_XYZ` definition
        domain and range scale support.
        """

        Lab_hdr = np.array([26.46461067, -24.61332600, -4.84796811])
        illuminant = np.array([0.31270, 0.32900])
        Y_s = 0.2
        Y_abs = 100
        XYZ = hdr_CIELab_to_XYZ(Lab_hdr, illuminant, Y_s, Y_abs)

        d_r = (('reference', 1, 1, 1), (1, 0.01, 1, 1), (100, 1, 100, 100))
        for scale, factor_a, factor_b, factor_c in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    hdr_CIELab_to_XYZ(Lab_hdr * factor_a, illuminant,
                                      Y_s * factor_b, Y_abs),
                    XYZ * factor_c,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_hdr_CIELab_to_XYZ(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.hdr_CIELab_to_XYZ` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab_hdr = np.array(case)
            illuminant = np.array(case[0:2])
            Y_s = case[0]
            Y_abs = case[0]
            hdr_CIELab_to_XYZ(Lab_hdr, illuminant, Y_s, Y_abs)


if __name__ == '__main__':
    unittest.main()
