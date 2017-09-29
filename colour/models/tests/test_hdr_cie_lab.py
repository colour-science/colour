#!/usr/bin/env python
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
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestXYZ_to_hdr_CIELab', 'TestHdr_CIELab_to_XYZ', 'TestExponent_hdr_CIELab'
]


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
            XYZ_to_hdr_CIELab(np.array([0.07049534, 0.10080000, 0.09558313])),
            np.array([24.90206646, -46.83127607, -10.14274843]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_hdr_CIELab(
                np.array([0.07049534, 0.10080000, 0.09558313]),
                np.array([0.44757, 0.40745])),
            np.array([24.90206646, -61.24983919, -83.63902870]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_hdr_CIELab(
                np.array([0.07049534, 0.10080000, 0.09558313]), Y_s=0.5),
            np.array([34.44227938, -36.51485775, -6.87279617]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_hdr_CIELab(
                np.array([0.07049534, 0.10080000, 0.09558313]), Y_abs=1000),
            np.array([32.39463250, -39.77445283, -7.66690737]),
            decimal=7)

    def test_n_dimensional_XYZ_to_hdr_CIELab(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.XYZ_to_hdr_CIELab` definition
        n-dimensions support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        illuminant = np.array([0.34570, 0.35850])
        Y_s = 0.2
        Y_abs = 100
        Lab_hdr = np.array([24.90206646, -46.83127607, -10.14274843])
        np.testing.assert_almost_equal(
            XYZ_to_hdr_CIELab(XYZ, illuminant, Y_s, Y_abs), Lab_hdr, decimal=7)

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
                np.array([24.90206646, -46.83127607, -10.14274843])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            hdr_CIELab_to_XYZ(
                np.array([24.90206646, -61.24983919, -83.63902870]),
                np.array([0.44757, 0.40745])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            hdr_CIELab_to_XYZ(
                np.array([34.44227938, -36.51485775, -6.87279617]), Y_s=0.5),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            hdr_CIELab_to_XYZ(
                np.array([32.39463250, -39.77445283, -7.66690737]),
                Y_abs=1000),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

    def test_n_dimensional_hdr_CIELab_to_XYZ(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.hdr_CIELab_to_XYZ` definition
        n-dimensions support.
        """

        Lab_hdr = np.array([24.90206646, -46.83127607, -10.14274843])
        illuminant = np.array([0.34570, 0.35850])
        Y_s = 0.2
        Y_abs = 100
        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        np.testing.assert_almost_equal(
            hdr_CIELab_to_XYZ(Lab_hdr, illuminant, Y_s, Y_abs), XYZ, decimal=7)

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
            exponent_hdr_CIELab(0.2, 100), 1.836019897814665, places=7)

        self.assertAlmostEqual(
            exponent_hdr_CIELab(0.4, 100), 1.326014370643925, places=7)

        self.assertAlmostEqual(
            exponent_hdr_CIELab(0.2, 1000), 1.224013265209777, places=7)

    def test_n_dimensional_exponent_hdr_CIELab(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.exponent_hdr_CIELab`
        definition n-dimensional arrays support.
        """

        Y_s = 0.2
        Y_abs = 100
        e = 1.836019897814665
        np.testing.assert_almost_equal(
            exponent_hdr_CIELab(Y_s, Y_abs), e, decimal=7)

        Y_s = np.tile(Y_s, 6)
        Y_abs = np.tile(Y_abs, 6)
        e = np.tile(e, 6)
        np.testing.assert_almost_equal(
            exponent_hdr_CIELab(Y_s, Y_abs), e, decimal=7)

        Y_s = np.reshape(Y_s, (2, 3))
        Y_abs = np.reshape(Y_abs, (2, 3))
        e = np.reshape(e, (2, 3))
        np.testing.assert_almost_equal(
            exponent_hdr_CIELab(Y_s, Y_abs), e, decimal=7)

        Y_s = np.reshape(Y_s, (2, 3, 1))
        Y_abs = np.reshape(Y_abs, (2, 3, 1))
        e = np.reshape(e, (2, 3, 1))
        np.testing.assert_almost_equal(
            exponent_hdr_CIELab(Y_s, Y_abs), e, decimal=7)

    @ignore_numpy_errors
    def test_nan_exponent_hdr_CIELab(self):
        """
        Tests :func:`colour.models.hdr_cie_lab.exponent_hdr_CIELab`
        definition nan support.
        """

        cases = np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        exponent_hdr_CIELab(cases, cases)


if __name__ == '__main__':
    unittest.main()
