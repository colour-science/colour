# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.cie_uvw` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import UVW_to_XYZ, XYZ_to_UVW
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_UVW', 'TestUVW_to_XYZ']


class TestXYZ_to_UVW(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_uvw.XYZ_to_UVW` definition unit tests
    methods.
    """

    def test_XYZ_to_UVW(self):
        """
        Tests :func:`colour.models.cie_uvw.XYZ_to_UVW` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([0.07049534, 0.10080000, 0.09558313]) * 100),
            np.array([-28.05797333, -0.88194493, 37.00411491]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([0.47097710, 0.34950000, 0.11301649]) * 100),
            np.array([85.91004857, 17.74103859, 64.73769793]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([0.25506814, 0.19150000, 0.08849752]) * 100),
            np.array([59.34788373, 8.59000007, 49.88513399]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                np.array([0.44757, 0.40745])),
            np.array([-50.56405108, -12.49600540, 37.00411491]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                np.array([0.31270, 0.32900])),
            np.array([-22.59840563, 5.45505477, 37.00411491]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                np.array([0.37208, 0.37529])),
            np.array([-33.35371445, -4.60753245, 37.00411491]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                np.array([0.37208, 0.37529, 0.10080])),
            np.array([-33.35371445, -4.60753245, 37.00411491]),
            decimal=7)

    def test_n_dimensional_XYZ_to_UVW(self):
        """
        Tests :func:`colour.models.cie_uvw.XYZ_to_UVW` definition n-dimensions
        support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
        illuminant = np.array([0.34570, 0.35850])
        UVW = np.array([-28.05797333, -0.88194493, 37.00411491])
        np.testing.assert_almost_equal(
            XYZ_to_UVW(XYZ, illuminant), UVW, decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        UVW = np.tile(UVW, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_UVW(XYZ, illuminant), UVW, decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_UVW(XYZ, illuminant), UVW, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        UVW = np.reshape(UVW, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_UVW(XYZ, illuminant), UVW, decimal=7)

    def test_domain_range_scale_XYZ_to_UVW(self):
        """
        Tests :func:`colour.models.cie_uvw.XYZ_to_UVW` definition domain and
        range scale support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
        illuminant = np.array([0.34570, 0.35850])
        UVW = XYZ_to_UVW(XYZ, illuminant)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_UVW(XYZ * factor, illuminant),
                    UVW * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_UVW(self):
        """
        Tests :func:`colour.models.cie_uvw.XYZ_to_UVW` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            illuminant = np.array(case[0:2])
            XYZ_to_UVW(XYZ, illuminant)


class TestUVW_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_uvw.UVW_to_XYZ` definition unit tests
    methods.
    """

    def test_UVW_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_uvw.UVW_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            UVW_to_XYZ(np.array([-28.05797333, -0.88194493, 37.00411491])),
            np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            UVW_to_XYZ(np.array([85.91004857, 17.74103859, 64.73769793])),
            np.array([0.47097710, 0.34950000, 0.11301649]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            UVW_to_XYZ(np.array([59.34788373, 8.59000007, 49.88513399])),
            np.array([0.25506814, 0.19150000, 0.08849752]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            UVW_to_XYZ(
                np.array([-50.56405108, -12.49600540, 37.00411491]),
                np.array([0.44757, 0.40745])),
            np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            UVW_to_XYZ(
                np.array([-22.59840563, 5.45505477, 37.00411491]),
                np.array([0.31270, 0.32900])),
            np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            UVW_to_XYZ(
                np.array([-33.35371445, -4.60753245, 37.00411491]),
                np.array([0.37208, 0.37529])),
            np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            UVW_to_XYZ(
                np.array([-33.35371445, -4.60753245, 37.00411491]),
                np.array([0.37208, 0.37529, 0.10080])),
            np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
            decimal=7)

    def test_n_dimensional_UVW_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_uvw.UVW_to_XYZ` definition n-dimensions
        support.
        """

        UVW = np.array([-28.05797333, -0.88194493, 37.00411491])
        illuminant = np.array([0.34570, 0.35850])
        XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
        np.testing.assert_almost_equal(
            UVW_to_XYZ(UVW, illuminant), XYZ, decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        UVW = np.tile(UVW, (6, 1))
        np.testing.assert_almost_equal(
            UVW_to_XYZ(UVW, illuminant), XYZ, decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            UVW_to_XYZ(UVW, illuminant), XYZ, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        UVW = np.reshape(UVW, (2, 3, 3))
        np.testing.assert_almost_equal(
            UVW_to_XYZ(UVW, illuminant), XYZ, decimal=7)

    def test_domain_range_scale_UVW_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_uvw.UVW_to_XYZ` definition domain and
        range scale support.
        """

        UVW = np.array([-28.05797333, -0.88194493, 37.00411491])
        illuminant = np.array([0.34570, 0.35850])
        XYZ = UVW_to_XYZ(UVW, illuminant)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    UVW_to_XYZ(UVW * factor, illuminant),
                    XYZ * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_UVW_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_uvw.UVW_to_XYZ` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            UVW = np.array(case)
            illuminant = np.array(case[0:2])
            UVW_to_XYZ(UVW, illuminant)


if __name__ == '__main__':
    unittest.main()
