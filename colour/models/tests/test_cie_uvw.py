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
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
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
            XYZ_to_UVW(np.array([0.20654008, 0.12197225, 0.05136952]) * 100),
            np.array([94.55035725, 11.55536523, 40.54757405]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([0.14222010, 0.23042768, 0.10495772]) * 100),
            np.array([-36.92762376, 28.90425105, 54.14071478]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(np.array([0.07818780, 0.06157201, 0.28099326]) * 100),
            np.array([-10.60111550, -41.94580000, 28.82134002]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
                np.array([0.44757, 0.40745])),
            np.array([63.90676310, -8.11466183, 40.54757405]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
                np.array([0.34570, 0.35850])),
            np.array([88.56798946, 4.61154385, 40.54757405]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_UVW(
                np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
                np.array([0.34570, 0.35850, 1.00000])),
            np.array([88.56798946, 4.61154385, 40.54757405]),
            decimal=7)

    def test_n_dimensional_XYZ_to_UVW(self):
        """
        Tests :func:`colour.models.cie_uvw.XYZ_to_UVW` definition n-dimensional
        support.
        """

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
        illuminant = np.array([0.31270, 0.32900])
        UVW = XYZ_to_UVW(XYZ, illuminant)

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

        XYZ = np.array([0.20654008, 0.12197225, 0.05136952]) * 100
        illuminant = np.array([0.31270, 0.32900])
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
            UVW_to_XYZ(np.array([94.55035725, 11.55536523, 40.54757405])),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            UVW_to_XYZ(np.array([-36.92762376, 28.90425105, 54.14071478])),
            np.array([0.14222010, 0.23042768, 0.10495772]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            UVW_to_XYZ(np.array([-10.60111550, -41.94580000, 28.82134002])),
            np.array([0.07818780, 0.06157201, 0.28099326]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            UVW_to_XYZ(
                np.array([63.90676310, -8.11466183, 40.54757405]),
                np.array([0.44757, 0.40745])),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            UVW_to_XYZ(
                np.array([88.56798946, 4.61154385, 40.54757405]),
                np.array([0.34570, 0.35850])),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            decimal=7)

        np.testing.assert_almost_equal(
            UVW_to_XYZ(
                np.array([88.56798946, 4.61154385, 40.54757405]),
                np.array([0.34570, 0.35850, 1.00000])),
            np.array([0.20654008, 0.12197225, 0.05136952]) * 100,
            decimal=7)

    def test_n_dimensional_UVW_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_uvw.UVW_to_XYZ` definition n-dimensional
        support.
        """

        UVW = np.array([94.55035725, 11.55536523, 40.54757405])
        illuminant = np.array([0.31270, 0.32900])
        XYZ = UVW_to_XYZ(UVW, illuminant)

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

        UVW = np.array([94.55035725, 11.55536523, 40.54757405])
        illuminant = np.array([0.31270, 0.32900])
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
