# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.osa_ucs` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_OSA_UCS, OSA_UCS_to_XYZ
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_OSA_UCS', 'TestOSA_UCS_to_XYZ']


class TestXYZ_to_OSA_UCS(unittest.TestCase):
    """
    Defines :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition unit tests
    methods.
    """

    def test_XYZ_to_OSA_UCS(self):
        """
        Tests :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_OSA_UCS(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100),
            np.array([-4.49006830, 0.70305936, 3.03463664]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_OSA_UCS(
                np.array([0.47097710, 0.34950000, 0.11301649]) * 100),
            np.array([1.45512585, 6.57345931, -8.02280578]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_OSA_UCS(
                np.array([0.25506814, 0.19150000, 0.08849752]) * 100),
            np.array([-1.88009027, 3.71899342, -5.98296399]),
            decimal=7)

    def test_n_dimensional_XYZ_to_OSA_UCS(self):
        """
        Tests :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition
        n-dimensions support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
        Ljg = np.array([-4.49006830, 0.70305936, 3.03463664])
        np.testing.assert_almost_equal(XYZ_to_OSA_UCS(XYZ), Ljg, decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        Ljg = np.tile(Ljg, (6, 1))
        np.testing.assert_almost_equal(XYZ_to_OSA_UCS(XYZ), Ljg, decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        Ljg = np.reshape(Ljg, (2, 3, 3))
        np.testing.assert_almost_equal(XYZ_to_OSA_UCS(XYZ), Ljg, decimal=7)

    def test_domain_range_scale_XYZ_to_OSA_UCS(self):
        """
        Tests :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition domain
        and range scale support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
        Ljg = XYZ_to_OSA_UCS(XYZ)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_OSA_UCS(XYZ * factor), Ljg * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_OSA_UCS(self):
        """
        Tests :func:`colour.models.osa_ucs.XYZ_to_OSA_UCS` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ_to_OSA_UCS(np.array(case))


class TestOSA_UCS_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition unit tests
    methods.
    """

    def test_OSA_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition.
        """

        np.testing.assert_allclose(
            OSA_UCS_to_XYZ(np.array([-4.49006830, 0.70305936, 3.03463664])),
            np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
            rtol=0.00001,
            atol=0.00001)

        np.testing.assert_allclose(
            OSA_UCS_to_XYZ(np.array([1.45512585, 6.57345931, -8.02280578])),
            np.array([0.47097710, 0.34950000, 0.11301649]) * 100,
            rtol=0.00001,
            atol=0.00001)

        np.testing.assert_allclose(
            OSA_UCS_to_XYZ(np.array([-1.88009027, 3.71899342, -5.98296399])),
            np.array([0.25506814, 0.19150000, 0.08849752]) * 100,
            rtol=0.00001,
            atol=0.00001)

    def test_n_dimensional_OSA_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition
        n-dimensions support.
        """

        Ljg = np.array([-4.49006830, 0.70305936, 3.03463664])
        XYZ = np.array([0.07049534, 0.10080000, 0.09558313]) * 100
        np.testing.assert_allclose(
            OSA_UCS_to_XYZ(Ljg), XYZ, rtol=0.00001, atol=0.00001)

        Ljg = np.tile(Ljg, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_allclose(
            OSA_UCS_to_XYZ(Ljg), XYZ, rtol=0.00001, atol=0.00001)

        Ljg = np.reshape(Ljg, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_allclose(
            OSA_UCS_to_XYZ(Ljg), XYZ, rtol=0.00001, atol=0.00001)

    def test_domain_range_scale_OSA_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition domain
        and range scale support.
        """

        Ljg = np.array([-4.49006830, 0.70305936, 3.03463664])
        XYZ = OSA_UCS_to_XYZ(Ljg)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    OSA_UCS_to_XYZ(Ljg * factor), XYZ * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_OSA_UCS_to_XYZ(self):
        """
        Tests :func:`colour.models.osa_ucs.OSA_UCS_to_XYZ` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            OSA_UCS_to_XYZ(np.array(case))


if __name__ == '__main__':
    unittest.main()
