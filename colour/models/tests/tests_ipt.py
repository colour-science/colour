#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.ipt` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_IPT, IPT_to_XYZ, IPT_hue_angle
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_IPT',
           'TestIPT_to_XYZ',
           'TestIPTHueAngle']


class TestXYZ_to_IPT(unittest.TestCase):
    """
    Defines :func:`colour.models.ipt.TestXYZ_to_IPT` definition unit tests
    methods.
    """

    def test_XYZ_to_IPT(self):
        """
        Tests :func:`colour.models.ipt.XYZ_to_IPT` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_IPT(np.array([0.07049534, 0.10080000, 0.09558313])),
            np.array([0.36571124, -0.11114798, 0.01594746]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_IPT(np.array([0.47097710, 0.34950000, 0.11301649])),
            np.array([0.59168030, 0.34150712, 0.33282621]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_IPT(np.array([0.25506814, 0.19150000, 0.08849752])),
            np.array([0.46626813, 0.25471184, 0.19904068]),
            decimal=7)

    def test_n_dimensional_XYZ_to_IPT(self):
        """
        Tests :func:`colour.models.ipt.XYZ_to_IPT` definition n-dimensions
        support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        IPT = np.array([0.36571124, -0.11114798, 0.01594746])
        np.testing.assert_almost_equal(
            XYZ_to_IPT(XYZ),
            IPT,
            decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        IPT = np.tile(IPT, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_IPT(XYZ),
            IPT,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        IPT = np.reshape(IPT, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_IPT(XYZ),
            IPT,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_IPT(self):
        """
        Tests :func:`colour.models.ipt.XYZ_to_IPT` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            XYZ_to_IPT(XYZ)


class TestIPT_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.ipt.IPT_to_XYZ` definition unit tests
    methods.
    """

    def test_IPT_to_XYZ(self):
        """
        Tests :func:`colour.models.ipt.IPT_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            IPT_to_XYZ(np.array([1.00300825, 0.01906918, -0.01369292])),
            np.array([0.96907232, 1.00000000, 1.12179215]),
            decimal=7)

        np.testing.assert_almost_equal(
            IPT_to_XYZ(np.array([0.73974548, 0.95333412, 1.71951212])),
            np.array([1.92001986, 1.00000000, -0.12413470]),
            decimal=7)

        np.testing.assert_almost_equal(
            IPT_to_XYZ(np.array([1.06406598, -0.08075812, -0.39625384])),
            np.array([1.01316770, 1.00000000, 2.11217686]),
            decimal=7)

    def test_n_dimensional_IPT_to_XYZ(self):
        """
        Tests :func:`colour.models.ipt.IPT_to_XYZ` definition n-dimensions
        support.
        """

        IPT = np.array([0.36571124, -0.11114798, 0.01594746])
        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        np.testing.assert_almost_equal(
            IPT_to_XYZ(IPT),
            XYZ,
            decimal=7)

        IPT = np.tile(IPT, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            IPT_to_XYZ(IPT),
            XYZ,
            decimal=7)

        IPT = np.reshape(IPT, (2, 3, 3))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            IPT_to_XYZ(IPT),
            XYZ,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_IPT_to_XYZ(self):
        """
        Tests :func:`colour.models.ipt.IPT_to_XYZ` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            IPT = np.array(case)
            IPT_to_XYZ(IPT)


class TestIPTHueAngle(unittest.TestCase):
    """
    Defines :func:`colour.models.ipt.IPT_hue_angle` definition unit tests
    methods.
    """

    def test_IPT_hue_angle(self):
        """
        Tests :func:`colour.models.ipt.IPT_hue_angle` definition.
        """

        np.testing.assert_almost_equal(
            IPT_hue_angle(np.array([0.07049534, 0.10080000, 0.09558313])),
            43.478309455309819,
            decimal=7)

        np.testing.assert_almost_equal(
            IPT_hue_angle(np.array([0.47097710, 0.34950000, 0.11301649])),
            17.919454543301892,
            decimal=7)

        np.testing.assert_almost_equal(
            IPT_hue_angle(np.array([0.25506814, 0.19150000, 0.08849752])),
            24.802982601941753,
            decimal=7)

    def test_n_dimensional_IPT_hue_angle(self):
        """
        Tests :func:`colour.models.ipt.IPT_hue_angle` definition n-dimensions
        support.
        """

        IPT = np.array([0.07049534, 0.10080000, 0.09558313])
        hue = 43.478309455309819
        np.testing.assert_almost_equal(
            IPT_hue_angle(IPT),
            hue,
            decimal=7)

        IPT = np.tile(IPT, (6, 1))
        hue = np.tile(hue, 6)
        np.testing.assert_almost_equal(
            IPT_hue_angle(IPT),
            hue,
            decimal=7)

        IPT = np.reshape(IPT, (2, 3, 3))
        hue = np.reshape(hue, (2, 3))
        np.testing.assert_almost_equal(
            IPT_hue_angle(IPT),
            hue,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_IPT_hue_angle(self):
        """
        Tests :func:`colour.models.ipt.IPT_hue_angle` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            IPT = np.array(case)
            IPT_hue_angle(IPT)


if __name__ == '__main__':
    unittest.main()
