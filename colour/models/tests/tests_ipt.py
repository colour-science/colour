#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.ipt` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import XYZ_to_IPT, IPT_to_XYZ, IPT_hue_angle

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
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
            XYZ_to_IPT(np.array([0.07049534, 0.1008, 0.09558313])),
            np.array([0.36571124, -0.11114798, 0.01594746]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_IPT(np.array([0.4709771, 0.3495, 0.11301649])),
            np.array([0.5916803, 0.34150712, 0.33282621]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_IPT(np.array([0.25506814, 0.1915, 0.08849752])),
            np.array([0.46626813, 0.25471184, 0.19904068]),
            decimal=7)


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
            np.array([0.96907232, 1., 1.12179215]),
            decimal=7)

        np.testing.assert_almost_equal(
            IPT_to_XYZ(np.array([0.73974548, 0.95333412, 1.71951212])),
            np.array([1.92001986, 1., -0.1241347]),
            decimal=7)

        np.testing.assert_almost_equal(
            IPT_to_XYZ(np.array([1.06406598, -0.08075812, -0.39625384])),
            np.array([1.0131677, 1., 2.11217686]),
            decimal=7)


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
            IPT_hue_angle(np.array([0.07049534, 0.1008, 0.09558313])),
            0.7588396531961388,
            decimal=7)

        np.testing.assert_almost_equal(
            IPT_hue_angle(np.array([0.4709771, 0.3495, 0.11301649])),
            0.3127534819420748,
            decimal=7)

        np.testing.assert_almost_equal(
            IPT_hue_angle(np.array([0.25506814, 0.1915, 0.08849752])),
            0.4328937107187537,
            decimal=7)
