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
            XYZ_to_IPT(np.array([0.96907232, 1, 1.12179215])),
            np.array([1.00300825, 0.01906918, -0.01369292]),
            decimal=7)
        np.testing.assert_almost_equal(
            XYZ_to_IPT(np.array([1.92001986, 1, -0.1241347])),
            np.array([0.73974548, 0.95333412, 1.71951212]),
            decimal=7)
        np.testing.assert_almost_equal(
            XYZ_to_IPT(np.array([1.0131677, 1, 2.11217686])),
            np.array([1.06406598, -0.08075812, -0.39625384]),
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
            np.array([0.9689994, 0.99995764, 1.1218432]),
            decimal=7)
        np.testing.assert_almost_equal(
            IPT_to_XYZ(np.array([0.73974548, 0.95333412, 1.71951212])),
            np.array([1.91998253, 0.99988784, -0.12416715]),
            decimal=7)
        np.testing.assert_almost_equal(
            IPT_to_XYZ(np.array([1.06406598, -0.08075812, -0.39625384])),
            np.array([1.0130757, 0.9999554, 2.11229678]),
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
            IPT_hue_angle(np.array([0.96907232, 1, 1.12179215])),
            0.84273584954373859,
            decimal=7)
        np.testing.assert_almost_equal(
            IPT_hue_angle(np.array([1.92001986, 1, -0.1241347])),
            -0.12350291631562464,
            decimal=7)
        np.testing.assert_almost_equal(
            IPT_hue_angle(np.array([1.0131677, 1, 2.11217686])),
            1.1286173302440385,
            decimal=7)
