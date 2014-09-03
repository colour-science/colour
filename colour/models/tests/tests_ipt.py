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
            XYZ_to_IPT(np.array([0.5, 0.5, 0.5])),
            np.array([0.738192, 0.0536732, 0.0359856]),
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
            IPT_to_XYZ(np.array([0.5, 0.5, 0.5])),
            np.array([0.4497109, 0.2694691, 0.0196303]),
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
            IPT_hue_angle(np.array([0.5, 0.5, 0.5])),
            0.78539812,
            decimal=7)
