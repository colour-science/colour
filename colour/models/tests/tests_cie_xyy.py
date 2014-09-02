#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.cie_xyy` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import XYZ_to_xyY, xyY_to_XYZ, xy_to_XYZ, XYZ_to_xy

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_xyY',
           'TestxyY_to_XYZ',
           'Testxy_to_XYZ',
           'TestXYZ_to_xy']


class TestXYZ_to_xyY(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.XYZ_to_xyY` definition unit tests
    methods.
    """

    def test_XYZ_to_xyY(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xyY` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0.1180583421, 0.1034, 0.0515089229])),
            np.array([0.4325, 0.3788, 0.1034]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0.0308690042, 0.032, 0.0268925666])),
            np.array([0.3439, 0.3565, 0.0320]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xyY(np.array([0, 0, 0]),
                       (0.34567, 0.35850)),
            np.array([0.34567, 0.35850, 0]),
            decimal=7)


class TestxyY_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.xyY_to_XYZ` definition unit tests
    methods.
    """

    def test_xyY_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xyY_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.4325, 0.3788, 0.1034])),
            np.array([0.11805834, 0.1034, 0.05150892]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.3439, 0.3565, 0.0320])),
            np.array([0.030869, 0.032, 0.02689257]),
            decimal=7)

        np.testing.assert_almost_equal(
            xyY_to_XYZ(np.array([0.4325, 0, 0.1034])),
            np.array([0., 0., 0.]),
            decimal=7)


class Testxy_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.xy_to_XYZ` definition unit tests
    methods.
    """

    def test_xy_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_xyy.xy_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            xy_to_XYZ((0.32207410281368043, 0.3315655001362353)),
            np.array([0.97137399, 1., 1.04462134]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_XYZ((0.32174206617150575, 0.337609723160027)),
            np.array([0.953, 1.000, 1.009]),
            decimal=7)

        np.testing.assert_almost_equal(
            xy_to_XYZ((0.4474327628361859, 0.4074979625101875)),
            np.array([1.098, 1.000, 0.356]),
            decimal=7)


class TestXYZ_to_xy(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_xyy.XYZ_to_xy` definition unit tests
    methods.
    """

    def test_XYZ_to_xy(self):
        """
        Tests :func:`colour.models.cie_xyy.XYZ_to_xy` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_xy((0.97137399, 1, 1.04462134)),
            (0.32207410281368043, 0.3315655001362353),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xy((0.953, 1.000, 1.009)),
            (0.32174206617150575, 0.337609723160027),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_xy((1.098, 1.000, 0.356)),
            (0.4474327628361859, 0.4074979625101875),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
