#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.cie_luv` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import (
    XYZ_to_Luv,
    Luv_to_XYZ,
    Luv_to_uv,
    Luv_uv_to_xy,
    Luv_to_LCHuv,
    LCHuv_to_Luv)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_Luv',
           'TestLuv_to_XYZ',
           'TestLuv_to_uv',
           'TestLuv_to_LCHuv',
           'TestLCHuv_to_Luv']


class TestXYZ_to_Luv(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.XYZ_to_Luv` definition unit tests
    methods.
    """

    def test_XYZ_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.XYZ_to_Luv` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([0.96907232, 1, 1.12179215])),
            np.array([100., -11.27488915, -29.36041662]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([1.92001986, 1, -0.1241347])),
            np.array([100., 331.44911128, 72.55258319]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([1.0131677, 1, 2.11217686])),
            np.array([100., -36.17788915, -111.00091702]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([1.0131677, 1, 2.11217686]),
                       (0.44757, 0.40745)),
            np.array([100., -97.02442861, -158.08546907]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([1.0131677, 1, 2.11217686]),
                       (1 / 3, 1 / 3)),
            np.array([100., -37.95520989, -92.29247371]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Luv(np.array([1.0131677, 1, 2.11217686]),
                       (0.31271, 0.32902)),
            np.array([100., -21.44928374, -85.33481874]),
            decimal=7)


class TestLuv_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.Luv_to_XYZ` definition unit tests
    methods.
    """

    def test_Luv_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([100, -11.27488915, -29.36041662])),
            np.array([0.96907232, 1., 1.12179215]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([100, 331.44911128, 72.55258319])),
            np.array([1.92001986, 1., -0.1241347]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([100, -36.17788915, -111.00091702])),
            np.array([1.0131677, 1., 2.11217686]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([100, -97.02442861, -158.08546907]),
                       (0.44757, 0.40745)),
            np.array([1.0131677, 1., 2.11217686]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([100, -37.95520989, -92.29247371]),
                       (1 / 3, 1 / 3)),
            np.array([1.0131677, 1., 2.11217686]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_XYZ(np.array([100, -21.44928374, -85.33481874]),
                       (0.31271, 0.32902)),
            np.array([1.0131677, 1., 2.11217686]),
            decimal=7)


class TestLuv_to_uv(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.Luv_to_uv` definition unit tests
    methods.
    """

    def test_Luv_to_uv(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_uv` definition.
        """

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([100, -11.27488915, -29.36041662])),
            (0.20048615433157738, 0.4654903849082484),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([100, 331.44911128, 72.55258319])),
            (0.46412000081619281, 0.54388500014670993),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([100, -36.17788915, -111.00091702])),
            (0.18133000048542355, 0.40268999998517152),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([100, -97.02442861, -158.08546907]),
                      (0.44757, 0.40745)),
            (0.18133000048503745, 0.40268999998707306),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([100, -37.95520989, -92.29247371]),
                      (1 / 3, 1 / 3)),
            (0.18133000048947367, 0.40268999998016192),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_uv(np.array([100, -21.44928374, -85.33481874]),
                      (0.31271, 0.32902)),
            (0.1813300004870092, 0.4026899999798475),
            decimal=7)


class TestLuv_to_LCHuv(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.Luv_to_LCHuv` definition unit tests
    methods.
    """

    def test_Luv_to_LCHuv(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_to_LCHuv` definition.
        """

        np.testing.assert_almost_equal(
            Luv_to_LCHuv(np.array([100, -11.27488915, -29.36041662])),
            np.array([100., 31.45086945, 248.99237865]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_LCHuv(np.array([100, 331.44911128, 72.55258319])),
            np.array([100., 339.2969064, 12.34702048]),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_to_LCHuv(np.array([100, -36.17788915, -111.00091702])),
            np.array([100., 116.74777618, 251.94795555]),
            decimal=7)


class TestLCHuv_to_Luv(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.LCHuv_to_Luv` definition unit tests
    methods.
    """

    def test_LCHuv_to_Luv(self):
        """
        Tests :func:`colour.models.cie_luv.LCHuv_to_Luv` definition.
        """

        np.testing.assert_almost_equal(
            LCHuv_to_Luv(np.array([100, 31.45086945, 248.99237865])),
            np.array([100., -11.27488915, -29.36041662]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHuv_to_Luv(np.array([100, 339.2969064, 12.34702048])),
            np.array([100., 331.44911128, 72.55258319]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHuv_to_Luv(np.array([100, 116.74777618, 251.94795555])),
            np.array([100., -36.17788915, -111.00091702]),
            decimal=7)


class TestLuv_uv_to_xy(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_luv.Luv_uv_to_xy` definition unit tests
    methods.
    """

    def test_Luv_uv_to_xy(self):
        """
        Tests :func:`colour.models.cie_luv.Luv_uv_to_xy` definition.
        """

        np.testing.assert_almost_equal(
            Luv_uv_to_xy((0.20048615433157738, 0.4654903849082484)),
            (0.31352792378977895, 0.32353408235422665),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_uv_to_xy((0.46412000081619281, 0.54388500014670993)),
            (0.6867305880410077, 0.3576684816384643),
            decimal=7)

        np.testing.assert_almost_equal(
            Luv_uv_to_xy((0.18133000048542355, 0.40268999998517152)),
            (0.2455958975694641, 0.2424039944946324),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
