#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.cie_lab` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import XYZ_to_Lab, Lab_to_XYZ, Lab_to_LCHab, LCHab_to_Lab

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestXYZ_to_Lab',
           'TestLab_to_XYZ',
           'TestLab_to_LCHab',
           'TestLCHab_to_Lab']


class TestXYZ_to_Lab(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_lab.XYZ_to_Lab` definition unit tests
    methods.
    """

    def test_XYZ_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.XYZ_to_Lab` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.07049534, 0.1008, 0.09558313])),
            np.array([37.9856291, -23.62302887, -4.41417036]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.4709771, 0.3495, 0.11301649])),
            np.array([65.7097188, 41.57577646, 37.78652063]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.25506814, 0.1915, 0.08849752])),
            np.array([50.86223896, 32.77078577, 20.25804815]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.07049534, 0.1008, 0.09558313]),
                       (0.44757, 0.40745)),
            np.array([37.9856291, -32.51333979, -35.96770745]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.07049534, 0.1008, 0.09558313]),
                       (0.31271, 0.32902)),
            np.array([37.9856291, -22.61718913, 4.19383056]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.07049534, 0.1008, 0.09558313]),
                       (0.37208, 0.37529)),
            np.array([37.9856291, -25.55521883, -11.26139386]),
            decimal=7)


class TestLab_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_lab.Lab_to_XYZ` definition unit tests
    methods.
    """

    def test_Lab_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([37.9856291, -23.62302887, -4.41417036])),
            np.array([0.07049534, 0.1008, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([65.7097188, 41.57577646, 37.78652063])),
            np.array([0.4709771, 0.3495, 0.11301649]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([50.86223896, 32.77078577, 20.25804815])),
            np.array([0.25506814, 0.1915, 0.08849752]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([37.9856291, -32.51333979, -35.96770745]),
                       (0.44757, 0.40745)),
            np.array([0.07049534, 0.1008, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([37.9856291, -22.61718913, 4.19383056]),
                       (0.31271, 0.32902)),
            np.array([0.07049534, 0.1008, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([37.9856291, -25.55521883, -11.26139386]),
                       (0.37208, 0.37529)),
            np.array([0.07049534, 0.1008, 0.09558313]),
            decimal=7)


class TestLab_to_LCHab(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_lab.Lab_to_LCHab` definition unit tests
    methods.
    """

    def test_Lab_to_LCHab(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_LCHab` definition.
        """

        np.testing.assert_almost_equal(
            Lab_to_LCHab(np.array([37.9856291, -23.62302887, -4.41417036])),
            np.array([37.9856291, 24.03190365, 190.58415972]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_LCHab(np.array([65.7097188, 41.57577646, 37.78652063])),
            np.array([65.7097188, 56.18154795, 42.26641468]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_LCHab(np.array([50.86223896, 32.77078577, 20.25804815])),
            np.array([50.86223896, 38.52678179, 31.7232794]),
            decimal=7)


class TestLCHab_to_Lab(unittest.TestCase):
    """
    Defines :func:`colour.models.cie_lab.LCHab_to_Lab` definition unit tests
    methods.
    """

    def test_LCHab_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.LCHab_to_Lab` definition.
        """

        np.testing.assert_almost_equal(
            LCHab_to_Lab(np.array([37.9856291, 24.03190365, 190.58415972])),
            np.array([37.9856291, -23.62302887, -4.41417036]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHab_to_Lab(np.array([65.7097188, 56.18154795, 42.26641468])),
            np.array([65.7097188, 41.57577646, 37.78652063]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHab_to_Lab(np.array([50.86223896, 38.52678179, 31.7232794])),
            np.array([50.86223896, 32.77078577, 20.25804815]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
