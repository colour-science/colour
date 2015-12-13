#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.cie_lab` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import XYZ_to_Lab, Lab_to_XYZ, Lab_to_LCHab, LCHab_to_Lab
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
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
            XYZ_to_Lab(np.array([0.07049534, 0.10080000, 0.09558313])),
            np.array([37.98562910, -23.62302887, -4.41417036]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.47097710, 0.34950000, 0.11301649])),
            np.array([65.70971880, 41.57577646, 37.78652063]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.25506814, 0.19150000, 0.08849752])),
            np.array([50.86223896, 32.77078577, 20.25804815]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.07049534, 0.10080000, 0.09558313]),
                       np.array([0.44757, 0.40745])),
            np.array([37.98562910, -32.51333979, -35.96770745]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.07049534, 0.10080000, 0.09558313]),
                       np.array([0.31271, 0.32902])),
            np.array([37.98562910, -22.61718913, 4.19383056]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.07049534, 0.10080000, 0.09558313]),
                       np.array([0.37208, 0.37529])),
            np.array([37.98562910, -25.55521883, -11.26139386]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([0.07049534, 0.10080000, 0.09558313]),
                       np.array([0.37208, 0.37529, 0.10080])),
            np.array([100., -54.91100935, -24.19758201]),
            decimal=7)

    def test_n_dimensional_XYZ_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.XYZ_to_Lab` definition n-dimensions
        support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        illuminant = np.array([0.34567, 0.35850])
        Lab = np.array([37.98562910, -23.62302887, -4.41417036])
        np.testing.assert_almost_equal(
            XYZ_to_Lab(XYZ, illuminant),
            Lab,
            decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        Lab = np.tile(Lab, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Lab(XYZ, illuminant),
            Lab,
            decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_Lab(XYZ, illuminant),
            Lab,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        Lab = np.reshape(Lab, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_Lab(XYZ, illuminant),
            Lab,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.XYZ_to_Lab` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            illuminant = np.array(case[0:2])
            XYZ_to_Lab(XYZ, illuminant)


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
            Lab_to_XYZ(np.array([37.98562910, -23.62302887, -4.41417036])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([65.70971880, 41.57577646, 37.78652063])),
            np.array([0.47097710, 0.34950000, 0.11301649]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([50.86223896, 32.77078577, 20.25804815])),
            np.array([0.25506814, 0.19150000, 0.08849752]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([37.98562910, -32.51333979, -35.96770745]),
                       np.array([0.44757, 0.40745])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([37.98562910, -22.61718913, 4.19383056]),
                       np.array([0.31271, 0.32902])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([37.98562910, -25.55521883, -11.26139386]),
                       np.array([0.37208, 0.37529])),
            np.array([0.07049534, 0.10080000, 0.09558313]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([37.98562910, -25.55521883, -11.26139386]),
                       np.array([0.37208, 0.37529, 0.10080])),
            np.array([0.00710593, 0.01016064, 0.00963478]),
            decimal=7)

    def test_n_dimensional_Lab_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_XYZ` definition n-dimensions
        support.
        """

        Lab = np.array([37.98562910, -23.62302887, -4.41417036])
        illuminant = np.array([0.34567, 0.35850])
        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        np.testing.assert_almost_equal(
            Lab_to_XYZ(Lab, illuminant),
            XYZ,
            decimal=7)

        Lab = np.tile(Lab, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            Lab_to_XYZ(Lab, illuminant),
            XYZ,
            decimal=7)

        illuminant = np.tile(illuminant, (6, 1))
        np.testing.assert_almost_equal(
            Lab_to_XYZ(Lab, illuminant),
            XYZ,
            decimal=7)

        Lab = np.reshape(Lab, (2, 3, 3))
        illuminant = np.reshape(illuminant, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            Lab_to_XYZ(Lab, illuminant),
            XYZ,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_Lab_to_XYZ(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_XYZ` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab = np.array(case)
            illuminant = np.array(case[0:2])
            Lab_to_XYZ(Lab, illuminant)


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
            Lab_to_LCHab(np.array([37.98562910, -23.62302887, -4.41417036])),
            np.array([37.98562910, 24.03190365, 190.58415972]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_LCHab(np.array([65.70971880, 41.57577646, 37.78652063])),
            np.array([65.70971880, 56.18154795, 42.26641468]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_LCHab(np.array([50.86223896, 32.77078577, 20.25804815])),
            np.array([50.86223896, 38.52678179, 31.7232794]),
            decimal=7)

    def test_n_dimensional_Lab_to_LCHab(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_LCHab` definition
        n-dimensional arrays support.
        """

        Lab = np.array([37.98562910, -23.62302887, -4.41417036])
        LCHab = np.array([37.98562910, 24.03190365, 190.58415972])
        np.testing.assert_almost_equal(
            Lab_to_LCHab(Lab),
            LCHab,
            decimal=7)

        Lab = np.tile(Lab, (6, 1))
        LCHab = np.tile(LCHab, (6, 1))
        np.testing.assert_almost_equal(
            Lab_to_LCHab(Lab),
            LCHab,
            decimal=7)

        Lab = np.reshape(Lab, (2, 3, 3))
        LCHab = np.reshape(LCHab, (2, 3, 3))
        np.testing.assert_almost_equal(
            Lab_to_LCHab(Lab),
            LCHab,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_Lab_to_LCHab(self):
        """
        Tests :func:`colour.models.cie_lab.Lab_to_LCHab` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab = np.array(case)
            Lab_to_LCHab(Lab)


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
            LCHab_to_Lab(np.array([37.98562910, 24.03190365, 190.58415972])),
            np.array([37.98562910, -23.62302887, -4.41417036]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHab_to_Lab(np.array([65.70971880, 56.18154795, 42.26641468])),
            np.array([65.70971880, 41.57577646, 37.78652063]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHab_to_Lab(np.array([50.86223896, 38.52678179, 31.72327940])),
            np.array([50.86223896, 32.77078577, 20.25804815]),
            decimal=7)

    def test_n_dimensional_LCHab_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.LCHab_to_Lab` definition
        n-dimensional arrays support.
        """

        LCHab = np.array([37.98562910, 24.03190365, 190.58415972])
        Lab = np.array([37.98562910, -23.62302887, -4.41417036])
        np.testing.assert_almost_equal(
            LCHab_to_Lab(LCHab),
            Lab,
            decimal=7)

        LCHab = np.tile(LCHab, (6, 1))
        Lab = np.tile(Lab, (6, 1))
        np.testing.assert_almost_equal(
            LCHab_to_Lab(LCHab),
            Lab,
            decimal=7)

        LCHab = np.reshape(LCHab, (2, 3, 3))
        Lab = np.reshape(Lab, (2, 3, 3))
        np.testing.assert_almost_equal(
            LCHab_to_Lab(LCHab),
            Lab,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_LCHab_to_Lab(self):
        """
        Tests :func:`colour.models.cie_lab.LCHab_to_Lab` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            LCHab = np.array(case)
            LCHab_to_Lab(LCHab)


if __name__ == '__main__':
    unittest.main()
