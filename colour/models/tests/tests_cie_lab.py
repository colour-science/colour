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
            XYZ_to_Lab(np.array([0.96907232, 1, 1.12179215])),
            np.array([100., 0.83871284, -21.55579303]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([1.92001986, 1, -0.1241347])),
            np.array([100., 129.04406346, 406.69765889]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([1.0131677, 1, 2.11217686])),
            np.array([100., 8.32281957, -73.58297716]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([1.0131677, 1, 2.11217686]),
                       (0.44757, 0.40745)),
            np.array([100., -13.29228089, -162.12804888]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([1.0131677, 1, 2.11217686]),
                       (1 / 3, 1 / 3)),
            np.array([100., 2.18505384, -56.60990888]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_Lab(np.array([1.0131677, 1, 2.11217686]),
                       (0.31271, 0.32902)),
            np.array([100., 10.76832763, -49.42733157]),
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
            Lab_to_XYZ(np.array([100, 0.83871284, -21.55579303])),
            np.array([0.96907232, 1., 1.12179215]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([100, 129.04406346, 406.69765889])),
            np.array([1.92001986, 1., -0.1241347]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([100, 8.32281957, -73.58297716])),
            np.array([1.0131677, 1., 2.11217686]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([100, -13.29228089, -162.12804888]),
                       (0.44757, 0.40745)),
            np.array([1.0131677, 1., 2.11217686]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([100, 2.18505384, -56.60990888]),
                       (1 / 3, 1 / 3)),
            np.array([1.0131677, 1., 2.11217686]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_XYZ(np.array([100, 10.76832763, -49.42733157]),
                       (0.31271, 0.32902)),
            np.array([1.0131677, 1., 2.11217686]),
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
            Lab_to_LCHab(np.array([100, 0.83871284, -21.55579303])),
            np.array([100., 21.57210357, 272.2281935]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_LCHab(np.array([100, 129.04406346, 406.69765889])),
            np.array([100., 426.67945353, 72.39590835]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_LCHab(np.array([100, 8.32281957, -73.58297716])),
            np.array([100., 74.05216981, 276.45318193]),
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
            LCHab_to_Lab(np.array([100, 21.57210357, 272.2281935])),
            np.array([100., 0.83871284, -21.55579303]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHab_to_Lab(np.array([100, 426.67945353, 72.39590835])),
            np.array([100., 129.04406346, 406.69765889]),
            decimal=7)

        np.testing.assert_almost_equal(
            LCHab_to_Lab(np.array([100, 74.05216981, 276.45318193])),
            np.array([100., 8.32281957, -73.58297716]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
