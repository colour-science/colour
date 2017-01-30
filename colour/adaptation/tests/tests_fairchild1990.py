# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.adaptation.fairchild1990` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.adaptation import chromatic_adaptation_Fairchild1990
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestChromaticAdaptationFairchild1990']


class TestChromaticAdaptationFairchild1990(unittest.TestCase):
    """
    Defines :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition unit tests methods.
    """

    def test_chromatic_adaptation_Fairchild1990(self):
        """
        Tests :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition.
        """

        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(
                np.array([0.07049534, 0.10080000, 0.09558313]) * 100,
                np.array([1.09846607, 1.00000000, 0.35582280]) * 100,
                np.array([0.95042855, 1.00000000, 1.08890037]) * 100,
                200),
            np.array([8.35782287, 10.21428897, 29.25065668]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(
                np.array([0.47097710, 0.34950000, 0.11301649]) * 100,
                np.array([0.99092745, 1.00000000, 0.85313273]) * 100,
                np.array([1.01679082, 1.00000000, 0.67610122]) * 100,
                200),
            np.array([49.00577034, 35.03909328, 8.95647114]),
            decimal=7)

        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(
                np.array([0.25506814, 0.19150000, 0.08849752]) * 100,
                np.array([0.98070597, 1.00000000, 1.18224949]) * 100,
                np.array([0.92833635, 1.00000000, 1.03664720]) * 100,
                200),
            np.array([24.79473034, 19.13024207, 7.75984317]),
            decimal=7)

    def test_n_dimensional_chromatic_adaptation_Fairchild1990(self):
        """
        Tests :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition n-dimensional arrays support.
        """

        XYZ_1 = np.array([19.53, 23.07, 24.97])
        XYZ_n = np.array([111.15, 100.00, 35.20])
        XYZ_r = np.array([94.81, 100.00, 107.30])
        Y_n = 200
        XYZ_c = np.array([23.32526349, 23.32455819, 76.11593750])
        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n),
            XYZ_c,
            decimal=7)

        XYZ_1 = np.tile(XYZ_1, (6, 1))
        XYZ_c = np.tile(XYZ_c, (6, 1))
        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n),
            XYZ_c,
            decimal=7)

        XYZ_n = np.tile(XYZ_n, (6, 1))
        XYZ_r = np.tile(XYZ_r, (6, 1))
        Y_n = np.tile(Y_n, 6)
        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n),
            XYZ_c,
            decimal=7)

        XYZ_1 = np.reshape(XYZ_1, (2, 3, 3))
        XYZ_n = np.reshape(XYZ_n, (2, 3, 3))
        XYZ_r = np.reshape(XYZ_r, (2, 3, 3))
        Y_n = np.reshape(Y_n, (2, 3))
        XYZ_c = np.reshape(XYZ_c, (2, 3, 3))
        np.testing.assert_almost_equal(
            chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n),
            XYZ_c,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_chromatic_adaptation_Fairchild1990(self):
        """
        Tests :func:`colour.adaptation.fairchild1990.\
chromatic_adaptation_Fairchild1990` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ_1 = np.array(case)
            XYZ_n = np.array(case)
            XYZ_r = np.array(case)
            Y_n = case[0]
            try:
                chromatic_adaptation_Fairchild1990(XYZ_1, XYZ_n, XYZ_r, Y_n)
            except np.linalg.linalg.LinAlgError:
                import traceback
                from colour.utilities import warning

                warning(traceback.format_exc())


if __name__ == '__main__':
    unittest.main()
