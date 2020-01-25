# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.temperature.mccamy1992` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.temperature import xy_to_CCT_McCamy1992, CCT_to_xy_McCamy1992
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['Testxy_to_CCT_McCamy1992', 'TestCCT_to_xy_McCamy1992']


class Testxy_to_CCT_McCamy1992(unittest.TestCase):
    """
    Defines :func:`colour.temperature.mccamy1992.xy_to_CCT_McCamy1992`
    definition unit tests methods.
    """

    def test_xy_to_CCT_McCamy1992(self):
        """
        Tests :func:`colour.temperature.mccamy1992.xy_to_CCT_McCamy1992`
        definition.
        """

        self.assertAlmostEqual(
            xy_to_CCT_McCamy1992(np.array([0.31270, 0.32900])),
            6505.08059131,
            places=7)

        self.assertAlmostEqual(
            xy_to_CCT_McCamy1992(np.array([0.44757, 0.40745])),
            2857.28961266,
            places=7)

        self.assertAlmostEqual(
            xy_to_CCT_McCamy1992(
                np.array([0.252520939374083, 0.252220883926284])),
            19501.61953130,
            places=7)

    def test_n_dimensional_xy_to_CCT_McCamy1992(self):
        """
        Tests :func:`colour.temperature.mccamy1992.xy_to_CCT_McCamy1992`
        definition n-dimensional arrays support.
        """

        xy = np.array([0.31270, 0.32900])
        CCT = xy_to_CCT_McCamy1992(xy)

        xy = np.tile(xy, (6, 1))
        CCT = np.tile(CCT, 6)
        np.testing.assert_almost_equal(
            xy_to_CCT_McCamy1992(xy), CCT, decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        CCT = np.reshape(CCT, (2, 3))
        np.testing.assert_almost_equal(
            xy_to_CCT_McCamy1992(xy), CCT, decimal=7)

    @ignore_numpy_errors
    def test_nan_xy_to_CCT_McCamy1992(self):
        """
        Tests :func:`colour.temperature.mccamy1992.xy_to_CCT_McCamy1992`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            xy_to_CCT_McCamy1992(case)


class TestCCT_to_xy_McCamy1992(unittest.TestCase):
    """
    Defines :func:`colour.temperature.mccamy1992.CCT_to_xy_McCamy1992`
    definition units tests methods.
    """

    def test_CCT_to_xy_McCamy1992(self):
        """
        Tests :func:`colour.temperature.mccamy1992.CCT_to_xy_McCamy1992`
        definition.
        """

        np.testing.assert_almost_equal(
            CCT_to_xy_McCamy1992(6505.08059131, {'method': 'Nelder-Mead'}),
            np.array([0.31269945, 0.32900411]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_xy_McCamy1992(2857.28961266, {'method': 'Nelder-Mead'}),
            np.array([0.42350314, 0.36129253]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_xy_McCamy1992(19501.61953130, {'method': 'Nelder-Mead'}),
            np.array([0.11173782, 0.36987375]),
            decimal=7)

    def test_n_dimensional_CCT_to_xy_McCamy1992(self):
        """
        Tests :func:`colour.temperature.mccamy1992.CCT_to_xy_McCamy1992`
        definition n-dimensional arrays support.
        """

        CCT = 6505.08059131
        xy = CCT_to_xy_McCamy1992(CCT)

        CCT = np.tile(CCT, 6)
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(
            CCT_to_xy_McCamy1992(CCT), xy, decimal=7)

        CCT = np.reshape(CCT, (2, 3))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(
            CCT_to_xy_McCamy1992(CCT), xy, decimal=7)

    @ignore_numpy_errors
    def test_nan_CCT_to_xy_McCamy1992(self):
        """
        Tests :func:`colour.temperature.mccamy1992.CCT_to_xy_McCamy1992`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            CCT_to_xy_McCamy1992(case)


if __name__ == '__main__':
    unittest.main()
