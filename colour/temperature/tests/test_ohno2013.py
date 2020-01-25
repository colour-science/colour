# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.temperature.ohno2013` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.colorimetry import STANDARD_OBSERVERS_CMFS
from colour.temperature import CCT_to_uv_Ohno2013, uv_to_CCT_Ohno2013
from colour.temperature.ohno2013 import (
    planckian_table, planckian_table_minimal_distance_index)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestPlanckianTable', 'TestPlanckianTableMinimalDistanceIndex',
    'Testuv_to_CCT_Ohno2013', 'TestCCT_to_uv_Ohno2013'
]

PLANCKIAN_TABLE = np.array([
    [1000.00000000, 0.44796288, 0.35462962, 0.25373557],
    [1001.11111111, 0.44770303, 0.35465214, 0.25348315],
    [1002.22222222, 0.44744348, 0.35467461, 0.25323104],
    [1003.33333333, 0.44718423, 0.35469704, 0.25297924],
    [1004.44444444, 0.44692529, 0.35471942, 0.25272774],
    [1005.55555556, 0.44666666, 0.35474175, 0.25247656],
    [1006.66666667, 0.44640833, 0.35476404, 0.25222569],
    [1007.77777778, 0.44615030, 0.35478628, 0.25197512],
    [1008.88888889, 0.44589258, 0.35480848, 0.25172487],
    [1010.00000000, 0.44563516, 0.35483063, 0.25147492],
])


class TestPlanckianTable(unittest.TestCase):
    """
    Defines :func:`colour.temperature.ohno2013.planckian_table` definition
    units tests methods.
    """

    def test_planckian_table(self):
        """
        Tests :func:`colour.temperature.ohno2013.planckian_table` definition.
        """

        cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']

        np.testing.assert_almost_equal(
            [(x.Ti, x.ui, x.vi, x.di) for x in planckian_table(
                np.array([0.1978, 0.3122]), cmfs, 1000, 1010, 10)],
            PLANCKIAN_TABLE)


class TestPlanckianTableMinimalDistanceIndex(unittest.TestCase):
    """
    Defines :func:`colour.temperature.ohno2013.\
planckian_table_minimal_distance_index` definition unit tests methods.
    """

    def test_planckian_table_minimal_distance_index(self):
        """
        Tests :func:`colour.temperature.ohno2013.\
planckian_table_minimal_distance_index` definition.
        """

        cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        self.assertEqual(
            planckian_table_minimal_distance_index(
                planckian_table(
                    np.array([0.1978, 0.3122]), cmfs, 1000, 1010, 10)), 9)


class Testuv_to_CCT_Ohno2013(unittest.TestCase):
    """
    Defines :func:`colour.temperature.ohno2013.uv_to_CCT_Ohno2013` definition
    units tests methods.
    """

    def test_uv_to_CCT_Ohno2013(self):
        """
        Tests :func:`colour.temperature.ohno2013.uv_to_CCT_Ohno2013`
        definition.
        """

        cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        np.testing.assert_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.1978, 0.3122]), cmfs),
            np.array([6507.47380460, 0.00322335]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.4328, 0.2883]), cmfs),
            np.array([1041.68315360, -0.06737802]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.2927, 0.2722]), cmfs, iterations=4),
            np.array([2452.15316417, -0.08437064]),
            decimal=7)

    def test_n_dimensional_uv_to_CCT_Ohno2013(self):
        """
        Tests :func:`colour.temperature.ohno2013.uv_to_CCT_Ohno2013` definition
        n-dimensional arrays support.
        """

        uv = np.array([0.1978, 0.3122])
        CCT_D_uv = uv_to_CCT_Ohno2013(uv)

        uv = np.tile(uv, (6, 1))
        CCT_D_uv = np.tile(CCT_D_uv, (6, 1))
        np.testing.assert_almost_equal(
            uv_to_CCT_Ohno2013(uv), CCT_D_uv, decimal=7)

        uv = np.reshape(uv, (2, 3, 2))
        CCT_D_uv = np.reshape(CCT_D_uv, (2, 3, 2))
        np.testing.assert_almost_equal(
            uv_to_CCT_Ohno2013(uv), CCT_D_uv, decimal=7)

    @ignore_numpy_errors
    def test_nan_uv_to_CCT_Ohno2013(self):
        """
        Tests :func:`colour.temperature.ohno2013.uv_to_CCT_Ohno2013` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            uv = np.array(case)
            uv_to_CCT_Ohno2013(uv)


class TestCCT_to_uv_Ohno2013(unittest.TestCase):
    """
    Defines :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013` definition
    units tests methods.
    """

    def test_CCT_to_uv_Ohno2013(self):
        """
        Tests :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013`
        definition.
        """

        cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        np.testing.assert_almost_equal(
            CCT_to_uv_Ohno2013(np.array([6507.47380460, 0.00322335]), cmfs),
            np.array([0.19779997, 0.31219997]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_uv_Ohno2013(np.array([1041.68315360, -0.06737802]), cmfs),
            np.array([0.43279885, 0.28830013]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_uv_Ohno2013(np.array([2452.15316417, -0.08437064]), cmfs),
            np.array([0.29247364, 0.27215157]),
            decimal=7)

    def test_n_dimensional_CCT_to_uv_Ohno2013(self):
        """
        Tests :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013` definition
        n-dimensional arrays support.
        """

        cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        CCT_D_uv = np.array([6507.47380460, 0.00322335])
        uv = CCT_to_uv_Ohno2013(CCT_D_uv, cmfs)

        CCT_D_uv = np.tile(CCT_D_uv, (6, 1))
        uv = np.tile(uv, (6, 1))
        np.testing.assert_almost_equal(
            CCT_to_uv_Ohno2013(CCT_D_uv, cmfs), uv, decimal=7)

        CCT_D_uv = np.reshape(CCT_D_uv, (2, 3, 2))
        uv = np.reshape(uv, (2, 3, 2))
        np.testing.assert_almost_equal(
            CCT_to_uv_Ohno2013(CCT_D_uv, cmfs), uv, decimal=7)

    @ignore_numpy_errors
    def test_nan_CCT_to_uv_Ohno2013(self):
        """
        Tests :func:`colour.temperature.ohno2013.CCT_to_uv_Ohno2013` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            CCT_D_uv = np.array(case)
            CCT_to_uv_Ohno2013(CCT_D_uv)


if __name__ == '__main__':
    unittest.main()
