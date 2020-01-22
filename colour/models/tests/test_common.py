# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.common` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import Jab_to_JCh, JCh_to_Jab
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestJab_to_JCh', 'TestJCh_to_Jab']


class TestJab_to_JCh(unittest.TestCase):
    """
    Defines :func:`colour.models.common.Jab_to_JCh` definition unit tests
    methods.
    """

    def test_Jab_to_JCh(self):
        """
        Tests :func:`colour.models.common.Jab_to_JCh` definition.
        """

        np.testing.assert_almost_equal(
            Jab_to_JCh(np.array([41.52787529, 52.63858304, 26.92317922])),
            np.array([41.52787529, 59.12425901, 27.08848784]),
            decimal=7)

        np.testing.assert_almost_equal(
            Jab_to_JCh(np.array([55.11636304, -41.08791787, 30.91825778])),
            np.array([55.11636304, 51.42135412, 143.03889556]),
            decimal=7)

        np.testing.assert_almost_equal(
            Jab_to_JCh(np.array([29.80565520, 20.01830466, -48.34913874])),
            np.array([29.80565520, 52.32945383, 292.49133666]),
            decimal=7)

    def test_n_dimensional_Jab_to_JCh(self):
        """
        Tests :func:`colour.models.common.Jab_to_JCh` definition n-dimensional
        arrays support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        LCHab = Jab_to_JCh(Lab)

        Lab = np.tile(Lab, (6, 1))
        LCHab = np.tile(LCHab, (6, 1))
        np.testing.assert_almost_equal(Jab_to_JCh(Lab), LCHab, decimal=7)

        Lab = np.reshape(Lab, (2, 3, 3))
        LCHab = np.reshape(LCHab, (2, 3, 3))
        np.testing.assert_almost_equal(Jab_to_JCh(Lab), LCHab, decimal=7)

    def test_domain_range_scale_Jab_to_JCh(self):
        """
        Tests :func:`colour.models.common.Jab_to_JCh` definition domain and
        range scale support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        LCHab = Jab_to_JCh(Lab)

        d_r = (('reference', 1, 1), (1, 0.01, np.array([0.01, 0.01, 1 / 360])),
               (100, 1, np.array([1, 1, 1 / 3.6])))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    Jab_to_JCh(Lab * factor_a), LCHab * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_Jab_to_JCh(self):
        """
        Tests :func:`colour.models.common.Jab_to_JCh` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab = np.array(case)
            Jab_to_JCh(Lab)


class TestJCh_to_Jab(unittest.TestCase):
    """
    Defines :func:`colour.models.common.JCh_to_Jab` definition unit tests
    methods.
    """

    def test_JCh_to_Jab(self):
        """
        Tests :func:`colour.models.common.JCh_to_Jab` definition.
        """

        np.testing.assert_almost_equal(
            JCh_to_Jab(np.array([41.52787529, 59.12425901, 27.08848784])),
            np.array([41.52787529, 52.63858304, 26.92317922]),
            decimal=7)

        np.testing.assert_almost_equal(
            JCh_to_Jab(np.array([55.11636304, 51.42135412, 143.03889556])),
            np.array([55.11636304, -41.08791787, 30.91825778]),
            decimal=7)

        np.testing.assert_almost_equal(
            JCh_to_Jab(np.array([29.80565520, 52.32945383, 292.49133666])),
            np.array([29.80565520, 20.01830466, -48.34913874]),
            decimal=7)

    def test_n_dimensional_JCh_to_Jab(self):
        """
        Tests :func:`colour.models.common.JCh_to_Jab` definition n-dimensional
        arrays support.
        """

        LCHab = np.array([41.52787529, 59.12425901, 27.08848784])
        Lab = JCh_to_Jab(LCHab)

        LCHab = np.tile(LCHab, (6, 1))
        Lab = np.tile(Lab, (6, 1))
        np.testing.assert_almost_equal(JCh_to_Jab(LCHab), Lab, decimal=7)

        LCHab = np.reshape(LCHab, (2, 3, 3))
        Lab = np.reshape(Lab, (2, 3, 3))
        np.testing.assert_almost_equal(JCh_to_Jab(LCHab), Lab, decimal=7)

    def test_domain_range_scale_JCh_to_Jab(self):
        """
        Tests :func:`colour.models.common.JCh_to_Jab` definition domain and
        range scale support.
        """

        LCHab = np.array([41.52787529, 59.12425901, 27.08848784])
        Lab = JCh_to_Jab(LCHab)

        d_r = (('reference', 1, 1), (1, np.array([0.01, 0.01, 1 / 360]), 0.01),
               (100, np.array([1, 1, 1 / 3.6]), 1))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    JCh_to_Jab(LCHab * factor_a), Lab * factor_b, decimal=7)

    @ignore_numpy_errors
    def test_nan_JCh_to_Jab(self):
        """
        Tests :func:`colour.models.common.JCh_to_Jab` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            LCHab = np.array(case)
            JCh_to_Jab(LCHab)


if __name__ == '__main__':
    unittest.main()
