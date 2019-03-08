# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.din99` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import Lab_to_DIN99, DIN99_to_Lab
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLab_to_DIN99', 'TestDIN99_to_Lab']


class TestLab_to_DIN99(unittest.TestCase):
    """
    Defines :func:`colour.models.din99.Lab_to_DIN99` definition unit tests
    methods.
    """

    def test_Lab_to_DIN99(self):
        """
        Tests :func:`colour.models.din99.Lab_to_DIN99` definition.
        """

        np.testing.assert_almost_equal(
            Lab_to_DIN99(np.array([41.52787529, 52.63858304, 26.92317922])),
            np.array([53.22821988, 28.41634656, 3.89839552]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_DIN99(np.array([55.11636304, -41.08791787, 30.91825778])),
            np.array([66.08943912, -17.35290106, 16.09690691]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_DIN99(np.array([29.80565520, 20.01830466, -48.34913874])),
            np.array([40.71533366, 3.48714163, -21.45321411]),
            decimal=7)

    def test_n_dimensional_Lab_to_DIN99(self):
        """
        Tests :func:`colour.models.din99.Lab_to_DIN99` definition n-dimensions
        support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        Lab_99 = np.array([53.22821988, 28.41634656, 3.89839552])
        np.testing.assert_almost_equal(Lab_to_DIN99(Lab), Lab_99, decimal=7)

        Lab = np.tile(Lab, (6, 1))
        Lab_99 = np.tile(Lab_99, (6, 1))
        np.testing.assert_almost_equal(Lab_to_DIN99(Lab), Lab_99, decimal=7)

        Lab = np.reshape(Lab, (2, 3, 3))
        Lab_99 = np.reshape(Lab_99, (2, 3, 3))
        np.testing.assert_almost_equal(Lab_to_DIN99(Lab), Lab_99, decimal=7)

    def test_domain_range_scale_Lab_to_DIN99(self):
        """
        Tests :func:`colour.models.din99.Lab_to_DIN99` definition domain and
        range scale support.
        """

        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        Lab_99 = Lab_to_DIN99(Lab)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    Lab_to_DIN99(Lab * factor), Lab_99 * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_Lab_to_DIN99(self):
        """
        Tests :func:`colour.models.din99.Lab_to_DIN99` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            Lab_to_DIN99(np.array(case))


class TestDIN99_to_Lab(unittest.TestCase):
    """
    Defines :func:`colour.models.din99.DIN99_to_Lab` definition unit tests
    methods.
    """

    def test_DIN99_to_Lab(self):
        """
        Tests :func:`colour.models.din99.DIN99_to_Lab` definition.
        """

        np.testing.assert_almost_equal(
            DIN99_to_Lab(np.array([53.22821988, 28.41634656, 3.89839552])),
            np.array([41.52787529, 52.63858304, 26.92317922]),
            decimal=7)

        np.testing.assert_almost_equal(
            DIN99_to_Lab(np.array([66.08943912, -17.35290106, 16.09690691])),
            np.array([55.11636304, -41.08791787, 30.91825778]),
            decimal=7)

        np.testing.assert_almost_equal(
            DIN99_to_Lab(np.array([40.71533366, 3.48714163, -21.45321411])),
            np.array([29.80565520, 20.01830466, -48.34913874]),
            decimal=7)

    def test_n_dimensional_DIN99_to_Lab(self):
        """
        Tests :func:`colour.models.din99.DIN99_to_Lab` definition n-dimensions
        support.
        """

        Lab_99 = np.array([53.22821988, 28.41634656, 3.89839552])
        Lab = np.array([41.52787529, 52.63858304, 26.92317922])
        np.testing.assert_almost_equal(DIN99_to_Lab(Lab_99), Lab, decimal=7)

        Lab_99 = np.tile(Lab_99, (6, 1))
        Lab = np.tile(Lab, (6, 1))
        np.testing.assert_almost_equal(DIN99_to_Lab(Lab_99), Lab, decimal=7)

        Lab_99 = np.reshape(Lab_99, (2, 3, 3))
        Lab = np.reshape(Lab, (2, 3, 3))
        np.testing.assert_almost_equal(DIN99_to_Lab(Lab_99), Lab, decimal=7)

    def test_domain_range_scale_DIN99_to_Lab(self):
        """
        Tests :func:`colour.models.din99.DIN99_to_Lab` definition domain and
        range scale support.
        """

        Lab_99 = np.array([53.22821988, 28.41634656, 3.89839552])
        Lab = DIN99_to_Lab(Lab_99)

        d_r = (('reference', 1), (1, 0.01), (100, 1))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    DIN99_to_Lab(Lab_99 * factor), Lab * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_DIN99_to_Lab(self):
        """
        Tests :func:`colour.models.din99.DIN99_to_Lab` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            DIN99_to_Lab(np.array(case))


if __name__ == '__main__':
    unittest.main()
