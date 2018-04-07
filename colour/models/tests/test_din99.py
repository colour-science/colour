# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.din99` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.models import Lab_to_DIN99
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLab_to_DIN99']


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
            Lab_to_DIN99(np.array([37.98562910, -23.62907688, -4.41746615])),
            np.array([49.60101649, -16.23145730, 1.07618123]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_DIN99(np.array([60.25740000, -34.00990000, 36.26770000])),
            np.array([70.57378525, -13.18189095, 17.98538208]),
            decimal=7)

        np.testing.assert_almost_equal(
            Lab_to_DIN99(np.array([22.72330000, 20.09040000, -46.69400000])),
            np.array([32.36697919, 3.83443742, -21.01059563]),
            decimal=7)

    def test_n_dimensional_Lab_to_DIN99(self):
        """
        Tests :func:`colour.models.din99.Lab_to_DIN99` definition n-dimensions
        support.
        """

        Lab = np.array([37.98562910, -23.62907688, -4.41746615])
        Lab_99 = np.array([49.60101649, -16.23145730, 1.07618123])
        np.testing.assert_almost_equal(Lab_to_DIN99(Lab), Lab_99, decimal=7)

        Lab = np.tile(Lab, (6, 1))
        Lab_99 = np.tile(Lab_99, (6, 1))
        np.testing.assert_almost_equal(Lab_to_DIN99(Lab), Lab_99, decimal=7)

        np.testing.assert_almost_equal(Lab_to_DIN99(Lab), Lab_99, decimal=7)

        Lab = np.reshape(Lab, (2, 3, 3))
        Lab_99 = np.reshape(Lab_99, (2, 3, 3))
        np.testing.assert_almost_equal(Lab_to_DIN99(Lab), Lab_99, decimal=7)

    def test_domain_range_scale_Lab_to_DIN99(self):
        """
        Tests :func:`colour.models.din99.Lab_to_DIN99` definition
        domain and range scale support.
        """

        Lab = np.array([37.98562910, -23.62907688, -4.41746615])
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


if __name__ == '__main__':
    unittest.main()
