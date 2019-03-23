# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.algebra.extrapolation` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.algebra import (Extrapolator, LinearInterpolator,
                            CubicSplineInterpolator, PchipInterpolator)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestExtrapolator']


class TestExtrapolator(unittest.TestCase):
    """
    Defines :func:`colour.algebra.extrapolation.Extrapolator` class units
    tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('interpolator', )

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Extrapolator))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ()

        for method in required_methods:
            self.assertIn(method, dir(Extrapolator))

    def test___call__(self):
        """
        Tests :func:`colour.algebra.extrapolation.Extrapolator.__call__`
        method.
        """

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([5, 6, 7]), np.array([5, 6, 7])))
        np.testing.assert_almost_equal(extrapolator((4, 8)), (4, 8))
        self.assertEqual(extrapolator(4), 4)

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([3, 4, 5]), np.array([1, 2, 3])),
            method='Constant')
        np.testing.assert_almost_equal(
            extrapolator((0.1, 0.2, 8, 9)), (1, 1, 3, 3))
        self.assertEqual(extrapolator(0.1), 1.)

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([3, 4, 5]), np.array([1, 2, 3])),
            method='Constant',
            left=0)
        np.testing.assert_almost_equal(
            extrapolator((0.1, 0.2, 8, 9)), (0, 0, 3, 3))
        self.assertEqual(extrapolator(0.1), 0)

        extrapolator = Extrapolator(
            LinearInterpolator(np.array([3, 4, 5]), np.array([1, 2, 3])),
            method='Constant',
            right=0)
        np.testing.assert_almost_equal(
            extrapolator((0.1, 0.2, 8, 9)), (1, 1, 0, 0))
        self.assertEqual(extrapolator(9), 0)

        extrapolator = Extrapolator(
            CubicSplineInterpolator(
                np.array([3, 4, 5, 6]), np.array([1, 2, 3, 4])))
        np.testing.assert_almost_equal(
            extrapolator((0.1, 0.2, 8.0, 9.0)), (-1.9, -1.8, 6.0, 7.0))
        self.assertEqual(extrapolator(9), 7)

        extrapolator = Extrapolator(
            PchipInterpolator(np.array([3, 4, 5]), np.array([1, 2, 3])))
        np.testing.assert_almost_equal(
            extrapolator((0.1, 0.2, 8.0, 9.0)), (-1.9, -1.8, 6.0, 7.0))
        self.assertEqual(extrapolator(9), 7.)

    @ignore_numpy_errors
    def test_nan__call__(self):
        """
        Tests :func:`colour.algebra.extrapolation.Extrapolator.__call__`
        method nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            extrapolator = Extrapolator(
                LinearInterpolator(np.array(case), np.array(case)))
            extrapolator(case[0])


if __name__ == '__main__':
    unittest.main()
