#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.algebra.regression` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.algebra import linear_regression

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Y', 'X1', 'X2', 'TestLinearRegression']

Y = np.array([1, 2, 1, 3, 2, 3, 3, 4, 4, 3, 5, 3, 3, 2, 4, 5, 5, 5, 4, 3])
X1 = np.array([
    40,
    45,
    38,
    50,
    48,
    55,
    53,
    55,
    58,
    40,
    55,
    48,
    45,
    55,
    60,
    60,
    60,
    65,
    50,
    58])
X2 = np.array([
    25,
    20,
    30,
    30,
    28,
    30,
    34,
    36,
    32,
    34,
    38,
    28,
    30,
    36,
    34,
    38,
    42,
    38,
    34,
    38])


class TestLinearRegression(unittest.TestCase):
    """
    Defines :func:`colour.algebra.regression.linear_regression` definition
    unit tests methods.
    """

    def test_linear_regression(self):
        """
        Tests :func:`colour.algebra.regression.linear_regression` definition.
        """

        np.testing.assert_almost_equal(
            linear_regression(Y),
            [0.1406015, 1.77368421],
            decimal=7)
        np.testing.assert_almost_equal(
            linear_regression(Y, X1),
            [0.12777065, -3.38129694],
            decimal=7)
        np.testing.assert_almost_equal(
            linear_regression(Y, tuple(zip(X1, X2))),
            [0.08640901, 0.08760164, -4.10358123],
            decimal=7)


if __name__ == '__main__':
    unittest.main()
