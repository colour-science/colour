#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.conversion_functions.bt_1886`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.conversion_functions import BT_1886_EOCF
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestBT_1886_EOCF']


class TestBT_1886_EOCF(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.conversion_functions.bt_1886.BT_1886_EOCF`
    definition unit tests methods.
    """

    def test_BT_1886_EOCF(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.bt_1886.\
BT_1886_EOCF` definition.
        """

        self.assertAlmostEqual(
            BT_1886_EOCF(0.00),
            64.0,
            places=7)

        self.assertAlmostEqual(
            BT_1886_EOCF(0.50),
            350.82249515639683,
            places=7)

        self.assertAlmostEqual(
            BT_1886_EOCF(1.00),
            939.99999999999989,
            places=7)

    def test_n_dimensional_BT_1886_EOCF(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.bt_1886.\
BT_1886_EOCF` definition n-dimensional arrays support.
        """

        V = 0.50
        L = 350.82249515639683
        np.testing.assert_almost_equal(
            BT_1886_EOCF(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            BT_1886_EOCF(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            BT_1886_EOCF(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            BT_1886_EOCF(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_BT_1886_EOCF(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.bt_1886.\
BT_1886_EOCF` definition nan support.
        """

        BT_1886_EOCF(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
