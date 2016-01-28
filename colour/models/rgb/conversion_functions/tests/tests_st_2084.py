#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.conversion_functions.st_2084`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.conversion_functions import ST_2084_EOCF, ST_2084_OECF
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestST_2084_EOCF']


class TestST_2084_EOCF(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.conversion_functions.st_2084.ST_2084_EOCF`
    definition unit tests methods.
    """

    def test_ST_2084_EOCF(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.st_2084.\
ST_2084_EOCF` definition.
        """

        self.assertAlmostEqual(
            ST_2084_EOCF(0.00),
            0.0,
            places=7)

        self.assertAlmostEqual(
            ST_2084_EOCF(0.50),
            92.245708994065268,
            places=7)

        self.assertAlmostEqual(
            ST_2084_EOCF(1.00),
            10000.0,
            places=7)

    def test_n_dimensional_ST_2084_EOCF(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.st_2084.\
ST_2084_EOCF` definition n-dimensional arrays support.
        """

        V = 0.50
        Y = 92.245708994065268
        np.testing.assert_almost_equal(
            ST_2084_EOCF(V),
            Y,
            decimal=7)

        V = np.tile(V, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_almost_equal(
            ST_2084_EOCF(V),
            Y,
            decimal=7)

        V = np.reshape(V, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_almost_equal(
            ST_2084_EOCF(V),
            Y,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_almost_equal(
            ST_2084_EOCF(V),
            Y,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_ST_2084_EOCF(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.st_2084.\
ST_2084_EOCF` definition nan support.
        """

        ST_2084_EOCF(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestST_2084_OECF(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.conversion_functions.st_2084.ST_2084_OECF`
    definition unit tests methods.
    """

    def test_ST_2084_OECF(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.st_2084.\
ST_2084_OECF` definition.
        """

        self.assertAlmostEqual(
            ST_2084_OECF(0.00),
            7.3095590257839665e-07,
            places=7)

        self.assertAlmostEqual(
            ST_2084_OECF(92.245708994065268),
            0.50000000000000067,
            places=7)

        self.assertAlmostEqual(
            ST_2084_OECF(10000),
            1.0,
            places=7)

    def test_n_dimensional_ST_2084_OECF(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.st_2084.\
ST_2084_OECF` definition n-dimensional arrays support.
        """

        Y = 92.245708994065268
        V = 0.50000000000000067
        np.testing.assert_almost_equal(
            ST_2084_OECF(Y),
            V,
            decimal=7)

        Y = np.tile(Y, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            ST_2084_OECF(Y),
            V,
            decimal=7)

        Y = np.reshape(Y, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            ST_2084_OECF(Y),
            V,
            decimal=7)

        Y = np.reshape(Y, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            ST_2084_OECF(Y),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_ST_2084_OECF(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.st_2084.\
ST_2084_OECF` definition nan support.
        """

        ST_2084_OECF(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
