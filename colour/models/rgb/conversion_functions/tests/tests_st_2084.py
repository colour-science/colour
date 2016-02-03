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

        self.assertAlmostEqual(
            ST_2084_EOCF(0.5, 5000),
            46.122854497032634,
            places=7)

    def test_n_dimensional_ST_2084_EOCF(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.st_2084.\
ST_2084_EOCF` definition n-dimensional arrays support.
        """

        N = 0.50
        C = 92.245708994065268
        np.testing.assert_almost_equal(
            ST_2084_EOCF(N),
            C,
            decimal=7)

        N = np.tile(N, 6)
        C = np.tile(C, 6)
        np.testing.assert_almost_equal(
            ST_2084_EOCF(N),
            C,
            decimal=7)

        N = np.reshape(N, (2, 3))
        C = np.reshape(C, (2, 3))
        np.testing.assert_almost_equal(
            ST_2084_EOCF(N),
            C,
            decimal=7)

        N = np.reshape(N, (2, 3, 1))
        C = np.reshape(C, (2, 3, 1))
        np.testing.assert_almost_equal(
            ST_2084_EOCF(N),
            C,
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

        self.assertAlmostEqual(
            ST_2084_OECF(92.245708994065268, 5000),
            0.57071924098752402,
            places=7)

    def test_n_dimensional_ST_2084_OECF(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.st_2084.\
ST_2084_OECF` definition n-dimensional arrays support.
        """

        C = 92.245708994065268
        N = 0.50000000000000067
        np.testing.assert_almost_equal(
            ST_2084_OECF(C),
            N,
            decimal=7)

        C = np.tile(C, 6)
        N = np.tile(N, 6)
        np.testing.assert_almost_equal(
            ST_2084_OECF(C),
            N,
            decimal=7)

        C = np.reshape(C, (2, 3))
        N = np.reshape(N, (2, 3))
        np.testing.assert_almost_equal(
            ST_2084_OECF(C),
            N,
            decimal=7)

        C = np.reshape(C, (2, 3, 1))
        N = np.reshape(N, (2, 3, 1))
        np.testing.assert_almost_equal(
            ST_2084_OECF(C),
            N,
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
