#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.st_2084`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import oecf_ST2084, eocf_ST2084
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestOecf_ST2084',
           'TestEocf_ST2084']


class TestOecf_ST2084(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.st_2084.oecf_ST2084`
    definition unit tests methods.
    """

    def test_oecf_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
oecf_ST2084` definition.
        """

        self.assertAlmostEqual(
            oecf_ST2084(0.0),
            7.3095590257839665e-07,
            places=7)

        self.assertAlmostEqual(
            oecf_ST2084(0.18),
            0.079420969944927269,
            places=7)

        self.assertAlmostEqual(
            oecf_ST2084(1),
            0.14994573210018022,
            places=7)

        self.assertAlmostEqual(
            oecf_ST2084(5000, 5000),
            1.0,
            places=7)

    def test_n_dimensional_oecf_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
oecf_ST2084` definition n-dimensional arrays support.
        """

        C = 0.18
        N = 0.079420969944927269
        np.testing.assert_almost_equal(
            oecf_ST2084(C),
            N,
            decimal=7)

        C = np.tile(C, 6)
        N = np.tile(N, 6)
        np.testing.assert_almost_equal(
            oecf_ST2084(C),
            N,
            decimal=7)

        C = np.reshape(C, (2, 3))
        N = np.reshape(N, (2, 3))
        np.testing.assert_almost_equal(
            oecf_ST2084(C),
            N,
            decimal=7)

        C = np.reshape(C, (2, 3, 1))
        N = np.reshape(N, (2, 3, 1))
        np.testing.assert_almost_equal(
            oecf_ST2084(C),
            N,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_oecf_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
oecf_ST2084` definition nan support.
        """

        oecf_ST2084(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEocf_ST2084(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.st_2084.eocf_ST2084`
    definition unit tests methods.
    """

    def test_eocf_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
eocf_ST2084` definition.
        """

        self.assertAlmostEqual(
            eocf_ST2084(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            eocf_ST2084(0.079420969944927269),
            0.18,
            places=7)

        self.assertAlmostEqual(
            eocf_ST2084(0.14994573210018022),
            1.0,
            places=7)

        self.assertAlmostEqual(
            eocf_ST2084(1.0, 5000),
            5000.0,
            places=7)

    def test_n_dimensional_eocf_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
eocf_ST2084` definition n-dimensional arrays support.
        """

        N = 0.18
        C = 1.7385804910848062
        np.testing.assert_almost_equal(
            eocf_ST2084(N),
            C,
            decimal=7)

        N = np.tile(N, 6)
        C = np.tile(C, 6)
        np.testing.assert_almost_equal(
            eocf_ST2084(N),
            C,
            decimal=7)

        N = np.reshape(N, (2, 3))
        C = np.reshape(C, (2, 3))
        np.testing.assert_almost_equal(
            eocf_ST2084(N),
            C,
            decimal=7)

        N = np.reshape(N, (2, 3, 1))
        C = np.reshape(C, (2, 3, 1))
        np.testing.assert_almost_equal(
            eocf_ST2084(N),
            C,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_eocf_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
eocf_ST2084` definition nan support.
        """

        eocf_ST2084(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
