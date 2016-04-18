#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.conversion_functions.sRGB`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.conversion_functions import oecf_sRGB, eocf_sRGB
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestOecf_sRGB',
           'TestEocf_sRGB']


class TestOecf_sRGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.conversion_functions.sRGB.oecf_sRGB`
    definition unit tests methods.
    """

    def test_oecf_sRGB(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.sRGB.\
oecf_sRGB` definition.
        """

        self.assertAlmostEqual(
            oecf_sRGB(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            oecf_sRGB(0.18),
            0.46135612950044164,
            places=7)

        self.assertAlmostEqual(
            oecf_sRGB(1.0),
            1.0,
            places=7)

    def test_n_dimensional_oecf_sRGB(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.sRGB.\
oecf_sRGB` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.46135612950044164
        np.testing.assert_almost_equal(
            oecf_sRGB(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            oecf_sRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            oecf_sRGB(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            oecf_sRGB(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_oecf_sRGB(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.sRGB.\
oecf_sRGB` definition nan support.
        """

        oecf_sRGB(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEocf_sRGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.conversion_functions.sRGB.eocf_sRGB`
    definition unit tests methods.
    """

    def test_eocf_sRGB(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.sRGB.\
eocf_sRGB` definition.
        """

        self.assertAlmostEqual(
            eocf_sRGB(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            eocf_sRGB(0.46135612950044164),
            0.18,
            places=7)

        self.assertAlmostEqual(
            eocf_sRGB(1.0),
            1.0,
            places=7)

    def test_n_dimensional_eocf_sRGB(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.sRGB.\
eocf_sRGB` definition n-dimensional arrays support.
        """

        V = 0.46135612950044164
        L = 0.18
        np.testing.assert_almost_equal(
            eocf_sRGB(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            eocf_sRGB(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            eocf_sRGB(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            eocf_sRGB(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_eocf_sRGB(self):
        """
        Tests :func:`colour.models.rgb.conversion_functions.sRGB.\
eocf_sRGB` definition nan support.
        """

        eocf_sRGB(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
