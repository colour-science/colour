#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.bt_709`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import oetf_BT709, eotf_BT709
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestOetf_BT709',
           'TestEotf_BT709']


class TestOetf_BT709(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.bt_709.oetf_BT709`
    definition unit tests methods.
    """

    def test_oetf_BT709(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_709.\
oetf_BT709` definition.
        """

        self.assertAlmostEqual(
            oetf_BT709(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            oetf_BT709(0.18),
            0.409007728864150,
            places=7)

        self.assertAlmostEqual(
            oetf_BT709(1.0),
            1.0,
            places=7)

    def test_n_dimensional_oetf_BT709(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_709.\
oetf_BT709` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.409007728864150
        np.testing.assert_almost_equal(
            oetf_BT709(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            oetf_BT709(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            oetf_BT709(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            oetf_BT709(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_BT709(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_709.\
oetf_BT709` definition nan support.
        """

        oetf_BT709(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_BT709(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.bt_709.eotf_BT709`
    definition unit tests methods.
    """

    def test_eotf_BT709(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_709.\
eotf_BT709` definition.
        """

        self.assertAlmostEqual(
            eotf_BT709(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            eotf_BT709(0.409007728864150),
            0.18,
            places=7)

        self.assertAlmostEqual(
            eotf_BT709(1.0),
            1.0,
            places=7)

    def test_n_dimensional_eotf_BT709(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_709.\
eotf_BT709` definition n-dimensional arrays support.
        """

        V = 0.409007728864150
        L = 0.18
        np.testing.assert_almost_equal(
            eotf_BT709(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            eotf_BT709(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            eotf_BT709(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            eotf_BT709(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_BT709(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_709.\
eotf_BT709` definition nan support.
        """

        eotf_BT709(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
