#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.bt_2020`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import oecf_BT2020, eocf_BT2020
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestOecf_BT2020',
           'TestEocf_BT2020']


class TestOecf_BT2020(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.bt_2020.oecf_BT2020`
    definition unit tests methods.
    """

    def test_oecf_BT2020(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_2020.\
oecf_BT2020` definition.
        """

        self.assertAlmostEqual(
            oecf_BT2020(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            oecf_BT2020(0.18),
            0.4090077288641504,
            places=7)

        self.assertAlmostEqual(
            oecf_BT2020(1.0),
            1.0,
            places=7)

    def test_n_dimensional_oecf_BT2020(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_2020.\
oecf_BT2020` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 0.4090077288641504
        np.testing.assert_almost_equal(
            oecf_BT2020(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            oecf_BT2020(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            oecf_BT2020(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            oecf_BT2020(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_oecf_BT2020(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_2020.\
oecf_BT2020` definition nan support.
        """

        oecf_BT2020(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEocf_BT2020(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.bt_2020.eocf_BT2020`
    definition unit tests methods.
    """

    def test_eocf_BT2020(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_2020.\
eocf_BT2020` definition.
        """

        self.assertAlmostEqual(
            eocf_BT2020(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            eocf_BT2020(0.4090077288641504),
            0.18,
            places=7)

        self.assertAlmostEqual(
            eocf_BT2020(1.0),
            1.0,
            places=7)

    def test_n_dimensional_eocf_BT2020(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_2020.\
eocf_BT2020` definition n-dimensional arrays support.
        """

        V = 0.4090077288641504
        L = 0.18
        np.testing.assert_almost_equal(
            eocf_BT2020(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            eocf_BT2020(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            eocf_BT2020(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            eocf_BT2020(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_eocf_BT2020(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_2020.\
eocf_BT2020` definition nan support.
        """

        eocf_BT2020(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
