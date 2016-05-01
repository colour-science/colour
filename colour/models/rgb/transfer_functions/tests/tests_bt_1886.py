#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.bt_1886`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import eocf_BT1886
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestEocf_BT1886']


class TestEocf_BT1886(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.bt_1886.eocf_BT1886`
    definition unit tests methods.
    """

    def test_eocf_BT1886(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_1886.\
eocf_BT1886` definition.
        """

        self.assertAlmostEqual(
            eocf_BT1886(0.0),
            64.0,
            places=7)

        self.assertAlmostEqual(
            eocf_BT1886(0.18),
            136.58617957264661,
            places=7)

        self.assertAlmostEqual(
            eocf_BT1886(1.0),
            939.99999999999989,
            places=7)

    def test_n_dimensional_eocf_BT1886(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_1886.\
eocf_BT1886` definition n-dimensional arrays support.
        """

        V = 0.18
        L = 136.58617957264661
        np.testing.assert_almost_equal(
            eocf_BT1886(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            eocf_BT1886(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            eocf_BT1886(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            eocf_BT1886(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_eocf_BT1886(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.bt_1886.\
eocf_BT1886` definition nan support.
        """

        eocf_BT1886(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
