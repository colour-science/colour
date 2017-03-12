#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.dci_p3`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    oetf_DCIP3,
    eotf_DCIP3)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestOetf_DCIP3',
           'TestEotf_DCIP3']


class TestOetf_DCIP3(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.dci_p3.\
oetf_DCIP3` definition unit tests methods.
    """

    def test_oetf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
oetf_DCIP3` definition.
        """

        self.assertAlmostEqual(
            oetf_DCIP3(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            oetf_DCIP3(0.18),
            461.99220597,
            places=7)

        self.assertAlmostEqual(
            oetf_DCIP3(1.0),
            893.44598341,
            places=7)

    def test_n_dimensional_oetf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
oetf_DCIP3` definition n-dimensional arrays support.
        """

        L = 0.18
        V = 461.99220597
        np.testing.assert_almost_equal(
            oetf_DCIP3(L),
            V,
            decimal=7)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(
            oetf_DCIP3(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(
            oetf_DCIP3(L),
            V,
            decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(
            oetf_DCIP3(L),
            V,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
oetf_DCIP3` definition nan support.
        """

        oetf_DCIP3(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_DCIP3(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.dci_p3.
eotf_DCIP3` definition unit tests methods.
    """

    def test_eotf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
eotf_DCIP3` definition.
        """

        self.assertAlmostEqual(
            eotf_DCIP3(0.0),
            0.0,
            places=7)

        self.assertAlmostEqual(
            eotf_DCIP3(461.99220597),
            0.18,
            places=7)

        self.assertAlmostEqual(
            eotf_DCIP3(893.44598341),
            1.0,
            places=7)

    def test_n_dimensional_eotf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
eotf_DCIP3` definition n-dimensional arrays support.
        """

        V = 461.99220597
        L = 0.18
        np.testing.assert_almost_equal(
            eotf_DCIP3(V),
            L,
            decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(
            eotf_DCIP3(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(
            eotf_DCIP3(V),
            L,
            decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(
            eotf_DCIP3(V),
            L,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_DCIP3(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.dci_p3.\
eotf_DCIP3` definition nan support.
        """

        eotf_DCIP3(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
