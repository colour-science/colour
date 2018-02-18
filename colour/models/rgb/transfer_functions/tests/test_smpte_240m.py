# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.smpte_240m`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import oetf_SMPTE240M, eotf_SMPTE240M
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestOetf_SMPTE240M', 'TestEotf_SMPTE240M']


class TestOetf_SMPTE240M(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.smpte_240m.\
oetf_SMPTE240M` definition unit tests methods.
    """

    def test_oetf_SMPTE240M(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.smpte_240m.\
oetf_SMPTE240M` definition.
        """

        self.assertAlmostEqual(oetf_SMPTE240M(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            oetf_SMPTE240M(0.02), 0.080000000000000, places=7)

        self.assertAlmostEqual(
            oetf_SMPTE240M(0.18), 0.402285796753870, places=7)

        self.assertAlmostEqual(oetf_SMPTE240M(1.0), 1.0, places=7)

    def test_n_dimensional_oetf_SMPTE240M(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.smpte_240m.\
oetf_SMPTE240M` definition n-dimensional arrays support.
        """

        L_c = 0.18
        V_c = 0.402285796753870
        np.testing.assert_almost_equal(oetf_SMPTE240M(L_c), V_c, decimal=7)

        L_c = np.tile(L_c, 6)
        V_c = np.tile(V_c, 6)
        np.testing.assert_almost_equal(oetf_SMPTE240M(L_c), V_c, decimal=7)

        L_c = np.reshape(L_c, (2, 3))
        V_c = np.reshape(V_c, (2, 3))
        np.testing.assert_almost_equal(oetf_SMPTE240M(L_c), V_c, decimal=7)

        L_c = np.reshape(L_c, (2, 3, 1))
        V_c = np.reshape(V_c, (2, 3, 1))
        np.testing.assert_almost_equal(oetf_SMPTE240M(L_c), V_c, decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_SMPTE240M(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.smpte_240m.\
oetf_SMPTE240M` definition nan support.
        """

        oetf_SMPTE240M(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_SMPTE240M(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.smpte_240m.\
eotf_SMPTE240M` definition unit tests methods.
    """

    def test_eotf_SMPTE240M(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.smpte_240m.\
eotf_SMPTE240M` definition.
        """

        self.assertAlmostEqual(eotf_SMPTE240M(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_SMPTE240M(0.080000000000000), 0.02, places=7)

        self.assertAlmostEqual(
            eotf_SMPTE240M(0.402285796753870), 0.18, places=7)

        self.assertAlmostEqual(eotf_SMPTE240M(1.0), 1.0, places=7)

    def test_n_dimensional_eotf_SMPTE240M(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.smpte_240m.\
eotf_SMPTE240M` definition n-dimensional arrays support.
        """

        V = 0.402285796753870
        L = 0.18
        np.testing.assert_almost_equal(eotf_SMPTE240M(V), L, decimal=7)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(eotf_SMPTE240M(V), L, decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(eotf_SMPTE240M(V), L, decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_SMPTE240M(V), L, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_SMPTE240M(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.smpte_240m.\
eotf_SMPTE240M` definition nan support.
        """

        eotf_SMPTE240M(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
