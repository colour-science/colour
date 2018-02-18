# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.itur_bt_601`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import oetf_BT601, oetf_reverse_BT601
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestOetf_BT601', 'TestOetf_reverse_BT601']


class TestOetf_BT601(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_601.oetf_BT601`
    definition unit tests methods.
    """

    def test_oetf_BT601(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_BT601` definition.
        """

        self.assertAlmostEqual(oetf_BT601(0.0), 0.0, places=7)

        self.assertAlmostEqual(oetf_BT601(0.015), 0.067500000000000, places=7)

        self.assertAlmostEqual(oetf_BT601(0.18), 0.409007728864150, places=7)

        self.assertAlmostEqual(oetf_BT601(1.0), 1.0, places=7)

    def test_n_dimensional_oetf_BT601(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_BT601` definition n-dimensional arrays support.
        """

        L = 0.18
        Es = 0.409007728864150
        np.testing.assert_almost_equal(oetf_BT601(L), Es, decimal=7)

        L = np.tile(L, 6)
        Es = np.tile(Es, 6)
        np.testing.assert_almost_equal(oetf_BT601(L), Es, decimal=7)

        L = np.reshape(L, (2, 3))
        Es = np.reshape(Es, (2, 3))
        np.testing.assert_almost_equal(oetf_BT601(L), Es, decimal=7)

        L = np.reshape(L, (2, 3, 1))
        Es = np.reshape(Es, (2, 3, 1))
        np.testing.assert_almost_equal(oetf_BT601(L), Es, decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_BT601(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_BT601` definition nan support.
        """

        oetf_BT601(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestOetf_reverse_BT601(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_reverse_BT601` definition unit tests methods.
    """

    def test_oetf_reverse_BT601(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_reverse_BT601` definition.
        """

        self.assertAlmostEqual(oetf_reverse_BT601(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            oetf_reverse_BT601(0.067500000000000), 0.015, places=7)

        self.assertAlmostEqual(
            oetf_reverse_BT601(0.409007728864150), 0.18, places=7)

        self.assertAlmostEqual(oetf_reverse_BT601(1.0), 1.0, places=7)

    def test_n_dimensional_oetf_reverse_BT601(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_reverse_BT601` definition n-dimensional arrays support.
        """

        E = 0.409007728864150
        L = 0.18
        np.testing.assert_almost_equal(oetf_reverse_BT601(E), L, decimal=7)

        E = np.tile(E, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(oetf_reverse_BT601(E), L, decimal=7)

        E = np.reshape(E, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(oetf_reverse_BT601(E), L, decimal=7)

        E = np.reshape(E, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(oetf_reverse_BT601(E), L, decimal=7)

    @ignore_numpy_errors
    def test_nan_oetf_reverse_BT601(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_601.\
oetf_reverse_BT601` definition nan support.
        """

        oetf_reverse_BT601(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
