# -*- coding: utf-8 -*-
"""
Defines the unit tests for the
:mod:`colour.models.rgb.transfer_functions.itur_bt_1886` module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (
    eotf_inverse_BT1886,
    eotf_BT1886,
)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestEotf_inverse_BT1886',
    'TestEotf_BT1886',
]


class TestEotf_inverse_BT1886(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition unit tests methods.
    """

    def test_eotf_inverse_BT1886(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition.
        """

        self.assertAlmostEqual(eotf_inverse_BT1886(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_inverse_BT1886(0.016317514686316), 0.18, places=7)

        self.assertAlmostEqual(eotf_inverse_BT1886(1.0), 1.0, places=7)

    def test_n_dimensional_eotf_inverse_BT1886(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition n-dimensional arrays support.
        """

        L = 0.016317514686316
        V = eotf_inverse_BT1886(L)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(eotf_inverse_BT1886(L), V, decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(eotf_inverse_BT1886(L), V, decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_inverse_BT1886(L), V, decimal=7)

    def test_domain_range_scale_eotf_inverse_BT1886(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition domain and range scale support.
        """

        L = 0.18
        V = eotf_inverse_BT1886(L)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_inverse_BT1886(L * factor), V * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_inverse_BT1886(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_inverse_BT1886` definition nan support.
        """

        eotf_inverse_BT1886(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_BT1886(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition unit tests methods.
    """

    def test_eotf_BT1886(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition.
        """

        self.assertAlmostEqual(eotf_BT1886(0.0), 0.0, places=7)

        self.assertAlmostEqual(eotf_BT1886(0.18), 0.016317514686316, places=7)

        self.assertAlmostEqual(eotf_BT1886(1.0), 1.0, places=7)

    def test_n_dimensional_eotf_BT1886(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition n-dimensional arrays support.
        """

        V = 0.18
        L = eotf_BT1886(V)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(eotf_BT1886(V), L, decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(eotf_BT1886(V), L, decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_BT1886(V), L, decimal=7)

    def test_domain_range_scale_eotf_BT1886(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition domain and range scale support.
        """

        V = 0.016317514686316
        L = eotf_BT1886(V)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_BT1886(V * factor), L * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_BT1886(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.itur_bt_1886.\
eotf_BT1886` definition nan support.
        """

        eotf_BT1886(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
