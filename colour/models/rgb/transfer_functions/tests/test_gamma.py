# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.gamma`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import gamma_function
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestGammaFunction']


class TestGammaFunction(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.gamma.gamma_function`
    definition unit tests methods.
    """

    def test_gamma_function(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.gamma.\
gamma_function` definition.
        """

        self.assertAlmostEqual(gamma_function(0.0, 2.2), 0.0, places=7)

        self.assertAlmostEqual(
            gamma_function(0.18, 2.2), 0.022993204992707, places=7)

        self.assertAlmostEqual(
            gamma_function(0.022993204992707, 1.0 / 2.2), 0.18, places=7)

        self.assertAlmostEqual(
            gamma_function(-0.18, 2.0), 0.0323999999999998, places=7)

        np.testing.assert_array_equal(gamma_function(-0.18, 2.2), np.nan)

        self.assertAlmostEqual(
            gamma_function(-0.18, 2.2, 'Mirror'), -0.022993204992707, places=7)

        self.assertAlmostEqual(
            gamma_function(-0.18, 2.2, 'Preserve'), -0.18, places=7)

        self.assertAlmostEqual(
            gamma_function(-0.18, 2.2, 'Clamp'), 0, places=7)

        np.testing.assert_array_equal(gamma_function(-0.18, -2.2), np.nan)

        self.assertAlmostEqual(
            gamma_function(0.0, -2.2, 'Mirror'), 0.0, places=7)

        self.assertAlmostEqual(
            gamma_function(0.0, 2.2, 'Preserve'), 0.0, places=7)

        self.assertAlmostEqual(gamma_function(0.0, 2.2, 'Clamp'), 0, places=7)

    def test_n_dimensional_gamma_function(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.gamma.\
gamma_function` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = gamma_function(a, 2.2)

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_almost_equal(gamma_function(a, 2.2), a_p, decimal=7)

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(gamma_function(a, 2.2), a_p, decimal=7)

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(gamma_function(a, 2.2), a_p, decimal=7)

        a = -0.18
        a_p = -0.022993204992707
        np.testing.assert_almost_equal(
            gamma_function(a, 2.2, 'Mirror'), a_p, decimal=7)

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_almost_equal(
            gamma_function(a, 2.2, 'Mirror'), a_p, decimal=7)

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(
            gamma_function(a, 2.2, 'Mirror'), a_p, decimal=7)

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            gamma_function(a, 2.2, 'Mirror'), a_p, decimal=7)

        a = -0.18
        a_p = -0.18
        np.testing.assert_almost_equal(
            gamma_function(a, 2.2, 'Preserve'), a_p, decimal=7)

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_almost_equal(
            gamma_function(a, 2.2, 'Preserve'), a_p, decimal=7)

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(
            gamma_function(a, 2.2, 'Preserve'), a_p, decimal=7)

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            gamma_function(a, 2.2, 'Preserve'), a_p, decimal=7)

        a = -0.18
        a_p = 0.0
        np.testing.assert_almost_equal(
            gamma_function(a, 2.2, 'Clamp'), a_p, decimal=7)

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_almost_equal(
            gamma_function(a, 2.2, 'Clamp'), a_p, decimal=7)

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(
            gamma_function(a, 2.2, 'Clamp'), a_p, decimal=7)

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(
            gamma_function(a, 2.2, 'Clamp'), a_p, decimal=7)

    def test_raise_exception_gamma_function(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.gamma.\
gamma_function` definition raised exception.
        """

        self.assertRaises(ValueError, gamma_function, 0.18, 1, 'Undefined')

    @ignore_numpy_errors
    def test_nan_gamma_function(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.gamma.\
gamma_function` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]

        for case in cases:
            gamma_function(case, case)


if __name__ == '__main__':
    unittest.main()
