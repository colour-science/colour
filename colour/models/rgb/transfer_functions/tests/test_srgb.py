# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.models.rgb.transfer_functions.sRGB`
module.
"""

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import eotf_inverse_sRGB, eotf_sRGB
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestEotf_inverse_sRGB',
    'TestEotf_sRGB',
]


class TestEotf_inverse_sRGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.sRGB.eotf_inverse_sRGB`
    definition unit tests methods.
    """

    def test_eotf_inverse_sRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sRGB.\
eotf_inverse_sRGB` definition.
        """

        self.assertAlmostEqual(eotf_inverse_sRGB(0.0), 0.0, places=7)

        self.assertAlmostEqual(
            eotf_inverse_sRGB(0.18), 0.461356129500442, places=7)

        self.assertAlmostEqual(eotf_inverse_sRGB(1.0), 1.0, places=7)

    def test_n_dimensional_eotf_inverse_sRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sRGB.\
eotf_inverse_sRGB` definition n-dimensional arrays support.
        """

        L = 0.18
        V = eotf_inverse_sRGB(L)

        L = np.tile(L, 6)
        V = np.tile(V, 6)
        np.testing.assert_almost_equal(eotf_inverse_sRGB(L), V, decimal=7)

        L = np.reshape(L, (2, 3))
        V = np.reshape(V, (2, 3))
        np.testing.assert_almost_equal(eotf_inverse_sRGB(L), V, decimal=7)

        L = np.reshape(L, (2, 3, 1))
        V = np.reshape(V, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_inverse_sRGB(L), V, decimal=7)

    def test_domain_range_scale_eotf_inverse_sRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sRGB.\
eotf_inverse_sRGB` definition domain and range scale support.
        """

        L = 0.18
        V = eotf_inverse_sRGB(L)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_inverse_sRGB(L * factor), V * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_inverse_sRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sRGB.\
eotf_inverse_sRGB` definition nan support.
        """

        eotf_inverse_sRGB(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_sRGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.sRGB.eotf_sRGB`
    definition unit tests methods.
    """

    def test_eotf_sRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sRGB.\
eotf_sRGB` definition.
        """

        self.assertAlmostEqual(eotf_sRGB(0.0), 0.0, places=7)

        self.assertAlmostEqual(eotf_sRGB(0.461356129500442), 0.18, places=7)

        self.assertAlmostEqual(eotf_sRGB(1.0), 1.0, places=7)

    def test_n_dimensional_eotf_sRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sRGB.\
eotf_sRGB` definition n-dimensional arrays support.
        """

        V = 0.461356129500442
        L = eotf_sRGB(V)

        V = np.tile(V, 6)
        L = np.tile(L, 6)
        np.testing.assert_almost_equal(eotf_sRGB(V), L, decimal=7)

        V = np.reshape(V, (2, 3))
        L = np.reshape(L, (2, 3))
        np.testing.assert_almost_equal(eotf_sRGB(V), L, decimal=7)

        V = np.reshape(V, (2, 3, 1))
        L = np.reshape(L, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_sRGB(V), L, decimal=7)

    def test_domain_range_scale_eotf_sRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sRGB.\
eotf_sRGB` definition domain and range scale support.
        """

        V = 0.461356129500442
        L = eotf_sRGB(V)

        d_r = (('reference', 1), ('1', 1), ('100', 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_sRGB(V * factor), L * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_sRGB(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.sRGB.\
eotf_sRGB` definition nan support.
        """

        eotf_sRGB(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
