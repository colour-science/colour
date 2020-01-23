# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.st_2084`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import (eotf_inverse_ST2084,
                                                  eotf_ST2084)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestEotf_inverse_ST2084', 'TestEotf_ST2084']


class TestEotf_inverse_ST2084(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition unit tests methods.
    """

    def test_eotf_inverse_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition.
        """

        self.assertAlmostEqual(
            eotf_inverse_ST2084(0.0), 0.000000730955903, places=7)

        self.assertAlmostEqual(
            eotf_inverse_ST2084(100), 0.508078421517399, places=7)

        self.assertAlmostEqual(
            eotf_inverse_ST2084(400), 0.652578597563067, places=7)

        self.assertAlmostEqual(eotf_inverse_ST2084(5000, 5000), 1.0, places=7)

    def test_n_dimensional_eotf_inverse_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition n-dimensional arrays support.
        """

        C = 100
        N = eotf_inverse_ST2084(C)

        C = np.tile(C, 6)
        N = np.tile(N, 6)
        np.testing.assert_almost_equal(eotf_inverse_ST2084(C), N, decimal=7)

        C = np.reshape(C, (2, 3))
        N = np.reshape(N, (2, 3))
        np.testing.assert_almost_equal(eotf_inverse_ST2084(C), N, decimal=7)

        C = np.reshape(C, (2, 3, 1))
        N = np.reshape(N, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_inverse_ST2084(C), N, decimal=7)

    def test_domain_range_scale_eotf_inverse_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition domain and range scale support.
        """

        C = 100
        N = eotf_inverse_ST2084(C)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_inverse_ST2084(C * factor), N * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_inverse_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_inverse_ST2084` definition nan support.
        """

        eotf_inverse_ST2084(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestEotf_ST2084(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.st_2084.eotf_ST2084`
    definition unit tests methods.
    """

    def test_eotf_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_ST2084` definition.
        """

        self.assertAlmostEqual(eotf_ST2084(0.0), 0.0, places=7)

        self.assertAlmostEqual(eotf_ST2084(0.508078421517399), 100, places=7)

        self.assertAlmostEqual(eotf_ST2084(0.652578597563067), 400, places=7)

        self.assertAlmostEqual(eotf_ST2084(1.0, 5000), 5000.0, places=7)

    def test_n_dimensional_eotf_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_ST2084` definition n-dimensional arrays support.
        """

        N = 0.508078421517399
        C = eotf_ST2084(N)

        N = np.tile(N, 6)
        C = np.tile(C, 6)
        np.testing.assert_almost_equal(eotf_ST2084(N), C, decimal=7)

        N = np.reshape(N, (2, 3))
        C = np.reshape(C, (2, 3))
        np.testing.assert_almost_equal(eotf_ST2084(N), C, decimal=7)

        N = np.reshape(N, (2, 3, 1))
        C = np.reshape(C, (2, 3, 1))
        np.testing.assert_almost_equal(eotf_ST2084(N), C, decimal=7)

    def test_domain_range_scale_eotf_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_ST2084` definition domain and range scale support.
        """

        N = 0.508078421517399
        C = eotf_ST2084(N)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    eotf_ST2084(N * factor), C * factor, decimal=7)

    @ignore_numpy_errors
    def test_nan_eotf_ST2084(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.st_2084.\
eotf_ST2084` definition nan support.
        """

        eotf_ST2084(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


if __name__ == '__main__':
    unittest.main()
