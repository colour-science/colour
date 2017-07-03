#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.gamma`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import function_gamma
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestFunctionGamma']


class TestFunctionGamma(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.gamma.function_gamma`
    definition unit tests methods.
    """

    def test_function_gamma(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.gamma.\
function_gamma` definition.
        """

        self.assertAlmostEqual(function_gamma(0.0, 2.2), 0.0, places=7)

        self.assertAlmostEqual(
            function_gamma(0.18, 2.2), 0.022993204992707, places=7)

        self.assertAlmostEqual(
            function_gamma(0.022993204992707, 1.0 / 2.2), 0.18, places=7)

        self.assertAlmostEqual(
            function_gamma(-0.18, 2.0), 0.0323999999999998, places=7)

        np.testing.assert_array_equal(function_gamma(-0.18, 2.2), np.nan)

        self.assertAlmostEqual(
            function_gamma(-0.18, 2.2, 'Mirror'), -0.022993204992707, places=7)

        self.assertAlmostEqual(
            function_gamma(-0.18, 2.2, 'Preserve'), -0.18, places=7)

        self.assertAlmostEqual(
            function_gamma(-0.18, 2.2, 'Clamp'), 0, places=7)

    def test_n_dimensional_function_gamma(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.gamma.\
function_gamma` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = 0.022993204992707
        np.testing.assert_almost_equal(function_gamma(a, 2.2), a_p, decimal=7)

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_almost_equal(function_gamma(a, 2.2), a_p, decimal=7)

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(function_gamma(a, 2.2), a_p, decimal=7)

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(function_gamma(a, 2.2), a_p, decimal=7)

    @ignore_numpy_errors
    def test_nan_function_gamma(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.gamma.\
function_gamma` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]

        for case in cases:
            function_gamma(case, case)


if __name__ == '__main__':
    unittest.main()
