#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.linear`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import function_linear
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestFunctionLinear']


class TestFunctionLinear(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.linear.\
function_linear` definition unit tests methods.
    """

    def test_function_linear(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.linear.\
function_linear` definition.
        """

        self.assertEqual(function_linear(0.0), 0.0)

        self.assertEqual(function_linear(0.18), 0.18)

        self.assertEqual(function_linear(1.0), 1.0)

    def test_n_dimensional_function_linear(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.linear.\
function_linear` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = 0.18
        np.testing.assert_almost_equal(function_linear(a), a_p, decimal=7)

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_almost_equal(function_linear(a), a_p, decimal=7)

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(function_linear(a), a_p, decimal=7)

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(function_linear(a), a_p, decimal=7)

    @ignore_numpy_errors
    def test_nan_function_linear(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.linear.\
function_linear` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]

        for case in cases:
            function_linear(case)


if __name__ == '__main__':
    unittest.main()
