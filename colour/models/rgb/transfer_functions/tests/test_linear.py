# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.transfer_functions.linear`
module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.models.rgb.transfer_functions import linear_function
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLinearFunction']


class TestLinearFunction(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.transfer_functions.linear.\
linear_function` definition unit tests methods.
    """

    def test_linear_function(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.linear.\
linear_function` definition.
        """

        self.assertEqual(linear_function(0.0), 0.0)

        self.assertEqual(linear_function(0.18), 0.18)

        self.assertEqual(linear_function(1.0), 1.0)

    def test_n_dimensional_linear_function(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.linear.\
linear_function` definition n-dimensional arrays support.
        """

        a = 0.18
        a_p = 0.18
        np.testing.assert_almost_equal(linear_function(a), a_p, decimal=7)

        a = np.tile(a, 6)
        a_p = np.tile(a_p, 6)
        np.testing.assert_almost_equal(linear_function(a), a_p, decimal=7)

        a = np.reshape(a, (2, 3))
        a_p = np.reshape(a_p, (2, 3))
        np.testing.assert_almost_equal(linear_function(a), a_p, decimal=7)

        a = np.reshape(a, (2, 3, 1))
        a_p = np.reshape(a_p, (2, 3, 1))
        np.testing.assert_almost_equal(linear_function(a), a_p, decimal=7)

    @ignore_numpy_errors
    def test_nan_linear_function(self):
        """
        Tests :func:`colour.models.rgb.transfer_functions.linear.\
linear_function` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]

        for case in cases:
            linear_function(case)


if __name__ == '__main__':
    unittest.main()
