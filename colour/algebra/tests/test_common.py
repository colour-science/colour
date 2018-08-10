# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.algebra.common` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.algebra import spow

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestSpow']


class TestSpow(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.spow` definition unit
    tests methods.
    """

    def test_spow(self):
        """
        Tests :func:`colour.algebra.common.spow` definition.
        """

        self.assertEqual(spow(2, 2), 4.0)

        self.assertEqual(spow(-2, 2), -4.0)

        np.testing.assert_almost_equal(
            spow([2, -2, -2, 0], [2, 2, 0.15, 0]),
            np.array([4.00000000, -4.00000000, -1.10956947, 0.00000000]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
