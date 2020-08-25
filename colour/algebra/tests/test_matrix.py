# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.algebra.matrix` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.algebra import is_identity

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestIsIdentity']


class TestIsIdentity(unittest.TestCase):
    """
    Defines :func:`colour.algebra.matrix.is_identity` definition unit tests
    methods.
    """

    def test_is_identity(self):
        """
        Tests :func:`colour.algebra.matrix.is_identity` definition.
        """

        self.assertTrue(
            is_identity(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape([3, 3])))

        self.assertFalse(
            is_identity(np.array([1, 2, 0, 0, 1, 0, 0, 0, 1]).reshape([3, 3])))

        self.assertTrue(
            is_identity(np.array([1, 0, 0, 1]).reshape([2, 2]), n=2))

        self.assertFalse(
            is_identity(np.array([1, 2, 0, 1]).reshape([2, 2]), n=2))


if __name__ == '__main__':
    unittest.main()
