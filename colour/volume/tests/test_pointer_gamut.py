# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.volume.pointer_gamut` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.volume import is_within_pointer_gamut
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestIsWithinPointerGamut']


class TestIsWithinPointerGamut(unittest.TestCase):
    """
    Defines :func:`colour.volume.pointer_gamut.is_within_pointer_gamut`
    definition unit tests methods.
    """

    def test_is_within_pointer_gamut(self):
        """
        Tests :func:`colour.volume.pointer_gamut.is_within_pointer_gamut`
        definition.
        """

        self.assertTrue(
            is_within_pointer_gamut(np.array([0.3205, 0.4131, 0.5100])))

        self.assertFalse(
            is_within_pointer_gamut(np.array([0.0005, 0.0031, 0.0010])))

        self.assertTrue(
            is_within_pointer_gamut(np.array([0.4325, 0.3788, 0.1034])))

        self.assertFalse(
            is_within_pointer_gamut(np.array([0.0025, 0.0088, 0.0340])))

    def test_n_dimensional_is_within_pointer_gamut(self):
        """
        Tests :func:`colour.volume.pointer_gamut.is_within_pointer_gamut`
        definition n-dimensional arrays support.
        """

        a = np.array([0.3205, 0.4131, 0.5100])
        b = is_within_pointer_gamut(a)

        a = np.tile(a, (6, 1))
        b = np.tile(b, 6)
        np.testing.assert_almost_equal(is_within_pointer_gamut(a), b)

        a = np.reshape(a, (2, 3, 3))
        b = np.reshape(b, (2, 3))
        np.testing.assert_almost_equal(is_within_pointer_gamut(a), b)

    @ignore_numpy_errors
    def test_nan_is_within_pointer_gamut(self):
        """
        Tests :func:`colour.volume.pointer_gamut.is_within_pointer_gamut`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            is_within_pointer_gamut(case)


if __name__ == '__main__':
    unittest.main()
