#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.notation.triplet` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.notation.triplet import (
    RGB_to_HEX,
    HEX_to_RGB)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRGB_to_HEX',
           'TestHEX_to_RGB']


class TestRGB_to_HEX(unittest.TestCase):
    """
    Defines :func:`colour.notation.triplet.RGB_to_HEX` definition unit tests
    methods.
    """

    def test_RGB_to_HEX(self):
        """
        Tests :func:`colour.notation.triplet.RGB_to_HEX` definition.
        """

        self.assertEqual(
            RGB_to_HEX(np.array([0.25, 0.60, 0.05])),
            '#3f990c')
        self.assertEqual(
            RGB_to_HEX(np.array([0, 0, 0])),
            '#000000')
        self.assertEqual(
            RGB_to_HEX(np.array([1, 1, 1])),
            '#ffffff')


class TestHEX_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.notation.triplet.HEX_to_RGB` definition unit tests
    methods.
    """

    def test_HEX_to_RGB(self):
        """
        Tests :func:`colour.notation.triplet.HEX_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            HEX_to_RGB('#3f990c'),
            np.array([0.25, 0.60, 0.05]),
            decimal=2)
        np.testing.assert_almost_equal(
            HEX_to_RGB('#000000'),
            np.array([0., 0., 0.]),
            decimal=2)
        np.testing.assert_almost_equal(
            HEX_to_RGB('#ffffff'),
            np.array([1., 1., 1.]),
            decimal=2)


if __name__ == '__main__':
    unittest.main()
