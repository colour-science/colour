# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.algebra.common` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.algebra import is_spow_enabled, set_spow_enable, spow_enable, spow

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestSpow']


class TestIsSpowEnabled(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.is_spow_enabled` definition unit
    tests methods.
    """

    def test_is_spow_enabled(self):
        """
        Tests :func:`colour.algebra.common.is_spow_enabled` definition.
        """

        with spow_enable(True):
            self.assertTrue(is_spow_enabled())

        with spow_enable(False):
            self.assertFalse(is_spow_enabled())


class TestSetSpowEnabled(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.set_spow_enable` definition unit
    tests methods.
    """

    def test_set_spow_enable(self):
        """
        Tests :func:`colour.algebra.common.set_spow_enable` definition.
        """

        with spow_enable(is_spow_enabled()):
            set_spow_enable(True)
            self.assertTrue(is_spow_enabled())

        with spow_enable(is_spow_enabled()):
            set_spow_enable(False)
            self.assertFalse(is_spow_enabled())


class TestSpowEnable(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.spow_enable` definition unit
    tests methods.
    """

    def test_spow_enable(self):
        """
        Tests :func:`colour.algebra.common.spow_enable` definition.
        """

        with spow_enable(True):
            self.assertTrue(is_spow_enabled())

        with spow_enable(False):
            self.assertFalse(is_spow_enabled())


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

        with spow_enable(True):
            np.testing.assert_almost_equal(
                spow(-2, 0.15), -1.10956947, decimal=7)

        with spow_enable(False):
            np.testing.assert_equal(spow(-2, 0.15), np.nan)


if __name__ == '__main__':
    unittest.main()
