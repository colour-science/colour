# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.algebra.common` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.algebra import (is_spow_enabled, set_spow_enable, spow_enable,
                            spow, smoothstep_function)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestIsSpowEnabled', 'TestSetSpowEnabled', 'TestSpowEnable', 'TestSpow',
    'TestSmoothstepFunction'
]


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

        @spow_enable(True)
        def fn_a():
            """
            :func:`spow_enable` unit tests :func:`fn_a` definition.
            """

            self.assertTrue(is_spow_enabled())

        fn_a()

        @spow_enable(False)
        def fn_b():
            """
            :func:`spow_enable` unit tests :func:`fn_b` definition.
            """

            self.assertFalse(is_spow_enabled())

        fn_b()


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


class TestSmoothstepFunction(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.smoothstep_function` definition unit
    tests methods.
    """

    def test_smoothstep_function(self):
        """
        Tests :func:`colour.algebra.common.smoothstep_function` definition.
        """

        self.assertEqual(smoothstep_function(0.5), 0.5)
        self.assertEqual(smoothstep_function(0.25), 0.15625)
        self.assertEqual(smoothstep_function(0.75), 0.84375)

        x = np.linspace(-2, 2, 5)
        np.testing.assert_almost_equal(
            smoothstep_function(x),
            np.array([28.00000, 5.00000, 0.00000, 1.00000, -4.00000]))
        np.testing.assert_almost_equal(
            smoothstep_function(x, -2, 2, clip=True),
            np.array([0.00000, 0.15625, 0.50000, 0.84375, 1.00000]))


if __name__ == '__main__':
    unittest.main()
