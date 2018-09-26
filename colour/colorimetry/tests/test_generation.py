# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.colorimetry.generation` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.colorimetry.generation import constant_spd, zeros_spd, ones_spd

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestConstantSpd', 'TestZerosSpd', 'TestOnesSpd']


class TestConstantSpd(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.generation.constant_spd` definition unit
    tests methods.
    """

    def test_constant_spd(self):
        """
        Tests :func:`colour.colorimetry.generation.constant_spd`
        definition.
        """

        spd = constant_spd(np.pi)

        self.assertAlmostEqual(spd[360], np.pi, places=7)

        self.assertAlmostEqual(spd[555], np.pi, places=7)

        self.assertAlmostEqual(spd[780], np.pi, places=7)


class TestZerosSpd(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.generation.zeros_spd` definition unit
    tests methods.
    """

    def test_zeros_spd(self):
        """
        Tests :func:`colour.colorimetry.generation.zeros_spd`
        definition.
        """

        spd = zeros_spd()

        self.assertEqual(spd[360], 0)

        self.assertEqual(spd[555], 0)

        self.assertEqual(spd[780], 0)


class TestOnesSpd(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.generation.ones_spd` definition unit
    tests methods.
    """

    def test_ones_spd(self):
        """
        Tests :func:`colour.colorimetry.generation.ones_spd`
        definition.
        """

        spd = ones_spd()

        self.assertEqual(spd[360], 1)

        self.assertEqual(spd[555], 1)

        self.assertEqual(spd[780], 1)


if __name__ == '__main__':
    unittest.main()
