# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.colorimetry.generation` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.colorimetry.generation import (
    constant_spd, zeros_spd, ones_spd, gaussian_spd_normal, gaussian_spd_fwhm,
    single_led_spd_Ohno2005, multi_led_spd_Ohno2005)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestConstantSpd', 'TestZerosSpd', 'TestOnesSpd', 'TestGaussianSpdNormal',
    'TestGaussianSpdFwhm', 'TestSingleLedSpdOhno2005',
    'TestMultiLedSpdOhno2005'
]


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


class TestGaussianSpdNormal(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.generation.gaussian_spd_normal`
    definition unit tests methods.
    """

    def test_gaussian_spd_normal(self):
        """
        Tests :func:`colour.colorimetry.generation.gaussian_spd_normal`
        definition.
        """

        spd = gaussian_spd_normal(555, 25)

        self.assertAlmostEqual(spd[530], 0.606530659712633, places=7)

        self.assertAlmostEqual(spd[555], 1, places=7)

        self.assertAlmostEqual(spd[580], 0.606530659712633, places=7)


class TestGaussianSpdFwhm(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.generation.gaussian_spd_fwhm` definition
    unit tests methods.
    """

    def test_gaussian_spd_fwhm(self):
        """
        Tests :func:`colour.colorimetry.generation.gaussian_spd_fwhm`
        definition.
        """

        spd = gaussian_spd_fwhm(555, 25)

        self.assertAlmostEqual(spd[530], 0.367879441171443, places=7)

        self.assertAlmostEqual(spd[555], 1, places=7)

        self.assertAlmostEqual(spd[580], 0.367879441171443, places=7)


class TestSingleLedSpdOhno2005(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.generation.single_led_spd_Ohno2005`
    definition unit tests methods.
    """

    def test_single_led_spd_Ohno2005(self):
        """
        Tests :func:`colour.colorimetry.generation.single_led_spd_Ohno2005`
        definition.
        """

        spd = single_led_spd_Ohno2005(555, 25)

        self.assertAlmostEqual(spd[530], 0.127118445056538, places=7)

        self.assertAlmostEqual(spd[555], 1, places=7)

        self.assertAlmostEqual(spd[580], 0.127118445056538, places=7)


class TestMultiLedSpdOhno2005(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.generation.multi_led_spd_Ohno2005`
    definition unit tests methods.
    """

    def test_multi_led_spd_Ohno2005(self):
        """
        Tests :func:`colour.colorimetry.generation.multi_led_spd_Ohno2005`
        definition.
        """

        spd = multi_led_spd_Ohno2005(
            np.array([457, 530, 615]),
            np.array([20, 30, 20]),
            np.array([0.731, 1.000, 1.660]),
        )

        self.assertAlmostEqual(spd[500], 0.129513248576116, places=7)

        self.assertAlmostEqual(spd[570], 0.059932156222703, places=7)

        self.assertAlmostEqual(spd[640], 0.116433257970624, places=7)

        spd = multi_led_spd_Ohno2005(
            np.array([457, 530, 615]),
            np.array([20, 30, 20]),
        )

        self.assertAlmostEqual(spd[500], 0.130394510062799, places=7)

        self.assertAlmostEqual(spd[570], 0.058539618824187, places=7)

        self.assertAlmostEqual(spd[640], 0.070140708922879, places=7)


if __name__ == '__main__':
    unittest.main()
