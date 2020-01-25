# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.colorimetry.photometry` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.colorimetry import (ILLUMINANTS_SDS, LIGHT_SOURCES_SDS,
                                luminous_flux, luminous_efficiency,
                                luminous_efficacy, sd_zeros)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestLuminousFlux', 'TestLuminousEfficiency', 'TestLuminousEfficacy'
]


class TestLuminousFlux(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.photometry.luminous_flux` definition unit
    tests methods.
    """

    def test_luminous_flux(self):
        """
        Tests :func:`colour.colorimetry.photometry.luminous_flux` definition.
        """

        self.assertAlmostEqual(
            luminous_flux(ILLUMINANTS_SDS['FL2'].copy().normalise()),
            28588.73612977,
            places=7)

        self.assertAlmostEqual(
            luminous_flux(LIGHT_SOURCES_SDS['Neodimium Incandescent']),
            23807.65552737,
            places=7)

        self.assertAlmostEqual(
            luminous_flux(LIGHT_SOURCES_SDS['F32T8/TL841 (Triphosphor)']),
            13090.06759053,
            places=7)


class TestLuminousEfficiency(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.photometry.luminous_efficiency`
    definition unit tests methods.
    """

    def test_luminous_efficiency(self):
        """
        Tests :func:`colour.colorimetry.photometry.luminous_efficiency`
        definition.
        """

        self.assertAlmostEqual(
            luminous_efficiency(ILLUMINANTS_SDS['FL2'].copy().normalise()),
            0.49317624,
            places=7)

        self.assertAlmostEqual(
            luminous_efficiency(LIGHT_SOURCES_SDS['Neodimium Incandescent']),
            0.19943936,
            places=7)

        self.assertAlmostEqual(
            luminous_efficiency(
                LIGHT_SOURCES_SDS['F32T8/TL841 (Triphosphor)']),
            0.51080919,
            places=7)


class TestLuminousEfficacy(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.photometry.luminous_efficacy`
    definition unit tests methods.
    """

    def test_luminous_efficacy(self):
        """
        Tests :func:`colour.colorimetry.photometry.luminous_efficacy`
        definition.
        """

        self.assertAlmostEqual(
            luminous_efficacy(ILLUMINANTS_SDS['FL2'].copy().normalise()),
            336.83937176,
            places=7)

        self.assertAlmostEqual(
            luminous_efficacy(LIGHT_SOURCES_SDS['Neodimium Incandescent']),
            136.21708032,
            places=7)

        self.assertAlmostEqual(
            luminous_efficacy(LIGHT_SOURCES_SDS['F32T8/TL841 (Triphosphor)']),
            348.88267549,
            places=7)

        sd = sd_zeros()
        sd[555] = 1
        self.assertAlmostEqual(luminous_efficacy(sd), 683.00000000, places=7)


if __name__ == '__main__':
    unittest.main()
