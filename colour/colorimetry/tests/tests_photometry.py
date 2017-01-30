#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.colorimetry.photometry` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.colorimetry import (
    ILLUMINANTS_RELATIVE_SPDS,
    LIGHT_SOURCES_RELATIVE_SPDS,
    luminous_flux,
    luminous_efficiency,
    luminous_efficacy,
    zeros_spd)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestLuminousFlux',
           'TestLuminousEfficiency',
           'TestLuminousEfficacy']


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
            luminous_flux(
                ILLUMINANTS_RELATIVE_SPDS['F2'].clone().normalise()),
            28588.73612977,
            places=7)

        self.assertAlmostEqual(
            luminous_flux(LIGHT_SOURCES_RELATIVE_SPDS[
                'Neodimium Incandescent']),
            23807.65552737,
            places=7)

        self.assertAlmostEqual(
            luminous_flux(LIGHT_SOURCES_RELATIVE_SPDS[
                'F32T8/TL841 (Triphosphor)']),
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
            luminous_efficiency(
                ILLUMINANTS_RELATIVE_SPDS['F2'].clone().normalise()),
            0.49317624,
            places=7)

        self.assertAlmostEqual(
            luminous_efficiency(LIGHT_SOURCES_RELATIVE_SPDS[
                'Neodimium Incandescent']),
            0.19943936,
            places=7)

        self.assertAlmostEqual(
            luminous_efficiency(LIGHT_SOURCES_RELATIVE_SPDS[
                'F32T8/TL841 (Triphosphor)']),
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
            luminous_efficacy(
                ILLUMINANTS_RELATIVE_SPDS['F2'].clone().normalise()),
            336.83937176,
            places=7)

        self.assertAlmostEqual(
            luminous_efficacy(LIGHT_SOURCES_RELATIVE_SPDS[
                'Neodimium Incandescent']),
            136.21708032,
            places=7)

        self.assertAlmostEqual(
            luminous_efficacy(LIGHT_SOURCES_RELATIVE_SPDS[
                'F32T8/TL841 (Triphosphor)']),
            348.88267549,
            places=7)

        spd = zeros_spd()
        spd[555] = 1
        self.assertAlmostEqual(
            luminous_efficacy(spd),
            683.00000000,
            places=7)


if __name__ == '__main__':
    unittest.main()
