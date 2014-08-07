# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines units tests for :mod:`colour.colorimetry.luminance` module.
"""

from __future__ import unicode_literals

import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.colorimetry.luminance import (
    luminance_newhall1943,
    luminance_1976,
    luminance_ASTM_D1535_08)

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["TestLuminanceNewhall1943",
           "TestLuminance1976",
           "TestLuminanceASTM_D1535_08"]


class TestLuminanceNewhall1943(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.luminance_newhall1943`
    definition units tests methods.
    """

    def test_luminance_newhall1943(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_newhall1943`
        definition.
        """

        self.assertAlmostEqual(
            luminance_newhall1943(3.74629715382),
            10.4089874577,
            places=7)
        self.assertAlmostEqual(
            luminance_newhall1943(8.64728711385),
            71.3174801757,
            places=7)
        self.assertAlmostEqual(
            luminance_newhall1943(1.52569021578),
            2.06998750444,
            places=7)


class TestLuminance1976(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.luminance_1976` definition
    units tests methods.
    """

    def test_luminance_1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_1976` definition.
        """

        self.assertAlmostEqual(
            luminance_1976(37.9856290977, 100.),
            10.08,
            places=7)
        self.assertAlmostEqual(
            luminance_1976(80.0444155585, 100.),
            56.76,
            places=7)
        self.assertAlmostEqual(
            luminance_1976(99.3467279026, 100.),
            98.32,
            places=7)


class TestLuminanceASTM_D1535_08(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.luminance.luminance_ASTM_D1535_08`
    definition units tests methods.
    """

    def test_luminance_1976(self):
        """
        Tests :func:`colour.colorimetry.luminance.luminance_ASTM_D1535_08`
        definition.
        """

        self.assertAlmostEqual(
            luminance_ASTM_D1535_08(3.74629715382),
            10.1488096782,
            places=7)
        self.assertAlmostEqual(
            luminance_ASTM_D1535_08(8.64728711385),
            69.5324092373,
            places=7)
        self.assertAlmostEqual(
            luminance_ASTM_D1535_08(1.52569021578),
            2.01830631474,
            places=7)


if __name__ == "__main__":
    unittest.main()
