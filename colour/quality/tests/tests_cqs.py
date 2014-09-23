#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.quality.cqs` module.
"""

from __future__ import division, unicode_literals

import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.quality import colour_quality_scale
from colour.colorimetry import LIGHT_SOURCES_RELATIVE_SPDS

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestColourQualityScale']


class TestColourQualityScale(unittest.TestCase):
    """
    Defines :func:`colour.quality.cqs.colour_quality_scale` definition unit
    tests methods.
    """

    def test_colour_quality_scale(self):
        """
        Tests :func:`colour.quality.cqs.colour_quality_scale` definition.
        """

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_RELATIVE_SPDS.get('Neodimium Incandescent')),
            77.,
            delta=0.5)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_RELATIVE_SPDS.get('F32T8/TL841 (Triphosphor)')),
            85.,
            delta=0.5)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_RELATIVE_SPDS.get('H38HT-100 (Mercury)')),
            53.,
            delta=0.5)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_RELATIVE_SPDS.get('SDW-T 100W/LV (Super HPS)')),
            85.,
            delta=0.5)


if __name__ == '__main__':
    unittest.main()
