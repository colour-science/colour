#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.quality.cqs` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.quality import colour_quality_scale
from colour.colorimetry import (
    ILLUMINANTS_RELATIVE_SPDS,
    LIGHT_SOURCES_RELATIVE_SPDS)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2016 - Colour Developers'
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
                ILLUMINANTS_RELATIVE_SPDS.get('F1')),
            75.334361226715345,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                ILLUMINANTS_RELATIVE_SPDS.get('F2')),
            64.678111793396397,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_RELATIVE_SPDS.get('Neodimium Incandescent')),
            87.655549804699419,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_RELATIVE_SPDS.get('F32T8/TL841 (Triphosphor)')),
            83.175799064274571,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_RELATIVE_SPDS.get('H38HT-100 (Mercury)')),
            22.847928690340929,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_RELATIVE_SPDS.get('Luxeon WW 2880')),
            84.880575409680162,
            places=7)


if __name__ == '__main__':
    unittest.main()
