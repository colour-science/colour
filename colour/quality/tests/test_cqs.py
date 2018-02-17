# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.quality.cqs` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.quality import colour_quality_scale
from colour.colorimetry import (ILLUMINANTS_RELATIVE_SPDS,
                                LIGHT_SOURCES_RELATIVE_SPDS)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
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
            colour_quality_scale(ILLUMINANTS_RELATIVE_SPDS['F1']),
            75.342591389578701,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(ILLUMINANTS_RELATIVE_SPDS['F2']),
            64.686339173112856,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_RELATIVE_SPDS['Neodimium Incandescent']),
            87.655035241231985,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_RELATIVE_SPDS['F32T8/TL841 (Triphosphor)']),
            83.179881092827671,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_RELATIVE_SPDS['H38HT-100 (Mercury)']),
            22.870604734960732,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_RELATIVE_SPDS['Luxeon WW 2880']),
            84.879524259605077,
            places=7)


if __name__ == '__main__':
    unittest.main()
