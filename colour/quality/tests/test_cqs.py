# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.quality.cqs` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.quality import colour_quality_scale
from colour.colorimetry import ILLUMINANTS_SDS, LIGHT_SOURCES_SDS

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
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
            colour_quality_scale(ILLUMINANTS_SDS['FL1']),
            74.933405395713180,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                ILLUMINANTS_SDS['FL1'], method='NIST CQS 7.4'),
            75.332008182589348,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(ILLUMINANTS_SDS['FL2']),
            64.017283509280588,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                ILLUMINANTS_SDS['FL2'], method='NIST CQS 7.4'),
            64.686339173112856,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(LIGHT_SOURCES_SDS['Neodimium Incandescent']),
            89.693921013642381,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_SDS['Neodimium Incandescent'],
                method='NIST CQS 7.4'),
            87.655035241231985,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_SDS['F32T8/TL841 (Triphosphor)']),
            84.878441814420910,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_SDS['F32T8/TL841 (Triphosphor)'],
                method='NIST CQS 7.4'),
            83.179881092827671,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(LIGHT_SOURCES_SDS['H38HT-100 (Mercury)']),
            19.836071708638958,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_SDS['H38HT-100 (Mercury)'],
                method='NIST CQS 7.4'),
            22.860610106043985,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(LIGHT_SOURCES_SDS['Luxeon WW 2880']),
            86.491761709787994,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(
                LIGHT_SOURCES_SDS['Luxeon WW 2880'], method='NIST CQS 7.4'),
            84.879524259605077,
            places=7)


if __name__ == '__main__':
    unittest.main()
