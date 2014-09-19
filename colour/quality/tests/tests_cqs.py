#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.quality.cri` module.
"""

from __future__ import division, unicode_literals

import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.quality import colour_quality_scale

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestColourQualityScale']


class TestColourQualityScale(unittest.TestCase):
    """
    Defines :func:`colour.quality.cqs.colour_quality_scale`
    definition unit tests methods.
    """

    def test_colour_quality_scale(self):
        """
        Tests :func:`colour.quality.cri.colour_rendering_index` definition.
        """

        self.assertAlmostEqual(
            colour_quality_scale(ILLUMINANTS_RELATIVE_SPDS.get('F2')),
            64.1507331494,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(ILLUMINANTS_RELATIVE_SPDS.get('A')),
            99.9978916846,
            places=7)

        # self.assertAlmostEqual(
        #     colour_quality_scale(SpectralPowerDistribution(
        #         'Sample',
        #         SAMPLE_SPD_DATA)),
        #     70.805836753503698,
        #     places=7)


if __name__ == '__main__':
    unittest.main()
