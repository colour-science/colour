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
from colour.colorimetry import (
    ILLUMINANTS_RELATIVE_SPDS,
    SpectralPowerDistribution)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestColourQualityScale']

SAMPLE_SPD_DATA = {
    380: 0.005883458,
    385: 0.003153768,
    390: 0.002428677,
    395: 0.005087091,
    400: 0.003232818,
    405: 0.003487637,
    410: 0.00369248,
    415: 0.005209241,
    420: 0.007479133,
    425: 0.013097946,
    430: 0.023971671,
    435: 0.04330206,
    440: 0.082721167,
    445: 0.141231868,
    450: 0.234004159,
    455: 0.342052302,
    460: 0.439128498,
    465: 0.448697658,
    470: 0.375497636,
    475: 0.278293162,
    480: 0.194531977,
    485: 0.141683533,
    490: 0.112335847,
    495: 0.103018715,
    500: 0.114389761,
    505: 0.145538096,
    510: 0.189716767,
    515: 0.251895814,
    520: 0.310723777,
    525: 0.359981034,
    530: 0.382088604,
    535: 0.376106021,
    540: 0.346534325,
    545: 0.308036724,
    550: 0.260159458,
    555: 0.216220023,
    560: 0.174484972,
    565: 0.135613979,
    570: 0.10873008,
    575: 0.085992361,
    580: 0.068631638,
    585: 0.058752863,
    590: 0.052765794,
    595: 0.055485988,
    600: 0.072911542,
    605: 0.153199437,
    610: 0.387537401,
    615: 0.817543224,
    620: 1.,
    625: 0.647943604,
    630: 0.213755255,
    635: 0.037105254,
    640: 0.017615101,
    645: 0.014653118,
    650: 0.013849082,
    655: 0.014657157,
    660: 0.013470592,
    665: 0.014247679,
    670: 0.012157914,
    675: 0.012093381,
    680: 0.011553132,
    685: 0.010619949,
    690: 0.010147794,
    695: 0.008642115,
    700: 0.009513861,
    705: 0.007869817,
    710: 0.008414764,
    715: 0.007418679,
    720: 0.006377112,
    725: 0.005564827,
    730: 0.005900156,
    735: 0.004168186,
    740: 0.004222223,
    745: 0.003457765,
    750: 0.003368788,
    755: 0.002989992,
    760: 0.003670466,
    765: 0.00340568,
    770: 0.002611528,
    775: 0.002588501,
    780: 0.002936635}


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
            58.2579269222,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(ILLUMINANTS_RELATIVE_SPDS.get('A')),
            98.1216443680,
            places=7)

        self.assertAlmostEqual(
            colour_quality_scale(SpectralPowerDistribution(
                'Sample',
                SAMPLE_SPD_DATA)),
            66.3696990636,
            places=7)


if __name__ == '__main__':
    unittest.main()
