#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.colorimetry.illuminants` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.colorimetry import D_illuminant_relative_spd

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['D60_SPD_DATA',
           'TestD_illuminantRelativeSpd']

D60_SPD_DATA = {
    300: 0.029370758174922885,
    310: 2.619241317964963,
    320: 15.71689061312826,
    330: 28.774580263919134,
    340: 31.86483993666198,
    350: 36.3774264446741,
    360: 38.68311546316286,
    370: 42.717548461834966,
    380: 41.45494057963752,
    390: 46.60531924327943,
    400: 72.27859383884864,
    410: 80.44059999279465,
    420: 82.91502693894319,
    430: 77.67626397731756,
    440: 95.68127430379398,
    450: 107.95482086750596,
    460: 109.55918680507406,
    470: 107.75814070682792,
    480: 109.6714042353418,
    490: 103.70787331005091,
    500: 105.23219857523205,
    510: 104.42766692185435,
    520: 102.52293357805246,
    530: 106.05267087904782,
    540: 103.31515403458198,
    550: 103.53859891732658,
    560: 100.0,
    570: 96.7514214188669,
    580: 96.71282250154032,
    590: 89.92147908442617,
    600: 91.99979329504407,
    610: 92.09870955067275,
    620: 90.64600269701035,
    630: 86.52648272486029,
    640: 87.57918623550152,
    650: 83.97614003583296,
    660: 84.72407422805772,
    670: 87.49349084772983,
    680: 83.48307015694974,
    690: 74.17245111876663,
    700: 76.62038531099138,
    710: 79.05184907375583,
    720: 65.47137071741646,
    730: 74.10607902725252,
    740: 79.5271204277263,
    750: 67.30716277162384,
    760: 49.273538206159095,
    770: 70.89241211789025,
    780: 67.16399622630497,
    790: 68.17137071741647,
    800: 62.9898086167058,
    810: 54.990892361077115,
    820: 60.82560067091317,
    830: 63.89349586226156}


class TestD_illuminantRelativeSpd(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.illuminants.D_illuminant_relative_spd`
    definition unit tests methods.
    """

    def test_D_illuminant_relative_spd(self):
        """
        Tests :func:`colour.colorimetry.illuminants.D_illuminant_relative_spd`
        definition.
        """

        np.testing.assert_almost_equal(
            sorted(D_illuminant_relative_spd(
                (0.32168, 0.33767)).data.values()),
            sorted(D60_SPD_DATA.values()),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
