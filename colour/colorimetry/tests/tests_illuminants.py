#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.colorimetry.illuminants` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.colorimetry import D_illuminant_relative_spd

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['D60_SPD_DATA',
           'TestD_illuminantRelativeSpd']

D60_SPD_DATA = {
    300: 0.02937075817492288549,
    310: 2.61924131796496295621,
    320: 15.71689061312826041217,
    330: 28.77458026391913392672,
    340: 31.86483993666197989114,
    350: 36.37742644467410002562,
    360: 38.68311546316286353431,
    370: 42.71754846183496567846,
    380: 41.45494057963752254636,
    390: 46.60531924327943187336,
    400: 72.27859383884863575531,
    410: 80.44059999279464534538,
    420: 82.91502693894318554158,
    430: 77.67626397731756071607,
    440: 95.68127430379398390414,
    450: 107.95482086750595840385,
    460: 109.55918680507406293145,
    470: 107.75814070682791623312,
    480: 109.67140423534179660692,
    490: 103.70787331005091402858,
    500: 105.23219857523204723293,
    510: 104.42766692185435317697,
    520: 102.52293357805245932468,
    530: 106.05267087904782385976,
    540: 103.31515403458197965847,
    550: 103.53859891732658127239,
    560: 100.00000000000000000000,
    570: 96.75142141886689728381,
    580: 96.71282250154031601141,
    590: 89.92147908442616710545,
    600: 91.99979329504407132845,
    610: 92.09870955067275133388,
    620: 90.64600269701034562786,
    630: 86.52648272486028702133,
    640: 87.57918623550152403823,
    650: 83.97614003583295527733,
    660: 84.72407422805771659569,
    670: 87.49349084772983076164,
    680: 83.48307015694973642894,
    690: 74.17245111876663088424,
    700: 76.62038531099138083391,
    710: 79.05184907375583236444,
    720: 65.47137071741646252576,
    730: 74.10607902725251960874,
    740: 79.52712042772630240961,
    750: 67.30716277162383676114,
    760: 49.27353820615909540948,
    770: 70.89241211789024532663,
    780: 67.16399622630497390219,
    790: 68.17137071741646536793,
    800: 62.98980861670580111422,
    810: 54.99089236107711542445,
    820: 60.82560067091316824417,
    830: 63.89349586226155963686, }


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
                np.array([0.32168, 0.33767])).data.values()),
            sorted(D60_SPD_DATA.values()),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
