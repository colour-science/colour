# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.colorimetry.illuminants` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.colorimetry import (ILLUMINANTS_SPDS, SpectralShape,
                                spd_CIE_standard_illuminant_A,
                                spd_CIE_illuminant_D_series)
from colour.temperature import CCT_to_xy_CIE_D

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'A_DATA', 'TestSpdCIEStandardIlluminantA', 'TestSpd_CIEIlluminantDSeries'
]

A_DATA = np.array([
    6.14461778,
    6.94719899,
    7.82134941,
    8.76980228,
    9.79509961,
    10.89957616,
    12.08534536,
    13.35428726,
    14.70803845,
    16.14798414,
    17.67525215,
    19.29070890,
    20.99495729,
    22.78833636,
    24.67092269,
    26.64253337,
    28.70273044,
    30.85082676,
    33.08589297,
    35.40676571,
    37.81205669,
    40.30016269,
    42.86927625,
    45.51739693,
    48.24234315,
    51.04176432,
    53.91315329,
    56.85385894,
    59.86109896,
    62.93197247,
    66.06347275,
    69.25249966,
    72.49587199,
    75.79033948,
    79.13259455,
    82.51928367,
    85.94701837,
    89.41238582,
    92.91195891,
    96.44230599,
    100.00000000,
    103.58162718,
    107.18379528,
    110.80314124,
    114.43633837,
    118.08010305,
    121.73120094,
    125.38645263,
    129.04273891,
    132.69700551,
    136.34626740,
    139.98761262,
    143.61820577,
    147.23529096,
    150.83619449,
    154.41832708,
    157.97918574,
    161.51635535,
    165.02750987,
    168.51041325,
    171.96292009,
    175.38297597,
    178.76861756,
    182.11797252,
    185.42925911,
    188.70078570,
    191.93094995,
    195.11823798,
    198.26122323,
    201.35856531,
    204.40900861,
    207.41138086,
    210.36459156,
    213.26763031,
    216.11956508,
    218.91954039,
    221.66677545,
    224.36056217,
    227.00026327,
    229.58531022,
    232.11520118,
    234.58949900,
    237.00782911,
    239.36987744,
    241.67538835,
    243.92416253,
    246.11605493,
    248.25097273,
    250.32887325,
    252.34976196,
    254.31369047,
    256.22075454,
    258.07109218,
    259.86488167,
    261.60233977,
])


class TestSpdCIEStandardIlluminantA(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.illuminants.\
spd_CIE_standard_illuminant_A` definition unit tests methods.
    """

    def test_spd_CIE_standard_illuminant_A(self):
        """
        Tests :func:`colour.colorimetry.illuminants.\
spd_CIE_standard_illuminant_A` definition.
        """

        np.testing.assert_almost_equal(
            spd_CIE_standard_illuminant_A(SpectralShape(360, 830, 5)).values,
            A_DATA,
            decimal=7)


class TestSpd_CIEIlluminantDSeries(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.illuminants.spd_CIE_illuminant_D_series`
    definition unit tests methods.
    """

    def test_spd_CIE_illuminant_D_series(self):
        """
        Tests :func:`colour.colorimetry.illuminants.\
spd_CIE_illuminant_D_series` definition.
        """

        for name, CCT, tolerance in (
            ('D50', 5000, 0.001),
            ('D55', 5500, 0.001),
            ('D65', 6500, 0.00001),
            ('D75', 7500, 0.0001),
        ):
            CCT = CCT * 1.4388 / 1.4380
            xy = CCT_to_xy_CIE_D(CCT)
            spd_r = ILLUMINANTS_SPDS[name]
            spd_t = spd_CIE_illuminant_D_series(xy)

            np.testing.assert_allclose(
                spd_r.values,
                spd_t[spd_r.wavelengths],
                rtol=tolerance,
                atol=tolerance)


if __name__ == '__main__':
    unittest.main()
