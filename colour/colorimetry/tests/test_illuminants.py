# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.colorimetry.illuminants` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.colorimetry import (D_illuminant_relative_spd,
                                CIE_standard_illuminant_A_function,
                                ILLUMINANTS_SPDS)
from colour.temperature import CCT_to_xy_CIE_D

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'A_DATA', 'TestD_illuminantRelativeSpd',
    'TestCIEStandardIlluminantAFunction'
]

A_DATA = np.array([
    6.144617784123856,
    6.947198985402079,
    7.821349414981689,
    8.769802283876084,
    9.795099608867382,
    10.899576157801631,
    12.085345363411140,
    13.354287257777719,
    14.708038449875502,
    16.147984141480254,
    17.675252152303049,
    19.290708903610355,
    20.994957290865997,
    22.788336360042955,
    24.670922689127945,
    26.642533365850888,
    28.702730444663004,
    30.850826760279453,
    33.085892971502503,
    35.406765707340028,
    37.812056687427500,
    40.300162690239553,
    42.869276245337296,
    45.517396929746482,
    48.242343153313406,
    51.041764323370195,
    53.913153285099291,
    56.853858940467404,
    59.861098955389089,
    62.931972471732792,
    66.063472747830630,
    69.252499658171686,
    72.495871989904842,
    75.790339480551324,
    79.132594547909648,
    82.519283669449450,
    85.947018374529335,
    89.412385818490364,
    92.911958913061213,
    96.442305992552875,
    100.000000000000000,
    103.581627181740913,
    107.183795282900803,
    110.803141239869944,
    114.436338369157482,
    118.080103054962819,
    121.731200940444666,
    125.386452630022220,
    129.042738912085099,
    132.697005513293647,
    136.346267397171545,
    139.987612621004047,
    143.618205766130785,
    147.235290957601734,
    150.836194489858400,
    154.418327075631083,
    157.979185735603039,
    161.516355346632452,
    165.027509866420871,
    168.510413252511256,
    171.962920093394303,
    175.382975969299480,
    178.768617559985131,
    182.117972516492841,
    185.429259113445994,
    188.700785698018507,
    191.930949951225926,
    195.118237976664375,
    198.261223231287033,
    201.358565312239051,
    204.409008613197130,
    207.411380863071741,
    210.364591559332979,
    213.267630307635471,
    216.119565078810581,
    218.919540393725441,
    221.666775445909082,
    224.360562171292912,
    227.000263273843757,
    229.585310215328150,
    232.115201176917310,
    234.589498999828919,
    237.007829111703302,
    239.369877444937316,
    241.675388352737258,
    243.924162528208456,
    246.116054931382024,
    248.250972728666568,
    250.328873248841006,
    252.349761959321199,
    254.313690466103111,
    256.220754540440964,
    258.071092175015735,
    259.864881672054366,
])


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

        for name, CCT, tolerance in (
            ('D50', 5000, 0.001),
            ('D55', 5500, 0.001),
            ('D65', 6500, 0.00001),
            ('D75', 7500, 0.0001),
        ):
            CCT = CCT * 1.4388 / 1.4380
            xy = CCT_to_xy_CIE_D(CCT)
            spd_r = ILLUMINANTS_SPDS[name]
            spd_t = D_illuminant_relative_spd(xy)

            np.testing.assert_allclose(
                spd_r.values,
                spd_t[spd_r.wavelengths],
                rtol=tolerance,
                atol=tolerance)


class TestCIEStandardIlluminantAFunction(unittest.TestCase):
    """
    Defines :func:`colour.colorimetry.illuminants.\
CIE_standard_illuminant_A_function` definition unit tests methods.
    """

    def test_CIE_standard_illuminant_A_function(self):
        """
        Tests :func:`colour.colorimetry.illuminants.\
CIE_standard_illuminant_A_function` definition.
        """

        np.testing.assert_almost_equal(
            CIE_standard_illuminant_A_function(np.arange(360, 830, 5)),
            A_DATA,
            decimal=7)


if __name__ == '__main__':
    unittest.main()
