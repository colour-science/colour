# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.quality.cri` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.quality import CRI_Specification, colour_rendering_index
from colour.colorimetry import ILLUMINANTS_SDS, SpectralDistribution
from colour.quality.cri import TCS_ColorimetryData, TCS_ColourQualityScaleData

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestColourRenderingIndex']

SAMPLE_SD_DATA = {
    380: 0.00588346,
    385: 0.00315377,
    390: 0.00242868,
    395: 0.00508709,
    400: 0.00323282,
    405: 0.00348764,
    410: 0.00369248,
    415: 0.00520924,
    420: 0.00747913,
    425: 0.01309795,
    430: 0.02397167,
    435: 0.04330206,
    440: 0.08272117,
    445: 0.14123187,
    450: 0.23400416,
    455: 0.34205230,
    460: 0.43912850,
    465: 0.44869766,
    470: 0.37549764,
    475: 0.27829316,
    480: 0.19453198,
    485: 0.14168353,
    490: 0.11233585,
    495: 0.10301871,
    500: 0.11438976,
    505: 0.14553810,
    510: 0.18971677,
    515: 0.25189581,
    520: 0.31072378,
    525: 0.35998103,
    530: 0.38208860,
    535: 0.37610602,
    540: 0.34653432,
    545: 0.30803672,
    550: 0.26015946,
    555: 0.21622002,
    560: 0.17448497,
    565: 0.13561398,
    570: 0.10873008,
    575: 0.08599236,
    580: 0.06863164,
    585: 0.05875286,
    590: 0.05276579,
    595: 0.05548599,
    600: 0.07291154,
    605: 0.15319944,
    610: 0.38753740,
    615: 0.81754322,
    620: 1.00000000,
    625: 0.64794360,
    630: 0.21375526,
    635: 0.03710525,
    640: 0.01761510,
    645: 0.01465312,
    650: 0.01384908,
    655: 0.01465716,
    660: 0.01347059,
    665: 0.01424768,
    670: 0.01215791,
    675: 0.01209338,
    680: 0.01155313,
    685: 0.01061995,
    690: 0.01014779,
    695: 0.00864212,
    700: 0.00951386,
    705: 0.00786982,
    710: 0.00841476,
    715: 0.00741868,
    720: 0.00637711,
    725: 0.00556483,
    730: 0.00590016,
    735: 0.00416819,
    740: 0.00422222,
    745: 0.00345776,
    750: 0.00336879,
    755: 0.00298999,
    760: 0.00367047,
    765: 0.00340568,
    770: 0.00261153,
    775: 0.00258850,
    780: 0.00293663
}


class TestColourRenderingIndex(unittest.TestCase):
    """
    Defines :func:`colour.quality.cri.colour_rendering_index`
    definition unit tests methods.
    """

    def test_colour_rendering_index(self):
        """
        Tests :func:`colour.quality.cri.colour_rendering_index` definition.
        """

        self.assertAlmostEqual(
            colour_rendering_index(ILLUMINANTS_SDS['FL1']),
            75.821550491976069,
            places=7)

        self.assertAlmostEqual(
            colour_rendering_index(ILLUMINANTS_SDS['FL2']),
            64.151520202968015,
            places=7)

        self.assertAlmostEqual(
            colour_rendering_index(ILLUMINANTS_SDS['A']),
            99.996732643006169,
            places=7)

        self.assertAlmostEqual(
            colour_rendering_index(SpectralDistribution(SAMPLE_SD_DATA)),
            70.813839034481575,
            places=7)

        specification_r = CRI_Specification(
            name='FL1',
            Q_a=75.821550491976069,
            Q_as={
                1:
                    TCS_ColourQualityScaleData(
                        name='TCS01', Q_a=69.146993939830651),
                2:
                    TCS_ColourQualityScaleData(
                        name='TCS02', Q_a=83.621259237346464),
                3:
                    TCS_ColourQualityScaleData(
                        name='TCS03', Q_a=92.103176674726171),
                4:
                    TCS_ColourQualityScaleData(
                        name='TCS04', Q_a=72.659091884741599),
                5:
                    TCS_ColourQualityScaleData(
                        name='TCS05', Q_a=73.875477056438399),
                6:
                    TCS_ColourQualityScaleData(
                        name='TCS06', Q_a=79.561558275578435),
                7:
                    TCS_ColourQualityScaleData(
                        name='TCS07', Q_a=82.238690395582836),
                8:
                    TCS_ColourQualityScaleData(
                        name='TCS08', Q_a=53.366156471564040),
                9:
                    TCS_ColourQualityScaleData(
                        name='TCS09', Q_a=-47.429377845680563),
                10:
                    TCS_ColourQualityScaleData(
                        name='TCS10', Q_a=61.459446471229136),
                11:
                    TCS_ColourQualityScaleData(
                        name='TCS11', Q_a=67.488681008259860),
                12:
                    TCS_ColourQualityScaleData(
                        name='TCS12', Q_a=74.912512331077920),
                13:
                    TCS_ColourQualityScaleData(
                        name='TCS13', Q_a=72.752285379884341),
                14:
                    TCS_ColourQualityScaleData(
                        name='TCS14', Q_a=94.874668874605561)
            },
            colorimetry_data=([
                TCS_ColorimetryData(
                    name='TCS01',
                    XYZ=np.array([31.20331782, 29.74794094, 23.46699878]),
                    uv=np.array([0.22783485, 0.32581236]),
                    UVW=np.array([25.41292946, 8.7380799, 60.46264324])),
                TCS_ColorimetryData(
                    name='TCS02',
                    XYZ=np.array([26.7148606, 29.25905082, 14.00052658]),
                    uv=np.array([0.21051808, 0.34585016]),
                    UVW=np.array([11.89933381, 24.70828029, 60.03594595])),
                TCS_ColorimetryData(
                    name='TCS03',
                    XYZ=np.array([24.20882878, 31.45657446, 9.19966949]),
                    uv=np.array([0.18492146, 0.36042609]),
                    UVW=np.array([-8.1256375, 37.50367669, 61.91819636])),
                TCS_ColorimetryData(
                    name='TCS04',
                    XYZ=np.array([20.75294445, 29.43766611, 19.42067879]),
                    uv=np.array([0.15946018, 0.33928696]),
                    UVW=np.array([-27.55034027, 19.3247759, 60.19238635])),
                TCS_ColorimetryData(
                    name='TCS05',
                    XYZ=np.array([25.10593837, 30.60666903, 37.97182815]),
                    uv=np.array([0.1678986, 0.30702797]),
                    UVW=np.array([-21.29395798, -6.63937802, 61.20095038])),
                TCS_ColorimetryData(
                    name='TCS06',
                    XYZ=np.array([27.45144596, 28.84090621, 55.21254008]),
                    uv=np.array([0.17549196, 0.27656177]),
                    UVW=np.array([-14.89855536, -30.53194048, 59.66720711])),
                TCS_ColorimetryData(
                    name='TCS07',
                    XYZ=np.array([31.31828342, 28.49717267, 51.6262295]),
                    uv=np.array([0.20414276, 0.27863076]),
                    UVW=np.array([6.89004159, -28.68174855, 59.36140901])),
                TCS_ColorimetryData(
                    name='TCS08',
                    XYZ=np.array([33.82293106, 29.92724727, 44.13391542]),
                    uv=np.array([0.21993884, 0.29190983]),
                    UVW=np.array([19.28664162, -18.59403459, 60.61796747])),
                TCS_ColorimetryData(
                    name='TCS09',
                    XYZ=np.array([14.74628552, 9.07383027, 4.24660598]),
                    uv=np.array([0.36055908, 0.33279417]),
                    UVW=np.array([74.80963996, 8.5929104, 35.14390587])),
                TCS_ColorimetryData(
                    name='TCS10',
                    XYZ=np.array([53.38914901, 60.57766887, 10.90608553]),
                    uv=np.array([0.21467884, 0.36537604]),
                    UVW=np.array([20.46081762, 54.70909846, 81.18478519])),
                TCS_ColorimetryData(
                    name='TCS11',
                    XYZ=np.array([12.25740399, 19.77026703, 14.05795099]),
                    uv=np.array([0.13969138, 0.33796747]),
                    UVW=np.array([-35.99235266, 15.30093833, 50.59960948])),
                TCS_ColorimetryData(
                    name='TCS12',
                    XYZ=np.array([5.7789513, 5.68760485, 24.7298429]),
                    uv=np.array([0.13985629, 0.20646843]),
                    UVW=np.array([-19.30294902, -39.56090075, 27.62550537])),
                TCS_ColorimetryData(
                    name='TCS13',
                    XYZ=np.array([56.63907549, 57.50545374, 39.17805]),
                    uv=np.array([0.21852443, 0.33280062]),
                    UVW=np.array([23.92148732, 18.87040227, 79.49608348])),
                TCS_ColorimetryData(
                    name='TCS14',
                    XYZ=np.array([9.42594992, 12.07064446, 5.06612415]),
                    uv=np.array([0.18330936, 0.35211232]),
                    UVW=np.array([-6.1238664, 19.9332886, 40.34780872]))
            ], [
                TCS_ColorimetryData(
                    name='TCS01',
                    XYZ=np.array([33.04406274, 29.80722965, 24.25677014]),
                    uv=np.array([0.23905009, 0.32345089]),
                    UVW=np.array([32.11150701, 8.4025005, 60.51407102])),
                TCS_ColorimetryData(
                    name='TCS02',
                    XYZ=np.array([27.53323581, 28.90793444, 14.76248032]),
                    uv=np.array([0.21789532, 0.34316182]),
                    UVW=np.array([15.26809231, 23.59761128, 59.72655443])),
                TCS_ColorimetryData(
                    name='TCS03',
                    XYZ=np.array([23.95435019, 30.44568852, 9.80615507]),
                    uv=np.array([0.18785584, 0.35814374]),
                    UVW=np.array([-8.23625903, 36.01892099, 61.06360597])),
                TCS_ColorimetryData(
                    name='TCS04',
                    XYZ=np.array([20.43638681, 29.46870454, 21.05101191]),
                    uv=np.array([0.15552214, 0.33638794]),
                    UVW=np.array([-33.43495759, 18.48941724, 60.21950681])),
                TCS_ColorimetryData(
                    name='TCS05',
                    XYZ=np.array([24.95728556, 30.82110743, 39.94366077]),
                    uv=np.array([0.16443476, 0.30460412]),
                    UVW=np.array([-26.96894169, -6.51619473, 61.38315768])),
                TCS_ColorimetryData(
                    name='TCS06',
                    XYZ=np.array([28.15012994, 29.75806707, 57.21213255]),
                    uv=np.array([0.17426171, 0.27632333]),
                    UVW=np.array([-18.84311713, -28.6517426, 60.4714316])),
                TCS_ColorimetryData(
                    name='TCS07',
                    XYZ=np.array([33.29875981, 29.36956806, 52.57688943]),
                    uv=np.array([0.21089415, 0.27901355]),
                    UVW=np.array([9.89894512, -26.38829247, 60.13281743])),
                TCS_ColorimetryData(
                    name='TCS08',
                    XYZ=np.array([37.64824851, 31.35309879, 44.88145951]),
                    uv=np.array([0.23435348, 0.29275098]),
                    UVW=np.array([29.03544464, -16.09146766, 61.83156811])),
                TCS_ColorimetryData(
                    name='TCS09',
                    XYZ=np.array([20.68888349, 11.28578196, 4.29031293]),
                    uv=np.array([0.40797112, 0.33382225]),
                    UVW=np.array([106.54776544, 10.69454868, 39.07688665])),
                TCS_ColorimetryData(
                    name='TCS10',
                    XYZ=np.array([55.03120276, 59.04411929, 11.87116937]),
                    uv=np.array([0.22546691, 0.36286219]),
                    UVW=np.array([28.44874091, 52.32328834, 80.34916371])),
                TCS_ColorimetryData(
                    name='TCS11',
                    XYZ=np.array([12.13121967, 20.3541437, 15.18142797]),
                    uv=np.array([0.1336819, 0.33644357]),
                    UVW=np.array([-43.01323833, 15.77519367, 51.25863839])),
                TCS_ColorimetryData(
                    name='TCS12',
                    XYZ=np.array([6.19245638, 6.41273455, 27.30815071]),
                    uv=np.array([0.13439371, 0.20876154]),
                    UVW=np.array([-24.4374163, -39.8151001, 29.44665364])),
                TCS_ColorimetryData(
                    name='TCS13',
                    XYZ=np.array([58.97285728, 57.14500933, 40.86401249]),
                    uv=np.array([0.22709381, 0.33008264]),
                    UVW=np.array([29.75220364, 17.84629845, 79.29404816])),
                TCS_ColorimetryData(
                    name='TCS14',
                    XYZ=np.array([9.34392028, 11.70931454, 5.33816006]),
                    uv=np.array([0.1859504, 0.34953505]),
                    UVW=np.array([-6.3492713, 19.0078073, 39.7697741]))
            ]))

        specification_t = colour_rendering_index(
            ILLUMINANTS_SDS['FL1'], additional_data=True)

        np.testing.assert_almost_equal(
            [
                data.Q_a
                for _index, data in sorted(specification_r.Q_as.items())
            ],
            [
                data.Q_a
                for _index, data in sorted(specification_t.Q_as.items())
            ],
            decimal=7,
        )


if __name__ == '__main__':
    unittest.main()
