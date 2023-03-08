# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.quality.ssi` module."""

from __future__ import annotations

import unittest

from colour.colorimetry import SDS_ILLUMINANTS, SpectralDistribution
from colour.quality import spectral_similarity_index

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestSpectralSimilarityIndex",
]

DATA_HMI: dict = {
    300: 0.000000000000000,
    301: 0.000000000000000,
    302: 0.000000000000000,
    303: 0.000000000000000,
    304: 0.000000000000000,
    305: 0.000000000000000,
    306: 0.000000000000000,
    307: 0.000000000000000,
    308: 0.000000000000000,
    309: 0.000000000000000,
    310: 0.000000000000000,
    311: 0.000000000000000,
    312: 0.000000000000000,
    313: 0.000000000000000,
    314: 0.000000000000000,
    315: 0.000000000000000,
    316: 0.000000000000000,
    317: 0.000000000000000,
    318: 0.000000000000000,
    319: 0.000000000000000,
    320: 0.000000000000000,
    321: 0.000000000000000,
    322: 0.000000000000000,
    323: 0.000000000000000,
    324: 0.000000000000000,
    325: 0.000000000000000,
    326: 0.000000000000000,
    327: 0.000000000000000,
    328: 0.000000000000000,
    329: 0.000000000000000,
    330: 0.000000000000000,
    331: 0.000000000000000,
    332: 0.000000000000000,
    333: 0.000000000000000,
    334: 0.000000000000000,
    335: 0.000000000000000,
    336: 0.000000000000000,
    337: 0.000000000000000,
    338: 0.000000000000000,
    339: 0.000000000000000,
    340: 0.000000000000000,
    341: 0.000000000000000,
    342: 0.000000000000000,
    343: 0.000000000000000,
    344: 0.000000000000000,
    345: 0.000000000000000,
    346: 0.000000000000000,
    347: 0.000000000000000,
    348: 0.000000000000000,
    349: 0.000000000000000,
    350: 0.000000000000000,
    351: 0.000000000000000,
    352: 0.000000000000000,
    353: 0.000000000000000,
    354: 0.000000000000000,
    355: 0.000000000000000,
    356: 0.000000000000000,
    357: 0.000000000000000,
    358: 0.000000000000000,
    359: 0.000000000000000,
    360: 0.000000000000000,
    361: 0.000000000000000,
    362: 0.000000000000000,
    363: 0.000000000000000,
    364: 0.000000000000000,
    365: 0.000000000000000,
    366: 0.000000000000000,
    367: 0.000000000000000,
    368: 0.000000000000000,
    369: 0.000000000000000,
    370: 0.000000000000000,
    371: 0.000000000000000,
    372: 0.000000000000000,
    373: 0.000000000000000,
    374: 0.000000000000000,
    375: 0.000000000000000,
    376: 0.000000000000000,
    377: 0.000000000000000,
    378: 0.000000000000000,
    379: 0.000000000000000,
    380: 1.204633204633200,
    381: 1.101029601029600,
    382: 0.997425997425997,
    383: 1.008365508365510,
    384: 1.019305019305020,
    385: 1.070141570141570,
    386: 1.120978120978120,
    387: 1.177606177606180,
    388: 1.234234234234230,
    389: 1.162162162162160,
    390: 1.090090090090090,
    391: 1.153796653796660,
    392: 1.217503217503220,
    393: 1.166023166023170,
    394: 1.114543114543110,
    395: 1.180823680823680,
    396: 1.247104247104250,
    397: 1.190476190476190,
    398: 1.133848133848130,
    399: 1.137065637065640,
    400: 1.140283140283140,
    401: 1.220077220077220,
    402: 1.299871299871300,
    403: 1.325611325611320,
    404: 1.351351351351350,
    405: 1.415701415701410,
    406: 1.480051480051480,
    407: 1.486486486486490,
    408: 1.492921492921490,
    409: 1.544401544401540,
    410: 1.595881595881600,
    411: 1.583011583011580,
    412: 1.570141570141570,
    413: 1.608751608751610,
    414: 1.647361647361650,
    415: 1.679536679536680,
    416: 1.711711711711710,
    417: 1.718146718146710,
    418: 1.724581724581720,
    419: 1.743886743886740,
    420: 1.763191763191760,
    421: 1.776061776061770,
    422: 1.788931788931790,
    423: 1.711711711711710,
    424: 1.634491634491630,
    425: 1.550836550836550,
    426: 1.467181467181470,
    427: 1.409266409266410,
    428: 1.351351351351350,
    429: 1.318532818532820,
    430: 1.285714285714290,
    431: 1.299227799227800,
    432: 1.312741312741310,
    433: 1.396396396396400,
    434: 1.480051480051480,
    435: 1.518661518661520,
    436: 1.557271557271560,
    437: 1.486486486486490,
    438: 1.415701415701420,
    439: 1.277348777348780,
    440: 1.138996138996140,
    441: 1.046975546975550,
    442: 0.954954954954955,
    443: 0.939510939510940,
    444: 0.924066924066924,
    445: 0.935006435006435,
    446: 0.945945945945946,
    447: 0.962676962676962,
    448: 0.979407979407979,
    449: 0.984555984555984,
    450: 0.989703989703990,
    451: 1.014800514800520,
    452: 1.039897039897040,
    453: 1.069498069498070,
    454: 1.099099099099100,
    455: 1.133848133848140,
    456: 1.168597168597170,
    457: 1.195624195624200,
    458: 1.222651222651220,
    459: 1.212998712998710,
    460: 1.203346203346200,
    461: 1.182110682110680,
    462: 1.160875160875160,
    463: 1.135135135135140,
    464: 1.109395109395110,
    465: 1.102960102960110,
    466: 1.096525096525100,
    467: 1.075289575289580,
    468: 1.054054054054050,
    469: 1.059202059202060,
    470: 1.064350064350060,
    471: 1.064993564993570,
    472: 1.065637065637070,
    473: 1.057915057915060,
    474: 1.050193050193050,
    475: 1.049549549549550,
    476: 1.048906048906050,
    477: 1.042471042471050,
    478: 1.036036036036040,
    479: 1.048262548262550,
    480: 1.060489060489060,
    481: 1.052767052767050,
    482: 1.045045045045050,
    483: 1.023166023166020,
    484: 1.001287001287000,
    485: 1.001287001287000,
    486: 1.001287001287000,
    487: 1.003217503217510,
    488: 1.005148005148010,
    489: 1.013513513513510,
    490: 1.021879021879020,
    491: 1.014800514800510,
    492: 1.007722007722010,
    493: 1.005791505791510,
    494: 1.003861003861000,
    495: 0.994208494208493,
    496: 0.984555984555985,
    497: 0.986486486486487,
    498: 0.988416988416988,
    499: 1.009009009009010,
    500: 1.029601029601030,
    501: 1.054697554697550,
    502: 1.079794079794080,
    503: 1.099099099099100,
    504: 1.118404118404120,
    505: 1.105534105534110,
    506: 1.092664092664090,
    507: 1.068854568854570,
    508: 1.045045045045050,
    509: 1.030244530244540,
    510: 1.015444015444020,
    511: 1.014157014157020,
    512: 1.012870012870010,
    513: 1.010939510939510,
    514: 1.009009009009010,
    515: 0.992277992277993,
    516: 0.975546975546976,
    517: 0.956241956241957,
    518: 0.936936936936937,
    519: 0.924710424710425,
    520: 0.912483912483913,
    521: 0.894465894465895,
    522: 0.876447876447876,
    523: 0.884813384813385,
    524: 0.893178893178893,
    525: 0.907979407979408,
    526: 0.922779922779923,
    527: 0.946589446589447,
    528: 0.970398970398970,
    529: 0.983268983268983,
    530: 0.996138996138996,
    531: 0.992921492921493,
    532: 0.989703989703990,
    533: 0.969111969111970,
    534: 0.948519948519949,
    535: 0.933719433719434,
    536: 0.918918918918919,
    537: 0.929214929214929,
    538: 0.939510939510939,
    539: 0.959459459459459,
    540: 0.979407979407979,
    541: 1.083011583011580,
    542: 1.186615186615190,
    543: 1.416988416988420,
    544: 1.647361647361650,
    545: 1.724581724581720,
    546: 1.801801801801800,
    547: 1.756756756756760,
    548: 1.711711711711710,
    549: 1.505791505791510,
    550: 1.299871299871300,
    551: 1.150579150579150,
    552: 1.001287001287000,
    553: 0.985199485199485,
    554: 0.969111969111969,
    555: 0.965250965250966,
    556: 0.961389961389962,
    557: 0.963320463320464,
    558: 0.965250965250965,
    559: 0.982625482625483,
    560: 1.000000000000000,
    561: 1.047619047619050,
    562: 1.095238095238100,
    563: 1.131274131274140,
    564: 1.167310167310170,
    565: 1.187902187902190,
    566: 1.208494208494210,
    567: 1.196267696267700,
    568: 1.184041184041180,
    569: 1.171814671814670,
    570: 1.159588159588160,
    571: 1.176962676962680,
    572: 1.194337194337190,
    573: 1.305019305019310,
    574: 1.415701415701420,
    575: 1.518661518661520,
    576: 1.621621621621620,
    577: 1.647361647361640,
    578: 1.673101673101670,
    579: 1.589446589446590,
    580: 1.505791505791510,
    581: 1.394465894465890,
    582: 1.283140283140280,
    583: 1.212355212355210,
    584: 1.141570141570140,
    585: 1.167953667953670,
    586: 1.194337194337190,
    587: 1.222651222651220,
    588: 1.250965250965250,
    589: 1.251608751608750,
    590: 1.252252252252250,
    591: 1.232303732303730,
    592: 1.212355212355210,
    593: 1.205920205920200,
    594: 1.199485199485200,
    595: 1.231016731016730,
    596: 1.262548262548260,
    597: 1.281209781209780,
    598: 1.299871299871300,
    599: 1.274774774774770,
    600: 1.249678249678250,
    601: 1.198198198198200,
    602: 1.146718146718150,
    603: 1.105534105534110,
    604: 1.064350064350060,
    605: 1.064993564993570,
    606: 1.065637065637070,
    607: 1.079150579150580,
    608: 1.092664092664090,
    609: 1.092020592020590,
    610: 1.091377091377090,
    611: 1.085585585585580,
    612: 1.079794079794080,
    613: 1.072072072072070,
    614: 1.064350064350060,
    615: 1.059202059202060,
    616: 1.054054054054050,
    617: 1.051480051480050,
    618: 1.048906048906050,
    619: 1.028314028314030,
    620: 1.007722007722010,
    621: 1.007078507078510,
    622: 1.006435006435010,
    623: 1.012226512226520,
    624: 1.018018018018020,
    625: 1.006435006435010,
    626: 0.994851994851995,
    627: 0.965250965250966,
    628: 0.935649935649936,
    629: 0.903474903474903,
    630: 0.871299871299871,
    631: 0.855212355212355,
    632: 0.839124839124839,
    633: 0.837194337194337,
    634: 0.835263835263835,
    635: 0.839768339768340,
    636: 0.844272844272844,
    637: 0.848777348777348,
    638: 0.853281853281853,
    639: 0.862290862290862,
    640: 0.871299871299871,
    641: 0.868725868725869,
    642: 0.866151866151866,
    643: 0.869369369369369,
    644: 0.872586872586873,
    645: 0.863577863577864,
    646: 0.854568854568855,
    647: 0.837194337194338,
    648: 0.819819819819820,
    649: 0.802445302445302,
    650: 0.785070785070785,
    651: 0.776061776061776,
    652: 0.767052767052767,
    653: 0.781209781209781,
    654: 0.795366795366795,
    655: 0.823680823680824,
    656: 0.851994851994852,
    657: 0.864864864864865,
    658: 0.877734877734878,
    659: 0.887387387387387,
    660: 0.897039897039897,
    661: 0.888674388674388,
    662: 0.880308880308880,
    663: 0.871299871299871,
    664: 0.862290862290862,
    665: 0.871943371943372,
    666: 0.881595881595882,
    667: 0.909266409266410,
    668: 0.936936936936937,
    669: 0.953667953667954,
    670: 0.970398970398970,
    671: 0.960746460746461,
    672: 0.951093951093951,
    673: 0.923423423423424,
    674: 0.895752895752896,
    675: 0.852638352638353,
    676: 0.809523809523810,
    677: 0.787644787644788,
    678: 0.765765765765766,
    679: 0.770270270270271,
    680: 0.774774774774775,
    681: 0.803088803088803,
    682: 0.831402831402831,
    683: 0.842985842985843,
    684: 0.854568854568855,
    685: 0.835263835263836,
    686: 0.815958815958816,
    687: 0.776061776061776,
    688: 0.736164736164736,
    689: 0.704633204633204,
    690: 0.673101673101673,
    691: 0.679536679536680,
    692: 0.685971685971686,
    693: 0.718790218790219,
    694: 0.751608751608752,
    695: 0.791505791505791,
    696: 0.831402831402831,
    697: 0.841698841698842,
    698: 0.851994851994852,
    699: 0.810810810810811,
    700: 0.769626769626770,
    701: 0.716216216216216,
    702: 0.662805662805663,
    703: 0.622265122265123,
    704: 0.581724581724582,
    705: 0.546975546975547,
    706: 0.512226512226512,
    707: 0.503217503217503,
    708: 0.494208494208494,
    709: 0.480051480051480,
    710: 0.465894465894466,
    711: 0.460746460746461,
    712: 0.455598455598456,
    713: 0.449806949806950,
    714: 0.444015444015444,
    715: 0.437580437580438,
    716: 0.431145431145431,
    717: 0.439510939510940,
    718: 0.447876447876448,
    719: 0.454954954954955,
    720: 0.462033462033462,
    721: 0.471685971685971,
    722: 0.481338481338481,
    723: 0.488416988416988,
    724: 0.495495495495495,
    725: 0.490347490347490,
    726: 0.485199485199485,
    727: 0.476190476190476,
    728: 0.467181467181467,
    729: 0.462676962676963,
    730: 0.458172458172458,
    731: 0.442084942084942,
    732: 0.425997425997426,
    733: 0.422136422136422,
    734: 0.418275418275418,
    735: 0.411196911196911,
    736: 0.404118404118404,
    737: 0.415701415701415,
    738: 0.427284427284427,
    739: 0.427927927927928,
    740: 0.428571428571429,
    741: 0.415701415701416,
    742: 0.402831402831403,
    743: 0.415057915057915,
    744: 0.427284427284427,
    745: 0.413770913770913,
    746: 0.400257400257400,
    747: 0.404761904761904,
    748: 0.409266409266409,
    749: 0.421492921492922,
    750: 0.433719433719434,
    751: 0.443371943371944,
    752: 0.453024453024453,
    753: 0.494208494208494,
    754: 0.535392535392535,
    755: 0.544401544401544,
    756: 0.553410553410553,
    757: 0.566924066924066,
    758: 0.580437580437580,
    759: 0.569498069498069,
    760: 0.558558558558559,
    761: 0.547619047619048,
    762: 0.536679536679537,
    763: 0.584298584298585,
    764: 0.631917631917632,
    765: 0.656370656370657,
    766: 0.680823680823681,
    767: 0.659588159588160,
    768: 0.638352638352638,
    769: 0.646718146718146,
    770: 0.655083655083655,
    771: 0.617760617760617,
    772: 0.580437580437580,
    773: 0.545045045045045,
    774: 0.509652509652510,
    775: 0.488416988416988,
    776: 0.467181467181467,
    777: 0.441441441441441,
    778: 0.415701415701416,
    779: 0.433075933075933,
    780: 0.450450450450450,
    781: 0.000000000000000,
    782: 0.000000000000000,
    783: 0.000000000000000,
    784: 0.000000000000000,
    785: 0.000000000000000,
    786: 0.000000000000000,
    787: 0.000000000000000,
    788: 0.000000000000000,
    789: 0.000000000000000,
    790: 0.000000000000000,
    791: 0.000000000000000,
    792: 0.000000000000000,
    793: 0.000000000000000,
    794: 0.000000000000000,
    795: 0.000000000000000,
    796: 0.000000000000000,
    797: 0.000000000000000,
    798: 0.000000000000000,
    799: 0.000000000000000,
    800: 0.000000000000000,
    801: 0.000000000000000,
    802: 0.000000000000000,
    803: 0.000000000000000,
    804: 0.000000000000000,
    805: 0.000000000000000,
    806: 0.000000000000000,
    807: 0.000000000000000,
    808: 0.000000000000000,
    809: 0.000000000000000,
    810: 0.000000000000000,
    811: 0.000000000000000,
    812: 0.000000000000000,
    813: 0.000000000000000,
    814: 0.000000000000000,
    815: 0.000000000000000,
    816: 0.000000000000000,
    817: 0.000000000000000,
    818: 0.000000000000000,
    819: 0.000000000000000,
    820: 0.000000000000000,
    821: 0.000000000000000,
    822: 0.000000000000000,
    823: 0.000000000000000,
    824: 0.000000000000000,
    825: 0.000000000000000,
    826: 0.000000000000000,
    827: 0.000000000000000,
    828: 0.000000000000000,
    829: 0.000000000000000,
    830: 0.000000000000000,
}


class TestSpectralSimilarityIndex(unittest.TestCase):
    """
    Define :func:`colour.quality.ssi.spectral_similarity_index`
    definition unit tests methods.
    """

    def test_spectral_similarity_index(self):
        """Test :func:`colour.quality.ssi.spectral_similarity_index` definition."""

        self.assertEqual(
            spectral_similarity_index(
                SDS_ILLUMINANTS["C"], SDS_ILLUMINANTS["D65"]
            ),
            94.0,
        )
        self.assertEqual(
            spectral_similarity_index(
                SpectralDistribution(DATA_HMI), SDS_ILLUMINANTS["D50"]
            ),
            72.0,
        )

    def test_spectral_similarity_continuous(self):
        """
        Test :func:`colour.quality.ssi.spectral_similarity_index` for
        producing continuous values.
        """

        # Test values were computed at ed2e90
        self.assertAlmostEqual(
            spectral_similarity_index(
                SDS_ILLUMINANTS["C"], SDS_ILLUMINANTS["D65"], continuous=True
            ),
            94.178,
            places=2,
        )
        self.assertAlmostEqual(
            spectral_similarity_index(
                SpectralDistribution(DATA_HMI),
                SDS_ILLUMINANTS["D50"],
                continuous=True,
            ),
            71.772,
            places=2,
        )


if __name__ == "__main__":
    unittest.main()
