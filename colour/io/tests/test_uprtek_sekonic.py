# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.io.uprtek_sekonic` module.
"""

from __future__ import annotations

import json
import numpy as np
import os
import unittest

from colour.colorimetry import SpectralDistribution
from colour.hints import Any, Dict, Optional
from colour.io import (
    SpectralDistribution_UPRTek,
    SpectralDistribution_Sekonic,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013-2021 - Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"
__all__ = [
    "RESOURCES_DIRECTORY",
    "AbstractSpectralDistributionTest",
    "TestSpectralDistributionUprTek",
    "TestSpectralDistributionSekonic",
]

RESOURCES_DIRECTORY: str = os.path.join(os.path.dirname(__file__), "resources")


class AbstractSpectralDistributionTest(unittest.TestCase):
    """
    Defines :class:`colour.SpectralDistribution_UPRTek`,
    :class:`colour.SpectralDistribution_Sekonic` classes common unit tests
    methods.
    """

    def __init__(self, *args: Any):
        """
        Create an instance of the class.

        Other Parameters
        ----------------
        args
            Arguments.
        """

        super(AbstractSpectralDistributionTest, self).__init__(*args)

        self._sd_factory: Any = None
        self._path: Optional[str] = None
        self._spectral_data: Optional[Dict] = None

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = (
            "mapping",
            "path",
            "header",
            "spectral_quantity",
            "reflection_geometry",
            "transmission_geometry",
            "bandwidth_FWHM",
            "bandwidth_corrected",
            "metadata",
        )

        for attribute in required_attributes:
            self.assertIn(attribute, dir(SpectralDistribution_UPRTek))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ("__init__", "read", "write")

        for method in required_methods:
            self.assertIn(method, dir(SpectralDistribution_UPRTek))

    def test_read(self):
        """
        Tests :meth:`colour.SpectralDistribution_UPRTek.read` and
        :meth:`colour.SpectralDistribution_Sekonic.read` methods.
        """

        if self._sd_factory is None:
            return

        sd = self._sd_factory(os.path.join(RESOURCES_DIRECTORY, self._path)).read()

        sd_r = SpectralDistribution(self._spectral_data)

        np.testing.assert_array_equal(sd_r.domain, sd.domain)
        np.testing.assert_almost_equal(sd_r.values, sd.values, decimal=6)

        for key, value in self._header.items():
            for specification in sd.header.mapping.elements:
                if key == specification.element:
                    if key == "Comments":
                        self.assertDictEqual(json.loads(sd.header.comments), value)
                    else:
                        self.assertEqual(
                            getattr(sd.header, specification.attribute), value
                        )


class TestSpectralDistributionUprTek(AbstractSpectralDistributionTest):
    """
    Defines :class:`colour.SpectralDistribution_UPRTek` class unit tests
    methods.
    """

    def __init__(self, *args: Any):
        """
        Create an instance of the class.

        Other Parameters
        ----------------
        args
            Arguments.
        """

        super(TestSpectralDistributionUprTek, self).__init__(*args)

        self._sd_factory = SpectralDistribution_UPRTek
        self._path = "ESPD2021_0104_231446.xls"
        self._spectral_data = {
            380: 0.030267,
            381: 0.030267,
            382: 0.030267,
            383: 0.029822,
            384: 0.028978,
            385: 0.028623,
            386: 0.030845,
            387: 0.035596,
            388: 0.039231,
            389: 0.039064,
            390: 0.035223,
            391: 0.031580,
            392: 0.029181,
            393: 0.027808,
            394: 0.026256,
            395: 0.024526,
            396: 0.022557,
            397: 0.020419,
            398: 0.018521,
            399: 0.018149,
            400: 0.019325,
            401: 0.021666,
            402: 0.024045,
            403: 0.026473,
            404: 0.029076,
            405: 0.031840,
            406: 0.033884,
            407: 0.034038,
            408: 0.032302,
            409: 0.030383,
            410: 0.029426,
            411: 0.029979,
            412: 0.032614,
            413: 0.037204,
            414: 0.042279,
            415: 0.046029,
            416: 0.048698,
            417: 0.053064,
            418: 0.059530,
            419: 0.070840,
            420: 0.087678,
            421: 0.110043,
            422: 0.136705,
            423: 0.165180,
            424: 0.199071,
            425: 0.241976,
            426: 0.293837,
            427: 0.359177,
            428: 0.434192,
            429: 0.523828,
            430: 0.632578,
            431: 0.758893,
            432: 0.915528,
            433: 1.096489,
            434: 1.307487,
            435: 1.557125,
            436: 1.838779,
            437: 2.183382,
            438: 2.586251,
            439: 3.054022,
            440: 3.625659,
            441: 4.279538,
            442: 5.055838,
            443: 5.919301,
            444: 6.869926,
            445: 7.940298,
            446: 9.090219,
            447: 10.336670,
            448: 11.619895,
            449: 12.939739,
            450: 14.206918,
            451: 15.396660,
            452: 16.430536,
            453: 17.267374,
            454: 17.912292,
            455: 18.261185,
            456: 18.404581,
            457: 18.288025,
            458: 18.002302,
            459: 17.570372,
            460: 17.011297,
            461: 16.411137,
            462: 15.779440,
            463: 15.168951,
            464: 14.585364,
            465: 14.057872,
            466: 13.575768,
            467: 13.144953,
            468: 12.737307,
            469: 12.346188,
            470: 11.967313,
            471: 11.590308,
            472: 11.209807,
            473: 10.815372,
            474: 10.406748,
            475: 10.007284,
            476: 9.627886,
            477: 9.279286,
            478: 8.958391,
            479: 8.663115,
            480: 8.427362,
            481: 8.238759,
            482: 8.110200,
            483: 8.011048,
            484: 7.939125,
            485: 7.900343,
            486: 7.880703,
            487: 7.887271,
            488: 7.907047,
            489: 7.939895,
            490: 7.977298,
            491: 8.013443,
            492: 8.056756,
            493: 8.112617,
            494: 8.181398,
            495: 8.256148,
            496: 8.332609,
            497: 8.418014,
            498: 8.513148,
            499: 8.616785,
            500: 8.719036,
            501: 8.817776,
            502: 8.914417,
            503: 9.011255,
            504: 9.105255,
            505: 9.193217,
            506: 9.274889,
            507: 9.350751,
            508: 9.423820,
            509: 9.490992,
            510: 9.553215,
            511: 9.608335,
            512: 9.653841,
            513: 9.691347,
            514: 9.727146,
            515: 9.767722,
            516: 9.809064,
            517: 9.842565,
            518: 9.867527,
            519: 9.887219,
            520: 9.906105,
            521: 9.920433,
            522: 9.929304,
            523: 9.932856,
            524: 9.935204,
            525: 9.937991,
            526: 9.938448,
            527: 9.936127,
            528: 9.930192,
            529: 9.922665,
            530: 9.913944,
            531: 9.905774,
            532: 9.898767,
            533: 9.894219,
            534: 9.891479,
            535: 9.883711,
            536: 9.862693,
            537: 9.829168,
            538: 9.795257,
            539: 9.767633,
            540: 9.747380,
            541: 9.729669,
            542: 9.714886,
            543: 9.701355,
            544: 9.688311,
            545: 9.673670,
            546: 9.657027,
            547: 9.633310,
            548: 9.603127,
            549: 9.567823,
            550: 9.534049,
            551: 9.504526,
            552: 9.484178,
            553: 9.471739,
            554: 9.455969,
            555: 9.429557,
            556: 9.396450,
            557: 9.368848,
            558: 9.344832,
            559: 9.313942,
            560: 9.273922,
            561: 9.240767,
            562: 9.220987,
            563: 9.210749,
            564: 9.195800,
            565: 9.173392,
            566: 9.143906,
            567: 9.109710,
            568: 9.078232,
            569: 9.052593,
            570: 9.023234,
            571: 8.984895,
            572: 8.950663,
            573: 8.935179,
            574: 8.936305,
            575: 8.937272,
            576: 8.931671,
            577: 8.921451,
            578: 8.910289,
            579: 8.908619,
            580: 8.917888,
            581: 8.934530,
            582: 8.946784,
            583: 8.958764,
            584: 8.979334,
            585: 9.007913,
            586: 9.033543,
            587: 9.051113,
            588: 9.067842,
            589: 9.089899,
            590: 9.114546,
            591: 9.136106,
            592: 9.164270,
            593: 9.207536,
            594: 9.264211,
            595: 9.321528,
            596: 9.371778,
            597: 9.411209,
            598: 9.443729,
            599: 9.490623,
            600: 9.557871,
            601: 9.626752,
            602: 9.674832,
            603: 9.705856,
            604: 9.739429,
            605: 9.784062,
            606: 9.841268,
            607: 9.907084,
            608: 9.971845,
            609: 10.026823,
            610: 10.060076,
            611: 10.076903,
            612: 10.105914,
            613: 10.161287,
            614: 10.230108,
            615: 10.285982,
            616: 10.336598,
            617: 10.396016,
            618: 10.449015,
            619: 10.478296,
            620: 10.484620,
            621: 10.487537,
            622: 10.498996,
            623: 10.519572,
            624: 10.541495,
            625: 10.549863,
            626: 10.543288,
            627: 10.538241,
            628: 10.546865,
            629: 10.560687,
            630: 10.567954,
            631: 10.564369,
            632: 10.555919,
            633: 10.542054,
            634: 10.527417,
            635: 10.513332,
            636: 10.500641,
            637: 10.493341,
            638: 10.491714,
            639: 10.477033,
            640: 10.435987,
            641: 10.374922,
            642: 10.317416,
            643: 10.269583,
            644: 10.220937,
            645: 10.168004,
            646: 10.115719,
            647: 10.061740,
            648: 9.998492,
            649: 9.919030,
            650: 9.821223,
            651: 9.716800,
            652: 9.619915,
            653: 9.531602,
            654: 9.435769,
            655: 9.326644,
            656: 9.215940,
            657: 9.111384,
            658: 9.005102,
            659: 8.892046,
            660: 8.775783,
            661: 8.659118,
            662: 8.537835,
            663: 8.413469,
            664: 8.292587,
            665: 8.175849,
            666: 8.055606,
            667: 7.931369,
            668: 7.812479,
            669: 7.695505,
            670: 7.564718,
            671: 7.422195,
            672: 7.286375,
            673: 7.166087,
            674: 7.050159,
            675: 6.925609,
            676: 6.792675,
            677: 6.659946,
            678: 6.534333,
            679: 6.416044,
            680: 6.298086,
            681: 6.182296,
            682: 6.073105,
            683: 5.965933,
            684: 5.853682,
            685: 5.729931,
            686: 5.599877,
            687: 5.480670,
            688: 5.376213,
            689: 5.273221,
            690: 5.156234,
            691: 5.027091,
            692: 4.900242,
            693: 4.777046,
            694: 4.658288,
            695: 4.547010,
            696: 4.443560,
            697: 4.347722,
            698: 4.252159,
            699: 4.152643,
            700: 4.053906,
            701: 3.961853,
            702: 3.865061,
            703: 3.755302,
            704: 3.634861,
            705: 3.519360,
            706: 3.418803,
            707: 3.328571,
            708: 3.246458,
            709: 3.160225,
            710: 3.066386,
            711: 2.970290,
            712: 2.878098,
            713: 2.790311,
            714: 2.701265,
            715: 2.607646,
            716: 2.515490,
            717: 2.435313,
            718: 2.361505,
            719: 2.282271,
            720: 2.192500,
            721: 2.101594,
            722: 2.027356,
            723: 1.966553,
            724: 1.912948,
            725: 1.855193,
            726: 1.785138,
            727: 1.710667,
            728: 1.638785,
            729: 1.582385,
            730: 1.539228,
            731: 1.498548,
            732: 1.455407,
            733: 1.413034,
            734: 1.372021,
            735: 1.324772,
            736: 1.277157,
            737: 1.238888,
            738: 1.211113,
            739: 1.182541,
            740: 1.149382,
            741: 1.118490,
            742: 1.091204,
            743: 1.065539,
            744: 1.039564,
            745: 1.013148,
            746: 0.990818,
            747: 0.976522,
            748: 0.960074,
            749: 0.935639,
            750: 0.905095,
            751: 0.878893,
            752: 0.862828,
            753: 0.847588,
            754: 0.829938,
            755: 0.808772,
            756: 0.786338,
            757: 0.761752,
            758: 0.735873,
            759: 0.711232,
            760: 0.690947,
            761: 0.673476,
            762: 0.659236,
            763: 0.646735,
            764: 0.633802,
            765: 0.612864,
            766: 0.589102,
            767: 0.567989,
            768: 0.551288,
            769: 0.533479,
            770: 0.508426,
            771: 0.487143,
            772: 0.474126,
            773: 0.465145,
            774: 0.455158,
            775: 0.442994,
            776: 0.429114,
            777: 0.419402,
            778: 0.411766,
            779: 0.411766,
            780: 0.411766,
        }

        self._header = {
            "Manufacturer": "UPRTek",
            "CatalogNumber": None,
            "Description": None,
            "DocumentCreator": None,
            "UniqueIdentifier": None,
            "MeasurementEquipment": "CV600",
            "Laboratory": None,
            "ReportNumber": None,
            "ReportDate": "2021/01/04_23:14:46",
            "DocumentCreationDate": None,
            "Comments": {
                "Model Name": "CV600",
                "Serial Number": "19J00789",
                "Time": "2021/01/04_23:14:46",
                "Memo": [],
                "LUX": 695.154907,
                "fc": 64.605476,
                "CCT": 5198.0,
                "Duv": -0.00062,
                "I-Time": 12000.0,
                "X": 682.470886,
                "Y": 695.154907,
                "Z": 631.635071,
                "x": 0.339663,
                "y": 0.345975,
                "u'": 0.209915,
                "v'": 0.481087,
                "LambdaP": 456.0,
                "LambdaPValue": 18.404581,
                "CRI": 92.956993,
                "R1": 91.651062,
                "R2": 93.014732,
                "R3": 97.032013,
                "R4": 93.513229,
                "R5": 92.48259,
                "R6": 91.48687,
                "R7": 93.016129,
                "R8": 91.459312,
                "R9": 77.613075,
                "R10": 86.981613,
                "R11": 94.841324,
                "R12": 74.139542,
                "R13": 91.073837,
                "R14": 97.064323,
                "R15": 88.615669,
                "TLCI": 97.495056,
                "TLMF-A": 1.270032,
                "SSI-A": 44.881924,
                "Rf": 87.234917,
                "Rg": 98.510712,
                "IRR": 2.607891,
            },
        }


class TestSpectralDistributionSekonic(AbstractSpectralDistributionTest):
    """
    Defines :class:`colour.SpectralDistribution_Sekonic` class unit tests
    methods.
    """

    def __init__(self, *args: Any):
        """
        Create an instance of the class.

        Other Parameters
        ----------------
        args
            Arguments.
        """

        super(TestSpectralDistributionSekonic, self).__init__(*args)

        self._sd_factory = SpectralDistribution_Sekonic
        self._path = "RANDOM_001_02._3262K.csv"
        self._spectral_data = {
            380: 0.000000000000,
            381: 0.000000000000,
            382: 0.000000000000,
            383: 0.000000000000,
            384: 0.000000000000,
            385: 0.000000000000,
            386: 0.000000000000,
            387: 0.000000000000,
            388: 0.000000000000,
            389: 0.000000000000,
            390: 0.000000000000,
            391: 0.000000000000,
            392: 0.000002927853,
            393: 0.000006502053,
            394: 0.000009265275,
            395: 0.000011032038,
            396: 0.000011953731,
            397: 0.000012279555,
            398: 0.000012258756,
            399: 0.000012112181,
            400: 0.000011981365,
            401: 0.000011995159,
            402: 0.000012281144,
            403: 0.000012880828,
            404: 0.000013697349,
            405: 0.000014621435,
            406: 0.000015547508,
            407: 0.000016454918,
            408: 0.000017407952,
            409: 0.000018474588,
            410: 0.000019711053,
            411: 0.000021048536,
            412: 0.000022339967,
            413: 0.000023436902,
            414: 0.000024226094,
            415: 0.000024806555,
            416: 0.000025354624,
            417: 0.000026046688,
            418: 0.000027027134,
            419: 0.000028330132,
            420: 0.000029966144,
            421: 0.000031945645,
            422: 0.000034267265,
            423: 0.000036904559,
            424: 0.000039828374,
            425: 0.000043010186,
            426: 0.000046453275,
            427: 0.000050200390,
            428: 0.000054296306,
            429: 0.000058792350,
            430: 0.000063819272,
            431: 0.000069569738,
            432: 0.000076238801,
            433: 0.000084002051,
            434: 0.000092899616,
            435: 0.000102907434,
            436: 0.000114000723,
            437: 0.000126147745,
            438: 0.000139350668,
            439: 0.000153605943,
            440: 0.000168909683,
            441: 0.000185196404,
            442: 0.000202212090,
            443: 0.000219666821,
            444: 0.000237270768,
            445: 0.000254752871,
            446: 0.000271882804,
            447: 0.000288435520,
            448: 0.000304183195,
            449: 0.000318816456,
            450: 0.000331902935,
            451: 0.000342996238,
            452: 0.000351659779,
            453: 0.000357679965,
            454: 0.000361089711,
            455: 0.000361937127,
            456: 0.000360277918,
            457: 0.000356289936,
            458: 0.000350250222,
            459: 0.000342438580,
            460: 0.000333143020,
            461: 0.000322732056,
            462: 0.000311622134,
            463: 0.000300230284,
            464: 0.000288942829,
            465: 0.000277946005,
            466: 0.000267342635,
            467: 0.000257235020,
            468: 0.000247702759,
            469: 0.000238719338,
            470: 0.000230227481,
            471: 0.000222169925,
            472: 0.000214497733,
            473: 0.000207189034,
            474: 0.000200227427,
            475: 0.000193596818,
            476: 0.000187307058,
            477: 0.000181425072,
            478: 0.000176026821,
            479: 0.000171187712,
            480: 0.000166981976,
            481: 0.000163483521,
            482: 0.000160765063,
            483: 0.000158896932,
            484: 0.000157875169,
            485: 0.000157608956,
            486: 0.000158002527,
            487: 0.000158960844,
            488: 0.000160401178,
            489: 0.000162251439,
            490: 0.000164439844,
            491: 0.000166898695,
            492: 0.000169602441,
            493: 0.000172551969,
            494: 0.000175748704,
            495: 0.000179197523,
            496: 0.000182933160,
            497: 0.000187002632,
            498: 0.000191452826,
            499: 0.000196314068,
            500: 0.000201534538,
            501: 0.000207037185,
            502: 0.000212744897,
            503: 0.000218581801,
            504: 0.000224477379,
            505: 0.000230361940,
            506: 0.000236165870,
            507: 0.000241834379,
            508: 0.000247346645,
            509: 0.000252687139,
            510: 0.000257840526,
            511: 0.000262814428,
            512: 0.000267655065,
            513: 0.000272412435,
            514: 0.000277135783,
            515: 0.000281845685,
            516: 0.000286527647,
            517: 0.000291164964,
            518: 0.000295740523,
            519: 0.000300232059,
            520: 0.000304612651,
            521: 0.000308855029,
            522: 0.000312933233,
            523: 0.000316833000,
            524: 0.000320547697,
            525: 0.000324070978,
            526: 0.000327409187,
            527: 0.000330665527,
            528: 0.000333987991,
            529: 0.000337524747,
            530: 0.000341368344,
            531: 0.000345327600,
            532: 0.000349117006,
            533: 0.000352450879,
            534: 0.000355126103,
            535: 0.000357231562,
            536: 0.000358921068,
            537: 0.000360348407,
            538: 0.000361620390,
            539: 0.000362726772,
            540: 0.000363639323,
            541: 0.000364331092,
            542: 0.000364891835,
            543: 0.000365620159,
            544: 0.000366836379,
            545: 0.000368854904,
            546: 0.000371746690,
            547: 0.000375265605,
            548: 0.000379145116,
            549: 0.000383122213,
            550: 0.000387050648,
            551: 0.000390928035,
            552: 0.000394761097,
            553: 0.000398556062,
            554: 0.000402294856,
            555: 0.000405925355,
            556: 0.000409392873,
            557: 0.000412643829,
            558: 0.000415688555,
            559: 0.000418639625,
            560: 0.000421619130,
            561: 0.000424748578,
            562: 0.000428094878,
            563: 0.000431627472,
            564: 0.000435305585,
            565: 0.000439088471,
            566: 0.000442934950,
            567: 0.000446803198,
            568: 0.000450651161,
            569: 0.000454437046,
            570: 0.000458150520,
            571: 0.000461855903,
            572: 0.000465628196,
            573: 0.000469542429,
            574: 0.000473651045,
            575: 0.000477944268,
            576: 0.000482402043,
            577: 0.000487004407,
            578: 0.000491718296,
            579: 0.000496469554,
            580: 0.000501176575,
            581: 0.000505757635,
            582: 0.000510152080,
            583: 0.000514372950,
            584: 0.000518449873,
            585: 0.000522412360,
            586: 0.000526284566,
            587: 0.000530070101,
            588: 0.000533766986,
            589: 0.000537373126,
            590: 0.000540883630,
            591: 0.000544285693,
            592: 0.000547563192,
            593: 0.000550700177,
            594: 0.000553691818,
            595: 0.000556585495,
            596: 0.000559442851,
            597: 0.000562325818,
            598: 0.000565279392,
            599: 0.000568273535,
            600: 0.000571256795,
            601: 0.000574177830,
            602: 0.000576974649,
            603: 0.000579536776,
            604: 0.000581740285,
            605: 0.000583461253,
            606: 0.000584599038,
            607: 0.000585157890,
            608: 0.000585171976,
            609: 0.000584675174,
            610: 0.000583703280,
            611: 0.000582299544,
            612: 0.000580509542,
            613: 0.000578378676,
            614: 0.000575953862,
            615: 0.000573287893,
            616: 0.000570435368,
            617: 0.000567450887,
            618: 0.000564369780,
            619: 0.000561140885,
            620: 0.000557688472,
            621: 0.000553937047,
            622: 0.000549851626,
            623: 0.000545581162,
            624: 0.000541326357,
            625: 0.000537287910,
            626: 0.000533593295,
            627: 0.000530039892,
            628: 0.000526331889,
            629: 0.000522173534,
            630: 0.000517328095,
            631: 0.000511825143,
            632: 0.000505769160,
            633: 0.000499264686,
            634: 0.000492379884,
            635: 0.000485043478,
            636: 0.000477139401,
            637: 0.000468551356,
            638: 0.000459251489,
            639: 0.000449585932,
            640: 0.000439994939,
            641: 0.000430918619,
            642: 0.000422663987,
            643: 0.000415024377,
            644: 0.000407667656,
            645: 0.000400261633,
            646: 0.000392578833,
            647: 0.000384767627,
            648: 0.000377058517,
            649: 0.000369681919,
            650: 0.000362766819,
            651: 0.000356107164,
            652: 0.000349425798,
            653: 0.000342445448,
            654: 0.000335026474,
            655: 0.000327456160,
            656: 0.000320101273,
            657: 0.000313328317,
            658: 0.000307335460,
            659: 0.000301838503,
            660: 0.000296465587,
            661: 0.000290844997,
            662: 0.000284782291,
            663: 0.000278556399,
            664: 0.000272522098,
            665: 0.000267032796,
            666: 0.000262254383,
            667: 0.000257897831,
            668: 0.000253598962,
            669: 0.000248999364,
            670: 0.000243966802,
            671: 0.000238797031,
            672: 0.000233855622,
            673: 0.000229498852,
            674: 0.000225782627,
            675: 0.000222411400,
            676: 0.000219070076,
            677: 0.000215468172,
            678: 0.000211623279,
            679: 0.000207766803,
            680: 0.000204134776,
            681: 0.000200916242,
            682: 0.000197999922,
            683: 0.000195158325,
            684: 0.000192163920,
            685: 0.000188884194,
            686: 0.000185509256,
            687: 0.000182299933,
            688: 0.000179515657,
            689: 0.000177253518,
            690: 0.000175304012,
            691: 0.000173423585,
            692: 0.000171374879,
            693: 0.000169089981,
            694: 0.000166684200,
            695: 0.000164281839,
            696: 0.000161995718,
            697: 0.000159809686,
            698: 0.000157624905,
            699: 0.000155341069,
            700: 0.000152887544,
            701: 0.000150368738,
            702: 0.000147950719,
            703: 0.000145799495,
            704: 0.000143992351,
            705: 0.000142327044,
            706: 0.000140546414,
            707: 0.000138393327,
            708: 0.000135762792,
            709: 0.000132830304,
            710: 0.000129795619,
            711: 0.000126856787,
            712: 0.000124101163,
            713: 0.000121442732,
            714: 0.000118780568,
            715: 0.000116016025,
            716: 0.000113144888,
            717: 0.000110295317,
            718: 0.000107605832,
            719: 0.000105211519,
            720: 0.000103122693,
            721: 0.000101195699,
            722: 0.000099277633,
            723: 0.000097221695,
            724: 0.000095040108,
            725: 0.000092921349,
            726: 0.000091063630,
            727: 0.000089657653,
            728: 0.000088729350,
            729: 0.000088144145,
            730: 0.000087760782,
            731: 0.000087439126,
            732: 0.000087065731,
            733: 0.000086550441,
            734: 0.000085803600,
            735: 0.000084741441,
            736: 0.000083366656,
            737: 0.000081748578,
            738: 0.000079958285,
            739: 0.000078067504,
            740: 0.000076152413,
            741: 0.000074292504,
            742: 0.000072567469,
            743: 0.000071058574,
            744: 0.000069874128,
            745: 0.000069137976,
            746: 0.000068973786,
            747: 0.000069459609,
            748: 0.000070268186,
            749: 0.000070849754,
            750: 0.000070651688,
            751: 0.000069174901,
            752: 0.000066329500,
            753: 0.000062221166,
            754: 0.000056957157,
            755: 0.000050740506,
            756: 0.000044398927,
            757: 0.000039030732,
            758: 0.000035736208,
            759: 0.000035360736,
            760: 0.000037219921,
            761: 0.000040070787,
            762: 0.000042669857,
            763: 0.000043976099,
            764: 0.000044012377,
            765: 0.000043148953,
            766: 0.000041756259,
            767: 0.000040175455,
            768: 0.000038621456,
            769: 0.000037272272,
            770: 0.000036305886,
            771: 0.000035866044,
            772: 0.000035955240,
            773: 0.000036541740,
            774: 0.000037593938,
            775: 0.000038985072,
            776: 0.000040247214,
            777: 0.000040842820,
            778: 0.000040234852,
            779: 0.000038216305,
            780: 0.000035575547,
        }
        self._header = {
            "Manufacturer": "Sekonic",
            "CatalogNumber": None,
            "Description": None,
            "DocumentCreator": None,
            "UniqueIdentifier": None,
            "MeasurementEquipment": None,
            "Laboratory": None,
            "ReportNumber": None,
            "ReportDate": "15/03/2021 3:44:14 p.m.",
            "DocumentCreationDate": None,
            "Comments": {
                "Date Saved": "15/03/2021 3:44:14 p.m.",
                "Title": "RANDOM_001_02°_3262K",
                "Measuring Mode": "Ambient",
                "Viewing Angle [°]": 2,
                "Tcp [K]": 3262,
                "⊿uv": -0.0029,
                "Illuminance [lx]": 30.1,
                "Illuminance [fc]": 2.79,
                "Peak Wavelength [nm]": 608,
                "Tristimulus Value X": 32.1626,
                "Tristimulus Value Y": 30.0794,
                "Tristimulus Value Z": 15.0951,
                "CIE1931 x": 0.4159,
                "CIE1931 y": 0.3889,
                "CIE1931 z": 0.1952,
                "CIE1976 u'": 0.2434,
                "CIE1976 v'": 0.5121,
                "Dominant Wavelength [nm]": 583,
                "Purity [%]": 41.5,
                "PPFD [umolm⁻²s⁻¹]": 0.4,
                "CRI Ra": 87.5,
                "CRI R1": 87.6,
                "CRI R2": 94.5,
                "CRI R3": 96.8,
                "CRI R4": 85.8,
                "CRI R5": 87.3,
                "CRI R6": 92.3,
                "CRI R7": 86.4,
                "CRI R8": 69.8,
                "CRI R9": 31.2,
                "CRI R10": 85.6,
                "CRI R11": 85.1,
                "CRI R12": 75.6,
                "CRI R13": 89.6,
                "CRI R14": 98.8,
                "CRI R15": 82.5,
                "TM-30 Rf": 87,
                "TM-30 Rg": 98,
                "SSIt": 76,
                "SSId": 59,
                "SSI1": "---",
                "SSI2": "---",
                "TLCI": 79,
                "TLMF": "---",
                "TM-30 Color Vector Graphic": [
                    "Reference Illuminant x",
                    "Reference Illuminant y",
                    "Measured Illuminant x",
                    "Measured Illuminant y",
                ],
                "bin1": [0.9764469, 0.2157578, 0.8882475, 0.2021859],
                "bin2": [0.7906278, 0.6122971, 0.7113284, 0.6248878],
                "bin3": [0.5509713, 0.8345242, 0.4676899, 0.8666077],
                "bin4": [0.1428891, 0.9897387, 0.0935279, 1.002316],
                "bin5": [-0.176162, 0.9843612, -0.2043247, 0.9795201],
                "bin6": [-0.5853095, 0.81081, -0.5838909, 0.8375309],
                "bin7": [-0.7960986, 0.6051669, -0.7457092, 0.6149487],
                "bin8": [-0.951027, 0.309108, -0.9191595, 0.309686],
                "bin9": [-0.9854512, -0.1699584, -0.9329426, -0.2097975],
                "bin10": [-0.8461911, -0.5328795, -0.7660208, -0.6001526],
                "bin11": [-0.5824577, -0.812861, -0.4902966, -0.8897363],
                "bin12": [-0.2939128, -0.9558322, -0.2872024, -1.03006],
                "bin13": [0.1462545, -0.989247, 0.1026697, -1.040349],
                "bin14": [0.508388, -0.8611281, 0.4397461, -0.9682071],
                "bin15": [0.8469644, -0.5316497, 0.7729813, -0.6153884],
                "bin16": [0.9788596, -0.2045332, 0.9110764, -0.2976203],
            },
            "SpectralQuantity": "Irradiance",
        }


if __name__ == "__main__":
    unittest.main()
