# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.temperature.cct` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from itertools import permutations

from colour.colorimetry import STANDARD_OBSERVERS_CMFS
from colour.temperature import (
    CCT_to_uv_Ohno2013, CCT_to_uv_Robertson1968, CCT_to_uv_Krystek1985,
    uv_to_CCT_Ohno2013, uv_to_CCT_Robertson1968, CCT_to_xy_Kang2002,
    CCT_to_xy_CIE_D, xy_to_CCT_McCamy1992, xy_to_CCT_Hernandez1999)
from colour.temperature.cct import (planckian_table,
                                    planckian_table_minimal_distance_index)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestPlanckianTable', 'TestPlanckianTableMinimalDistanceIndex',
    'Testuv_to_CCT_Ohno2013', 'TestCCT_to_uv_Ohno2013',
    'Testuv_to_CCT_Robertson1968', 'TestCCT_to_uv_Robertson1968',
    'TestCCT_to_uv_Krystek1985', 'Testxy_to_CCT_McCamy1992',
    'Testxy_to_CCT_Hernandez1999', 'TestCCT_to_xy_Kang2002',
    'TestCCT_to_xy_CIE_D'
]

PLANCKIAN_TABLE = np.array([
    [1000.00000000, 0.44796288, 0.35462962, 0.25373557],
    [1001.11111111, 0.44770303, 0.35465214, 0.25348315],
    [1002.22222222, 0.44744348, 0.35467461, 0.25323104],
    [1003.33333333, 0.44718423, 0.35469704, 0.25297924],
    [1004.44444444, 0.44692529, 0.35471942, 0.25272774],
    [1005.55555556, 0.44666666, 0.35474175, 0.25247656],
    [1006.66666667, 0.44640833, 0.35476404, 0.25222569],
    [1007.77777778, 0.44615030, 0.35478628, 0.25197512],
    [1008.88888889, 0.44589258, 0.35480848, 0.25172487],
    [1010.00000000, 0.44563516, 0.35483063, 0.25147492],
])

TEMPERATURE_DUV_TO_UV = {
    (2000, -0.0500): np.array([0.309448284638118, 0.309263824757947]),
    (2000, -0.0250): np.array([0.307249142319059, 0.334166912378974]),
    (2000, 0.0000): np.array([0.305050000000000, 0.359070000000000]),
    (2000, 0.0250): np.array([0.302850857680941, 0.383973087621026]),
    (2000, 0.0500): np.array([0.300651715361882, 0.408876175242053]),
    (4500, -0.0500): np.array([0.249455133366328, 0.290111562671684]),
    (4500, -0.0250): np.array([0.233393122238719, 0.309269114669175]),
    (4500, 0.0000): np.array([0.217331111111111, 0.328426666666667]),
    (4500, 0.0250): np.array([0.201269099983503, 0.347584218664158]),
    (4500, 0.0500): np.array([0.185207088855895, 0.366741770661649]),
    (7000, -0.0500): np.array([0.239763542240142, 0.279200871249525]),
    (7000, -0.0250): np.array([0.218977485405785, 0.293091149910477]),
    (7000, 0.0000): np.array([0.198191428571429, 0.306981428571429]),
    (7000, 0.0250): np.array([0.177405371737072, 0.320871707232380]),
    (7000, 0.0500): np.array([0.156619314902715, 0.334761985893332]),
    (9500, -0.0500): np.array([0.235948844766320, 0.272619554367689]),
    (9500, -0.0250): np.array([0.213587053962107, 0.283797671920686]),
    (9500, 0.0000): np.array([0.191225263157895, 0.294975789473684]),
    (9500, 0.0250): np.array([0.168863472353682, 0.306153907026682]),
    (9500, 0.0500): np.array([0.146501681549470, 0.317332024579680]),
    (12000, -0.0500): np.array([0.233956908310164, 0.268393952067210]),
    (12000, -0.0250): np.array([0.210911787488415, 0.278085309366938]),
    (12000, 0.0000): np.array([0.187866666666667, 0.287776666666667]),
    (12000, 0.0250): np.array([0.164821545844918, 0.297468023966395]),
    (12000, 0.0500): np.array([0.141776425023169, 0.307159381266124]),
    (14500, -0.0500): np.array([0.232785809380768, 0.265479540863524]),
    (14500, -0.0250): np.array([0.209387387449005, 0.274283735949003]),
    (14500, 0.0000): np.array([0.185988965517241, 0.283087931034483]),
    (14500, 0.0250): np.array([0.162590543585478, 0.291892126119962]),
    (14500, 0.0500): np.array([0.139192121653715, 0.300696321205442]),
    (17000, -0.0500): np.array([0.232028466821727, 0.263383405240889]),
    (17000, -0.0250): np.array([0.208421880469687, 0.271613173208680]),
    (17000, 0.0000): np.array([0.184815294117647, 0.279842941176471]),
    (17000, 0.0250): np.array([0.161208707765607, 0.288072709144261]),
    (17000, 0.0500): np.array([0.137602121413567, 0.296302477112052]),
    (19500, -0.0500): np.array([0.231498602829451, 0.261824985205592]),
    (19500, -0.0250): np.array([0.207757250132674, 0.269657492602796]),
    (19500, 0.0000): np.array([0.184015897435897, 0.277490000000000]),
    (19500, 0.0250): np.array([0.160274544739121, 0.285322507397204]),
    (19500, 0.0500): np.array([0.136533192042344, 0.293155014794408]),
    (22000, -0.0500): np.array([0.231114936519607, 0.260621561540512]),
    (22000, -0.0250): np.array([0.207281559168895, 0.268169417133892]),
    (22000, 0.0000): np.array([0.183448181818182, 0.275717272727273]),
    (22000, 0.0250): np.array([0.159614804467469, 0.283265128320653]),
    (22000, 0.0500): np.array([0.135781427116756, 0.290812983914034]),
    (24500, -0.0500): np.array([0.230812633988541, 0.259664771591227]),
    (24500, -0.0250): np.array([0.206910092504474, 0.266990651101736]),
    (24500, 0.0000): np.array([0.183007551020408, 0.274316530612245]),
    (24500, 0.0250): np.array([0.159105009536342, 0.281642410122754]),
    (24500, 0.0500): np.array([0.135202468052276, 0.288968289633262]),
    (27000, -0.0500): np.array([0.230583187091274, 0.258895100975109]),
    (27000, -0.0250): np.array([0.206630667619711, 0.266055883820888]),
    (27000, 0.0000): np.array([0.182678148148148, 0.273216666666667]),
    (27000, 0.0250): np.array([0.158725628676585, 0.280377449512446]),
    (27000, 0.0500): np.array([0.134773109205022, 0.287538232358225]),
    (29500, -0.0500): np.array([0.230395500499851, 0.258258464459758]),
    (29500, -0.0250): np.array([0.206403428216027, 0.265285588162082]),
    (29500, 0.0000): np.array([0.182411355932203, 0.272312711864407]),
    (29500, 0.0250): np.array([0.158419283648380, 0.279339835566731]),
    (29500, 0.0500): np.array([0.134427211364556, 0.286366959269056]),
    (32000, -0.0500): np.array([0.230235978654155, 0.257721638699323]),
    (32000, -0.0250): np.array([0.206211114327078, 0.264635819349661]),
    (32000, 0.0000): np.array([0.182186250000000, 0.271550000000000]),
    (32000, 0.0250): np.array([0.158161385672923, 0.278464180650339]),
    (32000, 0.0500): np.array([0.134136521345845, 0.285378361300677]),
    (34500, -0.0500): np.array([0.230105666168844, 0.257266749246258]),
    (34500, -0.0250): np.array([0.206054789606161, 0.264089896362260]),
    (34500, 0.0000): np.array([0.182003913043478, 0.270913043478261]),
    (34500, 0.0250): np.array([0.157953036480796, 0.277736190594262]),
    (34500, 0.0500): np.array([0.133902159918113, 0.284559337710263]),
    (37000, -0.0500): np.array([0.229999835834901, 0.256877639136083]),
    (37000, -0.0250): np.array([0.205928431430964, 0.263628008757231]),
    (37000, 0.0000): np.array([0.181857027027027, 0.270378378378378]),
    (37000, 0.0250): np.array([0.157785622623090, 0.277128747999526]),
    (37000, 0.0500): np.array([0.133714218219153, 0.283879117620674]),
    (39500, -0.0500): np.array([0.229907042065651, 0.256537886918202]),
    (39500, -0.0250): np.array([0.205817888121433, 0.263224639661633]),
    (39500, 0.0000): np.array([0.181728734177215, 0.269911392405063]),
    (39500, 0.0250): np.array([0.157639580232997, 0.276598145148494]),
    (39500, 0.0500): np.array([0.133550426288780, 0.283284897891924]),
    (42000, -0.0500): np.array([0.229825016678223, 0.256238659086365]),
    (42000, -0.0250): np.array([0.205720365481969, 0.262869329543182]),
    (42000, 0.0000): np.array([0.181615714285714, 0.269500000000000]),
    (42000, 0.0250): np.array([0.157511063089460, 0.276130670456817]),
    (42000, 0.0500): np.array([0.133406411893206, 0.282761340913635]),
    (44500, -0.0500): np.array([0.229751988653572, 0.255973111790163]),
    (44500, -0.0250): np.array([0.205633690955999, 0.262553971625419]),
    (44500, 0.0000): np.array([0.181515393258427, 0.269134831460674]),
    (44500, 0.0250): np.array([0.157397095560855, 0.275715691295930]),
    (44500, 0.0500): np.array([0.133278797863282, 0.282296551131185]),
    (47000, -0.0500): np.array([0.229686555034366, 0.255735860408829]),
    (47000, -0.0250): np.array([0.205556149857609, 0.262272185523563]),
    (47000, 0.0000): np.array([0.181425744680851, 0.268808510638298]),
    (47000, 0.0250): np.array([0.157295339504093, 0.275344835753032]),
    (47000, 0.0500): np.array([0.133164934327336, 0.281881160867767]),
    (49500, -0.0500): np.array([0.229627590056716, 0.255522610251793]),
    (49500, -0.0250): np.array([0.205486370785934, 0.262018880883472]),
    (49500, 0.0000): np.array([0.181345151515151, 0.268515151515151]),
    (49500, 0.0250): np.array([0.157203932244369, 0.275011422146831]),
    (49500, 0.0500): np.array([0.133062712973587, 0.281507692778510])
}


class TestPlanckianTable(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cct.planckian_table` definition units
    tests methods.
    """

    def test_planckian_table(self):
        """
        Tests :func:`colour.temperature.cct.planckian_table` definition.
        """

        cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']

        np.testing.assert_almost_equal(
            [(x.Ti, x.ui, x.vi, x.di) for x in planckian_table(
                np.array([0.1978, 0.3122]), cmfs, 1000, 1010, 10)],
            PLANCKIAN_TABLE)


class TestPlanckianTableMinimalDistanceIndex(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cct.\
planckian_table_minimal_distance_index` definition unit tests methods.
    """

    def test_planckian_table_minimal_distance_index(self):
        """
        Tests :func:`colour.temperature.cct.\
planckian_table_minimal_distance_index` definition.
        """

        cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        self.assertEqual(
            planckian_table_minimal_distance_index(
                planckian_table(
                    np.array([0.1978, 0.3122]), cmfs, 1000, 1010, 10)), 9)


class Testuv_to_CCT_Ohno2013(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cct.uv_to_CCT_Ohno2013` definition units
    tests methods.
    """

    def test_uv_to_CCT_Ohno2013(self):
        """
        Tests :func:`colour.temperature.cct.uv_to_CCT_Ohno2013` definition.
        """

        cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        np.testing.assert_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.1978, 0.3122]), cmfs),
            np.array([6507.47380460, 0.00322335]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.4328, 0.2883]), cmfs),
            np.array([1041.68315360, -0.06737802]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.2927, 0.2722]), cmfs, iterations=4),
            np.array([2452.15316417, -0.08437064]),
            decimal=7)


class TestCCT_to_uv_Ohno2013(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cct.CCT_to_uv_Ohno2013` definition units
    tests methods.
    """

    def test_CCT_to_uv_Ohno2013(self):
        """
        Tests :func:`colour.temperature.cct.CCT_to_uv_Ohno2013` definition.
        """

        cmfs = STANDARD_OBSERVERS_CMFS['CIE 1931 2 Degree Standard Observer']
        np.testing.assert_almost_equal(
            CCT_to_uv_Ohno2013(6507.47380460, 0.00322335, cmfs),
            np.array([0.19779997, 0.31219997]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_uv_Ohno2013(1041.68315360, -0.06737802, cmfs),
            np.array([0.43279885, 0.28830013]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_uv_Ohno2013(2452.15316417, -0.08437064, cmfs),
            np.array([0.29247364, 0.27215157]),
            decimal=7)


class Testuv_to_CCT_Robertson1968(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cct.uv_to_CCT_Robertson1968` definition
    unit tests methods.
    """

    def test_uv_to_CCT_Robertson1968(self):
        """
        Tests :func:`colour.temperature.cct.uv_to_CCT_Robertson1968`
        definition.
        """

        for key, value in TEMPERATURE_DUV_TO_UV.items():
            np.testing.assert_allclose(
                uv_to_CCT_Robertson1968(value), key, atol=0.25)


class TestCCT_to_uv_Robertson1968(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cct.CCT_to_uv_Robertson1968` definition
    unit tests methods.
    """

    def test_CCT_to_uv_Robertson1968(self):
        """
        Tests :func:`colour.temperature.cct.CCT_to_uv_Robertson1968`
        definition.
        """

        for key, value in TEMPERATURE_DUV_TO_UV.items():
            np.testing.assert_almost_equal(
                CCT_to_uv_Robertson1968(*key), value, decimal=7)


class TestCCT_to_uv_Krystek1985(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cct.CCT_to_uv_Krystek1985` definition
    units tests methods.
    """

    def test_CCT_to_uv_Krystek1985(self):
        """
        Tests :func:`colour.temperature.cct.CCT_to_uv_Krystek1985` definition.
        """

        np.testing.assert_almost_equal(
            CCT_to_uv_Krystek1985(1000),
            np.array([0.223421163527869, 0.499258998136231]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_uv_Krystek1985(7000),
            np.array([0.183513095046506, 0.305827773965731]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_uv_Krystek1985(15000),
            np.array([0.182148234861937, 0.281354360914682]),
            decimal=7)

    def test_n_dimensional_CCT_to_uv_Krystek1985(self):
        """
        Tests :func:`colour.temperature.cct.CCT_to_uv_Krystek1985` definition
        n-dimensional arrays support.
        """

        CCT = 7000
        xy = np.array([0.183513095046506, 0.305827773965731])
        np.testing.assert_almost_equal(
            CCT_to_uv_Krystek1985(CCT), xy, decimal=7)

        CCT = np.tile(CCT, 6)
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(
            CCT_to_uv_Krystek1985(CCT), xy, decimal=7)

        CCT = np.reshape(CCT, (2, 3))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(
            CCT_to_uv_Krystek1985(CCT), xy, decimal=7)

    @ignore_numpy_errors
    def test_nan_CCT_to_uv_Krystek1985(self):
        """
        Tests :func:`colour.temperature.cct.CCT_to_uv_Krystek1985` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            CCT_to_uv_Krystek1985(case)


class Testxy_to_CCT_McCamy1992(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cct.xy_to_CCT_McCamy1992` definition
    unit tests methods.
    """

    def test_xy_to_CCT_McCamy1992(self):
        """
        Tests :func:`colour.temperature.cct.xy_to_CCT_McCamy1992` definition.
        """

        self.assertAlmostEqual(
            xy_to_CCT_McCamy1992(np.array([0.31270, 0.32900])),
            6505.08059131,
            places=7)

        self.assertAlmostEqual(
            xy_to_CCT_McCamy1992(np.array([0.44757, 0.40745])),
            2857.28961266,
            places=7)

        self.assertAlmostEqual(
            xy_to_CCT_McCamy1992(
                np.array([0.252520939374083, 0.252220883926284])),
            19501.61953130,
            places=7)

    def test_n_dimensional_xy_to_CCT_McCamy1992(self):
        """
        Tests :func:`colour.temperature.cct.xy_to_CCT_McCamy1992` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.31270, 0.32900])
        CCT = 6505.08059131
        np.testing.assert_almost_equal(
            xy_to_CCT_McCamy1992(xy), CCT, decimal=7)

        xy = np.tile(xy, (6, 1))
        CCT = np.tile(CCT, 6)
        np.testing.assert_almost_equal(
            xy_to_CCT_McCamy1992(xy), CCT, decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        CCT = np.reshape(CCT, (2, 3))
        np.testing.assert_almost_equal(
            xy_to_CCT_McCamy1992(xy), CCT, decimal=7)

    @ignore_numpy_errors
    def test_nan_xy_to_CCT_McCamy1992(self):
        """
        Tests :func:`colour.temperature.cct.xy_to_CCT_McCamy1992` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            xy_to_CCT_McCamy1992(case)


class Testxy_to_CCT_Hernandez1999(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cct.xy_to_CCT_Hernandez1999` definition
    unit tests methods.
    """

    def test_xy_to_CCT_Hernandez1999(self):
        """
        Tests :func:`colour.temperature.cct.xy_to_CCT_McCamy1992` definition.
        """

        self.assertAlmostEqual(
            xy_to_CCT_Hernandez1999(np.array([0.31270, 0.32900])),
            6500.74204318,
            places=7)

        self.assertAlmostEqual(
            xy_to_CCT_Hernandez1999(np.array([0.44757, 0.40745])),
            2790.64222533,
            places=7)

        self.assertAlmostEqual(
            xy_to_CCT_Hernandez1999(
                np.array([0.244162248213914, 0.240333674758318])),
            64448.11092565,
            places=7)

    def test_n_dimensional_xy_to_CCT_Hernandez1999(self):
        """
        Tests :func:`colour.temperature.cct.xy_to_CCT_Hernandez1999` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.31270, 0.32900])
        CCT = 6500.74204318
        np.testing.assert_almost_equal(
            xy_to_CCT_Hernandez1999(xy), CCT, decimal=7)

        xy = np.tile(xy, (6, 1))
        CCT = np.tile(CCT, 6)
        np.testing.assert_almost_equal(
            xy_to_CCT_Hernandez1999(xy), CCT, decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        CCT = np.reshape(CCT, (2, 3))
        np.testing.assert_almost_equal(
            xy_to_CCT_Hernandez1999(xy), CCT, decimal=7)

    @ignore_numpy_errors
    def test_nan_xy_to_CCT_Hernandez1999(self):
        """
        Tests :func:`colour.temperature.cct.xy_to_CCT_Hernandez1999` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            xy_to_CCT_Hernandez1999(case)


class TestCCT_to_xy_Kang2002(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cct.CCT_to_xy_Kang2002` definition units
    tests methods.
    """

    def test_CCT_to_xy_Kang2002(self):
        """
        Tests :func:`colour.temperature.cct.CCT_to_xy_Kang2002` definition.
        """

        np.testing.assert_almost_equal(
            CCT_to_xy_Kang2002(4000),
            np.array([0.380528282812500, 0.376733530961114]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_xy_Kang2002(7000),
            np.array([0.306374019533528, 0.316552869726577]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_xy_Kang2002(25000),
            np.array([0.25247299, 0.252254791243654]),
            decimal=7)

    def test_n_dimensional_CCT_to_xy_Kang2002(self):
        """
        Tests :func:`colour.temperature.cct.CCT_to_xy_Kang2002` definition
        n-dimensional arrays support.
        """

        CCT = 4000
        xy = np.array([0.380528282812500, 0.376733530961114])
        np.testing.assert_almost_equal(CCT_to_xy_Kang2002(CCT), xy, decimal=7)

        CCT = np.tile(CCT, 6)
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(CCT_to_xy_Kang2002(CCT), xy, decimal=7)

        CCT = np.reshape(CCT, (2, 3))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(CCT_to_xy_Kang2002(CCT), xy, decimal=7)

    @ignore_numpy_errors
    def test_nan_CCT_to_xy_Kang2002(self):
        """
        Tests :func:`colour.temperature.cct.CCT_to_xy_Kang2002` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            CCT_to_xy_Kang2002(case)


class TestCCT_to_xy_CIE_D(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cct.CCT_to_xy_CIE_D` definition
    unit tests methods.
    """

    def test_CCT_to_xy_CIE_D(self):
        """
        Tests :func:`colour.temperature.cct.CCT_to_xy_CIE_D` definition.
        """

        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(4000.0),
            np.array([0.382343625000000, 0.383766261015578]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(4000),
            np.array([0.382343625000000, 0.383766261015578]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(7000),
            np.array([0.305357431486880, 0.321646345474552]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(25000),
            np.array([0.24985367, 0.254799464210944]),
            decimal=7)

    def test_n_dimensional_CCT_to_xy_CIE_D(self):
        """
        Tests :func:`colour.temperature.cct.CCT_to_xy_CIE_D` definition
        n-dimensional arrays support.
        """

        CCT = 4000
        xy = np.array([0.382343625000000, 0.383766261015578])
        np.testing.assert_almost_equal(CCT_to_xy_CIE_D(CCT), xy, decimal=7)

        CCT = np.tile(CCT, 6)
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(CCT_to_xy_CIE_D(CCT), xy, decimal=7)

        CCT = np.reshape(CCT, (2, 3))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(CCT_to_xy_CIE_D(CCT), xy, decimal=7)

    @ignore_numpy_errors
    def test_nan_CCT_to_xy_CIE_D(self):
        """
        Tests :func:`colour.temperature.cct.CCT_to_xy_CIE_D` definition
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=2))
        for case in cases:
            CCT_to_xy_CIE_D(case)


if __name__ == '__main__':
    unittest.main()
