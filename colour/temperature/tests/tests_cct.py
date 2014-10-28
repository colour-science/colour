#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.temperature.cct` module.
"""

from __future__ import division, unicode_literals

import numpy as np

import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.colorimetry import STANDARD_OBSERVERS_CMFS
from colour.temperature import CCT_to_uv_Ohno2013, CCT_to_uv_Robertson1968
from colour.temperature import uv_to_CCT_Ohno2013, uv_to_CCT_Robertson1968
from colour.temperature import CCT_to_xy_Kang2002, CCT_to_xy_CIE_D
from colour.temperature import xy_to_CCT_McCamy1992, xy_to_CCT_Hernandez1999
from colour.temperature.cct import (
    PLANCKIAN_TABLE_TUVD,
    planckian_table,
    planckian_table_minimal_distance_index)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestPlanckianTable',
           'TestPlanckianTableMinimalDistanceIndex',
           'Testuv_to_CCT_Ohno2013',
           'TestCCT_to_uv_Ohno2013',
           'Testuv_to_CCT_Robertson1968',
           'TestCCT_to_uv_Robertson1968',
           'Testxy_to_CCT_McCamy1992',
           'Testxy_to_CCT_Hernandez1999',
           'TestCCT_to_xy_Kang2002',
           'TestCCT_to_xy_CIE_D']

PLANCKIAN_TABLE = [
    PLANCKIAN_TABLE_TUVD(Ti=1000.0, ui=0.44801089464064786,
                         vi=0.35462498085812383, di=0.25378213254223697),
    PLANCKIAN_TABLE_TUVD(Ti=1001.1111111111111, ui=0.44775082571296082,
                         vi=0.35464751834928576, di=0.25352949944454956),
    PLANCKIAN_TABLE_TUVD(Ti=1002.2222222222222, ui=0.4474910632226583,
                         vi=0.35467001024049455, di=0.25327717785676873),
    PLANCKIAN_TABLE_TUVD(Ti=1003.3333333333334, ui=0.44723160701057407,
                         vi=0.35469245651987269, di=0.253025167595974),
    PLANCKIAN_TABLE_TUVD(Ti=1004.4444444444445, ui=0.44697245691544729,
                         vi=0.35471485717585538, di=0.2527734684771403),
    PLANCKIAN_TABLE_TUVD(Ti=1005.5555555555555, ui=0.44671361277394045,
                         vi=0.35473721219718934, di=0.25252208031315576),
    PLANCKIAN_TABLE_TUVD(Ti=1006.6666666666666, ui=0.44645507442066867,
                         vi=0.35475952157292978, di=0.2522710029148514),
    PLANCKIAN_TABLE_TUVD(Ti=1007.7777777777778, ui=0.446196841688206,
                         vi=0.35478178529244025, di=0.2520202360910075),
    PLANCKIAN_TABLE_TUVD(Ti=1008.8888888888889, ui=0.44593891440711669,
                         vi=0.35480400334538981, di=0.2517697796483851),
    PLANCKIAN_TABLE_TUVD(Ti=1010.0, ui=0.44568129240596821,
                         vi=0.35482617572175207, di=0.25151963339173905)]

TEMPERATURE_DUV_TO_UV = {
    (2000, -0.05): (0.3094482846381184, 0.309263824757947),
    (2000, -0.025): (0.3072491423190592, 0.3341669123789735),
    (2000, 0.0): (0.30505, 0.35907),
    (2000, 0.02500000000000001): (0.3028508576809408, 0.3839730876210265),
    (2000, 0.05): (0.30065171536188157, 0.408876175242053),
    (4500, -0.05): (0.24945513336632763, 0.29011156267168375),
    (4500, -0.025): (0.23339312223871939, 0.30926911466917517),
    (4500, 0.0): (0.2173311111111111, 0.32842666666666664),
    (4500, 0.02500000000000001): (0.20126909998350284, 0.3475842186641581),
    (4500, 0.05): (0.1852070888558946, 0.36674177066164954),
    (7000, -0.05): (0.23976354224014196, 0.27920087124952486),
    (7000, -0.025): (0.21897748540578527, 0.2930911499104767),
    (7000, 0.0): (0.19819142857142857, 0.3069814285714285),
    (7000, 0.02500000000000001): (0.17740537173707188, 0.32087170723238034),
    (7000, 0.05): (0.1566193149027152, 0.33476198589333217),
    (9500, -0.05): (0.2359488447663197, 0.2726195543676886),
    (9500, -0.025): (0.2135870539621072, 0.2837976719206864),
    (9500, 0.0): (0.19122526315789473, 0.29497578947368425),
    (9500, 0.02500000000000001): (0.16886347235368224, 0.3061539070266821),
    (9500, 0.05): (0.14650168154946977, 0.3173320245796799),
    (12000, -0.05): (0.23395690831016422, 0.2683939520672096),
    (12000, -0.025): (0.21091178748841544, 0.27808530936693815),
    (12000, 0.0): (0.18786666666666668, 0.2877766666666667),
    (12000, 0.02500000000000001): (0.1648215458449179, 0.2974680239663952),
    (12000, 0.05): (0.14177642502316914, 0.30715938126612374),
    (14500, -0.05): (0.23278580938076784, 0.2654795408635239),
    (14500, -0.025): (0.2093873874490046, 0.27428373594900335),
    (14500, 0.0): (0.18598896551724137, 0.2830879310344828),
    (14500, 0.02500000000000001): (0.16259054358547811, 0.2918921261199623),
    (14500, 0.05): (0.1391921216537149, 0.30069632120544176),
    (17000, -0.05): (0.23202846682172715, 0.2633834052408889),
    (17000, -0.025): (0.2084218804696871, 0.2716131732086797),
    (17000, 0.0): (0.18481529411764705, 0.27984294117647057),
    (17000, 0.02500000000000001): (0.161208707765607, 0.2880727091442614),
    (17000, 0.05): (0.13760212141356695, 0.29630247711205226),
    (19500, -0.05): (0.2314986028294509, 0.26182498520559233),
    (19500, -0.025): (0.20775725013267415, 0.2696574926027962),
    (19500, 0.0): (0.18401589743589744, 0.27749),
    (19500, 0.02500000000000001): (0.16027454473912073, 0.28532250739720383),
    (19500, 0.05): (0.136533192042344, 0.2931550147944077),
    (22000, -0.05): (0.23111493651960752, 0.26062156154051147),
    (22000, -0.025): (0.20728155916889468, 0.2681694171338921),
    (22000, 0.0): (0.18344818181818182, 0.27571727272727276),
    (22000, 0.02500000000000001): (0.15961480446746895, 0.2832651283206534),
    (22000, 0.05): (0.13578142711675611, 0.29081298391403404),
    (24500, -0.05): (0.23081263398854066, 0.2596647715912274),
    (24500, -0.025): (0.20691009250447442, 0.26699065110173614),
    (24500, 0.0): (0.18300755102040817, 0.2743165306122449),
    (24500, 0.02500000000000001): (0.1591050095363419, 0.2816424101227537),
    (24500, 0.05): (0.13520246805227568, 0.28896828963326243),
    (27000, -0.05): (0.23058318709127387, 0.25889510097510876),
    (27000, -0.025): (0.206630667619711, 0.26605588382088774),
    (27000, 0.0): (0.18267814814814815, 0.27321666666666666),
    (27000, 0.02500000000000001): (0.15872562867658527, 0.2803774495124456),
    (27000, 0.05): (0.13477310920502242, 0.28753823235822457),
    (29500, -0.05): (0.2303955004998511, 0.2582584644597578),
    (29500, -0.025): (0.20640342821602725, 0.26528558816208225),
    (29500, 0.0): (0.18241135593220337, 0.27231271186440675),
    (29500, 0.02500000000000001): (0.15841928364837948, 0.27933983556673125),
    (29500, 0.05): (0.13442721136455563, 0.2863669592690557),
    (32000, -0.05): (0.23023597865415507, 0.2577216386993227),
    (32000, -0.025): (0.20621111432707753, 0.2646358193496613),
    (32000, 0.0): (0.18218625, 0.27154999999999996),
    (32000, 0.02500000000000001): (0.15816138567292246, 0.2784641806503386),
    (32000, 0.05): (0.13413652134584492, 0.28537836130067723),
    (34500, -0.05): (0.23010566616884381, 0.2572667492462583),
    (34500, -0.025): (0.20605478960616105, 0.2640898963622596),
    (34500, 0.0): (0.18200391304347827, 0.27091304347826084),
    (34500, 0.02500000000000001): (0.15795303648079548, 0.2777361905942621),
    (34500, 0.05): (0.13390215991811272, 0.28455933771026337),
    (37000, -0.05): (0.22999983583490138, 0.25687763913608286),
    (37000, -0.025): (0.20592843143096423, 0.2636280087572306),
    (37000, 0.0): (0.18185702702702705, 0.27037837837837836),
    (37000, 0.02500000000000001): (0.15778562262308987, 0.27712874799952614),
    (37000, 0.05): (0.13371421821915272, 0.28387911762067386),
    (39500, -0.05): (0.2299070420656508, 0.2565378869182022),
    (39500, -0.025): (0.205817888121433, 0.2632246396616327),
    (39500, 0.0): (0.1817287341772152, 0.26991139240506323),
    (39500, 0.02500000000000001): (0.15763958023299737, 0.2765981451484938),
    (39500, 0.05): (0.13355042628877958, 0.2832848978919243),
    (42000, -0.05): (0.229825016678223, 0.2562386590863649),
    (42000, -0.025): (0.20572036548196865, 0.26286932954318243),
    (42000, 0.0): (0.18161571428571427, 0.26949999999999996),
    (42000, 0.02500000000000001): (0.1575110630894599, 0.2761306704568175),
    (42000, 0.05): (0.13340641189320554, 0.28276134091363503),
    (44500, -0.05): (0.22975198865357158, 0.2559731117901632),
    (44500, -0.025): (0.20563369095599926, 0.26255397162541866),
    (44500, 0.0): (0.18151539325842697, 0.26913483146067413),
    (44500, 0.02500000000000001): (0.15739709556085466, 0.2757156912959296),
    (44500, 0.05): (0.13327879786328237, 0.28229655113118507),
    (47000, -0.05): (0.22968655503436627, 0.2557358604088288),
    (47000, -0.025): (0.20555614985760867, 0.26227218552356335),
    (47000, 0.0): (0.18142574468085104, 0.26880851063829786),
    (47000, 0.02500000000000001): (0.1572953395040934, 0.27534483575303237),
    (47000, 0.05): (0.1331649343273358, 0.28188116086776693),
    (49500, -0.05): (0.22962759005671562, 0.2555226102517927),
    (49500, -0.025): (0.20548637078593357, 0.2620188808834721),
    (49500, 0.0): (0.1813451515151515, 0.2685151515151515),
    (49500, 0.02500000000000001): (0.1572039322443694, 0.2750114221468309),
    (49500, 0.05): (0.13306271297358735, 0.2815076927785103)}


class TestPlanckianTable(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cct.planckian_table` definition units
    tests methods.
    """

    def test_planckian_table(self):
        """
        Tests :func:`colour.temperature.cct.planckian_table` definition.
        """

        cmfs = STANDARD_OBSERVERS_CMFS.get(
            'CIE 1931 2 Degree Standard Observer')
        to_tuple = lambda x: (x.Ti, x.ui, x.vi, x.di)
        np.testing.assert_almost_equal(
            [to_tuple(x) for x in planckian_table((0.1978, 0.3122),
                                                  cmfs,
                                                  1000,
                                                  1010,
                                                  10)],
            [to_tuple(x) for x in PLANCKIAN_TABLE])


class TestPlanckianTableMinimalDistanceIndex(unittest.TestCase):
    """
    Defines
    :func:`colour.temperature.cct.planckian_table_minimal_distance_index`
    definition unit tests methods.
    """

    def test_planckian_table_minimal_distance_index(self):
        """
        Tests
        :func:`colour.temperature.cct.planckian_table_minimal_distance_index`
        definition.
        """

        cmfs = STANDARD_OBSERVERS_CMFS.get(
            'CIE 1931 2 Degree Standard Observer')
        self.assertEqual(
            planckian_table_minimal_distance_index(
                planckian_table((0.1978, 0.3122),
                                cmfs,
                                1000,
                                1010,
                                10)),
            9)


class Testuv_to_CCT_Ohno2013(unittest.TestCase):
    """
    Defines :func:`colour.temperature.cct.uv_to_CCT_Ohno2013` definition units
    tests methods.
    """

    def test_uv_to_CCT_Ohno2013(self):
        """
        Tests :func:`colour.temperature.cct.uv_to_CCT_Ohno2013` definition.
        """

        cmfs = STANDARD_OBSERVERS_CMFS.get(
            'CIE 1931 2 Degree Standard Observer')
        np.testing.assert_almost_equal(
            uv_to_CCT_Ohno2013((0.1978, 0.3122), cmfs),
            (6507.5470349001507, 0.0032236908012382953),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_CCT_Ohno2013((0.4328, 0.2883), cmfs),
            (1041.8672179878763, -0.067377582642145384),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_CCT_Ohno2013((0.2927, 0.2722), cmfs, iterations=4),
            (2452.1932942782669, -0.084369982045528508),
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

        cmfs = STANDARD_OBSERVERS_CMFS.get(
            'CIE 1931 2 Degree Standard Observer')
        np.testing.assert_almost_equal(
            CCT_to_uv_Ohno2013(
                6507.4342201047066, 0.003223690901512735, cmfs),
            (0.19780034881616862, 0.31220050291046603),
            decimal=7)
        np.testing.assert_almost_equal(
            CCT_to_uv_Ohno2013(
                1041.849524611546, -0.067377582728534946, cmfs),
            (0.43280250331413772, 0.28829975758516474),
            decimal=7)
        np.testing.assert_almost_equal(
            CCT_to_uv_Ohno2013(
                2448.9489053326438, -0.084324704634692743, cmfs),
            (0.29256616302348853, 0.27221773141874955),
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
                uv_to_CCT_Robertson1968(value),
                key,
                atol=0.25)


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

        for i in range(2000, 49501, 2500):
            for j in np.arange(-0.05, 0.075, 0.025):
                np.testing.assert_almost_equal(
                    CCT_to_uv_Robertson1968(i, j),
                    TEMPERATURE_DUV_TO_UV.get((i, j)),
                    decimal=7)


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
            xy_to_CCT_McCamy1992((0.31271, 0.32902)),
            6504.38938305,
            places=7)
        self.assertAlmostEqual(
            xy_to_CCT_McCamy1992((0.44757, 0.40745)),
            2857.28961266,
            places=7)
        self.assertAlmostEqual(
            xy_to_CCT_McCamy1992(
                (0.25252093937408293, 0.252220883926284)),
            19501.6195313,
            places=7)


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
            xy_to_CCT_Hernandez1999((0.31271, 0.32902)),
            6500.04215334,
            places=7)
        self.assertAlmostEqual(
            xy_to_CCT_Hernandez1999((0.44757, 0.40745)),
            2790.64222533,
            places=7)
        self.assertAlmostEqual(
            xy_to_CCT_Hernandez1999(
                (0.24416224821391358, 0.24033367475831827)),
            64448.110925653324,
            places=7)


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
            CCT_to_xy_Kang2002(4000), (
                0.38052828281249995, 0.3767335309611144),
            decimal=7)
        np.testing.assert_almost_equal(
            CCT_to_xy_Kang2002(7000),
            (0.30637401953352766, 0.31655286972657715),
            decimal=7)
        np.testing.assert_almost_equal(
            CCT_to_xy_Kang2002(25000),
            (0.2524729944384, 0.2522547912436536),
            decimal=7)


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
            CCT_to_xy_CIE_D(4000),
            (0.38234362499999996, 0.3837662610155782),
            decimal=7)
        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(7000),
            (0.3053574314868805, 0.3216463454745523),
            decimal=7)
        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(25000),
            (0.2498536704, 0.25479946421094446),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
