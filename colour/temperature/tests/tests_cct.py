#!/usr/bin/env python
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
    CCT_to_uv_Ohno2013,
    CCT_to_uv_Robertson1968,
    uv_to_CCT_Ohno2013,
    uv_to_CCT_Robertson1968,
    CCT_to_xy_Kang2002,
    CCT_to_xy_CIE_D,
    xy_to_CCT_McCamy1992,
    xy_to_CCT_Hernandez1999)
from colour.temperature.cct import (
    planckian_table,
    planckian_table_minimal_distance_index)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
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

PLANCKIAN_TABLE = (
    (1000.00000000, 0.44801089, 0.35462498, 0.25378213),
    (1001.11111111, 0.44775083, 0.35464752, 0.25352950),
    (1002.22222222, 0.44749106, 0.35467001, 0.25327718),
    (1003.33333333, 0.44723161, 0.35469246, 0.25302517),
    (1004.44444444, 0.44697246, 0.35471486, 0.25277347),
    (1005.55555556, 0.44671361, 0.35473721, 0.25252208),
    (1006.66666667, 0.44645507, 0.35475952, 0.25227100),
    (1007.77777778, 0.44619684, 0.35478179, 0.25202024),
    (1008.88888889, 0.44593891, 0.35480400, 0.25176978),
    (1010.00000000, 0.44568129, 0.35482618, 0.25151963))

TEMPERATURE_DUV_TO_UV = {
    (2000, -0.0500): (0.3094482846381184, 0.3092638247579470),
    (2000, -0.0250): (0.3072491423190592, 0.3341669123789735),
    (2000, 0.0000): (0.3050500000000000, 0.3590700000000000),
    (2000, 0.0250): (0.3028508576809408, 0.3839730876210265),
    (2000, 0.0500): (0.3006517153618816, 0.4088761752420530),
    (4500, -0.0500): (0.2494551333663276, 0.2901115626716838),
    (4500, -0.0250): (0.2333931222387194, 0.3092691146691752),
    (4500, 0.0000): (0.2173311111111111, 0.3284266666666666),
    (4500, 0.0250): (0.2012690999835028, 0.3475842186641581),
    (4500, 0.0500): (0.1852070888558946, 0.3667417706616495),
    (7000, -0.0500): (0.2397635422401420, 0.2792008712495249),
    (7000, -0.0250): (0.2189774854057853, 0.2930911499104767),
    (7000, 0.0000): (0.1981914285714286, 0.3069814285714285),
    (7000, 0.0250): (0.1774053717370719, 0.3208717072323803),
    (7000, 0.0500): (0.1566193149027152, 0.3347619858933322),
    (9500, -0.0500): (0.2359488447663197, 0.2726195543676886),
    (9500, -0.0250): (0.2135870539621072, 0.2837976719206864),
    (9500, 0.0000): (0.1912252631578947, 0.2949757894736842),
    (9500, 0.0250): (0.1688634723536822, 0.3061539070266821),
    (9500, 0.0500): (0.1465016815494698, 0.3173320245796799),
    (12000, -0.0500): (0.2339569083101642, 0.2683939520672096),
    (12000, -0.0250): (0.2109117874884154, 0.2780853093669381),
    (12000, 0.0000): (0.1878666666666667, 0.2877766666666667),
    (12000, 0.0250): (0.1648215458449179, 0.2974680239663952),
    (12000, 0.0500): (0.1417764250231691, 0.3071593812661237),
    (14500, -0.0500): (0.2327858093807678, 0.2654795408635239),
    (14500, -0.0250): (0.2093873874490046, 0.2742837359490033),
    (14500, 0.0000): (0.1859889655172414, 0.2830879310344828),
    (14500, 0.0250): (0.1625905435854781, 0.2918921261199623),
    (14500, 0.0500): (0.1391921216537149, 0.3006963212054418),
    (17000, -0.0500): (0.2320284668217271, 0.2633834052408889),
    (17000, -0.0250): (0.2084218804696871, 0.2716131732086797),
    (17000, 0.0000): (0.1848152941176470, 0.2798429411764706),
    (17000, 0.0250): (0.1612087077656070, 0.2880727091442614),
    (17000, 0.0500): (0.1376021214135670, 0.2963024771120523),
    (19500, -0.0500): (0.2314986028294509, 0.2618249852055923),
    (19500, -0.0250): (0.2077572501326742, 0.2696574926027962),
    (19500, 0.0000): (0.1840158974358974, 0.2774900000000000),
    (19500, 0.0250): (0.1602745447391207, 0.2853225073972038),
    (19500, 0.0500): (0.1365331920423440, 0.2931550147944077),
    (22000, -0.0500): (0.2311149365196075, 0.2606215615405115),
    (22000, -0.0250): (0.2072815591688947, 0.2681694171338921),
    (22000, 0.0000): (0.1834481818181818, 0.2757172727272728),
    (22000, 0.0250): (0.1596148044674690, 0.2832651283206534),
    (22000, 0.0500): (0.1357814271167561, 0.2908129839140340),
    (24500, -0.0500): (0.2308126339885407, 0.2596647715912274),
    (24500, -0.0250): (0.2069100925044744, 0.2669906511017361),
    (24500, 0.0000): (0.1830075510204082, 0.2743165306122449),
    (24500, 0.0250): (0.1591050095363419, 0.2816424101227537),
    (24500, 0.0500): (0.1352024680522757, 0.2889682896332624),
    (27000, -0.0500): (0.2305831870912739, 0.2588951009751088),
    (27000, -0.0250): (0.2066306676197110, 0.2660558838208877),
    (27000, 0.0000): (0.1826781481481481, 0.2732166666666667),
    (27000, 0.0250): (0.1587256286765853, 0.2803774495124456),
    (27000, 0.0500): (0.1347731092050224, 0.2875382323582246),
    (29500, -0.0500): (0.2303955004998511, 0.2582584644597578),
    (29500, -0.0250): (0.2064034282160273, 0.2652855881620823),
    (29500, 0.0000): (0.1824113559322034, 0.2723127118644068),
    (29500, 0.0250): (0.1584192836483795, 0.2793398355667313),
    (29500, 0.0500): (0.1344272113645556, 0.2863669592690557),
    (32000, -0.0500): (0.2302359786541551, 0.2577216386993227),
    (32000, -0.0250): (0.2062111143270775, 0.2646358193496613),
    (32000, 0.0000): (0.1821862500000000, 0.2715500000000000),
    (32000, 0.0250): (0.1581613856729225, 0.2784641806503386),
    (32000, 0.0500): (0.1341365213458449, 0.2853783613006772),
    (34500, -0.0500): (0.2301056661688438, 0.2572667492462583),
    (34500, -0.0250): (0.2060547896061611, 0.2640898963622596),
    (34500, 0.0000): (0.1820039130434783, 0.2709130434782608),
    (34500, 0.0250): (0.1579530364807955, 0.2777361905942621),
    (34500, 0.0500): (0.1339021599181127, 0.2845593377102634),
    (37000, -0.0500): (0.2299998358349014, 0.2568776391360829),
    (37000, -0.0250): (0.2059284314309642, 0.2636280087572306),
    (37000, 0.0000): (0.1818570270270271, 0.2703783783783784),
    (37000, 0.0250): (0.1577856226230899, 0.2771287479995261),
    (37000, 0.0500): (0.1337142182191527, 0.2838791176206739),
    (39500, -0.0500): (0.2299070420656508, 0.2565378869182022),
    (39500, -0.0250): (0.2058178881214330, 0.2632246396616327),
    (39500, 0.0000): (0.1817287341772152, 0.2699113924050632),
    (39500, 0.0250): (0.1576395802329974, 0.2765981451484938),
    (39500, 0.0500): (0.1335504262887796, 0.2832848978919243),
    (42000, -0.0500): (0.2298250166782230, 0.2562386590863649),
    (42000, -0.0250): (0.2057203654819686, 0.2628693295431824),
    (42000, 0.0000): (0.1816157142857143, 0.2695000000000000),
    (42000, 0.0250): (0.1575110630894599, 0.2761306704568175),
    (42000, 0.0500): (0.1334064118932055, 0.2827613409136350),
    (44500, -0.0500): (0.2297519886535716, 0.2559731117901632),
    (44500, -0.0250): (0.2056336909559993, 0.2625539716254187),
    (44500, 0.0000): (0.1815153932584270, 0.2691348314606741),
    (44500, 0.0250): (0.1573970955608547, 0.2757156912959296),
    (44500, 0.0500): (0.1332787978632824, 0.2822965511311851),
    (47000, -0.0500): (0.2296865550343663, 0.2557358604088288),
    (47000, -0.0250): (0.2055561498576087, 0.2622721855235634),
    (47000, 0.0000): (0.1814257446808510, 0.2688085106382979),
    (47000, 0.0250): (0.1572953395040934, 0.2753448357530324),
    (47000, 0.0500): (0.1331649343273358, 0.2818811608677669),
    (49500, -0.0500): (0.2296275900567156, 0.2555226102517927),
    (49500, -0.0250): (0.2054863707859336, 0.2620188808834721),
    (49500, 0.0000): (0.1813451515151515, 0.2685151515151515),
    (49500, 0.0250): (0.1572039322443694, 0.2750114221468309),
    (49500, 0.0500): (0.1330627129735873, 0.2815076927785103)}


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
        unpack = lambda x: (x.Ti, x.ui, x.vi, x.di)
        np.testing.assert_almost_equal(
            [unpack(x) for x in planckian_table(np.array([0.1978, 0.3122]),
                                                cmfs,
                                                1000,
                                                1010,
                                                10)],
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

        cmfs = STANDARD_OBSERVERS_CMFS.get(
            'CIE 1931 2 Degree Standard Observer')
        self.assertEqual(
            planckian_table_minimal_distance_index(
                planckian_table(np.array([0.1978, 0.3122]),
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
            uv_to_CCT_Ohno2013(np.array([0.1978, 0.3122]), cmfs),
            np.array([6507.5470349001507, 0.0032236908012382953]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.4328, 0.2883]), cmfs),
            np.array([1041.8672179878763, -0.067377582642145384]),
            decimal=7)

        np.testing.assert_almost_equal(
            uv_to_CCT_Ohno2013(np.array([0.2927, 0.2722]), cmfs, iterations=4),
            np.array([2452.1932942782669, -0.084369982045528508]),
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
            np.array([0.19780034881616862, 0.31220050291046603]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_uv_Ohno2013(
                1041.849524611546, -0.067377582728534946, cmfs),
            np.array([0.43280250331413772, 0.28829975758516474]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_uv_Ohno2013(
                2448.9489053326438, -0.084324704634692743, cmfs),
            np.array([0.29256616302348853, 0.27221773141874955]),
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

        for key, value in TEMPERATURE_DUV_TO_UV.items():
            np.testing.assert_almost_equal(
                CCT_to_uv_Robertson1968(*key),
                value,
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
            xy_to_CCT_McCamy1992(np.array([0.31271, 0.32902])),
            6504.38938305,
            places=7)

        self.assertAlmostEqual(
            xy_to_CCT_McCamy1992(np.array([0.44757, 0.40745])),
            2857.28961266,
            places=7)

        self.assertAlmostEqual(
            xy_to_CCT_McCamy1992(
                np.array([0.25252093937408293, 0.252220883926284])),
            19501.6195313,
            places=7)

    def test_n_dimensional_xy_to_CCT_McCamy1992(self):
        """
        Tests :func:`colour.temperature.cct.xy_to_CCT_McCamy1992` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.31271, 0.32902])
        CCT = 6504.38938305
        np.testing.assert_almost_equal(
            xy_to_CCT_McCamy1992(xy),
            CCT,
            decimal=7)

        xy = np.tile(xy, (6, 1))
        CCT = np.tile(CCT, 6)
        np.testing.assert_almost_equal(
            xy_to_CCT_McCamy1992(xy),
            CCT,
            decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        CCT = np.reshape(CCT, (2, 3))
        np.testing.assert_almost_equal(
            xy_to_CCT_McCamy1992(xy),
            CCT,
            decimal=7)

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
            xy_to_CCT_Hernandez1999(np.array([0.31271, 0.32902])),
            6500.04215334,
            places=7)

        self.assertAlmostEqual(
            xy_to_CCT_Hernandez1999(np.array([0.44757, 0.40745])),
            2790.64222533,
            places=7)

        self.assertAlmostEqual(
            xy_to_CCT_Hernandez1999(
                np.array([0.24416224821391358, 0.24033367475831827])),
            64448.110925653324,
            places=7)

    def test_n_dimensional_xy_to_CCT_Hernandez1999(self):
        """
        Tests :func:`colour.temperature.cct.xy_to_CCT_Hernandez1999` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.31271, 0.32902])
        CCT = 6500.04215334
        np.testing.assert_almost_equal(
            xy_to_CCT_Hernandez1999(xy),
            CCT,
            decimal=7)

        xy = np.tile(xy, (6, 1))
        CCT = np.tile(CCT, 6)
        np.testing.assert_almost_equal(
            xy_to_CCT_Hernandez1999(xy),
            CCT,
            decimal=7)

        xy = np.reshape(xy, (2, 3, 2))
        CCT = np.reshape(CCT, (2, 3))
        np.testing.assert_almost_equal(
            xy_to_CCT_Hernandez1999(xy),
            CCT,
            decimal=7)

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
            np.array([0.38052828281249995, 0.3767335309611144]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_xy_Kang2002(7000),
            np.array([0.30637401953352766, 0.31655286972657715]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_xy_Kang2002(25000),
            np.array([0.2524729944384, 0.2522547912436536]),
            decimal=7)

    def test_n_dimensional_CCT_to_xy_Kang2002(self):
        """
        Tests :func:`colour.temperature.cct.CCT_to_xy_Kang2002` definition
        n-dimensional arrays support.
        """

        CCT = 4000
        xy = np.array([0.38052828281249995, 0.3767335309611144])
        np.testing.assert_almost_equal(
            CCT_to_xy_Kang2002(CCT),
            xy,
            decimal=7)

        CCT = np.tile(CCT, 6)
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(
            CCT_to_xy_Kang2002(CCT),
            xy,
            decimal=7)

        CCT = np.reshape(CCT, (2, 3))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(
            CCT_to_xy_Kang2002(CCT),
            xy,
            decimal=7)

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
            CCT_to_xy_CIE_D(4000),
            np.array([0.38234362499999996, 0.3837662610155782]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(7000),
            np.array([0.3053574314868805, 0.3216463454745523]),
            decimal=7)

        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(25000),
            np.array([0.2498536704, 0.25479946421094446]),
            decimal=7)

    def test_n_dimensional_CCT_to_xy_CIE_D(self):
        """
        Tests :func:`colour.temperature.cct.CCT_to_xy_CIE_D` definition
        n-dimensional arrays support.
        """

        CCT = 4000
        xy = np.array([0.38234362499999996, 0.3837662610155782])
        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(CCT),
            xy,
            decimal=7)

        CCT = np.tile(CCT, 6)
        xy = np.tile(xy, (6, 1))
        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(CCT),
            xy,
            decimal=7)

        CCT = np.reshape(CCT, (2, 3))
        xy = np.reshape(xy, (2, 3, 2))
        np.testing.assert_almost_equal(
            CCT_to_xy_CIE_D(CCT),
            xy,
            decimal=7)

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
