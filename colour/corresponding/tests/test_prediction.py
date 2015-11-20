# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.corresponding.prediction` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest

from colour.corresponding.prediction import (
    corresponding_chromaticities_prediction_VonKries,
    corresponding_chromaticities_prediction_CIE1994,
    corresponding_chromaticities_prediction_CMCCAT2000,
    corresponding_chromaticities_prediction_Fairchild1990)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['VONKRIES_PREDICTION_DATA',
           'CIE1994_PREDICTION_DATA',
           'CMCCAT2000_PREDICTION_DATA',
           'FAIRCHILD1990_PREDICTION_DATA',
           'TestCorrespondingChromaticitiesPredictionVonKries',
           'TestCorrespondingChromaticitiesPredictionCIE1994',
           'TestCorrespondingChromaticitiesPredictionCMCCAT2000',
           'TestCorrespondingChromaticitiesPredictionFairchild1990']

VONKRIES_PREDICTION_DATA = (
    ((0.199, 0.487), (0.19999423529586305, 0.47059613254211008)),
    ((0.420, 0.509), (0.41491385566838473, 0.50376620468564648)),
    ((0.249, 0.497), (0.24420233277981676, 0.48315486115101869)),
    ((0.302, 0.548), (0.30728774349955523, 0.5431744633939557)),
    ((0.290, 0.537), (0.29412976520244888, 0.53162770735036535)),
    ((0.257, 0.554), (0.26139917197581486, 0.54947653225319748)),
    ((0.192, 0.529), (0.19911324843871134, 0.51276966776408339)),
    ((0.129, 0.521), (0.14226621770541509, 0.49981254299758376)),
    ((0.133, 0.469), (0.13813459337807268, 0.44376807955209857)),
    ((0.158, 0.340), (0.15418827142190009, 0.33832267888004569)),
    ((0.178, 0.426), (0.17529792410406500, 0.40434393555126880)),
    ((0.231, 0.365), (0.21300472149984359, 0.35459526269438407)))

CIE1994_PREDICTION_DATA = (
    ((0.199, 0.487), (0.21576789078095915, 0.49889536782628013)),
    ((0.420, 0.509), (0.41026774846950392, 0.51382153142525100)),
    ((0.249, 0.497), (0.26057149736232083, 0.50514816868743939)),
    ((0.302, 0.548), (0.31180162297157377, 0.54406424977182377)),
    ((0.290, 0.537), (0.30161466896702105, 0.53537570660303413)),
    ((0.257, 0.554), (0.26968427399737788, 0.54960443119933100)),
    ((0.192, 0.529), (0.21264318516664438, 0.52563542480925651)),
    ((0.129, 0.521), (0.15235652639902811, 0.51978549083298708)),
    ((0.133, 0.469), (0.14477688728365004, 0.48429776036021355)),
    ((0.158, 0.340), (0.15808013304408230, 0.408581420625705350)),
    ((0.178, 0.426), (0.18926085249284605, 0.45675663679141737)),
    ((0.231, 0.365), (0.23947321852501582, 0.42049748344380927)))

CMCCAT2000_PREDICTION_DATA = (
    ((0.199, 0.487), (0.20001127363345125, 0.47053923035439771)),
    ((0.420, 0.509), (0.40763714782394150, 0.50467220647243904)),
    ((0.249, 0.497), (0.24330393312208579, 0.48344348308897162)),
    ((0.302, 0.548), (0.30273887270442312, 0.54277139137206465)),
    ((0.290, 0.537), (0.29051724229258280, 0.53154763369771674)),
    ((0.257, 0.554), (0.25867471956856769, 0.54876423978816113)),
    ((0.192, 0.529), (0.19899305339457718, 0.51276519453986324)),
    ((0.129, 0.521), (0.14332417618528126, 0.49967334301878746)),
    ((0.133, 0.469), (0.13842458403118293, 0.44267868804709914)),
    ((0.158, 0.340), (0.15335302779251891, 0.33012125331569470)),
    ((0.178, 0.426), (0.17538205783877137, 0.40172588918242474)),
    ((0.231, 0.365), (0.21377091995977762, 0.34868595411619269)))

FAIRCHILD1990_PREDICTION_DATA = (
    (((0.199, 0.487), (0.20055493444868086, 0.47015569961951620)),
     ((0.420, 0.509), (0.38921444989602733, 0.51400288137926686)),
     ((0.249, 0.497), (0.24157002919664858, 0.48603385038823715)),
     ((0.302, 0.548), (0.29061922859039019, 0.55028259309248062)),
     ((0.290, 0.537), (0.28061133916001751, 0.53834840347704793)),
     ((0.257, 0.554), (0.24837987154120195, 0.55574167922033590)),
     ((0.192, 0.529), (0.19440436120251137, 0.51586907768408741)),
     ((0.129, 0.521), (0.13919738297504267, 0.49957842695662330)),
     ((0.133, 0.469), (0.14094834961519132, 0.43603879023209541)),
     ((0.158, 0.340), (0.17024015473705922, 0.31112125051836670)),
     ((0.178, 0.426), (0.18410450515622714, 0.39314310122137930)),
     ((0.231, 0.365), (0.22925269280444158, 0.33803399579545795))))


class TestCorrespondingChromaticitiesPredictionVonKries(unittest.TestCase):
    """
    Defines :func:`colour.corresponding.prediction.\
corresponding_chromaticities_prediction_VonKries` definition unit tests
    methods.
    """

    def test_corresponding_chromaticities_prediction_VonKries(self):
        """
        Tests :func:`colour.corresponding.prediction.\
corresponding_chromaticities_prediction_VonKries` definition.
        """

        np.testing.assert_almost_equal(
            np.array([(p.uvp_m, p.uvp_p)
                      for p in
                      corresponding_chromaticities_prediction_VonKries()]),
            VONKRIES_PREDICTION_DATA,
            decimal=7)


class TestCorrespondingChromaticitiesPredictionCIE1994(unittest.TestCase):
    """
    Defines :func:`colour.corresponding.prediction.\
corresponding_chromaticities_prediction_CIE1994` definition unit tests methods.
    """

    def test_corresponding_chromaticities_prediction_CIE1994(self):
        """
        Tests :func:`colour.corresponding.prediction.\
corresponding_chromaticities_prediction_CIE1994` definition.
        """

        np.testing.assert_almost_equal(
            np.array([(p.uvp_m, p.uvp_p)
                      for p in
                      corresponding_chromaticities_prediction_CIE1994()]),
            CIE1994_PREDICTION_DATA,
            decimal=7)


class TestCorrespondingChromaticitiesPredictionCMCCAT2000(unittest.TestCase):
    """
    Defines :func:`colour.corresponding.prediction.\
corresponding_chromaticities_prediction_CMCCAT2000` definition unit tests
    methods.
    """

    def test_corresponding_chromaticities_prediction_CMCCAT2000(self):
        """
        Tests :func:`colour.corresponding.prediction.\
    corresponding_chromaticities_prediction_CMCCAT2000` definition.
        """

        np.testing.assert_almost_equal(
            np.array([(p.uvp_m, p.uvp_p)
                      for p in
                      corresponding_chromaticities_prediction_CMCCAT2000()]),
            CMCCAT2000_PREDICTION_DATA,
            decimal=7)


class TestCorrespondingChromaticitiesPredictionFairchild1990(
    unittest.TestCase):  # noqa
    """
    Defines :func:`colour.corresponding.prediction.\
corresponding_chromaticities_prediction_Fairchild1990` definition unit tests
    methods.
    """

    def test_corresponding_chromaticities_prediction_Fairchild1990(self):
        """
        Tests :func:`colour.corresponding.prediction.\
corresponding_chromaticities_prediction_Fairchild1990` definition.
        """

        np.testing.assert_almost_equal(
            np.array(
                [(p.uvp_m, p.uvp_p)
                 for p in
                 corresponding_chromaticities_prediction_Fairchild1990()]),
            FAIRCHILD1990_PREDICTION_DATA,
            decimal=7)
