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
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'FAIRCHILD1990_PREDICTION_DATA', 'CIE1994_PREDICTION_DATA',
    'CMCCAT2000_PREDICTION_DATA', 'VONKRIES_PREDICTION_DATA',
    'TestCorrespondingChromaticitiesPredictionFairchild1990',
    'TestCorrespondingChromaticitiesPredictionCIE1994',
    'TestCorrespondingChromaticitiesPredictionCMCCAT2000',
    'TestCorrespondingChromaticitiesPredictionVonKries'
]

FAIRCHILD1990_PREDICTION_DATA = np.array([
    [(0.199, 0.487), (0.200554934448681, 0.470155699619516)],
    [(0.420, 0.509), (0.389214449896027, 0.514002881379267)],
    [(0.249, 0.497), (0.241570029196649, 0.486033850388237)],
    [(0.302, 0.548), (0.290619228590390, 0.550282593092481)],
    [(0.290, 0.537), (0.280611339160018, 0.538348403477048)],
    [(0.257, 0.554), (0.248379871541202, 0.555741679220336)],
    [(0.192, 0.529), (0.194404361202511, 0.515869077684087)],
    [(0.129, 0.521), (0.139197382975043, 0.499578426956623)],
    [(0.133, 0.469), (0.140948349615191, 0.436038790232095)],
    [(0.158, 0.340), (0.170240154737059, 0.311121250518367)],
    [(0.178, 0.426), (0.184104505156227, 0.393143101221379)],
    [(0.231, 0.365), (0.229252692804442, 0.338033995795458)],
])

CIE1994_PREDICTION_DATA = np.array([
    [(0.199, 0.487), (0.215767890780959, 0.498895367826280)],
    [(0.420, 0.509), (0.410267748469504, 0.513821531425251)],
    [(0.249, 0.497), (0.260571497362321, 0.505148168687439)],
    [(0.302, 0.548), (0.311801622971574, 0.544064249771824)],
    [(0.290, 0.537), (0.301614668967021, 0.535375706603034)],
    [(0.257, 0.554), (0.269684273997378, 0.549604431199331)],
    [(0.192, 0.529), (0.212643185166644, 0.525635424809257)],
    [(0.129, 0.521), (0.152356526399028, 0.519785490832987)],
    [(0.133, 0.469), (0.144776887283650, 0.484297760360214)],
    [(0.158, 0.340), (0.158080133044082, 0.408581420625705)],
    [(0.178, 0.426), (0.189260852492846, 0.456756636791417)],
    [(0.231, 0.365), (0.239473218525016, 0.420497483443809)],
])

CMCCAT2000_PREDICTION_DATA = np.array([
    [(0.199, 0.487), (0.200011273633451, 0.470539230354398)],
    [(0.420, 0.509), (0.407637147823942, 0.504672206472439)],
    [(0.249, 0.497), (0.243303933122086, 0.483443483088972)],
    [(0.302, 0.548), (0.302738872704423, 0.542771391372065)],
    [(0.290, 0.537), (0.290517242292583, 0.531547633697717)],
    [(0.257, 0.554), (0.258674719568568, 0.548764239788161)],
    [(0.192, 0.529), (0.198993053394577, 0.512765194539863)],
    [(0.129, 0.521), (0.143324176185281, 0.499673343018787)],
    [(0.133, 0.469), (0.138424584031183, 0.442678688047099)],
    [(0.158, 0.340), (0.153353027792519, 0.330121253315695)],
    [(0.178, 0.426), (0.175382057838771, 0.401725889182425)],
    [(0.231, 0.365), (0.213770919959778, 0.348685954116193)],
])

VONKRIES_PREDICTION_DATA = np.array([
    [(0.199, 0.487), (0.199994235295863, 0.470596132542110)],
    [(0.420, 0.509), (0.414913855668385, 0.503766204685646)],
    [(0.249, 0.497), (0.244202332779817, 0.483154861151019)],
    [(0.302, 0.548), (0.307287743499555, 0.543174463393956)],
    [(0.290, 0.537), (0.294129765202449, 0.531627707350365)],
    [(0.257, 0.554), (0.261399171975815, 0.549476532253197)],
    [(0.192, 0.529), (0.199113248438711, 0.512769667764083)],
    [(0.129, 0.521), (0.142266217705415, 0.499812542997584)],
    [(0.133, 0.469), (0.138134593378073, 0.443768079552099)],
    [(0.158, 0.340), (0.154188271421900, 0.338322678880046)],
    [(0.178, 0.426), (0.175297924104065, 0.404343935551269)],
    [(0.231, 0.365), (0.213004721499844, 0.354595262694384)],
])


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
                [(p.uvp_m, p.uvp_p) for p in
                 corresponding_chromaticities_prediction_Fairchild1990()]),
            FAIRCHILD1990_PREDICTION_DATA,
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
            np.array(
                [(p.uvp_m, p.uvp_p)
                 for p in corresponding_chromaticities_prediction_CIE1994()]),
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
            np.array([
                (p.uvp_m, p.uvp_p)
                for p in corresponding_chromaticities_prediction_CMCCAT2000()
            ]),
            CMCCAT2000_PREDICTION_DATA,
            decimal=7)


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
            np.array(
                [(p.uvp_m, p.uvp_p)
                 for p in corresponding_chromaticities_prediction_VonKries()]),
            VONKRIES_PREDICTION_DATA,
            decimal=7)
