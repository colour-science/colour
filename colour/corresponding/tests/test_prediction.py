# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.corresponding.prediction` module.
"""

from __future__ import annotations

import numpy as np
import unittest

from colour.corresponding.prediction import (
    CorrespondingColourDataset,
    convert_experiment_results_Breneman1987,
)
from colour.corresponding import (
    corresponding_chromaticities_prediction_VonKries,
    corresponding_chromaticities_prediction_CIE1994,
    corresponding_chromaticities_prediction_CMCCAT2000,
    corresponding_chromaticities_prediction_Fairchild1990,
)
from colour.hints import NDArray

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'DATASET_CORRESPONDING_COLOUR_1',
    'DATA_PREDICTION_FAIRCHILD1990',
    'DATA_PREDICTION_CIE1994',
    'DATA_PREDICTION_CMCCAT2000',
    'DATA_PREDICTION_VONKRIES',
    'TestCorrespondingChromaticitiesPredictionFairchild1990',
    'TestCorrespondingChromaticitiesPredictionCIE1994',
    'TestCorrespondingChromaticitiesPredictionCMCCAT2000',
    'TestCorrespondingChromaticitiesPredictionVonKries',
]

DATASET_CORRESPONDING_COLOUR_1: CorrespondingColourDataset = (
    CorrespondingColourDataset(
        name=1,
        XYZ_r=np.array(
            [0.947368421052632, 1.000000000000000, 1.000000000000000]),
        XYZ_t=np.array(
            [1.107889733840304, 1.000000000000000, 0.334125475285171]),
        XYZ_cr=np.array([
            [372.358829568788565, 405.000000000000000, 345.746919917864489],
            [250.638506876227865, 135.000000000000000, 37.131630648330052],
            [456.541750503018136, 405.000000000000000, 267.487424547283638],
            [502.185218978102114, 405.000000000000000, 24.758211678831920],
            [164.036312849162016, 135.000000000000000, 24.511173184357535],
            [422.727888086642622, 405.000000000000000, 27.231498194945619],
            [110.245746691871460, 135.000000000000000, 53.846880907372366],
            [75.208733205374273, 135.000000000000000, 77.281669865642954],
            [258.414179104477626, 405.000000000000000, 479.480277185501166],
            [141.154411764705856, 135.000000000000000, 469.124999999999943],
            [380.757042253521092, 405.000000000000000, 700.193661971831261],
            [192.236301369863071, 135.000000000000000, 370.510273972602761],
        ]),
        XYZ_ct=np.array([
            [450.407919847328230, 405.000000000000000, 143.566316793893037],
            [267.090517241379359, 135.000000000000000, 11.831896551724059],
            [531.851235741444839, 405.000000000000000, 107.602186311787023],
            [603.033088235294031, 405.000000000000000, 7.444852941176350],
            [196.511090573012893, 135.000000000000000, 8.109981515711597],
            [526.868181818181711, 405.000000000000000, 8.468181818181574],
            [144.589483394833962, 135.000000000000000, 24.035977859778562],
            [108.161900369003689, 135.000000000000000, 36.178505535055272],
            [317.877906976744100, 405.000000000000000, 223.691860465116235],
            [126.960674157303373, 135.000000000000000, 192.792134831460686],
            [419.434826883910489, 405.000000000000000, 309.730142566191489],
            [185.180921052631589, 135.000000000000000, 151.430921052631589],
        ]),
        Y_r=np.array(1500),
        Y_t=np.array(1500),
        B_r=0.3,
        B_t=0.3,
        metadata={}))

DATA_PREDICTION_FAIRCHILD1990: NDArray = np.array([
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

DATA_PREDICTION_CIE1994: NDArray = np.array([
    [(0.199, 0.487), (0.261386136420622, 0.533662817418878)],
    [(0.420, 0.509), (0.451546322781523, 0.521730655299867)],
    [(0.249, 0.497), (0.310170032403198, 0.532080871352910)],
    [(0.302, 0.548), (0.362647256496429, 0.543095116682339)],
    [(0.290, 0.537), (0.342002205412566, 0.540654345195812)],
    [(0.257, 0.554), (0.320554395869614, 0.549100152511420)],
    [(0.192, 0.529), (0.249890733398992, 0.543032894366422)],
    [(0.129, 0.521), (0.184477356124214, 0.545089114658268)],
    [(0.133, 0.469), (0.179131489840910, 0.534176133801835)],
    [(0.158, 0.340), (0.173452103066259, 0.480817508459012)],
    [(0.178, 0.426), (0.226570643054285, 0.516988754071352)],
    [(0.231, 0.365), (0.272300292824155, 0.482318190580811)],
])

DATA_PREDICTION_CMCCAT2000: NDArray = np.array([
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

DATA_PREDICTION_VONKRIES: NDArray = np.array([
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


class TestConvertExperimentResultsBreneman1987(unittest.TestCase):
    """
    Defines :func:`colour.corresponding.prediction.\
convert_experiment_results_Breneman1987` definition unit tests
    methods.
    """

    def test_convert_experiment_results_Breneman1987(self):
        """
        Tests :func:`colour.corresponding.prediction.\
convert_experiment_results_Breneman1987` definition.
        """

        corresponding_colour_dataset = convert_experiment_results_Breneman1987(
            1)

        np.testing.assert_almost_equal(
            corresponding_colour_dataset.XYZ_r,
            DATASET_CORRESPONDING_COLOUR_1.XYZ_r,
            decimal=7)

        np.testing.assert_almost_equal(
            corresponding_colour_dataset.XYZ_t,
            DATASET_CORRESPONDING_COLOUR_1.XYZ_t,
            decimal=7)

        np.testing.assert_almost_equal(
            corresponding_colour_dataset.XYZ_cr,
            DATASET_CORRESPONDING_COLOUR_1.XYZ_cr,
            decimal=7)

        np.testing.assert_almost_equal(
            corresponding_colour_dataset.XYZ_ct,
            DATASET_CORRESPONDING_COLOUR_1.XYZ_ct,
            decimal=7)

        np.testing.assert_almost_equal(
            corresponding_colour_dataset.Y_r,
            DATASET_CORRESPONDING_COLOUR_1.Y_r,
            decimal=7)

        np.testing.assert_almost_equal(
            corresponding_colour_dataset.Y_t,
            DATASET_CORRESPONDING_COLOUR_1.Y_t,
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
                [(p.uv_m, p.uv_p) for p in
                 corresponding_chromaticities_prediction_Fairchild1990()]),
            DATA_PREDICTION_FAIRCHILD1990,
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
                [(p.uv_m, p.uv_p)
                 for p in corresponding_chromaticities_prediction_CIE1994()]),
            DATA_PREDICTION_CIE1994,
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
                (p.uv_m, p.uv_p)
                for p in corresponding_chromaticities_prediction_CMCCAT2000()
            ]),
            DATA_PREDICTION_CMCCAT2000,
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
                [(p.uv_m, p.uv_p)
                 for p in corresponding_chromaticities_prediction_VonKries()]),
            DATA_PREDICTION_VONKRIES,
            decimal=7)
