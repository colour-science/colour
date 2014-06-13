# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_transformations.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`color.spectrum.transformations` module.

**Others:**

"""

from __future__ import unicode_literals

import numpy
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import color.spectrum.spd
import color.spectrum.transformations

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["RELATIVE_SPD_DATA",
           "TestWavelength_to_XYZ",
           "TestSpectral_to_XYZ"]

RELATIVE_SPD_DATA = color.spectrum.spd.SpectralPowerDistribution("Custom", {340: 0.0000,
                                                                            345: 0.0000,
                                                                            350: 0.0000,
                                                                            355: 0.0000,
                                                                            360: 0.0000,
                                                                            365: 0.0000,
                                                                            370: 0.0000,
                                                                            375: 0.0000,
                                                                            380: 0.0000,
                                                                            385: 0.0000,
                                                                            390: 0.0000,
                                                                            395: 0.0000,
                                                                            400: 0.0641,
                                                                            405: 0.0650,
                                                                            410: 0.0654,
                                                                            415: 0.0652,
                                                                            420: 0.0645,
                                                                            425: 0.0629,
                                                                            430: 0.0605,
                                                                            435: 0.0581,
                                                                            440: 0.0562,
                                                                            445: 0.0551,
                                                                            450: 0.0543,
                                                                            455: 0.0539,
                                                                            460: 0.0537,
                                                                            465: 0.0538,
                                                                            470: 0.0541,
                                                                            475: 0.0547,
                                                                            480: 0.0559,
                                                                            485: 0.0578,
                                                                            490: 0.0603,
                                                                            495: 0.0629,
                                                                            500: 0.0651,
                                                                            505: 0.0667,
                                                                            510: 0.0680,
                                                                            515: 0.0691,
                                                                            520: 0.0705,
                                                                            525: 0.0720,
                                                                            530: 0.0736,
                                                                            535: 0.0753,
                                                                            540: 0.0772,
                                                                            545: 0.0791,
                                                                            550: 0.0809,
                                                                            555: 0.0833,
                                                                            560: 0.0870,
                                                                            565: 0.0924,
                                                                            570: 0.0990,
                                                                            575: 0.1061,
                                                                            580: 0.1128,
                                                                            585: 0.1190,
                                                                            590: 0.1251,
                                                                            595: 0.1308,
                                                                            600: 0.1360,
                                                                            605: 0.1403,
                                                                            610: 0.1439,
                                                                            615: 0.1473,
                                                                            620: 0.1511,
                                                                            625: 0.1550,
                                                                            630: 0.1590,
                                                                            635: 0.1634,
                                                                            640: 0.1688,
                                                                            645: 0.1753,
                                                                            650: 0.1828,
                                                                            655: 0.1909,
                                                                            660: 0.1996,
                                                                            665: 0.2088,
                                                                            670: 0.2187,
                                                                            675: 0.2291,
                                                                            680: 0.2397,
                                                                            685: 0.2505,
                                                                            690: 0.2618,
                                                                            695: 0.2733,
                                                                            700: 0.2852,
                                                                            705: 0.0000,
                                                                            710: 0.0000,
                                                                            715: 0.0000,
                                                                            720: 0.0000,
                                                                            725: 0.0000,
                                                                            730: 0.0000,
                                                                            735: 0.0000,
                                                                            740: 0.0000,
                                                                            745: 0.0000,
                                                                            750: 0.0000,
                                                                            755: 0.0000,
                                                                            760: 0.0000,
                                                                            765: 0.0000,
                                                                            770: 0.0000,
                                                                            775: 0.0000,
                                                                            780: 0.0000,
                                                                            785: 0.0000,
                                                                            790: 0.0000,
                                                                            795: 0.0000,
                                                                            800: 0.0000,
                                                                            805: 0.0000,
                                                                            810: 0.0000,
                                                                            815: 0.0000,
                                                                            820: 0.0000,
                                                                            825: 0.0000,
                                                                            830: 0.0000})

sRGB_LINEAR_COLORCHECKER_2005 = [[[0.4316, 0.3777, 0.1008],
                                  (0.11518474980142972, 0.1008, 0.050893725178713274),
                                  numpy.matrix([[0.45293517],
                                                [0.31732158],
                                                [0.26414773]])],
                                 [[0.4197, 0.3744, 0.34950000000000003],
                                  (0.39178725961538463, 0.34950000000000003, 0.19220633012820515),
                                  numpy.matrix([[0.77875824],
                                                [0.5772645],
                                                [0.50453169]])],
                                 [[0.276, 0.3016, 0.18359999999999999],
                                  (0.1680159151193634, 0.18359999999999999, 0.25713740053050399),
                                  numpy.matrix([[0.35505307],
                                                [0.47995567],
                                                [0.61088035]])],
                                 [[0.3703, 0.4499, 0.1325],
                                  (0.10905701266948212, 0.13250000000000001, 0.052952878417426061),
                                  numpy.matrix([[0.35179242],
                                                [0.42214077],
                                                [0.25258942]])],
                                 [[0.2999, 0.2856, 0.2304],
                                  (0.24193613445378148, 0.23039999999999999, 0.33438655462184863),
                                  numpy.matrix([[0.50809894],
                                                [0.50196494],
                                                [0.69048098]])],
                                 [[0.2848, 0.3911, 0.4178],
                                  (0.30424300690360523, 0.4178, 0.34622597801073896),
                                  numpy.matrix([[0.36240083],
                                                [0.74473539],
                                                [0.67467032]])],
                                 [[0.5295, 0.4055, 0.3118],
                                  (0.40714697903822439, 0.31180000000000002, 0.049980271270036999),
                                  numpy.matrix([[0.87944466],
                                                [0.48522956],
                                                [0.18327685]])],
                                 [[0.2305, 0.2106, 0.11259999999999999],
                                  (0.1232397910731244, 0.11259999999999999, 0.29882307692307686),
                                  numpy.matrix([[0.26605806],
                                                [0.35770363],
                                                [0.66743852]])],
                                 [[0.5012, 0.3273, 0.1938],
                                  (0.29676920256645278, 0.1938, 0.10154812098991754),
                                  numpy.matrix([[0.77782346],
                                                [0.32138719],
                                                [0.38060248]])],
                                 [[0.3319, 0.2482, 0.0637],
                                  (0.085181426269137786, 0.063700000000000007, 0.1077664383561644),
                                  numpy.matrix([[0.36729282],
                                                [0.22739265],
                                                [0.41412433]])],
                                 [[0.3984, 0.5008, 0.4446],
                                  (0.35369137380191684, 0.4446, 0.089488178913738003),
                                  numpy.matrix([[0.62266646],
                                                [0.7410742],
                                                [0.24626906]])],
                                 [[0.4957, 0.4427, 0.4357],
                                  (0.48786196069573068, 0.43569999999999998, 0.060625976959566286),
                                  numpy.matrix([[0.90369041],
                                                [0.63376348],
                                                [0.15395733]])],
                                 [[0.2018, 0.1692, 0.0575],
                                  (0.068578605200945636, 0.057500000000000002, 0.21375591016548467),
                                  numpy.matrix([[0.1384956],
                                                [0.24831912],
                                                [0.57681467]])],
                                 [[0.3253, 0.5032, 0.2318],
                                  (0.14985003974562797, 0.23180000000000001, 0.079001788553259192),
                                  numpy.matrix([[0.26252953],
                                                [0.58394952],
                                                [0.29070622]])],
                                 [[0.5686, 0.3303, 0.1257],
                                  (0.21638819255222524, 0.12570000000000001, 0.038474931880109003),
                                  numpy.matrix([[0.70564037],
                                                [0.19094729],
                                                [0.22335249]])],
                                 [[0.4697, 0.4734, 0.5981000000000001],
                                  (0.59342536966624426, 0.59810000000000008, 0.071888234051542058),
                                  numpy.matrix([[0.93451045],
                                                [0.77825294],
                                                [0.07655428]])],
                                 [[0.4159, 0.2688, 0.2009],
                                  (0.3108419270833333, 0.2009, 0.2356539062500001),
                                  numpy.matrix([[0.75715761],
                                                [0.32930283],
                                                [0.59045447]])],
                                 [[0.2131, 0.3023, 0.193],
                                  (0.13605127356930202, 0.193, 0.30938736354614615),
                                  numpy.matrix([[-0.48463915],
                                                [0.53412743],
                                                [0.66546058]])],
                                 [[0.3469, 0.3608, 0.9131],
                                  (0.87792236696230597, 0.91310000000000002, 0.73974259977827039),
                                  numpy.matrix([[0.96027764],
                                                [0.96170536],
                                                [0.95169688]])],
                                 [[0.344, 0.3584, 0.5893999999999999],
                                  (0.56571874999999983, 0.58939999999999992, 0.48941249999999997),
                                  numpy.matrix([[0.78565259],
                                                [0.79300245],
                                                [0.79387336]])],
                                 [[0.3432, 0.3581, 0.3632],
                                  (0.34808779670483109, 0.36320000000000002, 0.30295403518570241),
                                  numpy.matrix([[0.63023284],
                                                [0.63852418],
                                                [0.64028572]])],
                                 [[0.3446, 0.3579, 0.19149999999999998],
                                  (0.18438362671137187, 0.19149999999999998, 0.15918203408773396),
                                  numpy.matrix([[0.4732449],
                                                [0.47519512],
                                                [0.47670436]])],
                                 [[0.3401, 0.3548, 0.0883],
                                  (0.084641572717023675, 0.088300000000000003, 0.075931031567080032),
                                  numpy.matrix([[0.32315746],
                                                [0.32983556],
                                                [0.33640183]])],
                                 [[0.3406, 0.3537, 0.0311],
                                  (0.029948148148148147, 0.031099999999999999, 0.026879474130619162),
                                  numpy.matrix([[0.19104038],
                                                [0.19371002],
                                                [0.19903915]])]]


class TestWavelength_to_XYZ(unittest.TestCase):
    """
    Defines :func:`color.spectrum.transformations.wavelength_to_XYZ` definition units tests methods.
    """

    def test_wavelength_to_XYZ(self):
        """
        Tests :func:`color.spectrum.transformations.wavelength_to_XYZ` definition.
        """

        numpy.testing.assert_almost_equal(
            color.spectrum.transformations.wavelength_to_XYZ(480,
                                                             color.spectrum.STANDARD_OBSERVERS_CMFS.get(
                                                                 "CIE 1931 2 Degree Standard Observer")),
            numpy.matrix([0.09564, 0.13902, 0.81295]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.spectrum.transformations.wavelength_to_XYZ(480,
                                                             color.spectrum.STANDARD_OBSERVERS_CMFS.get(
                                                                 "CIE 2006 2 Degree Standard Observer")),
            numpy.matrix([0.08182895, 0.1788048, 0.7552379]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.spectrum.transformations.wavelength_to_XYZ(641.5,
                                                             color.spectrum.STANDARD_OBSERVERS_CMFS.get(
                                                                 "CIE 2006 2 Degree Standard Observer")),
            numpy.matrix([0.44575583, 0.18184213, 0.]).reshape((3, 1)),
            decimal=7)


class TestSpectral_to_XYZ(unittest.TestCase):
    """
    Defines :func:`color.spectrum.transformations.spectral_to_XYZ` definition units tests methods.
    """

    def test_spectral_to_XYZ(self):
        """
        Tests :func:`color.spectrum.transformations.spectral_to_XYZ` definition.
        """

        cmfs = color.spectrum.STANDARD_OBSERVERS_CMFS.get("CIE 1931 2 Degree Standard Observer")
        numpy.testing.assert_almost_equal(
            color.spectrum.transformations.spectral_to_XYZ(RELATIVE_SPD_DATA.zeros(*cmfs.shape),
                                                           cmfs,
                                                           color.spectrum.ILLUMINANTS_RELATIVE_SPDS.get(
                                                               "A").clone().zeros(*cmfs.shape)),
            numpy.matrix([14.46371626, 10.85832347, 2.04664796]).reshape((3, 1)),
            decimal=7)

        cmfs = color.spectrum.STANDARD_OBSERVERS_CMFS.get("CIE 1964 10 Degree Standard Observer")
        numpy.testing.assert_almost_equal(
            color.spectrum.transformations.spectral_to_XYZ(RELATIVE_SPD_DATA.zeros(*cmfs.shape),
                                                           cmfs,
                                                           color.spectrum.ILLUMINANTS_RELATIVE_SPDS.get(
                                                               "C").clone().zeros(*cmfs.shape)),
            numpy.matrix([10.7704252, 9.44870313, 6.62742289]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.spectrum.transformations.spectral_to_XYZ(RELATIVE_SPD_DATA.zeros(*cmfs.shape),
                                                           cmfs,
                                                           color.spectrum.ILLUMINANTS_RELATIVE_SPDS.get(
                                                               "F2").clone().zeros(*cmfs.shape)),
            numpy.matrix([11.57830745, 9.98744967, 3.95396539]).reshape((3, 1)),
            decimal=7)


if __name__ == "__main__":
    unittest.main()
