# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**tests_transformations.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines units tests for :mod:`color.transformations` module.

**Others:**

"""

from __future__ import unicode_literals

import numpy
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

import color.transformations

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["sRGB_LINEAR_COLORCHECKER_2005",
           "sRGB_TRANSFER_FUNCTION",
           "sRGB_INVERSE_TRANSFER_FUNCTION",
           "ACES_COLORCHECKER_2005",
           "TestXYZ_to_xyY",
           "TestxyY_to_XYZ",
           "Testxy_to_XYZ",
           "TestXYZ_to_xy",
           "TestXYZ_to_RGB",
           "TestRGB_to_XYZ",
           "TestxyY_to_RGB",
           "TestRGB_to_xyY",
           "TestXYZ_to_UCS",
           "TestUCS_to_XYZ",
           "TestUCS_to_uv",
           "TestUCS_uv_to_xy",
           "TestXYZ_to_UVW",
           "TestXYZ_to_Luv",
           "TestLuv_to_XYZ",
           "TestLuv_to_uv",
           "TestLuv_to_LCHuv",
           "TestLCHuv_to_Luv",
           "TestXYZ_to_Lab",
           "TestLab_to_XYZ",
           "TestLab_to_LCHab",
           "TestLCHab_to_Lab"]

sRGB_LINEAR_COLORCHECKER_2005 = [
    [[0.4316, 0.3777, 0.1008], (0.11518474980142972, 0.1008, 0.050893725178713274),
     numpy.matrix([[0.45293517],
                   [0.31732158],
                   [0.26414773]])],
    [[0.4197, 0.3744, 0.34950000000000003], (0.39178725961538463, 0.34950000000000003, 0.19220633012820515),
     numpy.matrix([[0.77875824],
                   [0.5772645],
                   [0.50453169]])],
    [[0.276, 0.3016, 0.18359999999999999], (0.1680159151193634, 0.18359999999999999, 0.25713740053050399),
     numpy.matrix([[0.35505307],
                   [0.47995567],
                   [0.61088035]])],
    [[0.3703, 0.4499, 0.1325], (0.10905701266948212, 0.13250000000000001, 0.052952878417426061),
     numpy.matrix([[0.35179242],
                   [0.42214077],
                   [0.25258942]])],
    [[0.2999, 0.2856, 0.2304], (0.24193613445378148, 0.23039999999999999, 0.33438655462184863),
     numpy.matrix([[0.50809894],
                   [0.50196494],
                   [0.69048098]])],
    [[0.2848, 0.3911, 0.4178], (0.30424300690360523, 0.4178, 0.34622597801073896),
     numpy.matrix([[0.36240083],
                   [0.74473539],
                   [0.67467032]])],
    [[0.5295, 0.4055, 0.3118], (0.40714697903822439, 0.31180000000000002, 0.049980271270036999),
     numpy.matrix([[0.87944466],
                   [0.48522956],
                   [0.18327685]])],
    [[0.2305, 0.2106, 0.11259999999999999], (0.1232397910731244, 0.11259999999999999, 0.29882307692307686),
     numpy.matrix([[0.26605806],
                   [0.35770363],
                   [0.66743852]])],
    [[0.5012, 0.3273, 0.1938], (0.29676920256645278, 0.1938, 0.10154812098991754),
     numpy.matrix([[0.77782346],
                   [0.32138719],
                   [0.38060248]])],
    [[0.3319, 0.2482, 0.0637], (0.085181426269137786, 0.063700000000000007, 0.1077664383561644),
     numpy.matrix([[0.36729282],
                   [0.22739265],
                   [0.41412433]])],
    [[0.3984, 0.5008, 0.4446], (0.35369137380191684, 0.4446, 0.089488178913738003),
     numpy.matrix([[0.62266646],
                   [0.7410742],
                   [0.24626906]])],
    [[0.4957, 0.4427, 0.4357], (0.48786196069573068, 0.43569999999999998, 0.060625976959566286),
     numpy.matrix([[0.90369041],
                   [0.63376348],
                   [0.15395733]])],
    [[0.2018, 0.1692, 0.0575], (0.068578605200945636, 0.057500000000000002, 0.21375591016548467),
     numpy.matrix([[0.1384956],
                   [0.24831912],
                   [0.57681467]])],
    [[0.3253, 0.5032, 0.2318], (0.14985003974562797, 0.23180000000000001, 0.079001788553259192),
     numpy.matrix([[0.26252953],
                   [0.58394952],
                   [0.29070622]])],
    [[0.5686, 0.3303, 0.1257], (0.21638819255222524, 0.12570000000000001, 0.038474931880109003),
     numpy.matrix([[0.70564037],
                   [0.19094729],
                   [0.22335249]])],
    [[0.4697, 0.4734, 0.5981000000000001], (0.59342536966624426, 0.59810000000000008, 0.071888234051542058),
     numpy.matrix([[0.93451045],
                   [0.77825294],
                   [0.07655428]])],
    [[0.4159, 0.2688, 0.2009], (0.3108419270833333, 0.2009, 0.2356539062500001),
     numpy.matrix([[0.75715761],
                   [0.32930283],
                   [0.59045447]])],
    [[0.2131, 0.3023, 0.193], (0.13605127356930202, 0.193, 0.30938736354614615),
     numpy.matrix([[-0.48463915],
                   [0.53412743],
                   [0.66546058]])],
    [[0.3469, 0.3608, 0.9131], (0.87792236696230597, 0.91310000000000002, 0.73974259977827039),
     numpy.matrix([[0.96027764],
                   [0.96170536],
                   [0.95169688]])],
    [[0.344, 0.3584, 0.5893999999999999], (0.56571874999999983, 0.58939999999999992, 0.48941249999999997),
     numpy.matrix([[0.78565259],
                   [0.79300245],
                   [0.79387336]])],
    [[0.3432, 0.3581, 0.3632], (0.34808779670483109, 0.36320000000000002, 0.30295403518570241),
     numpy.matrix([[0.63023284],
                   [0.63852418],
                   [0.64028572]])],
    [[0.3446, 0.3579, 0.19149999999999998], (0.18438362671137187, 0.19149999999999998, 0.15918203408773396),
     numpy.matrix([[0.4732449],
                   [0.47519512],
                   [0.47670436]])],
    [[0.3401, 0.3548, 0.0883], (0.084641572717023675, 0.088300000000000003, 0.075931031567080032),
     numpy.matrix([[0.32315746],
                   [0.32983556],
                   [0.33640183]])],
    [[0.3406, 0.3537, 0.0311], (0.029948148148148147, 0.031099999999999999, 0.026879474130619162),
     numpy.matrix([[0.19104038],
                   [0.19371002],
                   [0.19903915]])]]

sRGB_TRANSFER_FUNCTION = lambda x: x * 12.92 if x <= 0.0031308 else 1.055 * (x ** (1 / 2.4)) - 0.055

sRGB_INVERSE_TRANSFER_FUNCTION = lambda x: x / 12.92 if x <= 0.0031308 else ((x + 0.055) / 1.055) ** 2.4

ACES_COLORCHECKER_2005 = [[[0.4316, 0.3777, 0.1008],
                           (0.11518474980142972, 0.1008, 0.050893725178713274),
                           numpy.matrix([[0.11758989],
                                         [0.08781098],
                                         [0.06184838]])],
                          [[0.4197, 0.3744, 0.34950000000000003],
                           (0.39178725961538463, 0.34950000000000003, 0.19220633012820515),
                           numpy.matrix([[0.40073605],
                                         [0.31020146],
                                         [0.2334411]])],
                          [[0.276, 0.3016, 0.18359999999999999],
                           (0.1680159151193634, 0.18359999999999999, 0.25713740053050399),
                           numpy.matrix([[0.17949613],
                                         [0.20101795],
                                         [0.31109218]])],
                          [[0.3703, 0.4499, 0.1325],
                           (0.10905701266948212, 0.13250000000000001, 0.052952878417426061),
                           numpy.matrix([[0.1107181],
                                         [0.13503098],
                                         [0.06442476]])],
                          [[0.2999, 0.2856, 0.2304],
                           (0.24193613445378148, 0.23039999999999999, 0.33438655462184863),
                           numpy.matrix([[0.2575148],
                                         [0.23804357],
                                         [0.40454743]])],
                          [[0.2848, 0.3911, 0.4178],
                           (0.30424300690360523, 0.4178, 0.34622597801073896),
                           numpy.matrix([[0.31733562],
                                         [0.46758348],
                                         [0.41947022]])],
                          [[0.5295, 0.4055, 0.3118],
                           (0.40714697903822439, 0.31180000000000002, 0.049980271270036999),
                           numpy.matrix([[0.41040872],
                                         [0.23293505],
                                         [0.06167114]])],
                          [[0.2305, 0.2106, 0.11259999999999999],
                           (0.1232397910731244, 0.11259999999999999, 0.29882307692307686),
                           numpy.matrix([[0.13747056],
                                         [0.13033376],
                                         [0.36114764]])],
                          [[0.5012, 0.3273, 0.1938],
                           (0.29676920256645278, 0.1938, 0.10154812098991754),
                           numpy.matrix([[0.30304559],
                                         [0.13139056],
                                         [0.12344791]])],
                          [[0.3319, 0.2482, 0.0637],
                           (0.085181426269137786, 0.063700000000000007, 0.1077664383561644),
                           numpy.matrix([[0.09058405],
                                         [0.05847923],
                                         [0.13035265]])],
                          [[0.3984, 0.5008, 0.4446],
                           (0.35369137380191684, 0.4446, 0.089488178913738003),
                           numpy.matrix([[0.3547791],
                                         [0.44849679],
                                         [0.10971221]])],
                          [[0.4957, 0.4427, 0.4357],
                           (0.48786196069573068, 0.43569999999999998, 0.060625976959566286),
                           numpy.matrix([[0.49038927],
                                         [0.36515801],
                                         [0.07497681]])],
                          [[0.2018, 0.1692, 0.0575],
                           (0.068578605200945636, 0.057500000000000002, 0.21375591016548467),
                           numpy.matrix([[0.07890084],
                                         [0.07117527],
                                         [0.25824906]])],
                          [[0.3253, 0.5032, 0.2318],
                           (0.14985003974562797, 0.23180000000000001, 0.079001788553259192),
                           numpy.matrix([[0.15129818],
                                         [0.25515937],
                                         [0.09620886]])],
                          [[0.5686, 0.3303, 0.1257],
                           (0.21638819255222524, 0.12570000000000001, 0.038474931880109003),
                           numpy.matrix([[0.21960818],
                                         [0.06985597],
                                         [0.04703204]])],
                          [[0.4697, 0.4734, 0.5981000000000001],
                           (0.59342536966624426, 0.59810000000000008, 0.071888234051542058),
                           numpy.matrix([[0.5948559],
                                         [0.5382559],
                                         [0.08916818]])],
                          [[0.4159, 0.2688, 0.2009],
                           (0.3108419270833333, 0.2009, 0.2356539062500001),
                           numpy.matrix([[0.32368864],
                                         [0.15049668],
                                         [0.28535138]])],
                          [[0.2131, 0.3023, 0.193],
                           (0.13605127356930202, 0.193, 0.30938736354614615),
                           numpy.matrix([[0.14920707],
                                         [0.23648468],
                                         [0.37415686]])],
                          [[0.3469, 0.3608, 0.9131],
                           (0.87792236696230597, 0.91310000000000002, 0.73974259977827039),
                           numpy.matrix([[0.90989008],
                                         [0.91268206],
                                         [0.89651699]])],
                          [[0.344, 0.3584, 0.5893999999999999],
                           (0.56571874999999983, 0.58939999999999992, 0.48941249999999997),
                           numpy.matrix([[0.58690823],
                                         [0.59107342],
                                         [0.59307473]])],
                          [[0.3432, 0.3581, 0.3632],
                           (0.34808779670483109, 0.36320000000000002, 0.30295403518570241),
                           numpy.matrix([[0.36120089],
                                         [0.36465935],
                                         [0.36711553]])],
                          [[0.3446, 0.3579, 0.19149999999999998],
                           (0.18438362671137187, 0.19149999999999998, 0.15918203408773396),
                           numpy.matrix([[0.19128766],
                                         [0.19177359],
                                         [0.19289805]])],
                          [[0.3401, 0.3548, 0.0883],
                           (0.084641572717023675, 0.088300000000000003, 0.075931031567080032),
                           numpy.matrix([[0.08793956],
                                         [0.08892476],
                                         [0.09200134]])],
                          [[0.3406, 0.3537, 0.0311],
                           (0.029948148148148147, 0.031099999999999999, 0.026879474130619162),
                           numpy.matrix([[0.03111895],
                                         [0.03126787],
                                         [0.03256784]])]]


class TestXYZ_to_xyY(unittest.TestCase):
    """
    Defines :func:`color.transformations.XYZ_to_xyY` definition units tests methods.
    """

    def test_XYZ_to_xyY(self):
        """
        Tests :func:`color.transformations.XYZ_to_xyY` definition.
        """

        numpy.testing.assert_almost_equal(
            color.transformations.XYZ_to_xyY(numpy.matrix([11.80583421, 10.34, 5.15089229]).reshape((3, 1))),
            numpy.matrix([0.4325, 0.3788, 10.34]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.XYZ_to_xyY(numpy.matrix([3.08690042, 3.2, 2.68925666]).reshape((3, 1))),
            numpy.matrix([0.3439, 0.3565, 3.20]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.XYZ_to_xyY(numpy.matrix([0, 0, 0]).reshape((3, 1)), (0.34567, 0.35850)),
            numpy.matrix([0.34567, 0.35850, 0]).reshape((3, 1)),
            decimal=7)


class TestxyY_to_XYZ(unittest.TestCase):
    """
    Defines :func:`color.transformations.xyY_to_XYZ` definition units tests methods.
    """

    def test_xyY_to_XYZ(self):
        """
        Tests :func:`color.transformations.xyY_to_XYZ` definition.
        """

        numpy.testing.assert_almost_equal(
            color.transformations.xyY_to_XYZ(numpy.matrix([0.4325, 0.3788, 10.34]).reshape((3, 1))),
            numpy.matrix([11.80583421,
                          10.34,
                          5.15089229]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.xyY_to_XYZ(numpy.matrix([0.3439, 0.3565, 3.20]).reshape((3, 1))),
            numpy.matrix([3.08690042,
                          3.2,
                          2.68925666]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.xyY_to_XYZ(numpy.matrix([0.4325, 0., 10.34]).reshape((3, 1))),
            numpy.matrix([0.,
                          0.,
                          0.]).reshape((3, 1)),
            decimal=7)


class Testxy_to_XYZ(unittest.TestCase):
    """
    Defines :func:`color.transformations.xy_to_XYZ` definition units tests methods.
    """

    def test_xy_to_XYZ(self):
        """
        Tests :func:`color.transformations.xy_to_XYZ` definition.
        """

        numpy.testing.assert_almost_equal(
            color.transformations.xy_to_XYZ((0.32207410281368043,
                                             0.3315655001362353)),
            numpy.matrix([0.97137399,
                          1.,
                          1.04462134]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.xy_to_XYZ((0.32174206617150575,
                                             0.337609723160027)),
            numpy.matrix([0.953,
                          1.000,
                          1.009]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.xy_to_XYZ((0.4474327628361859,
                                             0.4074979625101875)),
            numpy.matrix([1.098,
                          1.000,
                          0.356]).reshape((3, 1)),
            decimal=7)


class TestXYZ_to_xy(unittest.TestCase):
    """
    Defines :func:`color.transformations.XYZ_to_xy` definition units tests methods.
    """

    def test_XYZ_to_xy(self):
        """
        Tests :func:`color.transformations.XYZ_to_xy` definition.
        """

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_xy((0.97137399,
                                                                           1.,
                                                                           1.04462134)),
                                          (0.32207410281368043, 0.3315655001362353),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_xy((0.953,
                                                                           1.000,
                                                                           1.009)),
                                          (0.32174206617150575, 0.337609723160027),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_xy((1.098,
                                                                           1.000,
                                                                           0.356)),
                                          (0.4474327628361859, 0.4074979625101875),
                                          decimal=7)


class TestXYZ_to_RGB(unittest.TestCase):
    """
    Defines :func:`color.transformations.XYZ_to_RGB` definition units tests methods.
    """

    def test_XYZ_to_RGB(self):
        """
        Tests :func:`color.transformations.XYZ_to_RGB` definition.
        """

        for xyY, XYZ, RGB in sRGB_LINEAR_COLORCHECKER_2005:
            numpy.testing.assert_almost_equal(color.transformations.XYZ_to_RGB(numpy.matrix(XYZ).reshape((3, 1)),
                                                                               (0.34567, 0.35850),
                                                                               (0.31271, 0.32902),
                                                                               "Bradford",
                                                                               numpy.matrix(
                                                                                   [3.24100326, -1.53739899,
                                                                                    -0.49861587,
                                                                                    -0.96922426, 1.87592999,
                                                                                    0.04155422,
                                                                                    0.05563942, -0.2040112,
                                                                                    1.05714897]).reshape((3, 3)),
                                                                               sRGB_TRANSFER_FUNCTION),
                                              RGB,
                                              decimal=7)

        for xyY, XYZ, RGB in ACES_COLORCHECKER_2005:
            numpy.testing.assert_almost_equal(color.transformations.XYZ_to_RGB(numpy.matrix(XYZ).reshape((3, 1)),
                                                                               (0.34567, 0.35850),
                                                                               (0.32168, 0.33767),
                                                                               "CAT02",
                                                                               numpy.matrix(
                                                                                   [1.04981102e+00, 0.00000000e+00,
                                                                                    -9.74845410e-05,
                                                                                    -4.95903023e-01, 1.37331305e+00,
                                                                                    9.82400365e-02,
                                                                                    0.00000000e+00, 0.00000000e+00,
                                                                                    9.91252022e-01]).reshape(
                                                                                   (3, 3))),
                                              RGB,
                                              decimal=7)


class TestRGB_to_XYZ(unittest.TestCase):
    """
    Defines :func:`color.transformations.RGB_to_XYZ` definition units tests methods.
    """

    def test_RGB_to_XYZ(self):
        """
        Tests :func:`color.transformations.RGB_to_XYZ` definition.
        """

        for xyY, XYZ, RGB in sRGB_LINEAR_COLORCHECKER_2005:
            numpy.testing.assert_almost_equal(color.transformations.RGB_to_XYZ(RGB,
                                                                               (0.31271, 0.32902),
                                                                               (0.34567, 0.35850),
                                                                               "Bradford",
                                                                               numpy.matrix(
                                                                                   [0.41238656, 0.35759149, 0.18045049,
                                                                                    0.21263682, 0.71518298, 0.0721802,
                                                                                    0.01933062, 0.11919716,
                                                                                    0.95037259]).reshape((3, 3)),
                                                                               sRGB_INVERSE_TRANSFER_FUNCTION),
                                              numpy.matrix(XYZ).reshape((3, 1)),
                                              decimal=7)

        for xyY, XYZ, RGB in ACES_COLORCHECKER_2005:
            numpy.testing.assert_almost_equal(color.transformations.RGB_to_XYZ(RGB,
                                                                               (0.32168, 0.33767),
                                                                               (0.34567, 0.35850),
                                                                               "CAT02",
                                                                               numpy.matrix(
                                                                                   [9.52552396e-01, 0.00000000e+00,
                                                                                    9.36786317e-05,
                                                                                    3.43966450e-01, 7.28166097e-01,
                                                                                    -7.21325464e-02,
                                                                                    0.00000000e+00, 0.00000000e+00,
                                                                                    1.00882518e+00]).reshape(
                                                                                   (3, 3))),
                                              numpy.matrix(XYZ).reshape((3, 1)),
                                              decimal=7)


class TestxyY_to_RGB(unittest.TestCase):
    """
    Defines :func:`color.transformations.xyY_to_RGB` definition units tests methods.
    """

    def test_xyY_to_RGB(self):
        """
        Tests :func:`color.transformations.xyY_to_RGB` definition.
        """

        for xyY, XYZ, RGB in sRGB_LINEAR_COLORCHECKER_2005:
            numpy.testing.assert_almost_equal(color.transformations.xyY_to_RGB(numpy.matrix(xyY).reshape((3, 1)),
                                                                               (0.34567, 0.35850),
                                                                               (0.31271, 0.32902),
                                                                               "Bradford",
                                                                               numpy.matrix(
                                                                                   [3.24100326, -1.53739899,
                                                                                    -0.49861587,
                                                                                    -0.96922426, 1.87592999,
                                                                                    0.04155422,
                                                                                    0.05563942, -0.2040112,
                                                                                    1.05714897]).reshape((3, 3)),
                                                                               sRGB_TRANSFER_FUNCTION),
                                              RGB,
                                              decimal=7)

        for xyY, XYZ, RGB in ACES_COLORCHECKER_2005:
            numpy.testing.assert_almost_equal(color.transformations.xyY_to_RGB(numpy.matrix(xyY).reshape((3, 1)),
                                                                               (0.34567, 0.35850),
                                                                               (0.32168, 0.33767),
                                                                               "CAT02",
                                                                               numpy.matrix(
                                                                                   [1.04981102e+00, 0.00000000e+00,
                                                                                    -9.74845410e-05,
                                                                                    -4.95903023e-01, 1.37331305e+00,
                                                                                    9.82400365e-02,
                                                                                    0.00000000e+00, 0.00000000e+00,
                                                                                    9.91252022e-01]).reshape(
                                                                                   (3, 3))),
                                              RGB,
                                              decimal=7)


class TestRGB_to_xyY(unittest.TestCase):
    """
    Defines :func:`color.transformations.RGB_to_xyY` definition units tests methods.
    """

    def test_RGB_to_xyY(self):
        """
        Tests :func:`color.transformations.RGB_to_xyY` definition.
        """

        for xyY, XYZ, RGB in sRGB_LINEAR_COLORCHECKER_2005:
            numpy.testing.assert_almost_equal(color.transformations.RGB_to_xyY(RGB,
                                                                               (0.31271, 0.32902),
                                                                               (0.34567, 0.35850),
                                                                               "Bradford",
                                                                               numpy.matrix(
                                                                                   [0.41238656, 0.35759149, 0.18045049,
                                                                                    0.21263682, 0.71518298, 0.0721802,
                                                                                    0.01933062, 0.11919716,
                                                                                    0.95037259]).reshape((3, 3)),
                                                                               sRGB_INVERSE_TRANSFER_FUNCTION),
                                              numpy.matrix(xyY).reshape((3, 1)),
                                              decimal=7)

        for xyY, XYZ, RGB in ACES_COLORCHECKER_2005:
            numpy.testing.assert_almost_equal(color.transformations.RGB_to_xyY(RGB,
                                                                               (0.32168, 0.33767),
                                                                               (0.34567, 0.35850),
                                                                               "CAT02",
                                                                               numpy.matrix(
                                                                                   [9.52552396e-01, 0.00000000e+00,
                                                                                    9.36786317e-05,
                                                                                    3.43966450e-01, 7.28166097e-01,
                                                                                    -7.21325464e-02,
                                                                                    0.00000000e+00, 0.00000000e+00,
                                                                                    1.00882518e+00]).reshape(
                                                                                   (3, 3))),
                                              numpy.matrix(xyY).reshape((3, 1)),
                                              decimal=7)


class TestXYZ_to_UCS(unittest.TestCase):
    """
    Defines :func:`color.transformations.XYZ_to_UCS` definition units tests methods.
    """

    def test_XYZ_to_UCS(self):
        """
        Tests :func:`color.transformations.XYZ_to_UCS` definition.
        """

        numpy.testing.assert_almost_equal(
            color.transformations.XYZ_to_UCS(numpy.matrix([11.80583421, 10.34, 5.15089229]).reshape((3, 1))),
            numpy.matrix([7.87055614, 10.34, 12.18252904]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.XYZ_to_UCS(numpy.matrix([3.08690042, 3.2, 2.68925666]).reshape((3, 1))),
            numpy.matrix([2.05793361, 3.2, 4.60117812]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.XYZ_to_UCS(numpy.matrix([0.96907232, 1., 1.12179215]).reshape((3, 1))),
            numpy.matrix([0.64604821, 1., 1.57635992]).reshape((3, 1)),
            decimal=7)


class TestUCS_to_XYZ(unittest.TestCase):
    """
    Defines :func:`color.transformations.UCS_to_XYZ` definition units tests methods.
    """

    def test_UCS_to_XYZ(self):
        """
        Tests :func:`color.transformations.UCS_to_XYZ` definition.
        """

        numpy.testing.assert_almost_equal(
            color.transformations.UCS_to_XYZ(numpy.matrix([7.87055614, 10.34, 12.18252904]).reshape((3, 1))),
            numpy.matrix([11.80583421, 10.34, 5.15089229]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.UCS_to_XYZ(numpy.matrix([2.05793361, 3.2, 4.60117812]).reshape((3, 1))),
            numpy.matrix([3.08690042, 3.2, 2.68925666]).reshape((3, 1)),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.UCS_to_XYZ(numpy.matrix([0.64604821, 1., 1.57635992]).reshape((3, 1))),
            numpy.matrix([0.96907232, 1., 1.12179215]).reshape((3, 1)),
            decimal=7)


class TestUCS_to_uv(unittest.TestCase):
    """
    Defines :func:`color.transformations.UCS_to_uv` definition units tests methods.
    """

    def test_UCS_to_uv(self):
        """
        Tests :func:`color.transformations.UCS_to_uv` definition.
        """

        numpy.testing.assert_almost_equal(
            color.transformations.UCS_to_uv(numpy.matrix([7.87055614, 10.34, 12.18252904]).reshape((3, 1))),
            (0.25895877609618834, 0.34020896328103534),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.UCS_to_uv(numpy.matrix([2.05793361, 3.2, 4.60117812]).reshape((3, 1))),
            (0.20873418076173886, 0.32457285074301517),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.UCS_to_uv(numpy.matrix([0.64604821, 1., 1.57635992]).reshape((3, 1))),
            (0.20048615319251942, 0.31032692311386395),
            decimal=7)


class TestUCS_uv_to_xy(unittest.TestCase):
    """
    Defines :func:`color.transformations.UCS_uv_to_xy` definition units tests methods.
    """

    def test_UCS_uv_to_xy(self):
        """
        Tests :func:`color.transformations.UCS_uv_to_xy` definition.
        """

        numpy.testing.assert_almost_equal(color.transformations.UCS_uv_to_xy((0.2033733344733139, 0.3140500001549052)),
                                          (0.32207410281368043, 0.33156550013623537),
                                          decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.UCS_uv_to_xy((0.20873418102926322, 0.32457285063327812)),
            (0.3439000000209443, 0.35650000010917804),
            decimal=7)

        numpy.testing.assert_almost_equal(
            color.transformations.UCS_uv_to_xy((0.25585459629500179, 0.34952813701502972)),
            (0.4474327628361858, 0.40749796251018744),
            decimal=7)


class TestXYZ_to_UVW(unittest.TestCase):
    """
    Defines :func:`color.transformations.XYZ_to_UVW` definition units tests methods.
    """

    def test_XYZ_to_UVW(self):
        """
        Tests :func:`color.transformations.XYZ_to_UVW` definition.
        """

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_UVW(numpy.matrix([0.96907232,
                                                                                         1.,
                                                                                         1.12179215]).reshape((3, 1))),
                                          numpy.matrix([-0.90199113, -1.56588889, 8.]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_UVW(numpy.matrix([1.92001986,
                                                                                         1.,
                                                                                         -0.1241347]).reshape((3, 1))),
                                          numpy.matrix([26.5159289, 3.8694711, 8.]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_UVW(numpy.matrix([1.0131677,
                                                                                         1.,
                                                                                         2.11217686]).reshape((3, 1))),
                                          numpy.matrix([-2.89423113, -5.92004891, 8.]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_UVW(numpy.matrix([1.0131677,
                                                                                         1.,
                                                                                         2.11217686]).reshape((3, 1)),
                                                                           (0.44757, 0.40745)),
                                          numpy.matrix([-7.76195429, -8.43122502, 8.]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_UVW(numpy.matrix([1.0131677,
                                                                                         1.,
                                                                                         2.11217686]).reshape((3, 1)),
                                                                           (1. / 3., 1. / 3.)),
                                          numpy.matrix([-3.03641679, -4.92226526, 8.]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_UVW(numpy.matrix([1.0131677,
                                                                                         1.,
                                                                                         2.11217686]).reshape((3, 1)),
                                                                           (0.31271, 0.32902)),
                                          numpy.matrix([-1.7159427, -4.55119033, 8]).reshape((3, 1)),
                                          decimal=7)


class TestXYZ_to_Luv(unittest.TestCase):
    """
    Defines :func:`color.transformations.XYZ_to_Luv` definition units tests methods.
    """

    def test_XYZ_to_Luv(self):
        """
        Tests :func:`color.transformations.XYZ_to_Luv` definition.
        """

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_Luv(numpy.matrix([0.96907232,
                                                                                         1.,
                                                                                         1.12179215]).reshape((3, 1))),
                                          numpy.matrix([100., -11.27488915, -29.36041662]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_Luv(numpy.matrix([1.92001986,
                                                                                         1.,
                                                                                         -0.1241347]).reshape((3, 1))),
                                          numpy.matrix([100., 331.44911128, 72.55258319]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_Luv(numpy.matrix([1.0131677,
                                                                                         1.,
                                                                                         2.11217686]).reshape((3, 1))),
                                          numpy.matrix([100., -36.17788915, -111.00091702]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_Luv(numpy.matrix([1.0131677,
                                                                                         1.,
                                                                                         2.11217686]).reshape((3, 1)),
                                                                           (0.44757, 0.40745)),
                                          numpy.matrix([100., -97.02442861, -158.08546907]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_Luv(numpy.matrix([1.0131677,
                                                                                         1.,
                                                                                         2.11217686]).reshape((3, 1)),
                                                                           (1. / 3., 1. / 3.)),
                                          numpy.matrix([100., -37.95520989, -92.29247371]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_Luv(numpy.matrix([1.0131677,
                                                                                         1.,
                                                                                         2.11217686]).reshape((3, 1)),
                                                                           (0.31271, 0.32902)),
                                          numpy.matrix([100., -21.44928374, -85.33481874]).reshape((3, 1)),
                                          decimal=7)


class TestLuv_to_XYZ(unittest.TestCase):
    """
    Defines :func:`color.transformations.Luv_to_XYZ` definition units tests methods.
    """

    def test_Luv_to_XYZ(self):
        """
        Tests :func:`color.transformations.Luv_to_XYZ` definition.
        """

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_XYZ(numpy.matrix([100.,
                                                                                         -11.27488915,
                                                                                         -29.36041662]).reshape(
            (3, 1))),
                                          numpy.matrix([0.96907232, 1., 1.12179215]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_XYZ(numpy.matrix([100.,
                                                                                         331.44911128,
                                                                                         72.55258319]).reshape((3, 1))),
                                          numpy.matrix([1.92001986, 1., -0.1241347]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_XYZ(numpy.matrix([100.,
                                                                                         -36.17788915,
                                                                                         -111.00091702]).reshape(
            (3, 1))),
                                          numpy.matrix([1.0131677, 1., 2.11217686]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_XYZ(numpy.matrix([100.,
                                                                                         -97.02442861,
                                                                                         -158.08546907]).reshape(
            (3, 1)),
                                                                           (0.44757, 0.40745)),
                                          numpy.matrix([1.0131677, 1., 2.11217686]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_XYZ(numpy.matrix([100.,
                                                                                         -37.95520989,
                                                                                         -92.29247371]).reshape((3, 1)),
                                                                           (1. / 3., 1. / 3.)),
                                          numpy.matrix([1.0131677, 1., 2.11217686]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_XYZ(numpy.matrix([100.,
                                                                                         -21.44928374,
                                                                                         -85.33481874]).reshape((3, 1)),
                                                                           (0.31271, 0.32902)),
                                          numpy.matrix([1.0131677, 1., 2.11217686]).reshape((3, 1)),
                                          decimal=7)


class TestLuv_to_uv(unittest.TestCase):
    """
    Defines :func:`color.transformations.Luv_to_uv` definition units tests methods.
    """

    def test_Luv_to_uv(self):
        """
        Tests :func:`color.transformations.Luv_to_uv` definition.
        """

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_uv(numpy.matrix([100.,
                                                                                        -11.27488915,
                                                                                        -29.36041662]).reshape((3, 1))),
                                          (0.20048615433157738, 0.4654903849082484),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_uv(numpy.matrix([100.,
                                                                                        331.44911128,
                                                                                        72.55258319]).reshape((3, 1))),
                                          (0.46412000081619281, 0.54388500014670993),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_uv(numpy.matrix([100.,
                                                                                        -36.17788915,
                                                                                        -111.00091702]).reshape(
            (3, 1))),
                                          (0.18133000048542355, 0.40268999998517152),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_uv(numpy.matrix([100.,
                                                                                        -97.02442861,
                                                                                        -158.08546907]).reshape((3, 1)),
                                                                          (0.44757, 0.40745)),
                                          (0.18133000048503745, 0.40268999998707306),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_uv(numpy.matrix([100.,
                                                                                        -37.95520989,
                                                                                        -92.29247371]).reshape((3, 1)),
                                                                          (1. / 3., 1. / 3.)),
                                          (0.18133000048947367, 0.40268999998016192),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_uv(numpy.matrix([100.,
                                                                                        -21.44928374,
                                                                                        -85.33481874]).reshape((3, 1)),
                                                                          (0.31271, 0.32902)),
                                          (0.1813300004870092, 0.4026899999798475),
                                          decimal=7)


class TestLuv_to_LCHuv(unittest.TestCase):
    """
    Defines :func:`color.transformations.Luv_to_LCHuv` definition units tests methods.
    """

    def test_Luv_to_LCHuv(self):
        """
        Tests :func:`color.transformations.Luv_to_LCHuv` definition.
        """

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_LCHuv(numpy.matrix([100.,
                                                                                           -11.27488915,
                                                                                           -29.36041662]).reshape(
            (3, 1))),
                                          numpy.matrix([100., 31.45086945, 248.99237865]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_LCHuv(numpy.matrix([100.,
                                                                                           331.44911128,
                                                                                           72.55258319]).reshape(
            (3, 1))),
                                          numpy.matrix([100., 339.2969064, 12.34702048]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_to_LCHuv(numpy.matrix([100.,
                                                                                           -36.17788915,
                                                                                           -111.00091702]).reshape(
            (3, 1))),
                                          numpy.matrix([100., 116.74777618, 251.94795555]).reshape((3, 1)),
                                          decimal=7)


class TestLCHuv_to_Luv(unittest.TestCase):
    """
    Defines :func:`color.transformations.LCHuv_to_Luv` definition units tests methods.
    """

    def test_LCHuv_to_Luv(self):
        """
        Tests :func:`color.transformations.LCHuv_to_Luv` definition.
        """

        numpy.testing.assert_almost_equal(color.transformations.LCHuv_to_Luv(numpy.matrix([100.,
                                                                                           31.45086945,
                                                                                           248.99237865]).reshape(
            (3, 1))),
                                          numpy.matrix([100., -11.27488915, -29.36041662]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.LCHuv_to_Luv(numpy.matrix([100.,
                                                                                           339.2969064,
                                                                                           12.34702048]).reshape(
            (3, 1))),
                                          numpy.matrix([100., 331.44911128, 72.55258319]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.LCHuv_to_Luv(numpy.matrix([100.,
                                                                                           116.74777618,
                                                                                           251.94795555]).reshape(
            (3, 1))),
                                          numpy.matrix([100., -36.17788915, -111.00091702]).reshape((3, 1)),
                                          decimal=7)


class TestLuv_uv_to_xy(unittest.TestCase):
    """
    Defines :func:`color.transformations.Luv_uv_to_xy` definition units tests methods.
    """

    def test_Luv_uv_to_xy(self):
        """
        Tests :func:`color.transformations.Luv_uv_to_xy` definition.
        """

        numpy.testing.assert_almost_equal(color.transformations.Luv_uv_to_xy((0.20048615433157738, 0.4654903849082484)),
                                          (0.31352792378977895, 0.32353408235422665),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_uv_to_xy((0.46412000081619281,
                                                                              0.54388500014670993)),
                                          (0.6867305880410077, 0.3576684816384643),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Luv_uv_to_xy((0.18133000048542355,
                                                                              0.40268999998517152)),
                                          (0.2455958975694641, 0.2424039944946324),
                                          decimal=7)


class TestXYZ_to_Lab(unittest.TestCase):
    """
    Defines :func:`color.transformations.XYZ_to_Lab` definition units tests methods.
    """

    def test_XYZ_to_Lab(self):
        """
        Tests :func:`color.transformations.XYZ_to_Lab` definition.
        """

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_Lab(numpy.matrix([0.96907232,
                                                                                         1.,
                                                                                         1.12179215]).reshape((3, 1))),
                                          numpy.matrix([100., 0.83871284, -21.55579303]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_Lab(numpy.matrix([1.92001986,
                                                                                         1.,
                                                                                         -0.1241347]).reshape((3, 1))),
                                          numpy.matrix([100., 129.04406346, 406.69765889]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_Lab(numpy.matrix([1.0131677,
                                                                                         1.,
                                                                                         2.11217686]).reshape((3, 1))),
                                          numpy.matrix([100., 8.32281957, -73.58297716]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_Lab(numpy.matrix([1.0131677,
                                                                                         1.,
                                                                                         2.11217686]).reshape((3, 1)),
                                                                           (0.44757, 0.40745)),
                                          numpy.matrix([100., -13.29228089, -162.12804888]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_Lab(numpy.matrix([1.0131677,
                                                                                         1.,
                                                                                         2.11217686]).reshape((3, 1)),
                                                                           (1. / 3., 1. / 3.)),
                                          numpy.matrix([100., 2.18505384, -56.60990888]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.XYZ_to_Lab(numpy.matrix([1.0131677,
                                                                                         1.,
                                                                                         2.11217686]).reshape((3, 1)),
                                                                           (0.31271, 0.32902)),
                                          numpy.matrix([100., 10.76832763, -49.42733157]).reshape((3, 1)),
                                          decimal=7)


class TestLab_to_XYZ(unittest.TestCase):
    """
    Defines :func:`color.transformations.Lab_to_XYZ` definition units tests methods.
    """

    def test_Lab_to_XYZ(self):
        """
        Tests :func:`color.transformations.Lab_to_XYZ` definition.
        """

        numpy.testing.assert_almost_equal(color.transformations.Lab_to_XYZ(numpy.matrix([100.,
                                                                                         0.83871284,
                                                                                         -21.55579303]).reshape(
            (3, 1))),
                                          numpy.matrix([0.96907232, 1., 1.12179215]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Lab_to_XYZ(numpy.matrix([100.,
                                                                                         129.04406346,
                                                                                         406.69765889]).reshape(
            (3, 1))),
                                          numpy.matrix([1.92001986, 1., -0.1241347]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Lab_to_XYZ(numpy.matrix([100.,
                                                                                         8.32281957,
                                                                                         -73.58297716]).reshape(
            (3, 1))),
                                          numpy.matrix([1.0131677, 1., 2.11217686]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Lab_to_XYZ(numpy.matrix([100.,
                                                                                         -13.29228089,
                                                                                         -162.12804888]).reshape(
            (3, 1)),
                                                                           (0.44757, 0.40745)),
                                          numpy.matrix([1.0131677, 1., 2.11217686]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Lab_to_XYZ(numpy.matrix([100.,
                                                                                         2.18505384,
                                                                                         -56.60990888]).reshape((3, 1)),
                                                                           (1. / 3., 1. / 3.)),
                                          numpy.matrix([1.0131677, 1., 2.11217686]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Lab_to_XYZ(numpy.matrix([100.,
                                                                                         10.76832763,
                                                                                         -49.42733157]).reshape((3, 1)),
                                                                           (0.31271, 0.32902)),
                                          numpy.matrix([1.0131677, 1., 2.11217686]).reshape((3, 1)),
                                          decimal=7)


class TestLab_to_LCHab(unittest.TestCase):
    """
    Defines :func:`color.transformations.Lab_to_LCHab` definition units tests methods.
    """

    def test_Lab_to_LCHab(self):
        """
        Tests :func:`color.transformations.Lab_to_LCHab` definition.
        """

        numpy.testing.assert_almost_equal(color.transformations.Lab_to_LCHab(numpy.matrix([100.,
                                                                                           0.83871284,
                                                                                           -21.55579303]).reshape(
            (3, 1))),
                                          numpy.matrix([100., 21.57210357, 272.2281935]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Lab_to_LCHab(numpy.matrix([100.,
                                                                                           129.04406346,
                                                                                           406.69765889]).reshape(
            (3, 1))),
                                          numpy.matrix([100., 426.67945353, 72.39590835]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.Lab_to_LCHab(numpy.matrix([100.,
                                                                                           8.32281957,
                                                                                           -73.58297716]).reshape(
            (3, 1))),
                                          numpy.matrix([100., 74.05216981, 276.45318193]).reshape((3, 1)),
                                          decimal=7)


class TestLCHab_to_Lab(unittest.TestCase):
    """
    Defines :func:`color.transformations.LCHab_to_Lab` definition units tests methods.
    """

    def test_LCHab_to_Lab(self):
        """
        Tests :func:`color.transformations.LCHab_to_Lab` definition.
        """

        numpy.testing.assert_almost_equal(color.transformations.LCHab_to_Lab(numpy.matrix([100.,
                                                                                           21.57210357,
                                                                                           272.2281935]).reshape(
            (3, 1))),
                                          numpy.matrix([100.,
                                                        0.83871284,
                                                        -21.55579303]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.LCHab_to_Lab(numpy.matrix([100.,
                                                                                           426.67945353,
                                                                                           72.39590835]).reshape(
            (3, 1))),
                                          numpy.matrix([100.,
                                                        129.04406346,
                                                        406.69765889]).reshape((3, 1)),
                                          decimal=7)

        numpy.testing.assert_almost_equal(color.transformations.LCHab_to_Lab(numpy.matrix([100.,
                                                                                           74.05216981,
                                                                                           276.45318193]).reshape(
            (3, 1))),
                                          numpy.matrix([100.,
                                                        8.32281957,
                                                        -73.58297716]).reshape((3, 1)),
                                          decimal=7)


if __name__ == "__main__":
    unittest.main()
