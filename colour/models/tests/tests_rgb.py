#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb` module.
"""

from __future__ import division, unicode_literals

import pickle
import numpy as np
import sys

if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest

from colour.models import (
    RGB_COLOURSPACES,
    RGB_Colourspace,
    XYZ_to_RGB,
    RGB_to_XYZ,
    RGB_to_RGB,
    normalised_primary_matrix)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['sRGB_LINEAR_COLORCHECKER_2005',
           'ACES_COLORCHECKER_2005',
           'sRGB_TRANSFER_FUNCTION',
           'sRGB_INVERSE_TRANSFER_FUNCTION',
           'TestRGB_COLOURSPACES',
           'TestRGB_Colourspace',
           'TestXYZ_to_RGB',
           'TestRGB_to_XYZ',
           'TestRGB_to_RGB']

sRGB_LINEAR_COLORCHECKER_2005 = [
    [[0.4316, 0.3777, 0.1008],
     (0.11518474980142972, 0.1008, 0.050893725178713274),
     np.array([0.45293517, 0.31732158, 0.26414773])],
    [[0.4197, 0.3744, 0.34950000000000003],
     (0.39178725961538463, 0.34950000000000003, 0.19220633012820515),
     np.array([0.77875824, 0.5772645, 0.50453169])],
    [[0.276, 0.3016, 0.18359999999999999],
     (0.1680159151193634, 0.18359999999999999, 0.25713740053050399),
     np.array([0.35505307, 0.47995567, 0.61088035])],
    [[0.3703, 0.4499, 0.1325],
     (0.10905701266948212, 0.13250000000000001, 0.052952878417426061),
     np.array([0.35179242, 0.42214077, 0.25258942])],
    [[0.2999, 0.2856, 0.2304],
     (0.24193613445378148, 0.23039999999999999, 0.33438655462184863),
     np.array([0.50809894, 0.50196494, 0.69048098])],
    [[0.2848, 0.3911, 0.4178],
     (0.30424300690360523, 0.4178, 0.34622597801073896),
     np.array([0.36240083, 0.74473539, 0.67467032])],
    [[0.5295, 0.4055, 0.3118],
     (0.40714697903822439, 0.31180000000000002, 0.049980271270036999),
     np.array([0.87944466, 0.48522956, 0.18327685])],
    [[0.2305, 0.2106, 0.11259999999999999],
     (0.1232397910731244, 0.11259999999999999, 0.29882307692307686),
     np.array([0.26605806, 0.35770363, 0.66743852])],
    [[0.5012, 0.3273, 0.1938],
     (0.29676920256645278, 0.1938, 0.10154812098991754),
     np.array([0.77782346, 0.32138719, 0.38060248])],
    [[0.3319, 0.2482, 0.0637],
     (0.085181426269137786, 0.063700000000000007, 0.1077664383561644),
     np.array([0.36729282, 0.22739265, 0.41412433])],
    [[0.3984, 0.5008, 0.4446],
     (0.35369137380191684, 0.4446, 0.089488178913738003),
     np.array([0.62266646, 0.7410742, 0.24626906])],
    [[0.4957, 0.4427, 0.4357],
     (0.48786196069573068, 0.43569999999999998, 0.060625976959566286),
     np.array([0.90369041, 0.63376348, 0.15395733])],
    [[0.2018, 0.1692, 0.0575],
     (0.068578605200945636, 0.057500000000000002, 0.21375591016548467),
     np.array([0.1384956, 0.24831912, 0.57681467])],
    [[0.3253, 0.5032, 0.2318],
     (0.14985003974562797, 0.23180000000000001, 0.079001788553259192),
     np.array([0.26252953, 0.58394952, 0.29070622])],
    [[0.5686, 0.3303, 0.1257],
     (0.21638819255222524, 0.12570000000000001, 0.038474931880109003),
     np.array([0.70564037, 0.19094729, 0.22335249])],
    [[0.4697, 0.4734, 0.5981000000000001],
     (0.59342536966624426, 0.59810000000000008, 0.071888234051542058),
     np.array([0.93451045, 0.77825294, 0.07655428])],
    [[0.4159, 0.2688, 0.2009],
     (0.3108419270833333, 0.2009, 0.2356539062500001),
     np.array([0.75715761, 0.32930283, 0.59045447])],
    [[0.2131, 0.3023, 0.193],
     (0.13605127356930202, 0.193, 0.30938736354614615),
     np.array([-0.48463915, 0.53412743, 0.66546058])],
    [[0.3469, 0.3608, 0.9131],
     (0.87792236696230597, 0.91310000000000002, 0.73974259977827039),
     np.array([0.96027764, 0.96170536, 0.95169688])],
    [[0.344, 0.3584, 0.5893999999999999],
     (0.56571874999999983, 0.58939999999999992, 0.48941249999999997),
     np.array([0.78565259, 0.79300245, 0.79387336])],
    [[0.3432, 0.3581, 0.3632],
     (0.34808779670483109, 0.36320000000000002, 0.30295403518570241),
     np.array([0.63023284, 0.63852418, 0.64028572])],
    [[0.3446, 0.3579, 0.19149999999999998],
     (0.18438362671137187, 0.19149999999999998, 0.15918203408773396),
     np.array([0.4732449, 0.47519512, 0.47670436])],
    [[0.3401, 0.3548, 0.0883],
     (0.084641572717023675, 0.088300000000000003, 0.075931031567080032),
     np.array([0.32315746, 0.32983556, 0.33640183])],
    [[0.3406, 0.3537, 0.0311],
     (0.029948148148148147, 0.031099999999999999, 0.026879474130619162),
     np.array([0.19104038, 0.19371002, 0.19903915])]]

ACES_COLORCHECKER_2005 = [
    [[0.4316, 0.3777, 0.1008],
     (0.11518474980142972, 0.1008, 0.050893725178713274),
     np.array([0.11758989, 0.08781098, 0.06184838])],
    [[0.4197, 0.3744, 0.34950000000000003],
     (0.39178725961538463, 0.34950000000000003, 0.19220633012820515),
     np.array([0.40073605, 0.31020146, 0.2334411])],
    [[0.276, 0.3016, 0.18359999999999999],
     (0.1680159151193634, 0.18359999999999999, 0.25713740053050399),
     np.array([0.17949613, 0.20101795, 0.31109218])],
    [[0.3703, 0.4499, 0.1325],
     (0.10905701266948212, 0.13250000000000001, 0.052952878417426061),
     np.array([0.1107181, 0.13503098, 0.06442476])],
    [[0.2999, 0.2856, 0.2304],
     (0.24193613445378148, 0.23039999999999999, 0.33438655462184863),
     np.array([0.2575148, 0.23804357, 0.40454743])],
    [[0.2848, 0.3911, 0.4178],
     (0.30424300690360523, 0.4178, 0.34622597801073896),
     np.array([0.31733562, 0.46758348, 0.41947022])],
    [[0.5295, 0.4055, 0.3118],
     (0.40714697903822439, 0.31180000000000002, 0.049980271270036999),
     np.array([0.41040872, 0.23293505, 0.06167114])],
    [[0.2305, 0.2106, 0.11259999999999999],
     (0.1232397910731244, 0.11259999999999999, 0.29882307692307686),
     np.array([0.13747056, 0.13033376, 0.36114764])],
    [[0.5012, 0.3273, 0.1938],
     (0.29676920256645278, 0.1938, 0.10154812098991754),
     np.array([0.30304559, 0.13139056, 0.12344791])],
    [[0.3319, 0.2482, 0.0637],
     (0.085181426269137786, 0.063700000000000007, 0.1077664383561644),
     np.array([0.09058405, 0.05847923, 0.13035265])],
    [[0.3984, 0.5008, 0.4446],
     (0.35369137380191684, 0.4446, 0.089488178913738003),
     np.array([0.3547791, 0.44849679, 0.10971221])],
    [[0.4957, 0.4427, 0.4357],
     (0.48786196069573068, 0.43569999999999998, 0.060625976959566286),
     np.array([0.49038927, 0.36515801, 0.07497681])],
    [[0.2018, 0.1692, 0.0575],
     (0.068578605200945636, 0.057500000000000002, 0.21375591016548467),
     np.array([0.07890084, 0.07117527, 0.25824906])],
    [[0.3253, 0.5032, 0.2318],
     (0.14985003974562797, 0.23180000000000001, 0.079001788553259192),
     np.array([0.15129818, 0.25515937, 0.09620886])],
    [[0.5686, 0.3303, 0.1257],
     (0.21638819255222524, 0.12570000000000001, 0.038474931880109003),
     np.array([0.21960818, 0.06985597, 0.04703204])],
    [[0.4697, 0.4734, 0.5981000000000001],
     (0.59342536966624426, 0.59810000000000008, 0.071888234051542058),
     np.array([0.5948559, 0.5382559, 0.08916818])],
    [[0.4159, 0.2688, 0.2009],
     (0.3108419270833333, 0.2009, 0.2356539062500001),
     np.array([0.32368864, 0.15049668, 0.28535138])],
    [[0.2131, 0.3023, 0.193],
     (0.13605127356930202, 0.193, 0.30938736354614615),
     np.array([0.14920707, 0.23648468, 0.37415686])],
    [[0.3469, 0.3608, 0.9131],
     (0.87792236696230597, 0.91310000000000002, 0.73974259977827039),
     np.array([0.90989008, 0.91268206, 0.89651699])],
    [[0.344, 0.3584, 0.5893999999999999],
     (0.56571874999999983, 0.58939999999999992, 0.48941249999999997),
     np.array([0.58690823, 0.59107342, 0.59307473])],
    [[0.3432, 0.3581, 0.3632],
     (0.34808779670483109, 0.36320000000000002, 0.30295403518570241),
     np.array([0.36120089, 0.36465935, 0.36711553])],
    [[0.3446, 0.3579, 0.19149999999999998],
     (0.18438362671137187, 0.19149999999999998, 0.15918203408773396),
     np.array([0.19128766, 0.19177359, 0.19289805])],
    [[0.3401, 0.3548, 0.0883],
     (0.084641572717023675, 0.088300000000000003, 0.075931031567080032),
     np.array([0.08793956, 0.08892476, 0.09200134])],
    [[0.3406, 0.3537, 0.0311],
     (0.029948148148148147, 0.031099999999999999, 0.026879474130619162),
     np.array([0.03111895, 0.03126787, 0.03256784])]]

sRGB_TRANSFER_FUNCTION = lambda x: (
    x * 12.92 if x <= 0.0031308 else 1.055 * (x ** (1 / 2.4)) - 0.055)

sRGB_INVERSE_TRANSFER_FUNCTION = lambda x: (
    x / 12.92 if x <= 0.0031308 else ((x + 0.055) / 1.055) ** 2.4)


class TestRGB_COLOURSPACES(unittest.TestCase):
    """
    Defines :attr:`colour.models.RGB_COLOURSPACES` attribute unit tests
    methods.
    """

    def test_transformation_matrices(self):
        """
        Tests the transformations matrices from the
        :attr:`colour.models.RGB_COLOURSPACES` attribute colourspace models.
        """

        XYZ_r = np.array([0.5, 0.5, 0.5]).reshape((3, 1))
        for colourspace in RGB_COLOURSPACES.values():
            M = normalised_primary_matrix(colourspace.primaries,
                                          colourspace.whitepoint)
            np.testing.assert_allclose(colourspace.RGB_to_XYZ_matrix,
                                       M,
                                       rtol=0.001,
                                       atol=0.001,
                                       verbose=False)

            RGB = np.dot(colourspace.XYZ_to_RGB_matrix, XYZ_r)
            XYZ = np.dot(colourspace.RGB_to_XYZ_matrix, RGB)
            np.testing.assert_almost_equal(XYZ_r, XYZ, decimal=7)

    def test_opto_electronic_conversion_functions(self):
        """
        Tests the opto-electronic conversion functions from the
        :attr:`colour.models.RGB_COLOURSPACES` attribute colourspace models.
        """

        aces_proxy_colourspaces = ('ACES RGB Proxy 10', 'ACES RGB Proxy 12')

        samples = np.linspace(0, 1, 1000)
        for colourspace in RGB_COLOURSPACES.values():
            if colourspace.name in aces_proxy_colourspaces:
                continue

            samples_oecf = [colourspace.transfer_function(sample)
                            for sample in samples]
            samples_invert_oecf = [
                colourspace.inverse_transfer_function(sample)
                for sample in samples_oecf]

            np.testing.assert_almost_equal(samples,
                                           samples_invert_oecf,
                                           decimal=7)

        for colourspace in aces_proxy_colourspaces:
            colourspace = RGB_COLOURSPACES.get(colourspace)
            samples_oecf = [colourspace.transfer_function(sample)
                            for sample in samples]
            samples_invert_oecf = [
                colourspace.inverse_transfer_function(sample)
                for sample in samples_oecf]

            np.testing.assert_allclose(samples,
                                       samples_invert_oecf,
                                       rtol=0.01,
                                       atol=0.01)

    def test_pickle(self):
        """
        Tests the ability of colourspace models to be pickled.
        """

        for colourspace in RGB_COLOURSPACES:
            pickle.dumps(colourspace)


class TestRGB_Colourspace(unittest.TestCase):
    """
    Defines :class:`colour.colour.models.RGB_Colourspace` class units
    tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('name',
                               'primaries',
                               'whitepoint',
                               'illuminant',
                               'RGB_to_XYZ_matrix',
                               'XYZ_to_RGB_matrix',
                               'transfer_function',
                               'inverse_transfer_function',)

        for attribute in required_attributes:
            self.assertIn(attribute, dir(RGB_Colourspace))


class TestXYZ_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.XYZ_to_RGB` definition unit tests
    methods.
    """

    def test_XYZ_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.XYZ_to_RGB` definition.
        """

        for xyY, XYZ, RGB in sRGB_LINEAR_COLORCHECKER_2005:
            np.testing.assert_almost_equal(
                XYZ_to_RGB(
                    np.array(XYZ),
                    (0.34567, 0.35850),
                    (0.31271, 0.32902),
                    np.array(
                        [3.24100326, -1.53739899, -0.49861587,
                         -0.96922426, 1.87592999, 0.04155422,
                         0.05563942, -0.2040112, 1.05714897]),
                    'Bradford',
                    sRGB_TRANSFER_FUNCTION),
                RGB,
                decimal=7)

        for xyY, XYZ, RGB in ACES_COLORCHECKER_2005:
            np.testing.assert_almost_equal(
                XYZ_to_RGB(
                    np.array(XYZ),
                    (0.34567, 0.35850),
                    (0.32168, 0.33767),
                    np.array(
                        [1.04981102e+00, 0.00000000e+00, -9.74845410e-05,
                         -4.95903023e-01, 1.37331305e+00, 9.82400365e-02,
                         0.00000000e+00, 0.00000000e+00, 9.91252022e-01]
                    )),
                RGB,
                decimal=7)


class TestRGB_to_XYZ(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.RGB_to_XYZ` definition unit tests
    methods.
    """

    def test_RGB_to_XYZ(self):
        """
        Tests :func:`colour.models.rgb.RGB_to_XYZ` definition.
        """

        for xyY, XYZ, RGB in sRGB_LINEAR_COLORCHECKER_2005:
            np.testing.assert_almost_equal(
                RGB_to_XYZ(
                    RGB,
                    (0.31271, 0.32902),
                    (0.34567, 0.35850),
                    np.array(
                        [0.41238656, 0.35759149, 0.18045049,
                         0.21263682, 0.71518298, 0.0721802,
                         0.01933062, 0.11919716, 0.95037259]),
                    'Bradford',
                    sRGB_INVERSE_TRANSFER_FUNCTION),
                np.array(XYZ),
                decimal=7)

        for xyY, XYZ, RGB in ACES_COLORCHECKER_2005:
            np.testing.assert_almost_equal(
                RGB_to_XYZ(
                    RGB,
                    (0.32168, 0.33767),
                    (0.34567, 0.35850),
                    np.array(
                        [9.52552396e-01, 0.00000000e+00, 9.36786317e-05,
                         3.43966450e-01, 7.28166097e-01, -7.21325464e-02,
                         0.00000000e+00, 0.00000000e+00, 1.00882518e+00])),
                np.array(XYZ),
                decimal=7)


class TestRGB_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.RGB_to_RGB` definition unit tests
    methods.
    """

    def test_RGB_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.RGB_to_RGB` definition.
        """

        aces_rgb_colourspace = RGB_COLOURSPACES.get('ACES RGB')
        sRGB_colourspace = RGB_COLOURSPACES.get('sRGB')

        np.testing.assert_almost_equal(
            RGB_to_RGB((0.35521588, 0.41, 0.24177934),
                       aces_rgb_colourspace,
                       sRGB_colourspace),
            np.array([0.33658567, 0.44096335, 0.21509975]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB((0.33658567, 0.44096335, 0.21509975),
                       sRGB_colourspace,
                       aces_rgb_colourspace),
            np.array([0.35521588, 0.41, 0.24177934]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB((0.35521588, 0.41, 0.24177934),
                       aces_rgb_colourspace,
                       sRGB_colourspace,
                       'Bradford'),
            np.array([0.33704409, 0.44133521, 0.21429761]),
            decimal=7)


if __name__ == '__main__':
    unittest.main()
