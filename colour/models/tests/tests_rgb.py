#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb` module.
"""

from __future__ import division, unicode_literals

import pickle
import numpy as np
import unittest
from itertools import permutations

from colour.models import (
    RGB_COLOURSPACES,
    RGB_Colourspace,
    XYZ_to_RGB,
    RGB_to_XYZ,
    RGB_to_RGB,
    normalised_primary_matrix)
from colour.models.dataset.srgb import (
    _srgb_OECF,
    _srgb_EOCF)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['sRGB_LINEAR_COLORCHECKER_2005',
           'ACES_COLORCHECKER_2005',
           'sRGB_OECF',
           'sRGB_EOCF',
           'TestRGB_COLOURSPACES',
           'TestRGB_Colourspace',
           'TestXYZ_to_RGB',
           'TestRGB_to_XYZ',
           'TestRGB_to_RGB']

sRGB_LINEAR_COLORCHECKER_2005 = (
    ((0.4316, 0.3777, 0.1008),
     (0.11518475, 0.10080000, 0.05089373),
     (0.45293517, 0.31732158, 0.26414773)),
    ((0.4197, 0.3744, 0.3495),
     (0.39178726, 0.34950000, 0.19220633),
     (0.77875824, 0.57726450, 0.50453169)),
    ((0.2760, 0.3016, 0.1836),
     (0.16801592, 0.18360000, 0.25713740),
     (0.35505307, 0.47995567, 0.61088035)),
    ((0.3703, 0.4499, 0.1325),
     (0.10905701, 0.13250000, 0.05295288),
     (0.35179242, 0.42214077, 0.25258942)),
    ((0.2999, 0.2856, 0.2304),
     (0.24193613, 0.23040000, 0.33438655),
     (0.50809894, 0.50196494, 0.69048098)),
    ((0.2848, 0.3911, 0.4178),
     (0.30424301, 0.41780000, 0.34622598),
     (0.36240083, 0.74473539, 0.67467032)),
    ((0.5295, 0.4055, 0.3118),
     (0.40714698, 0.31180000, 0.04998027),
     (0.87944466, 0.48522956, 0.18327685)),
    ((0.2305, 0.2106, 0.1126),
     (0.12323979, 0.11260000, 0.29882308),
     (0.26605806, 0.35770363, 0.66743852)),
    ((0.5012, 0.3273, 0.1938),
     (0.29676920, 0.19380000, 0.10154812),
     (0.77782346, 0.32138719, 0.38060248)),
    ((0.3319, 0.2482, 0.0637),
     (0.08518143, 0.06370000, 0.10776644),
     (0.36729282, 0.22739265, 0.41412433)),
    ((0.3984, 0.5008, 0.4446),
     (0.35369137, 0.44460000, 0.08948818),
     (0.62266646, 0.74107420, 0.24626906)),
    ((0.4957, 0.4427, 0.4357),
     (0.48786196, 0.43570000, 0.06062598),
     (0.90369041, 0.63376348, 0.15395733)),
    ((0.2018, 0.1692, 0.0575),
     (0.06857861, 0.05750000, 0.21375591),
     (0.13849560, 0.24831912, 0.57681467)),
    ((0.3253, 0.5032, 0.2318),
     (0.14985004, 0.23180000, 0.07900179),
     (0.26252953, 0.58394952, 0.29070622)),
    ((0.5686, 0.3303, 0.1257),
     (0.21638819, 0.12570000, 0.03847493),
     (0.70564037, 0.19094729, 0.22335249)),
    ((0.4697, 0.4734, 0.5981),
     (0.59342537, 0.59810000, 0.07188823),
     (0.93451045, 0.77825294, 0.07655428)),
    ((0.4159, 0.2688, 0.2009),
     (0.31084193, 0.20090000, 0.23565391),
     (0.75715761, 0.32930283, 0.59045447)),
    ((0.2131, 0.3023, 0.1930),
     (0.13605127, 0.19300000, 0.30938736),
     (-0.48463915, 0.53412743, 0.66546058)),
    ((0.3469, 0.3608, 0.9131),
     (0.87792237, 0.91310000, 0.73974260),
     (0.96027764, 0.96170536, 0.95169688)),
    ((0.3440, 0.3584, 0.5894),
     (0.56571875, 0.58940000, 0.48941250),
     (0.78565259, 0.79300245, 0.79387336)),
    ((0.3432, 0.3581, 0.3632),
     (0.34808780, 0.36320000, 0.30295404),
     (0.63023284, 0.63852418, 0.64028572)),
    ((0.3446, 0.3579, 0.1915),
     (0.18438363, 0.19150000, 0.15918203),
     (0.47324490, 0.47519512, 0.47670436)),
    ((0.3401, 0.3548, 0.0883),
     (0.08464157, 0.08830000, 0.07593103),
     (0.32315746, 0.32983556, 0.33640183)),
    ((0.3406, 0.3537, 0.0311),
     (0.02994815, 0.03110000, 0.02687947),
     (0.19104038, 0.19371002, 0.19903915)))

ACES_COLORCHECKER_2005 = (
    ((0.4316, 0.3777, 0.1008),
     (0.11518475, 0.10080000, 0.05089373),
     (0.11758989, 0.08781098, 0.06184838)),
    ((0.4197, 0.3744, 0.3495),
     (0.39178726, 0.34950000, 0.19220633),
     (0.40073605, 0.31020146, 0.23344110)),
    ((0.2760, 0.3016, 0.1836),
     (0.16801592, 0.18360000, 0.25713740),
     (0.17949613, 0.20101795, 0.31109218)),
    ((0.3703, 0.4499, 0.1325),
     (0.10905701, 0.13250000, 0.05295288),
     (0.11071810, 0.13503098, 0.06442476)),
    ((0.2999, 0.2856, 0.2304),
     (0.24193613, 0.23040000, 0.33438655),
     (0.25751480, 0.23804357, 0.40454743)),
    ((0.2848, 0.3911, 0.4178),
     (0.30424301, 0.41780000, 0.34622598),
     (0.31733562, 0.46758348, 0.41947022)),
    ((0.5295, 0.4055, 0.3118),
     (0.40714698, 0.31180000, 0.04998027),
     (0.41040872, 0.23293505, 0.06167114)),
    ((0.2305, 0.2106, 0.1126),
     (0.12323979, 0.11260000, 0.29882308),
     (0.13747056, 0.13033376, 0.36114764)),
    ((0.5012, 0.3273, 0.1938),
     (0.29676920, 0.19380000, 0.10154812),
     (0.30304559, 0.13139056, 0.12344791)),
    ((0.3319, 0.2482, 0.0637),
     (0.08518143, 0.06370000, 0.10776644),
     (0.09058405, 0.05847923, 0.13035265)),
    ((0.3984, 0.5008, 0.4446),
     (0.35369137, 0.44460000, 0.08948818),
     (0.35477910, 0.44849679, 0.10971221)),
    ((0.4957, 0.4427, 0.4357),
     (0.48786196, 0.43570000, 0.06062598),
     (0.49038927, 0.36515801, 0.07497681)),
    ((0.2018, 0.1692, 0.0575),
     (0.06857861, 0.05750000, 0.21375591),
     (0.07890084, 0.07117527, 0.25824906)),
    ((0.3253, 0.5032, 0.2318),
     (0.14985004, 0.23180000, 0.07900179),
     (0.15129818, 0.25515937, 0.09620886)),
    ((0.5686, 0.3303, 0.1257),
     (0.21638819, 0.12570000, 0.03847493),
     (0.21960818, 0.06985597, 0.04703204)),
    ((0.4697, 0.4734, 0.5981),
     (0.59342537, 0.59810000, 0.07188823),
     (0.59485590, 0.53825590, 0.08916818)),
    ((0.4159, 0.2688, 0.2009),
     (0.31084193, 0.20090000, 0.23565391),
     (0.32368864, 0.15049668, 0.28535138)),
    ((0.2131, 0.3023, 0.1930),
     (0.13605127, 0.19300000, 0.30938736),
     (0.14920707, 0.23648468, 0.37415686)),
    ((0.3469, 0.3608, 0.9131),
     (0.87792237, 0.91310000, 0.73974260),
     (0.90989008, 0.91268206, 0.89651699)),
    ((0.3440, 0.3584, 0.5894),
     (0.56571875, 0.58940000, 0.48941250),
     (0.58690823, 0.59107342, 0.59307473)),
    ((0.3432, 0.3581, 0.3632),
     (0.34808780, 0.36320000, 0.30295404),
     (0.36120089, 0.36465935, 0.36711553)),
    ((0.3446, 0.3579, 0.1915),
     (0.18438363, 0.19150000, 0.15918203),
     (0.19128766, 0.19177359, 0.19289805)),
    ((0.3401, 0.3548, 0.0883),
     (0.08464157, 0.08830000, 0.07593103),
     (0.08793956, 0.08892476, 0.09200134)),
    ((0.3406, 0.3537, 0.0311),
     (0.02994815, 0.03110000, 0.02687947),
     (0.03111895, 0.03126787, 0.03256784)))

sRGB_OECF = _srgb_OECF

sRGB_EOCF = _srgb_EOCF


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
                                       rtol=0.0001,
                                       atol=0.0001,
                                       verbose=False)

            RGB = np.dot(colourspace.XYZ_to_RGB_matrix, XYZ_r)
            XYZ = np.dot(colourspace.RGB_to_XYZ_matrix, RGB)
            np.testing.assert_almost_equal(XYZ_r, XYZ, decimal=7)

    def test_opto_electronic_conversion_functions(self):
        """
        Tests opto-electronic conversion functions from the
        :attr:`colour.models.RGB_COLOURSPACES` attribute colourspace models.
        """

        aces_proxy_colourspaces = ('ACESproxy', 'ACEScg')

        samples = np.linspace(0, 1, 1000)
        for colourspace in RGB_COLOURSPACES.values():
            if colourspace.name in aces_proxy_colourspaces:
                continue

            samples_oecf = [colourspace.OECF(sample)
                            for sample in samples]
            samples_inverse_oecf = [
                colourspace.EOCF(sample)
                for sample in samples_oecf]

            np.testing.assert_almost_equal(samples,
                                           samples_inverse_oecf,
                                           decimal=7)

        for colourspace in aces_proxy_colourspaces:
            colourspace = RGB_COLOURSPACES.get(colourspace)
            samples_oecf = [colourspace.OECF(sample)
                            for sample in samples]
            samples_inverse_oecf = [
                colourspace.EOCF(sample)
                for sample in samples_oecf]

            np.testing.assert_allclose(samples,
                                       samples_inverse_oecf,
                                       rtol=0.01,
                                       atol=0.01)

    def test_n_dimensional_opto_electronic_conversion_functions(self):
        """
        Tests opto-electronic conversion functions from the
        :attr:`colour.models.RGB_COLOURSPACES` attribute colourspace models
        n-dimensional arrays support.
        """

        for colourspace in RGB_COLOURSPACES.values():
            value_oecf = 0.5
            value_inverse_oecf = colourspace.EOCF(
                colourspace.OECF(value_oecf))
            np.testing.assert_almost_equal(
                value_oecf,
                value_inverse_oecf,
                decimal=7)

            value_oecf = np.tile(value_oecf, 6)
            value_inverse_oecf = np.tile(value_inverse_oecf, 6)
            np.testing.assert_almost_equal(
                value_oecf,
                value_inverse_oecf,
                decimal=7)

            value_oecf = np.reshape(value_oecf, (3, 2))
            value_inverse_oecf = np.reshape(value_inverse_oecf, (3, 2))
            np.testing.assert_almost_equal(
                value_oecf,
                value_inverse_oecf,
                decimal=7)

            value_oecf = np.reshape(value_oecf, (3, 2, 1))
            value_inverse_oecf = np.reshape(value_inverse_oecf, (3, 2, 1))
            np.testing.assert_almost_equal(
                value_oecf,
                value_inverse_oecf,
                decimal=7)

    @ignore_numpy_errors
    def test_nan_opto_electronic_conversion_functions(self):
        """
        Tests opto-electronic conversion functions from the
        :attr:`colour.models.RGB_COLOURSPACES` attribute colourspace models
        nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        for colourspace in RGB_COLOURSPACES.values():
            for case in cases:
                colourspace.OECF(case)
                colourspace.EOCF(case)

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
                               'OECF',
                               'EOCF',)

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

        for _xyY, XYZ, RGB in sRGB_LINEAR_COLORCHECKER_2005:
            np.testing.assert_almost_equal(
                XYZ_to_RGB(
                    np.array(XYZ),
                    np.array([0.34567, 0.35850]),
                    np.array([0.31271, 0.32902]),
                    np.array([[3.24100326, -1.53739899, -0.49861587],
                              [-0.96922426, 1.87592999, 0.04155422],
                              [0.05563942, -0.20401120, 1.05714897]]),
                    'Bradford',
                    sRGB_OECF),
                RGB,
                decimal=7)

        for _xyY, XYZ, RGB in ACES_COLORCHECKER_2005:
            np.testing.assert_almost_equal(
                XYZ_to_RGB(
                    np.array(XYZ),
                    np.array([0.34567, 0.35850]),
                    np.array([0.32168, 0.33767]),
                    np.array(
                        [[1.04981102e+00, 0.00000000e+00, -9.74845410e-05],
                         [-4.95903023e-01, 1.37331305e+00, 9.82400365e-02],
                         [0.00000000e+00, 0.00000000e+00, 9.91252022e-01]])),
                RGB,
                decimal=7)

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        W_R = np.array([0.34567, 0.35850])
        W_T = np.array([0.31271, 0.32902, 0.10080])
        M = np.array([[3.24100326, -1.53739899, -0.49861587],
                      [-0.96922426, 1.87592999, 0.04155422],
                      [0.05563942, -0.20401120, 1.05714897]])
        np.testing.assert_almost_equal(
            XYZ_to_RGB(XYZ, W_R, W_T, M),
            np.array([0.00110011, 0.01282112, 0.01173427]),
            decimal=7)

    def test_n_dimensional_XYZ_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.XYZ_to_RGB` definition n-dimensions
        support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        W_R = np.array([0.34567, 0.35850])
        W_T = np.array([0.31271, 0.32902])
        M = np.array([[3.24100326, -1.53739899, -0.49861587],
                      [-0.96922426, 1.87592999, 0.04155422],
                      [0.05563942, -0.20401120, 1.05714897]])
        RGB = np.array([0.01091381, 0.12719366, 0.11641136])
        np.testing.assert_almost_equal(
            XYZ_to_RGB(XYZ, W_R, W_T, M),
            RGB,
            decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_RGB(XYZ, W_R, W_T, M),
            RGB,
            decimal=7)

        W_R = np.tile(W_R, (6, 1))
        W_T = np.tile(W_T, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_RGB(XYZ, W_R, W_T, M),
            RGB,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        W_R = np.reshape(W_R, (2, 3, 2))
        W_T = np.reshape(W_T, (2, 3, 2))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_RGB(XYZ, W_R, W_T, M),
            RGB,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_XYZ_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.XYZ_to_RGB` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            XYZ = np.array(case)
            W_R = np.array(case[0:2])
            W_T = np.array(case[0:2])
            M = np.vstack((case, case, case)).reshape((3, 3))
            XYZ_to_RGB(XYZ, W_R, W_T, M)


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
                    np.array([0.31271, 0.32902]),
                    np.array([0.34567, 0.35850]),
                    np.array(
                        [[0.41238656, 0.35759149, 0.18045049],
                         [0.21263682, 0.71518298, 0.07218020],
                         [0.01933062, 0.11919716, 0.95037259]]),
                    'Bradford',
                    sRGB_EOCF),
                np.array(XYZ),
                decimal=7)

        for xyY, XYZ, RGB in ACES_COLORCHECKER_2005:
            np.testing.assert_almost_equal(
                RGB_to_XYZ(
                    RGB,
                    np.array([0.32168, 0.33767]),
                    np.array([0.34567, 0.35850]),
                    np.array(
                        [[9.52552396e-01, 0.00000000e+00, 9.36786317e-05],
                         [3.43966450e-01, 7.28166097e-01, -7.21325464e-02],
                         [0.00000000e+00, 0.00000000e+00, 1.00882518e+00]])),
                np.array(XYZ),
                decimal=7)

        RGB = np.array([0.86969452, 1.00516431, 1.41715848])
        W_R = np.array([0.31271, 0.32902])
        W_T = np.array([0.34567, 0.35850, 0.10080])
        M = np.array([
            [0.41238656, 0.35759149, 0.18045049],
            [0.21263682, 0.71518298, 0.07218020],
            [0.01933062, 0.11919716, 0.95037259]])
        np.testing.assert_almost_equal(
            RGB_to_XYZ(RGB, W_R, W_T, M),
            np.array([0.09757065, 0.10063053, 0.11347848]),
            decimal=7)

    def test_n_dimensional_RGB_to_XYZ(self):
        """
        Tests :func:`colour.models.rgb.RGB_to_XYZ` definition n-dimensions
        support.
        """

        RGB = np.array([0.86969452, 1.00516431, 1.41715848])
        W_R = np.array([0.31271, 0.32902])
        W_T = np.array([0.34567, 0.35850])
        M = np.array([
            [0.41238656, 0.35759149, 0.18045049],
            [0.21263682, 0.71518298, 0.07218020],
            [0.01933062, 0.11919716, 0.95037259]])
        XYZ = np.array([0.96796280, 0.99831871, 1.12577854])
        np.testing.assert_almost_equal(
            RGB_to_XYZ(RGB, W_R, W_T, M),
            XYZ,
            decimal=7)

        RGB = np.tile(RGB, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            RGB_to_XYZ(RGB, W_R, W_T, M),
            XYZ,
            decimal=7)

        W_R = np.tile(W_R, (6, 1))
        W_T = np.tile(W_T, (6, 1))
        np.testing.assert_almost_equal(
            RGB_to_XYZ(RGB, W_R, W_T, M),
            XYZ,
            decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        W_R = np.reshape(W_R, (2, 3, 2))
        W_T = np.reshape(W_T, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            RGB_to_XYZ(RGB, W_R, W_T, M),
            XYZ,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_XYZ(self):
        """
        Tests :func:`colour.models.rgb.RGB_to_XYZ` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            W_R = np.array(case[0:2])
            W_T = np.array(case[0:2])
            M = np.vstack((case, case, case)).reshape((3, 3))
            RGB_to_XYZ(RGB, W_R, W_T, M)


class TestRGB_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.RGB_to_RGB` definition unit tests
    methods.
    """

    def test_RGB_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.RGB_to_RGB` definition.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES.get('ACES2065-1')
        sRGB_colourspace = RGB_COLOURSPACES.get('sRGB')

        np.testing.assert_almost_equal(
            RGB_to_RGB(np.array([0.35521588, 0.41000000, 0.24177934]),
                       aces_2065_1_colourspace,
                       sRGB_colourspace),
            np.array([0.33658567, 0.44096335, 0.21509975]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB(np.array([0.33658567, 0.44096335, 0.21509975]),
                       sRGB_colourspace,
                       aces_2065_1_colourspace),
            np.array([0.35521588, 0.41000000, 0.24177934]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB(np.array([0.35521588, 0.41000000, 0.24177934]),
                       aces_2065_1_colourspace,
                       sRGB_colourspace,
                       'Bradford'),
            np.array([0.33704409, 0.44133521, 0.21429761]),
            decimal=7)

    def test_n_dimensional_RGB_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.RGB_to_RGB` definition n-dimensions
        support.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES.get('ACES2065-1')
        sRGB_colourspace = RGB_COLOURSPACES.get('sRGB')
        RGB_i = np.array([0.35521588, 0.41000000, 0.24177934])
        RGB_o = np.array([0.33658567, 0.44096335, 0.21509975])
        np.testing.assert_almost_equal(
            RGB_to_RGB(RGB_i, aces_2065_1_colourspace, sRGB_colourspace),
            RGB_o,
            decimal=7)

        RGB_i = np.tile(RGB_i, (6, 1))
        RGB_o = np.tile(RGB_o, (6, 1))
        np.testing.assert_almost_equal(
            RGB_to_RGB(RGB_i, aces_2065_1_colourspace, sRGB_colourspace),
            RGB_o,
            decimal=7)

        RGB_i = np.reshape(RGB_i, (2, 3, 3))
        RGB_o = np.reshape(RGB_o, (2, 3, 3))
        np.testing.assert_almost_equal(
            RGB_to_RGB(RGB_i, aces_2065_1_colourspace, sRGB_colourspace),
            RGB_o,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.RGB_to_RGB` definition nan support.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES.get('ACES2065-1')
        sRGB_colourspace = RGB_COLOURSPACES.get('sRGB')

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_RGB(RGB, aces_2065_1_colourspace, sRGB_colourspace)


if __name__ == '__main__':
    unittest.main()
