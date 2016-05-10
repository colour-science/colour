#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.models.rgb.rgb_colourspace` module.
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
    normalised_primary_matrix,
    oetf_sRGB,
    eotf_sRGB)
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestRGB_COLOURSPACES',
           'TestRGB_Colourspace',
           'TestXYZ_to_RGB',
           'TestRGB_to_XYZ',
           'TestRGB_to_RGB']


class TestRGB_COLOURSPACES(unittest.TestCase):
    """
    Defines :attr:`colour.models.rgb.rgb_colourspace.RGB_COLOURSPACES`
    attribute unit tests methods.
    """

    def test_transformation_matrices(self):
        """
        Tests the transformations matrices from the
        :attr:`colour.models.rgb.rgb_colourspace.RGB_COLOURSPACES` attribute
        colourspace models.
        """

        XYZ_r = np.array([0.5, 0.5, 0.5]).reshape((3, 1))
        for colourspace in RGB_COLOURSPACES.values():
            if colourspace.name == 'Adobe RGB (1998)':
                tolerance = 1e-5
            elif colourspace.name in ('ALEXA Wide Gamut RGB', 'V-Gamut'):
                tolerance = 1e-6
            elif colourspace.name == 'sRGB':
                tolerance = 1e-4
            else:
                tolerance = 1e-7

            M = normalised_primary_matrix(colourspace.primaries,
                                          colourspace.whitepoint)

            np.testing.assert_allclose(colourspace.RGB_to_XYZ_matrix,
                                       M,
                                       rtol=tolerance,
                                       atol=tolerance,
                                       verbose=False)

            RGB = np.dot(colourspace.XYZ_to_RGB_matrix, XYZ_r)
            XYZ = np.dot(colourspace.RGB_to_XYZ_matrix, RGB)
            np.testing.assert_almost_equal(XYZ_r, XYZ, decimal=7)

    def test_cctf(self):
        """
        Tests colour component transfer functions from the
        :attr:`colour.models.rgb.rgb_colourspace.RGB_COLOURSPACES` attribute
        colourspace models.
        """

        aces_proxy_colourspaces = ('ACESproxy', 'ACEScg')

        samples = np.linspace(0, 1, 1000)
        for colourspace in RGB_COLOURSPACES.values():
            encoding_cctf_s = colourspace.encoding_cctf(samples)
            decoding_cctf_s = colourspace.decoding_cctf(encoding_cctf_s)

            if colourspace.name not in aces_proxy_colourspaces:
                np.testing.assert_almost_equal(samples,
                                               decoding_cctf_s,
                                               decimal=7)
                # else:
                #     np.testing.assert_allclose(samples,
                #                                decoding_cctf_s,
                #                                rtol=0.01,
                #                                atol=0.01)

    def test_n_dimensional_cctf(self):
        """
        Tests colour component transfer functions from the
        :attr:`colour.models.rgb.rgb_colourspace.RGB_COLOURSPACES` attribute
        colourspace models n-dimensional arrays support.
        """

        for colourspace in RGB_COLOURSPACES.values():
            value_encoding_cctf = 0.5
            value_decoding_cctf = colourspace.decoding_cctf(
                colourspace.encoding_cctf(value_encoding_cctf))
            np.testing.assert_almost_equal(
                value_encoding_cctf,
                value_decoding_cctf,
                decimal=7)

            value_encoding_cctf = np.tile(value_encoding_cctf, 6)
            value_decoding_cctf = np.tile(value_decoding_cctf, 6)
            np.testing.assert_almost_equal(
                value_encoding_cctf,
                value_decoding_cctf,
                decimal=7)

            value_encoding_cctf = np.reshape(value_encoding_cctf, (3, 2))
            value_decoding_cctf = np.reshape(value_decoding_cctf, (3, 2))
            np.testing.assert_almost_equal(
                value_encoding_cctf,
                value_decoding_cctf,
                decimal=7)

            value_encoding_cctf = np.reshape(value_encoding_cctf, (3, 2, 1))
            value_decoding_cctf = np.reshape(value_decoding_cctf, (3, 2, 1))
            np.testing.assert_almost_equal(
                value_encoding_cctf,
                value_decoding_cctf,
                decimal=7)

    @ignore_numpy_errors
    def test_nan_cctf(self):
        """
        Tests colour component transfer functions from the
        :attr:`colour.models.rgb.rgb_colourspace.RGB_COLOURSPACES` attribute
        colourspace models nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        for colourspace in RGB_COLOURSPACES.values():
            for case in cases:
                colourspace.encoding_cctf(case)
                colourspace.decoding_cctf(case)

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
                               'encoding_cctf',
                               'decoding_cctf')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(RGB_Colourspace))


class TestXYZ_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.rgb_colourspace.XYZ_to_RGB` definition
    unit tests methods.
    """

    def test_XYZ_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.XYZ_to_RGB` definition.
        """

        np.testing.assert_almost_equal(
            XYZ_to_RGB(
                np.array([0.11518475, 0.10080000, 0.05089373]),
                np.array([0.34567, 0.35850]),
                np.array([0.31270, 0.32900]),
                np.array([[3.24100326, -1.53739899, -0.49861587],
                          [-0.96922426, 1.87592999, 0.04155422],
                          [0.05563942, -0.20401120, 1.05714897]]),
                'Bradford',
                oetf_sRGB),
            np.array([0.45293636, 0.31731850, 0.26417184]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_RGB(
                np.array([0.11518475, 0.10080000, 0.05089373]),
                np.array([0.34567, 0.35850]),
                np.array([0.32168, 0.33767]),
                np.array(
                    [[1.0498110175, 0.0000000000, -0.0000974845],
                     [-0.4959030231, 1.3733130458, 0.0982400361],
                     [0.0000000000, 0.0000000000, 0.9912520182]])),
            np.array([0.11758989, 0.08781098, 0.06184839]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_RGB(
                np.array([0.07049534, 0.10080000, 0.09558313]),
                np.array([0.34567, 0.35850]),
                np.array([0.31270, 0.32900, 0.10080]),
                np.array([[3.24100326, -1.53739899, -0.49861587],
                          [-0.96922426, 1.87592999, 0.04155422],
                          [0.05563942, -0.20401120, 1.05714897]])),
            np.array([0.00110030, 0.01282088, 0.01173622]),
            decimal=7)

    def test_n_dimensional_XYZ_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.XYZ_to_RGB` definition
        n-dimensions support.
        """

        XYZ = np.array([0.07049534, 0.10080000, 0.09558313])
        W_R = np.array([0.34567, 0.35850])
        W_T = np.array([0.31270, 0.32900])
        M = np.array([[3.24100326, -1.53739899, -0.49861587],
                      [-0.96922426, 1.87592999, 0.04155422],
                      [0.05563942, -0.20401120, 1.05714897]])
        RGB = np.array([0.01091567, 0.12719122, 0.11643074])
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
        Tests :func:`colour.models.rgb.rgb_colourspace.XYZ_to_RGB` definition
        nan support.
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
    Defines :func:`colour.models.rgb.rgb_colourspace.RGB_to_XYZ` definition
    unit tests methods.
    """

    def test_RGB_to_XYZ(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_to_XYZ` definition.
        """

        np.testing.assert_almost_equal(
            RGB_to_XYZ(
                np.array([0.45293636, 0.31731850, 0.26417184]),
                np.array([0.31270, 0.32900]),
                np.array([0.34567, 0.35850]),
                np.array(
                    [[0.41238656, 0.35759149, 0.18045049],
                     [0.21263682, 0.71518298, 0.07218020],
                     [0.01933062, 0.11919716, 0.95037259]]),
                'Bradford',
                eotf_sRGB),
            np.array([0.11518475, 0.10080000, 0.05089373]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_XYZ(
                np.array([0.11758989, 0.08781098, 0.06184839]),
                np.array([0.32168, 0.33767]),
                np.array([0.34567, 0.35850]),
                np.array(
                    [[0.9525523959, 0.0000000000, 0.0000936786],
                     [0.3439664498, 0.7281660966, -0.0721325464],
                     [0.0000000000, 0.0000000000, 1.0088251844]])),
            np.array([0.11518475, 0.10080000, 0.05089373]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_XYZ(
                np.array([0.86969452, 1.00516431, 1.41715848]),
                np.array([0.31270, 0.32900]),
                np.array([0.34567, 0.35850, 0.10080]),
                np.array([
                    [0.41238656, 0.35759149, 0.18045049],
                    [0.21263682, 0.71518298, 0.07218020],
                    [0.01933062, 0.11919716, 0.95037259]])),
            np.array([0.09756781, 0.10063048, 0.11346209]),
            decimal=7)

    def test_n_dimensional_RGB_to_XYZ(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_to_XYZ` definition
        n-dimensions support.
        """

        RGB = np.array([0.86969452, 1.00516431, 1.41715848])
        W_R = np.array([0.31270, 0.32900])
        W_T = np.array([0.34567, 0.35850])
        M = np.array([
            [0.41238656, 0.35759149, 0.18045049],
            [0.21263682, 0.71518298, 0.07218020],
            [0.01933062, 0.11919716, 0.95037259]])
        XYZ = np.array([0.96793459, 0.99831823, 1.12561593])
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
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_to_XYZ` definition
        nan support.
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
    Defines :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition
    unit tests methods.
    """

    def test_RGB_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES.get('ACES2065-1')
        sRGB_colourspace = RGB_COLOURSPACES.get('sRGB')

        np.testing.assert_almost_equal(
            RGB_to_RGB(np.array([0.35521588, 0.41000000, 0.24177934]),
                       aces_2065_1_colourspace,
                       sRGB_colourspace),
            np.array([0.33653829, 0.44097338, 0.21512063]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB(np.array([0.33653829, 0.44097338, 0.21512063]),
                       sRGB_colourspace,
                       aces_2065_1_colourspace),
            np.array([0.35521588, 0.41000000, 0.24177934]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB(np.array([0.35521588, 0.41000000, 0.24177934]),
                       aces_2065_1_colourspace,
                       sRGB_colourspace,
                       'Bradford'),
            np.array([0.33699675, 0.44134608, 0.21431681]),
            decimal=7)

    def test_n_dimensional_RGB_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition
        n-dimensions support.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES.get('ACES2065-1')
        sRGB_colourspace = RGB_COLOURSPACES.get('sRGB')
        RGB_i = np.array([0.35521588, 0.41000000, 0.24177934])
        RGB_o = np.array([0.33653829, 0.44097338, 0.21512063])
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
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition
        nan support.
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
