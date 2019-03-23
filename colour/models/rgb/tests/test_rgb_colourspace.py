# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.models.rgb.rgb_colourspace` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import pickle
import re
import six
import textwrap
import unittest
from copy import deepcopy
from itertools import permutations

from colour.models import (
    RGB_COLOURSPACES, RGB_Colourspace, XYZ_to_RGB, RGB_to_XYZ,
    RGB_to_RGB_matrix, RGB_to_RGB, chromatically_adapted_primaries,
    normalised_primary_matrix, oetf_sRGB, oetf_reverse_sRGB)
from colour.utilities import domain_range_scale, ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'TestRGB_COLOURSPACES', 'TestRGB_Colourspace', 'TestXYZ_to_RGB',
    'TestRGB_to_XYZ', 'TestRGB_to_RGB_matrix', 'TestRGB_to_RGB'
]


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

        XYZ_r = np.array([0.5, 0.5, 0.5]).reshape([3, 1])
        for colourspace in RGB_COLOURSPACES.values():
            # Instantiation transformation matrices.
            if colourspace.name in ('ProPhoto RGB', 'ERIMM RGB', 'RIMM RGB',
                                    'ROMM RGB'):
                tolerance = 1e-3
            elif colourspace.name in ('sRGB', 'DJI D-Gamut'):
                tolerance = 1e-4
            elif colourspace.name in ('Adobe RGB (1998)', ):
                tolerance = 1e-5
            elif colourspace.name in ('ALEXA Wide Gamut', 'V-Gamut',
                                      'REDWideGamutRGB'):
                tolerance = 1e-6
            else:
                tolerance = 1e-7

            M = normalised_primary_matrix(colourspace.primaries,
                                          colourspace.whitepoint)

            np.testing.assert_allclose(
                colourspace.RGB_to_XYZ_matrix,
                M,
                rtol=tolerance,
                atol=tolerance,
                verbose=False)

            RGB = np.dot(colourspace.XYZ_to_RGB_matrix, XYZ_r)
            XYZ = np.dot(colourspace.RGB_to_XYZ_matrix, RGB)
            np.testing.assert_allclose(
                XYZ_r, XYZ, rtol=tolerance, atol=tolerance, verbose=False)

            # Derived transformation matrices.
            colourspace = deepcopy(colourspace)
            colourspace.use_derived_transformation_matrices(True)
            RGB = np.dot(colourspace.XYZ_to_RGB_matrix, XYZ_r)
            XYZ = np.dot(colourspace.RGB_to_XYZ_matrix, RGB)
            np.testing.assert_almost_equal(XYZ_r, XYZ, decimal=7)

    def test_cctf(self):
        """
        Tests colour component transfer functions from the
        :attr:`colour.models.rgb.rgb_colourspace.RGB_COLOURSPACES` attribute
        colourspace models.
        """

        ignored_colourspaces = ('ACESproxy', 'DJI D-Gamut')

        samples = np.hstack(
            [np.linspace(0, 1, 1e5),
             np.linspace(0, 65504, 65504 * 10)])

        for colourspace in RGB_COLOURSPACES.values():
            if colourspace.name in ignored_colourspaces:
                continue

            encoding_cctf_s = colourspace.encoding_cctf(samples)
            decoding_cctf_s = colourspace.decoding_cctf(encoding_cctf_s)

            np.testing.assert_almost_equal(samples, decoding_cctf_s, decimal=7)

    def test_n_dimensional_cctf(self):
        """
        Tests colour component transfer functions from the
        :attr:`colour.models.rgb.rgb_colourspace.RGB_COLOURSPACES` attribute
        colourspace models n-dimensional arrays support.
        """

        for colourspace in RGB_COLOURSPACES.values():
            if colourspace.name in ('DJI D-Gamut', ):
                decimal = 6
            else:
                decimal = 7

            value_encoding_cctf = 0.5
            value_decoding_cctf = colourspace.decoding_cctf(
                colourspace.encoding_cctf(value_encoding_cctf))
            np.testing.assert_almost_equal(
                value_encoding_cctf, value_decoding_cctf, decimal=decimal)

            value_encoding_cctf = np.tile(value_encoding_cctf, 6)
            value_decoding_cctf = np.tile(value_decoding_cctf, 6)
            np.testing.assert_almost_equal(
                value_encoding_cctf, value_decoding_cctf, decimal=decimal)

            value_encoding_cctf = np.reshape(value_encoding_cctf, (3, 2))
            value_decoding_cctf = np.reshape(value_decoding_cctf, (3, 2))
            np.testing.assert_almost_equal(
                value_encoding_cctf, value_decoding_cctf, decimal=decimal)

            value_encoding_cctf = np.reshape(value_encoding_cctf, (3, 2, 1))
            value_decoding_cctf = np.reshape(value_decoding_cctf, (3, 2, 1))
            np.testing.assert_almost_equal(
                value_encoding_cctf, value_decoding_cctf, decimal=decimal)

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

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        p = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
        whitepoint = np.array([0.32168, 0.33767])
        RGB_to_XYZ_matrix = np.identity(3)
        XYZ_to_RGB_matrix = np.identity(3)
        self._colourspace = RGB_Colourspace(
            'RGB Colourspace', p, whitepoint, 'ACES', RGB_to_XYZ_matrix,
            XYZ_to_RGB_matrix, lambda x: x, lambda x: x)

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('name', 'primaries', 'whitepoint',
                               'whitepoint_name', 'RGB_to_XYZ_matrix',
                               'XYZ_to_RGB_matrix', 'encoding_cctf',
                               'decoding_cctf',
                               'use_derived_RGB_to_XYZ_matrix',
                               'use_derived_XYZ_to_RGB_matrix')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(RGB_Colourspace))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', '__repr__',
                            'use_derived_transformation_matrices',
                            'chromatically_adapt', 'copy')

        for method in required_methods:
            self.assertIn(method, dir(RGB_Colourspace))

    def test__str__(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_Colourspace.__str__`
        method.
        """

        # Skipping unit test on Python 2.7.
        if six.PY2:
            return

        self.assertEqual(
            re.sub(' at 0x\\w+>', '', str(self._colourspace)),
            textwrap.dedent("""
    RGB Colourspace
    ---------------

    Primaries          : [[  7.34700000e-01   2.65300000e-01]
                          [  0.00000000e+00   1.00000000e+00]
                          [  1.00000000e-04  -7.70000000e-02]]
    Whitepoint         : [ 0.32168  0.33767]
    Whitepoint Name    : ACES
    Encoding CCTF      : <function TestRGB_Colourspace.setUp.<locals>.<lambda>
    Decoding CCTF      : <function TestRGB_Colourspace.setUp.<locals>.<lambda>
    NPM                : [[ 1.  0.  0.]
                          [ 0.  1.  0.]
                          [ 0.  0.  1.]]
    NPM -1             : [[ 1.  0.  0.]
                          [ 0.  1.  0.]
                          [ 0.  0.  1.]]
    Derived NPM        : [[  9.52552396e-01   0.00000000e+00   9.36786317e-05]
                          [  3.43966450e-01   7.28166097e-01  -7.21325464e-02]
                          [  0.00000000e+00   0.00000000e+00   1.00882518e+00]]
    Derived NPM -1     : [[  1.04981102e+00   0.00000000e+00  -9.74845406e-05]
                          [ -4.95903023e-01   1.37331305e+00   9.82400361e-02]
                          [  0.00000000e+00   0.00000000e+00   9.91252018e-01]]
    Use Derived NPM    : False
    Use Derived NPM -1 : False""")[1:])

    def test__repr__(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_Colourspace.\
__repr__` method.
        """

        # Skipping unit test on Python 2.7.
        if six.PY2:
            return

        self.assertEqual(
            re.sub(' at 0x\\w+>', '', repr(self._colourspace)),
            textwrap.dedent("""
        RGB_Colourspace(RGB Colourspace,
                        [[  7.34700000e-01,   2.65300000e-01],
                         [  0.00000000e+00,   1.00000000e+00],
                         [  1.00000000e-04,  -7.70000000e-02]],
                        [ 0.32168,  0.33767],
                        ACES,
                        [[ 1.,  0.,  0.],
                         [ 0.,  1.,  0.],
                         [ 0.,  0.,  1.]],
                        [[ 1.,  0.,  0.],
                         [ 0.,  1.,  0.],
                         [ 0.,  0.,  1.]],
                        <function TestRGB_Colourspace.setUp.<locals>.<lambda>,
                        <function TestRGB_Colourspace.setUp.<locals>.<lambda>,
                        False,
                        False)""")[1:])

    def test_use_derived_transformation_matrices(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_Colourspace.\
use_derived_transformation_matrices` method.
        """

        np.testing.assert_array_equal(self._colourspace.RGB_to_XYZ_matrix,
                                      np.identity(3))
        np.testing.assert_array_equal(self._colourspace.XYZ_to_RGB_matrix,
                                      np.identity(3))

        self.assertTrue(
            self._colourspace.use_derived_transformation_matrices())

        np.testing.assert_almost_equal(
            self._colourspace.RGB_to_XYZ_matrix,
            np.array([
                [0.95255240, 0.00000000, 0.00009368],
                [0.34396645, 0.72816610, -0.07213255],
                [0.00000000, 0.00000000, 1.00882518],
            ]),
            decimal=7)
        np.testing.assert_almost_equal(
            self._colourspace.XYZ_to_RGB_matrix,
            np.array([
                [1.04981102, 0.00000000, -0.00009748],
                [-0.49590302, 1.37331305, 0.09824004],
                [0.00000000, 0.00000000, 0.99125202],
            ]),
            decimal=7)

        self._colourspace.use_derived_RGB_to_XYZ_matrix = False
        np.testing.assert_array_equal(self._colourspace.RGB_to_XYZ_matrix,
                                      np.identity(3))
        self._colourspace.use_derived_XYZ_to_RGB_matrix = False
        np.testing.assert_array_equal(self._colourspace.XYZ_to_RGB_matrix,
                                      np.identity(3))

    def test_chromatically_adapt(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_Colourspace.\
chromatically_adapt` method.
        """

        whitepoint_t = np.array([0.31270, 0.32900])
        colourspace = self._colourspace.chromatically_adapt(
            whitepoint_t, 'D50', 'Bradford')

        np.testing.assert_array_almost_equal(
            colourspace.primaries,
            np.array([
                [0.73485524, 0.26422533],
                [-0.00617091, 1.01131496],
                [0.01596756, -0.06423550],
            ]),
            decimal=7)
        np.testing.assert_array_almost_equal(
            colourspace.whitepoint, whitepoint_t, decimal=7)

        self.assertEqual(colourspace.whitepoint_name, 'D50')

        np.testing.assert_array_almost_equal(
            colourspace.primaries,
            chromatically_adapted_primaries(self._colourspace.primaries,
                                            self._colourspace.whitepoint,
                                            whitepoint_t, 'Bradford'),
            decimal=7)

        np.testing.assert_array_almost_equal(
            colourspace.RGB_to_XYZ_matrix,
            normalised_primary_matrix(colourspace.primaries,
                                      colourspace.whitepoint),
            decimal=7)

        np.testing.assert_array_almost_equal(
            colourspace.XYZ_to_RGB_matrix,
            np.linalg.inv(
                normalised_primary_matrix(colourspace.primaries,
                                          colourspace.whitepoint)),
            decimal=7)

    def test_copy(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_Colourspace.copy`
        method.
        """

        self.assertIsNot(self._colourspace.copy(), self)


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
                np.array([0.21638819, 0.12570000, 0.03847493]),
                np.array([0.34570, 0.35850]), np.array([0.31270, 0.32900]),
                np.array([
                    [3.24062548, -1.53720797, -0.49862860],
                    [-0.96893071, 1.87575606, 0.04151752],
                    [0.05571012, -0.20402105, 1.05699594],
                ]), 'Bradford', oetf_sRGB),
            np.array([0.70556599, 0.19109268, 0.22340812]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_RGB(
                np.array([0.21638819, 0.12570000, 0.03847493]),
                np.array([0.34570, 0.35850]), np.array([0.31270, 0.32900]),
                np.array([
                    [3.24062548, -1.53720797, -0.49862860],
                    [-0.96893071, 1.87575606, 0.04151752],
                    [0.05571012, -0.20402105, 1.05699594],
                ]), None, oetf_sRGB),
            np.array([0.72794579, 0.18180021, 0.17951580]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_RGB(
                np.array([0.21638819, 0.12570000, 0.03847493]),
                np.array([0.34570, 0.35850]), np.array([0.32168, 0.33767]),
                np.array([
                    [1.04981102, 0.00000000, -0.00009748],
                    [-0.49590302, 1.37331305, 0.09824004],
                    [0.00000000, 0.00000000, 0.99125202],
                ])),
            np.array([0.21959099, 0.06985815, 0.04703704]),
            decimal=7)

        np.testing.assert_almost_equal(
            XYZ_to_RGB(
                np.array([0.21638819, 0.12570000, 0.03847493]),
                np.array([0.34570, 0.35850]),
                np.array([0.31270, 0.32900, 1.00000]),
                np.array([
                    [3.24062548, -1.53720797, -0.49862860],
                    [-0.96893071, 1.87575606, 0.04151752],
                    [0.05571012, -0.20402105, 1.05699594],
                ])),
            np.array([0.45620801, 0.03079991, 0.04091883]),
            decimal=7)

    def test_n_dimensional_XYZ_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.XYZ_to_RGB` definition
        n-dimensions support.
        """

        XYZ = np.array([0.21638819, 0.12570000, 0.03847493])
        W_R = np.array([0.34570, 0.35850])
        W_T = np.array([0.31270, 0.32900])
        M = np.array([
            [3.24062548, -1.53720797, -0.49862860],
            [-0.96893071, 1.87575606, 0.04151752],
            [0.05571012, -0.20402105, 1.05699594],
        ])
        RGB = np.array([0.70556599, 0.19109268, 0.22340812])
        np.testing.assert_almost_equal(
            XYZ_to_RGB(XYZ, W_R, W_T, M, 'Bradford', oetf_sRGB),
            RGB,
            decimal=7)

        XYZ = np.tile(XYZ, (6, 1))
        RGB = np.tile(RGB, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_RGB(XYZ, W_R, W_T, M, 'Bradford', oetf_sRGB),
            RGB,
            decimal=7)

        W_R = np.tile(W_R, (6, 1))
        W_T = np.tile(W_T, (6, 1))
        np.testing.assert_almost_equal(
            XYZ_to_RGB(XYZ, W_R, W_T, M, 'Bradford', oetf_sRGB),
            RGB,
            decimal=7)

        XYZ = np.reshape(XYZ, (2, 3, 3))
        W_R = np.reshape(W_R, (2, 3, 2))
        W_T = np.reshape(W_T, (2, 3, 2))
        RGB = np.reshape(RGB, (2, 3, 3))
        np.testing.assert_almost_equal(
            XYZ_to_RGB(XYZ, W_R, W_T, M, 'Bradford', oetf_sRGB),
            RGB,
            decimal=7)

    def test_domain_range_scale_XYZ_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.XYZ_to_RGB` definition
        domain and range scale support.
        """

        XYZ = np.array([0.21638819, 0.12570000, 0.03847493])
        W_R = np.array([0.34570, 0.35850])
        W_T = np.array([0.31270, 0.32900])
        M = np.array([
            [3.24062548, -1.53720797, -0.49862860],
            [-0.96893071, 1.87575606, 0.04151752],
            [0.05571012, -0.20402105, 1.05699594],
        ])
        RGB = XYZ_to_RGB(XYZ, W_R, W_T, M)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    XYZ_to_RGB(XYZ * factor, W_R, W_T, M),
                    RGB * factor,
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
            M = np.vstack([case, case, case]).reshape([3, 3])
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
                np.array([0.70556599, 0.19109268, 0.22340812]),
                np.array([0.31270, 0.32900]), np.array([0.34570, 0.35850]),
                np.array([
                    [0.41240000, 0.35760000, 0.18050000],
                    [0.21260000, 0.71520000, 0.07220000],
                    [0.01930000, 0.11920000, 0.95050000],
                ]), 'Bradford', oetf_reverse_sRGB),
            np.array([0.21638819, 0.12570000, 0.03847493]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_XYZ(
                np.array([0.72794579, 0.18180021, 0.17951580]),
                np.array([0.31270, 0.32900]), np.array([0.34570, 0.35850]),
                np.array([
                    [0.41240000, 0.35760000, 0.18050000],
                    [0.21260000, 0.71520000, 0.07220000],
                    [0.01930000, 0.11920000, 0.95050000],
                ]), None, oetf_reverse_sRGB),
            np.array([0.21638819, 0.12570000, 0.03847493]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_XYZ(
                np.array([0.21959099, 0.06985815, 0.04703704]),
                np.array([0.32168, 0.33767]), np.array([0.34570, 0.35850]),
                np.array([
                    [0.95255240, 0.00000000, 0.00009368],
                    [0.34396645, 0.72816610, -0.07213255],
                    [0.00000000, 0.00000000, 1.00882518],
                ])),
            np.array([0.21638819, 0.12570000, 0.03847493]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_XYZ(
                np.array([0.45620801, 0.03079991, 0.04091883]),
                np.array([0.31270, 0.32900, 1.00000]),
                np.array([0.34570, 0.35850]),
                np.array([
                    [0.41240000, 0.35760000, 0.18050000],
                    [0.21260000, 0.71520000, 0.07220000],
                    [0.01930000, 0.11920000, 0.95050000],
                ])),
            np.array([0.21638819, 0.12570000, 0.03847493]),
            decimal=7)

    def test_n_dimensional_RGB_to_XYZ(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_to_XYZ` definition
        n-dimensions support.
        """

        RGB = np.array([0.70556599, 0.19109268, 0.22340812])
        W_R = np.array([0.31270, 0.32900])
        W_T = np.array([0.34570, 0.35850])
        M = np.array([
            [0.41240000, 0.35760000, 0.18050000],
            [0.21260000, 0.71520000, 0.07220000],
            [0.01930000, 0.11920000, 0.95050000],
        ])
        XYZ = np.array([0.21638819, 0.12570000, 0.03847493])
        np.testing.assert_almost_equal(
            RGB_to_XYZ(RGB, W_R, W_T, M, 'Bradford', oetf_reverse_sRGB),
            XYZ,
            decimal=7)

        RGB = np.tile(RGB, (6, 1))
        XYZ = np.tile(XYZ, (6, 1))
        np.testing.assert_almost_equal(
            RGB_to_XYZ(RGB, W_R, W_T, M, 'Bradford', oetf_reverse_sRGB),
            XYZ,
            decimal=7)

        W_R = np.tile(W_R, (6, 1))
        W_T = np.tile(W_T, (6, 1))
        np.testing.assert_almost_equal(
            RGB_to_XYZ(RGB, W_R, W_T, M, 'Bradford', oetf_reverse_sRGB),
            XYZ,
            decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        W_R = np.reshape(W_R, (2, 3, 2))
        W_T = np.reshape(W_T, (2, 3, 2))
        XYZ = np.reshape(XYZ, (2, 3, 3))
        np.testing.assert_almost_equal(
            RGB_to_XYZ(RGB, W_R, W_T, M, 'Bradford', oetf_reverse_sRGB),
            XYZ,
            decimal=7)

    def test_domain_range_scale_XYZ_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_to_XYZ` definition
        domain and range scale support.
        """

        RGB = np.array([0.45620801, 0.03079991, 0.04091883])
        W_R = np.array([0.31270, 0.32900])
        W_T = np.array([0.34570, 0.35850])
        M = np.array([
            [3.24062548, -1.53720797, -0.49862860],
            [-0.96893071, 1.87575606, 0.04151752],
            [0.05571012, -0.20402105, 1.05699594],
        ])
        XYZ = RGB_to_XYZ(RGB, W_R, W_T, M)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    RGB_to_XYZ(RGB * factor, W_R, W_T, M),
                    XYZ * factor,
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
            M = np.vstack([case, case, case]).reshape((3, 3))
            RGB_to_XYZ(RGB, W_R, W_T, M)


class TestRGB_to_RGB_matrix(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB_matrix`
    definition unit tests methods.
    """

    def test_RGB_to_RGB_matrix(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB_matrix`
        definition.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES['ACES2065-1']
        aces_cg_colourspace = RGB_COLOURSPACES['ACEScg']
        sRGB_colourspace = RGB_COLOURSPACES['sRGB']

        np.testing.assert_almost_equal(
            RGB_to_RGB_matrix(aces_2065_1_colourspace, sRGB_colourspace),
            np.array([
                [2.52164943, -1.13688855, -0.38491759],
                [-0.27521355, 1.36970515, -0.09439245],
                [-0.01592501, -0.14780637, 1.16380582],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB_matrix(sRGB_colourspace, aces_2065_1_colourspace),
            np.array([
                [0.43958564, 0.38392940, 0.17653274],
                [0.08953957, 0.81474984, 0.09568361],
                [0.01738718, 0.10873911, 0.87382059],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB_matrix(aces_2065_1_colourspace, aces_cg_colourspace,
                              'Bradford'),
            np.array([
                [1.45143932, -0.23651075, -0.21492857],
                [-0.07655377, 1.17622970, -0.09967593],
                [0.00831615, -0.00603245, 0.99771630],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB_matrix(aces_2065_1_colourspace, sRGB_colourspace,
                              'Bradford'),
            np.array([
                [2.52140089, -1.13399575, -0.38756186],
                [-0.27621406, 1.37259557, -0.09628236],
                [-0.01532020, -0.15299256, 1.16838720],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB_matrix(aces_2065_1_colourspace, sRGB_colourspace, None),
            np.array([
                [2.55809607, -1.11933692, -0.39181451],
                [-0.27771575, 1.36589396, -0.09353075],
                [-0.01711199, -0.14854588, 1.08104848],
            ]),
            decimal=7)


class TestRGB_to_RGB(unittest.TestCase):
    """
    Defines :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition
    unit tests methods.
    """

    def test_RGB_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES['ACES2065-1']
        sRGB_colourspace = RGB_COLOURSPACES['sRGB']

        np.testing.assert_almost_equal(
            RGB_to_RGB(
                np.array([0.21931722, 0.06950287, 0.04694832]),
                aces_2065_1_colourspace, sRGB_colourspace),
            np.array([0.45595289, 0.03040780, 0.04087313]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB(
                np.array([0.45595571, 0.03039702, 0.04087245]),
                sRGB_colourspace, aces_2065_1_colourspace),
            np.array([0.21931722, 0.06950287, 0.04694832]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB(
                np.array([0.21931722, 0.06950287, 0.04694832]),
                aces_2065_1_colourspace, sRGB_colourspace, 'Bradford'),
            np.array([0.45597530, 0.03030054, 0.04086041]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB(
                np.array([0.21931722, 0.06950287, 0.04694832]),
                aces_2065_1_colourspace, sRGB_colourspace, None),
            np.array([0.46484236, 0.02963459, 0.03667609]),
            decimal=7)

        aces_cg_colourspace = RGB_COLOURSPACES['ACEScg']
        aces_cc_colourspace = RGB_COLOURSPACES['ACEScc']

        np.testing.assert_almost_equal(
            RGB_to_RGB(
                np.array([0.21931722, 0.06950287, 0.04694832]),
                aces_cg_colourspace,
                aces_cc_colourspace,
                apply_decoding_cctf=True,
                apply_encoding_cctf=True),
            np.array([0.42985679, 0.33522924, 0.30292336]),
            decimal=7)

        np.testing.assert_almost_equal(
            RGB_to_RGB(
                np.array([0.46956438, 0.48137533, 0.43788601]),
                aces_cc_colourspace,
                sRGB_colourspace,
                apply_decoding_cctf=True,
                apply_encoding_cctf=True),
            np.array([0.60983062, 0.67896356, 0.50435764]),
            decimal=7)

    def test_n_dimensional_RGB_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition
        n-dimensions support.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES['ACES2065-1']
        sRGB_colourspace = RGB_COLOURSPACES['sRGB']
        RGB_i = np.array([0.21931722, 0.06950287, 0.04694832])
        RGB_o = np.array([0.45595289, 0.03040780, 0.04087313])
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

    def test_domain_range_scale_XYZ_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition
        domain and range scale support.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES['ACES2065-1']
        sRGB_colourspace = RGB_COLOURSPACES['sRGB']
        RGB_i = np.array([0.21931722, 0.06950287, 0.04694832])
        RGB_o = RGB_to_RGB(RGB_i, aces_2065_1_colourspace, sRGB_colourspace)

        d_r = (('reference', 1), (1, 1), (100, 100))
        for scale, factor in d_r:
            with domain_range_scale(scale):
                np.testing.assert_almost_equal(
                    RGB_to_RGB(RGB_i * factor, aces_2065_1_colourspace,
                               sRGB_colourspace),
                    RGB_o * factor,
                    decimal=7)

    @ignore_numpy_errors
    def test_nan_RGB_to_RGB(self):
        """
        Tests :func:`colour.models.rgb.rgb_colourspace.RGB_to_RGB` definition
        nan support.
        """

        aces_2065_1_colourspace = RGB_COLOURSPACES['ACES2065-1']
        sRGB_colourspace = RGB_COLOURSPACES['sRGB']

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            RGB = np.array(case)
            RGB_to_RGB(RGB, aces_2065_1_colourspace, sRGB_colourspace)


if __name__ == '__main__':
    unittest.main()
