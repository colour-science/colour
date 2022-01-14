# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.characterisation.correction`
module.
"""

from __future__ import annotations

import numpy as np
import unittest
from itertools import permutations
from numpy.linalg import LinAlgError

from colour.characterisation.correction import (
    matrix_augmented_Cheung2004,
    polynomial_expansion_Finlayson2015,
    polynomial_expansion_Vandermonde,
    matrix_colour_correction_Cheung2004,
    matrix_colour_correction_Finlayson2015,
    matrix_colour_correction_Vandermonde,
    colour_correction_Cheung2004,
    colour_correction_Finlayson2015,
    colour_correction_Vandermonde,
)
from colour.hints import NDArray
from colour.utilities import ignore_numpy_errors

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'MATRIX_TEST',
    'MATRIX_REFERENCE',
    'TestMatrixAugmentedCheung2004',
    'TestPolynomialExpansionFinlayson2015',
    'TestPolynomialExpansionVandermonde',
    'TestMatrixColourCorrectionCheung2004',
    'TestMatrixColourCorrectionFinlayson2015',
    'TestMatrixColourCorrectionVandermonde',
    'TestColourCorrectionCheung2004',
    'TestColourCorrectionFinlayson2015',
    'TestColourCorrectionVandermonde',
]

MATRIX_TEST: NDArray = np.array([
    [0.17224810, 0.09170660, 0.06416938],
    [0.49189645, 0.27802050, 0.21923399],
    [0.10999751, 0.18658946, 0.29938611],
    [0.11666120, 0.14327905, 0.05713804],
    [0.18988879, 0.18227649, 0.36056247],
    [0.12501329, 0.42223442, 0.37027445],
    [0.64785606, 0.22396782, 0.03365194],
    [0.06761093, 0.11076896, 0.39779139],
    [0.49101797, 0.09448929, 0.11623839],
    [0.11622386, 0.04425753, 0.14469986],
    [0.36867946, 0.44545230, 0.06028681],
    [0.61632937, 0.32323906, 0.02437089],
    [0.03016472, 0.06153243, 0.29014596],
    [0.11103655, 0.30553067, 0.08149137],
    [0.41162190, 0.05816656, 0.04845934],
    [0.73339206, 0.53075188, 0.02475212],
    [0.47347718, 0.08834792, 0.30310315],
    [0.00000000, 0.25187016, 0.35062450],
    [0.76809639, 0.78486240, 0.77808297],
    [0.53822392, 0.54307997, 0.54710883],
    [0.35458526, 0.35318419, 0.35524431],
    [0.17976704, 0.18000531, 0.17991488],
    [0.09351417, 0.09510603, 0.09675027],
    [0.03405071, 0.03295077, 0.03702047],
])

MATRIX_REFERENCE: NDArray = np.array([
    [0.15579559, 0.09715755, 0.07514556],
    [0.39113140, 0.25943419, 0.21266708],
    [0.12824821, 0.18463570, 0.31508023],
    [0.12028974, 0.13455659, 0.07408400],
    [0.19368988, 0.21158946, 0.37955964],
    [0.19957424, 0.36085439, 0.40678123],
    [0.48896605, 0.20691688, 0.05816533],
    [0.09775522, 0.16710693, 0.47147724],
    [0.39358649, 0.12233400, 0.10526425],
    [0.10780332, 0.07258529, 0.16151473],
    [0.27502671, 0.34705454, 0.09728099],
    [0.43980441, 0.26880559, 0.05430533],
    [0.05887212, 0.11126272, 0.38552469],
    [0.12705825, 0.25787860, 0.13566464],
    [0.35612929, 0.07933258, 0.05118732],
    [0.48131976, 0.42082843, 0.07120612],
    [0.34665585, 0.15170714, 0.24969804],
    [0.08261116, 0.24588716, 0.48707733],
    [0.66054904, 0.65941137, 0.66376412],
    [0.48051509, 0.47870296, 0.48230082],
    [0.33045354, 0.32904184, 0.33228886],
    [0.18001305, 0.17978567, 0.18004416],
    [0.10283975, 0.10424680, 0.10384975],
    [0.04742204, 0.04772203, 0.04914226],
])


class TestMatrixAugmentedCheung2004(unittest.TestCase):
    """
    Defines :func:`colour.characterisation.correction.\
matrix_augmented_Cheung2004` definition unit tests methods.
    """

    def test_matrix_augmented_Cheung2004(self):
        """
        Tests :func:`colour.characterisation.correction.\
matrix_augmented_Cheung2004` definition.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])

        polynomials = [
            np.array([0.17224810, 0.09170660, 0.06416938]),
            np.array(
                [0.17224810, 0.09170660, 0.06416938, 0.00101364, 1.00000000]),
            np.array([
                0.17224810, 0.09170660, 0.06416938, 0.01579629, 0.01105305,
                0.00588476, 1.00000000
            ]),
            np.array([
                0.17224810, 0.09170660, 0.06416938, 0.01579629, 0.01105305,
                0.00588476, 0.00101364, 1.00000000
            ]),
            np.array([
                0.17224810, 0.09170660, 0.06416938, 0.01579629, 0.01105305,
                0.00588476, 0.02966941, 0.00841010, 0.00411771, 1.00000000
            ]),
            np.array([
                0.17224810, 0.09170660, 0.06416938, 0.01579629, 0.01105305,
                0.00588476, 0.02966941, 0.00841010, 0.00411771, 0.00101364,
                1.00000000
            ]),
            np.array([
                0.17224810, 0.09170660, 0.06416938, 0.01579629, 0.01105305,
                0.00588476, 0.02966941, 0.00841010, 0.00411771, 0.00101364,
                0.00511050, 0.00077126, 0.00026423, 1.00000000
            ]),
            np.array([
                0.17224810, 0.09170660, 0.06416938, 0.01579629, 0.01105305,
                0.00588476, 0.02966941, 0.00841010, 0.00411771, 0.00101364,
                0.00272088, 0.00053967, 0.00070927, 0.00511050, 0.00077126,
                0.00026423
            ]),
            np.array([
                0.17224810, 0.09170660, 0.06416938, 0.01579629, 0.01105305,
                0.00588476, 0.02966941, 0.00841010, 0.00411771, 0.00101364,
                0.00272088, 0.00053967, 0.00070927, 0.00511050, 0.00077126,
                0.00026423, 1.00000000
            ]),
            np.array([
                0.17224810, 0.09170660, 0.06416938, 0.01579629, 0.01105305,
                0.00588476, 0.02966941, 0.00841010, 0.00411771, 0.00101364,
                0.00272088, 0.00053967, 0.00070927, 0.00190387, 0.00144862,
                0.00037762, 0.00511050, 0.00077126, 0.00026423
            ]),
            np.array([
                0.17224810, 0.09170660, 0.06416938, 0.01579629, 0.01105305,
                0.00588476, 0.02966941, 0.00841010, 0.00411771, 0.00101364,
                0.00272088, 0.00053967, 0.00070927, 0.00190387, 0.00144862,
                0.00037762, 0.00511050, 0.00077126, 0.00026423, 1.00000000
            ]),
            np.array([
                0.17224810, 0.09170660, 0.06416938, 0.01579629, 0.01105305,
                0.00588476, 0.02966941, 0.00841010, 0.00411771, 0.00101364,
                0.00272088, 0.00053967, 0.00070927, 0.00190387, 0.00144862,
                0.00037762, 0.00511050, 0.00077126, 0.00026423, 0.00017460,
                0.00009296, 0.00006504
            ]),
        ]

        for i, terms in enumerate([3, 5, 7, 8, 10, 11, 14, 16, 17, 19, 20,
                                   22]):
            np.testing.assert_almost_equal(
                matrix_augmented_Cheung2004(RGB, terms),
                polynomials[i],
                decimal=7)

    def test_raise_exception_matrix_augmented_Cheung2004(self):
        """
        Tests :func:`colour.characterisation.correction.\
matrix_augmented_Cheung2004` definition raised exception.
        """

        self.assertRaises(ValueError, matrix_augmented_Cheung2004,
                          np.array([0.17224810, 0.09170660, 0.06416938]), 4)

    @ignore_numpy_errors
    def test_nan_matrix_augmented_Cheung2004(self):
        """
        Tests :func:`colour.characterisation.correction.\
matrix_augmented_Cheung2004` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            matrix_augmented_Cheung2004(case)


class TestPolynomialExpansionFinlayson2015(unittest.TestCase):
    """
    Defines :func:`colour.characterisation.correction.\
polynomial_expansion_Finlayson2015` definition unit tests methods.
    """

    def test_polynomial_expansion_Finlayson2015(self):
        """
        Tests :func:`colour.characterisation.correction.\
polynomial_expansion_Finlayson2015` definition.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])

        polynomials = [
            [
                np.array([0.17224810, 0.09170660, 0.06416938]),
                np.array([0.17224810, 0.09170660, 0.06416938])
            ],
            [
                np.array([
                    0.17224810, 0.09170660, 0.06416938, 0.02966941, 0.00841010,
                    0.00411771, 0.01579629, 0.00588476, 0.01105305
                ]),
                np.array([
                    0.17224810, 0.09170660, 0.06416938, 0.12568328, 0.07671216,
                    0.10513350
                ])
            ],
            [
                np.array([
                    0.17224810, 0.09170660, 0.06416938, 0.02966941, 0.00841010,
                    0.00411771, 0.01579629, 0.00588476, 0.01105305, 0.00511050,
                    0.00077126, 0.00026423, 0.00144862, 0.00037762, 0.00070927,
                    0.00272088, 0.00053967, 0.00190387, 0.00101364
                ]),
                np.array([
                    0.17224810, 0.09170660, 0.06416938, 0.12568328, 0.07671216,
                    0.10513350, 0.11314930, 0.07228010, 0.08918053, 0.13960570,
                    0.08141598, 0.12394021, 0.10045255
                ])
            ],
            [
                np.array([
                    0.17224810, 0.09170660, 0.06416938, 0.02966941, 0.00841010,
                    0.00411771, 0.01579629, 0.00588476, 0.01105305, 0.00511050,
                    0.00077126, 0.00026423, 0.00144862, 0.00037762, 0.00070927,
                    0.00272088, 0.00053967, 0.00190387, 0.00101364, 0.00088027,
                    0.00007073, 0.00001696, 0.00046867, 0.00032794, 0.00013285,
                    0.00004949, 0.00004551, 0.00002423, 0.00024952, 0.00003463,
                    0.00012217, 0.00017460, 0.00009296, 0.00006504
                ]),
                np.array([
                    0.17224810, 0.09170660, 0.06416938, 0.12568328, 0.07671216,
                    0.10513350, 0.11314930, 0.07228010, 0.08918053, 0.13960570,
                    0.08141598, 0.12394021, 0.10045255, 0.14713499, 0.13456986,
                    0.10735915, 0.08387498, 0.08213618, 0.07016104, 0.11495009,
                    0.09819082, 0.08980545
                ])
            ],
        ]

        for i in range(4):
            np.testing.assert_almost_equal(
                polynomial_expansion_Finlayson2015(RGB, i + 1, False),
                polynomials[i][0],
                decimal=7)
            np.testing.assert_almost_equal(
                polynomial_expansion_Finlayson2015(RGB, i + 1, True),
                polynomials[i][1],
                decimal=7)

    def test_raise_exception_polynomial_expansion_Finlayson2015(self):
        """
        Tests :func:`colour.characterisation.correction.\
polynomial_expansion_Finlayson2015` definition raised exception.
        """

        self.assertRaises(ValueError, polynomial_expansion_Finlayson2015,
                          np.array([0.17224810, 0.09170660, 0.06416938]), 5)

    @ignore_numpy_errors
    def test_nan_polynomial_expansion_Finlayson2015(self):
        """
        Tests :func:`colour.characterisation.correction.\
polynomial_expansion_Finlayson2015` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            polynomial_expansion_Finlayson2015(case)


class TestPolynomialExpansionVandermonde(unittest.TestCase):
    """
    Defines :func:`colour.characterisation.correction.\
polynomial_expansion_Vandermonde` definition unit tests methods.
    """

    def test_polynomial_expansion_Vandermonde(self):
        """
        Tests :func:`colour.characterisation.correction.\
polynomial_expansion_Vandermonde` definition.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])

        polynomials = [
            np.array([0.17224810, 0.09170660, 0.06416938, 1.00000000]),
            np.array([
                0.02966941, 0.00841010, 0.00411771, 0.17224810, 0.09170660,
                0.06416938, 1.00000000
            ]),
            np.array([
                0.00511050, 0.00077126, 0.00026423, 0.02966941, 0.00841010,
                0.00411771, 0.17224810, 0.09170660, 0.06416938, 1.00000000
            ]),
            np.array([
                0.00088027, 0.00007073, 0.00001696, 0.00511050, 0.00077126,
                0.00026423, 0.02966941, 0.00841010, 0.00411771, 0.17224810,
                0.09170660, 0.06416938, 1.00000000
            ]),
        ]

        for i in range(4):
            np.testing.assert_almost_equal(
                polynomial_expansion_Vandermonde(RGB, i + 1),
                polynomials[i],
                decimal=7)

    @ignore_numpy_errors
    def test_nan_polynomial_expansion_Vandermonde(self):
        """
        Tests :func:`colour.characterisation.correction.\
polynomial_expansion_Vandermonde` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = set(permutations(cases * 3, r=3))
        for case in cases:
            polynomial_expansion_Vandermonde(case)


class TestMatrixColourCorrectionCheung2004(unittest.TestCase):
    """
    Defines :func:`colour.characterisation.correction.\
matrix_colour_correction_Cheung2004` definition unit tests methods.
    """

    def test_matrix_colour_correction_Cheung2004(self):
        """
        Tests :func:`colour.characterisation.correction.\
matrix_colour_correction_Cheung2004` definition.
        """

        np.testing.assert_almost_equal(
            matrix_colour_correction_Cheung2004(MATRIX_TEST, MATRIX_REFERENCE),
            np.array([
                [0.69822661, 0.03071629, 0.16210422],
                [0.06893498, 0.67579611, 0.16430385],
                [-0.06314956, 0.09212471, 0.97134152],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            matrix_colour_correction_Cheung2004(
                MATRIX_TEST, MATRIX_REFERENCE, terms=7),
            np.array([
                [
                    0.80512769, 0.04001012, -0.01255261, -0.41056170,
                    -0.28052094, 0.68417697, 0.02251728
                ],
                [
                    0.03270288, 0.71452384, 0.17581905, -0.00897913,
                    0.04900199, -0.17162742, 0.01688472
                ],
                [
                    -0.03973098, -0.07164767, 1.16401636, 0.29017859,
                    -0.88909018, 0.26675507, 0.02345109
                ],
            ]),
            decimal=7)

    @ignore_numpy_errors
    def test_nan_matrix_colour_correction_Cheung2004(self):
        """
        Tests :func:`colour.characterisation.correction.
    matrix_colour_correction_Cheung2004` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = list(set(permutations(cases * 3, r=3)))[0:4]
        for case in cases:
            try:
                matrix_colour_correction_Cheung2004(
                    np.vstack([case, case, case]),
                    np.transpose(np.vstack([case, case, case])))
            except LinAlgError:
                pass


class TestMatrixColourCorrectionFinlayson2015(unittest.TestCase):
    """
    Defines :func:`colour.characterisation.correction.\
matrix_colour_correction_Finlayson2015` definition unit tests methods.
    """

    def test_matrix_colour_correction_Finlayson2015(self):
        """
        Tests :func:`colour.characterisation.correction.\
matrix_colour_correction_Finlayson2015` definition.
        """

        np.testing.assert_almost_equal(
            matrix_colour_correction_Finlayson2015(MATRIX_TEST,
                                                   MATRIX_REFERENCE),
            np.array([
                [0.69822661, 0.03071629, 0.16210422],
                [0.06893498, 0.67579611, 0.16430385],
                [-0.06314956, 0.09212471, 0.97134152],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            matrix_colour_correction_Finlayson2015(
                MATRIX_TEST, MATRIX_REFERENCE, degree=3),
            np.array([
                [
                    2.87796213, 9.85720054, 2.99863978, 76.97227806,
                    73.73571500, -49.37563169, -48.70879206, -47.53280959,
                    29.88241815, -39.82871801, -37.11388282, 23.30393209,
                    3.81579802
                ],
                [
                    -0.78448243, 5.63631335, 0.95306110, 14.19762287,
                    20.60124427, -18.05512861, -14.52994195, -13.10606336,
                    10.53666341, -3.63132534, -12.49672335, 8.17401039,
                    3.37995231
                ],
                [
                    -2.39092600, 10.57193455, 4.16361285, 23.41748866,
                    58.26902059, -39.39669827, -26.63805785, -35.98397757,
                    21.25508558, -4.12726077, -34.31995017, 18.72796247,
                    7.33531009
                ],
            ]),
            decimal=7)

    @ignore_numpy_errors
    def test_nan_matrix_colour_correction_Finlayson2015(self):
        """
        Tests :func:`colour.characterisation.correction.
    matrix_colour_correction_Finlayson2015` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = list(set(permutations(cases * 3, r=3)))[0:4]
        for case in cases:
            try:
                matrix_colour_correction_Finlayson2015(
                    np.vstack([case, case, case]),
                    np.transpose(np.vstack([case, case, case])))
            except LinAlgError:
                pass


class TestMatrixColourCorrectionVandermonde(unittest.TestCase):
    """
    Defines :func:`colour.characterisation.correction.\
matrix_colour_correction_Vandermonde` definition unit tests methods.
    """

    def test_matrix_colour_correction_Vandermonde(self):
        """
        Tests :func:`colour.characterisation.correction.\
matrix_colour_correction_Vandermonde` definition.
        """

        np.testing.assert_almost_equal(
            matrix_colour_correction_Vandermonde(MATRIX_TEST,
                                                 MATRIX_REFERENCE),
            np.array([
                [0.66770040, 0.02514036, 0.12745797, 0.02485425],
                [0.03155494, 0.66896825, 0.12187874, 0.03043460],
                [-0.14502258, 0.07716975, 0.87841836, 0.06666049],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            matrix_colour_correction_Vandermonde(
                MATRIX_TEST, MATRIX_REFERENCE, degree=3),
            np.array([
                [
                    -0.04328223, -1.87886146, 1.83369170, -0.10798116,
                    1.06608177, -0.87495813, 0.75525839, -0.08558123,
                    0.15919076, 0.02404598
                ],
                [
                    0.00998152, 0.44525275, -0.53192490, 0.00904507,
                    -0.41034458, 0.36173334, 0.02904178, 0.78362950,
                    0.07894900, 0.01986479
                ],
                [
                    -1.66921744, 3.62954420, -2.96789849, 2.31451409,
                    -3.10767297, 1.85975390, -0.98795093, 0.85962796,
                    0.63591240, 0.07302317
                ],
            ]),
            decimal=7)

    @ignore_numpy_errors
    def test_nan_matrix_colour_correction_Vandermonde(self):
        """
        Tests :func:`colour.characterisation.correction.
    matrix_colour_correction_Vandermonde` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = list(set(permutations(cases * 3, r=3)))[0:4]
        for case in cases:
            try:
                matrix_colour_correction_Vandermonde(
                    np.vstack([case, case, case]),
                    np.transpose(np.vstack([case, case, case])))
            except LinAlgError:
                pass


class TestColourCorrectionCheung2004(unittest.TestCase):
    """
    Defines :func:`colour.characterisation.correction.\
colour_correction_Cheung2004` definition unit tests methods.
    """

    def test_colour_correction_Cheung2004(self):
        """
        Tests :func:`colour.characterisation.correction.\
colour_correction_Cheung2004` definition.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])

        np.testing.assert_almost_equal(
            colour_correction_Cheung2004(RGB, MATRIX_TEST, MATRIX_REFERENCE),
            np.array([0.13348722, 0.08439216, 0.05990144]),
            decimal=7)

        np.testing.assert_almost_equal(
            colour_correction_Cheung2004(
                RGB, MATRIX_TEST, MATRIX_REFERENCE, terms=7),
            np.array([0.15850295, 0.09871628, 0.08105752]),
            decimal=7)

    def test_n_dimensional_colour_correction_Cheung2004(self):
        """
        Tests :func:`colour.characterisation.correction.\
colour_correction_Cheung2004` definition n-dimensional support.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])
        RGB_c = colour_correction_Cheung2004(RGB, MATRIX_TEST,
                                             MATRIX_REFERENCE)

        RGB = np.tile(RGB, (6, 1))
        RGB_c = np.tile(RGB_c, (6, 1))
        np.testing.assert_almost_equal(
            colour_correction_Cheung2004(RGB, MATRIX_TEST, MATRIX_REFERENCE),
            RGB_c,
            decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        RGB_c = np.reshape(RGB_c, (2, 3, 3))
        np.testing.assert_almost_equal(
            colour_correction_Cheung2004(RGB, MATRIX_TEST, MATRIX_REFERENCE),
            RGB_c,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_colour_correction_Cheung2004(self):
        """
        Tests :func:`colour.characterisation.correction.\
colour_correction_Cheung2004` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = list(set(permutations(cases * 3, r=3)))[0:4]
        for case in cases:
            try:
                colour_correction_Cheung2004(
                    case, np.vstack([case, case, case]),
                    np.transpose(np.vstack([case, case, case])))
            except LinAlgError:
                pass


class TestColourCorrectionFinlayson2015(unittest.TestCase):
    """
    Defines :func:`colour.characterisation.correction.\
colour_correction_Finlayson2015` definition unit tests methods.
    """

    def test_colour_correction_Finlayson2015(self):
        """
        Tests :func:`colour.characterisation.correction.\
colour_correction_Finlayson2015` definition.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])

        np.testing.assert_almost_equal(
            colour_correction_Finlayson2015(RGB, MATRIX_TEST,
                                            MATRIX_REFERENCE),
            np.array([0.13348722, 0.08439216, 0.05990144]),
            decimal=7)

        np.testing.assert_almost_equal(
            colour_correction_Finlayson2015(
                RGB, MATRIX_TEST, MATRIX_REFERENCE, degree=3),
            np.array([0.13914542, 0.08602124, 0.06422973]),
            decimal=7)

    def test_n_dimensional_colour_correction_Finlayson2015(self):
        """
        Tests :func:`colour.characterisation.correction.\
colour_correction_Finlayson2015` definition n-dimensional support.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])
        RGB_c = colour_correction_Finlayson2015(RGB, MATRIX_TEST,
                                                MATRIX_REFERENCE)

        RGB = np.tile(RGB, (6, 1))
        RGB_c = np.tile(RGB_c, (6, 1))
        np.testing.assert_almost_equal(
            colour_correction_Finlayson2015(RGB, MATRIX_TEST,
                                            MATRIX_REFERENCE),
            RGB_c,
            decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        RGB_c = np.reshape(RGB_c, (2, 3, 3))
        np.testing.assert_almost_equal(
            colour_correction_Finlayson2015(RGB, MATRIX_TEST,
                                            MATRIX_REFERENCE),
            RGB_c,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_colour_correction_Finlayson2015(self):
        """
        Tests :func:`colour.characterisation.correction.\
colour_correction_Finlayson2015` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = list(set(permutations(cases * 3, r=3)))[0:4]
        for case in cases:
            try:
                colour_correction_Finlayson2015(
                    case, np.vstack([case, case, case]),
                    np.transpose(np.vstack([case, case, case])))
            except LinAlgError:
                pass


class TestColourCorrectionVandermonde(unittest.TestCase):
    """
    Defines :func:`colour.characterisation.correction.\
colour_correction_Vandermonde` definition unit tests methods.
    """

    def test_colour_correction_Vandermonde(self):
        """
        Tests :func:`colour.characterisation.correction.\
colour_correction_Vandermonde` definition.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])

        np.testing.assert_almost_equal(
            colour_correction_Vandermonde(RGB, MATRIX_TEST, MATRIX_REFERENCE),
            np.array([0.15034881, 0.10503956, 0.10512517]),
            decimal=7)

        np.testing.assert_almost_equal(
            colour_correction_Vandermonde(
                RGB, MATRIX_TEST, MATRIX_REFERENCE, degree=3),
            np.array([0.15747814, 0.10035799, 0.06616709]),
            decimal=7)

    def test_n_dimensional_colour_correction_Vandermonde(self):
        """
        Tests :func:`colour.characterisation.correction.\
colour_correction_Vandermonde` definition n-dimensional support.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])
        RGB_c = colour_correction_Vandermonde(RGB, MATRIX_TEST,
                                              MATRIX_REFERENCE)

        RGB = np.tile(RGB, (6, 1))
        RGB_c = np.tile(RGB_c, (6, 1))
        np.testing.assert_almost_equal(
            colour_correction_Vandermonde(RGB, MATRIX_TEST, MATRIX_REFERENCE),
            RGB_c,
            decimal=7)

        RGB = np.reshape(RGB, (2, 3, 3))
        RGB_c = np.reshape(RGB_c, (2, 3, 3))
        np.testing.assert_almost_equal(
            colour_correction_Vandermonde(RGB, MATRIX_TEST, MATRIX_REFERENCE),
            RGB_c,
            decimal=7)

    @ignore_numpy_errors
    def test_nan_colour_correction_Vandermonde(self):
        """
        Tests :func:`colour.characterisation.correction.\
colour_correction_Vandermonde` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = list(set(permutations(cases * 3, r=3)))[0:4]
        for case in cases:
            try:
                colour_correction_Vandermonde(
                    case,
                    np.vstack([case, case, case]),
                    np.transpose(np.vstack([case, case, case])),
                )
            except LinAlgError:
                pass


if __name__ == '__main__':
    unittest.main()
