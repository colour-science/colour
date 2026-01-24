"""
Define the unit tests for the
:mod:`colour.characterisation.correction.finlayson2015` module.
"""

from __future__ import annotations

import contextlib
import platform
import typing
from itertools import product

import numpy as np
import pytest
from numpy.linalg import LinAlgError

from colour.characterisation.correction import (
    apply_matrix_colour_correction_Finlayson2015,
    colour_correction_Finlayson2015,
    matrix_colour_correction_Finlayson2015,
    polynomial_expansion_Finlayson2015,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS

if typing.TYPE_CHECKING:
    from colour.hints import NDArrayFloat

from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "MATRIX_TEST",
    "MATRIX_REFERENCE",
    "TestPolynomialExpansionFinlayson2015",
    "TestMatrixColourCorrectionFinlayson2015",
    "TestApplyMatrixColourCorrectionFinlayson2015",
    "TestColourCorrectionFinlayson2015",
]

MATRIX_TEST: NDArrayFloat = np.array(
    [
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
    ]
)

MATRIX_REFERENCE: NDArrayFloat = np.array(
    [
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
    ]
)


class TestPolynomialExpansionFinlayson2015:
    """
    Define :func:`colour.characterisation.correction.finlayson2015.\
polynomial_expansion_Finlayson2015` definition unit tests methods.
    """

    def test_polynomial_expansion_Finlayson2015(self) -> None:
        """
        Test :func:`colour.characterisation.correction.finlayson2015.\
polynomial_expansion_Finlayson2015` definition.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])

        polynomials = [
            [
                np.array([0.17224810, 0.09170660, 0.06416938]),
                np.array([0.17224810, 0.09170660, 0.06416938]),
            ],
            [
                np.array(
                    [
                        0.17224810,
                        0.09170660,
                        0.06416938,
                        0.02966941,
                        0.00841010,
                        0.00411771,
                        0.01579629,
                        0.00588476,
                        0.01105305,
                    ]
                ),
                np.array(
                    [
                        0.17224810,
                        0.09170660,
                        0.06416938,
                        0.12568328,
                        0.07671216,
                        0.10513350,
                    ]
                ),
            ],
            [
                np.array(
                    [
                        0.17224810,
                        0.09170660,
                        0.06416938,
                        0.02966941,
                        0.00841010,
                        0.00411771,
                        0.01579629,
                        0.00588476,
                        0.01105305,
                        0.00511050,
                        0.00077126,
                        0.00026423,
                        0.00144862,
                        0.00037762,
                        0.00070927,
                        0.00272088,
                        0.00053967,
                        0.00190387,
                        0.00101364,
                    ]
                ),
                np.array(
                    [
                        0.17224810,
                        0.09170660,
                        0.06416938,
                        0.12568328,
                        0.07671216,
                        0.10513350,
                        0.11314930,
                        0.07228010,
                        0.08918053,
                        0.13960570,
                        0.08141598,
                        0.12394021,
                        0.10045255,
                    ]
                ),
            ],
            [
                np.array(
                    [
                        0.17224810,
                        0.09170660,
                        0.06416938,
                        0.02966941,
                        0.00841010,
                        0.00411771,
                        0.01579629,
                        0.00588476,
                        0.01105305,
                        0.00511050,
                        0.00077126,
                        0.00026423,
                        0.00144862,
                        0.00037762,
                        0.00070927,
                        0.00272088,
                        0.00053967,
                        0.00190387,
                        0.00101364,
                        0.00088027,
                        0.00007073,
                        0.00001696,
                        0.00046867,
                        0.00032794,
                        0.00013285,
                        0.00004949,
                        0.00004551,
                        0.00002423,
                        0.00024952,
                        0.00003463,
                        0.00012217,
                        0.00017460,
                        0.00009296,
                        0.00006504,
                    ]
                ),
                np.array(
                    [
                        0.17224810,
                        0.09170660,
                        0.06416938,
                        0.12568328,
                        0.07671216,
                        0.10513350,
                        0.11314930,
                        0.07228010,
                        0.08918053,
                        0.13960570,
                        0.08141598,
                        0.12394021,
                        0.10045255,
                        0.14713499,
                        0.13456986,
                        0.10735915,
                        0.08387498,
                        0.08213618,
                        0.07016104,
                        0.11495009,
                        0.09819082,
                        0.08980545,
                    ]
                ),
            ],
        ]

        for i in range(4):
            np.testing.assert_allclose(
                polynomial_expansion_Finlayson2015(RGB, i + 1, False),
                polynomials[i][0],
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )
            np.testing.assert_allclose(
                polynomial_expansion_Finlayson2015(RGB, i + 1, True),
                polynomials[i][1],
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    def test_raise_exception_polynomial_expansion_Finlayson2015(self) -> None:
        """
        Test :func:`colour.characterisation.correction.finlayson2015.\
polynomial_expansion_Finlayson2015` definition raised exception.
        """

        pytest.raises(
            ValueError,
            polynomial_expansion_Finlayson2015,
            np.array([0.17224810, 0.09170660, 0.06416938]),
            5,
        )

    @ignore_numpy_errors
    def test_nan_polynomial_expansion_Finlayson2015(self) -> None:
        """
        Test :func:`colour.characterisation.correction.finlayson2015.\
polynomial_expansion_Finlayson2015` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        polynomial_expansion_Finlayson2015(cases)


class TestMatrixColourCorrectionFinlayson2015:
    """
    Define :func:`colour.characterisation.correction.finlayson2015.\
matrix_colour_correction_Finlayson2015` definition unit tests methods.
    """

    def test_matrix_colour_correction_Finlayson2015(self) -> None:
        """
        Test :func:`colour.characterisation.correction.finlayson2015.\
matrix_colour_correction_Finlayson2015` definition.
        """

        np.testing.assert_allclose(
            matrix_colour_correction_Finlayson2015(MATRIX_TEST, MATRIX_REFERENCE),
            np.array(
                [
                    [0.69822661, 0.03071629, 0.16210422],
                    [0.06893498, 0.67579611, 0.16430385],
                    [-0.06314956, 0.09212471, 0.97134152],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            matrix_colour_correction_Finlayson2015(
                MATRIX_TEST, MATRIX_REFERENCE, degree=3
            ),
            np.array(
                [
                    [
                        2.87796213,
                        9.85720054,
                        2.99863978,
                        76.97227806,
                        73.73571500,
                        -49.37563169,
                        -48.70879206,
                        -47.53280959,
                        29.88241815,
                        -39.82871801,
                        -37.11388282,
                        23.30393209,
                        3.81579802,
                    ],
                    [
                        -0.78448243,
                        5.63631335,
                        0.95306110,
                        14.19762287,
                        20.60124427,
                        -18.05512861,
                        -14.52994195,
                        -13.10606336,
                        10.53666341,
                        -3.63132534,
                        -12.49672335,
                        8.17401039,
                        3.37995231,
                    ],
                    [
                        -2.39092600,
                        10.57193455,
                        4.16361285,
                        23.41748866,
                        58.26902059,
                        -39.39669827,
                        -26.63805785,
                        -35.98397757,
                        21.25508558,
                        -4.12726077,
                        -34.31995017,
                        18.72796247,
                        7.33531009,
                    ],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    @pytest.mark.skipif(
        platform.system() in ("Darwin", "Linux"),
        reason="Hangs on macOS and Linux",
    )
    def test_nan_matrix_colour_correction_Finlayson2015(
        self,
    ) -> None:  # pragma: no cover
        """
        Test :func:`colour.characterisation.correction.finlayson2015.\
        matrix_colour_correction_Finlayson2015` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            with contextlib.suppress(LinAlgError):
                matrix_colour_correction_Finlayson2015(
                    np.vstack([case, case, case]),
                    np.transpose(np.vstack([case, case, case])),
                )


class TestApplyMatrixColourCorrectionFinlayson2015:
    """
    Define :func:`colour.characterisation.correction.finlayson2015.\
apply_matrix_colour_correction_Finlayson2015` definition unit tests methods.
    """

    def test_apply_matrix_colour_correction_Finlayson2015(self) -> None:
        """
        Test :func:`colour.characterisation.correction.finlayson2015.\
apply_matrix_colour_correction_Finlayson2015` definition.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])

        np.testing.assert_allclose(
            apply_matrix_colour_correction_Finlayson2015(
                RGB,
                np.array(
                    [
                        [0.69822661, 0.03071629, 0.16210422],
                        [0.06893498, 0.67579611, 0.16430385],
                        [-0.06314956, 0.09212471, 0.97134152],
                    ]
                ),
            ),
            np.array([0.13348722, 0.08439216, 0.05990144]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_apply_matrix_colour_correction_Finlayson2015(self) -> None:
        """
        Test :func:`colour.characterisation.correction.finlayson2015.\
apply_matrix_colour_correction_Finlayson2015` definition n-dimensional support.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])
        CCM = np.array(
            [
                [0.69822661, 0.03071629, 0.16210422],
                [0.06893498, 0.67579611, 0.16430385],
                [-0.06314956, 0.09212471, 0.97134152],
            ]
        )
        RGB_c = apply_matrix_colour_correction_Finlayson2015(RGB, CCM)

        RGB = np.tile(RGB, (6, 1))
        RGB_c = np.tile(RGB_c, (6, 1))
        np.testing.assert_allclose(
            apply_matrix_colour_correction_Finlayson2015(RGB, CCM),
            RGB_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB = np.reshape(RGB, (2, 3, 3))
        RGB_c = np.reshape(RGB_c, (2, 3, 3))
        np.testing.assert_allclose(
            apply_matrix_colour_correction_Finlayson2015(RGB, CCM),
            RGB_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    @pytest.mark.skipif(
        platform.system() in ("Darwin", "Linux"),
        reason="Hangs on macOS and Linux",
    )
    def test_nan_apply_matrix_colour_correction_Finlayson2015(
        self,
    ) -> None:  # pragma: no cover
        """
                Test :func:`colour.characterisation.correction.finlayson2015.
        apply_matrix_colour_correction_Finlayson2015` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            with contextlib.suppress(LinAlgError):
                apply_matrix_colour_correction_Finlayson2015(
                    case,
                    np.vstack([case, case, case]),
                )


class TestColourCorrectionFinlayson2015:
    """
    Define :func:`colour.characterisation.correction.finlayson2015.\
colour_correction_Finlayson2015` definition unit tests methods.
    """

    def test_colour_correction_Finlayson2015(self) -> None:
        """
        Test :func:`colour.characterisation.correction.finlayson2015.\
colour_correction_Finlayson2015` definition.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])

        np.testing.assert_allclose(
            colour_correction_Finlayson2015(RGB, MATRIX_TEST, MATRIX_REFERENCE),
            np.array([0.13348722, 0.08439216, 0.05990144]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            colour_correction_Finlayson2015(
                RGB, MATRIX_TEST, MATRIX_REFERENCE, degree=3
            ),
            np.array([0.13914542, 0.08602124, 0.06422973]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_colour_correction_Finlayson2015(self) -> None:
        """
        Test :func:`colour.characterisation.correction.finlayson2015.\
colour_correction_Finlayson2015` definition n-dimensional support.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])
        RGB_c = colour_correction_Finlayson2015(RGB, MATRIX_TEST, MATRIX_REFERENCE)

        RGB = np.tile(RGB, (6, 1))
        RGB_c = np.tile(RGB_c, (6, 1))
        np.testing.assert_allclose(
            colour_correction_Finlayson2015(RGB, MATRIX_TEST, MATRIX_REFERENCE),
            RGB_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB = np.reshape(RGB, (2, 3, 3))
        RGB_c = np.reshape(RGB_c, (2, 3, 3))
        np.testing.assert_allclose(
            colour_correction_Finlayson2015(RGB, MATRIX_TEST, MATRIX_REFERENCE),
            RGB_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    @pytest.mark.skipif(
        platform.system() in ("Darwin", "Linux"),
        reason="Hangs on macOS and Linux",
    )
    def test_nan_colour_correction_Finlayson2015(self) -> None:  # pragma: no cover
        """
                Test :func:`colour.characterisation.correction.finlayson2015.
        colour_correction_Finlayson2015` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            with contextlib.suppress(LinAlgError):
                colour_correction_Finlayson2015(
                    case,
                    np.vstack([case, case, case]),
                    np.transpose(np.vstack([case, case, case])),
                )
