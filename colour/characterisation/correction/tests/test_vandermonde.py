"""
Define the unit tests for the
:mod:`colour.characterisation.correction.vandermonde` module.
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
    apply_matrix_colour_correction_Vandermonde,
    colour_correction_Vandermonde,
    matrix_colour_correction_Vandermonde,
    polynomial_expansion_Vandermonde,
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
    "TestPolynomialExpansionVandermonde",
    "TestMatrixColourCorrectionVandermonde",
    "TestApplyMatrixColourCorrectionVandermonde",
    "TestColourCorrectionVandermonde",
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


class TestPolynomialExpansionVandermonde:
    """
    Define :func:`colour.characterisation.correction.vandermonde.\
polynomial_expansion_Vandermonde` definition unit tests methods.
    """

    def test_polynomial_expansion_Vandermonde(self) -> None:
        """
        Test :func:`colour.characterisation.correction.vandermonde.\
polynomial_expansion_Vandermonde` definition.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])

        polynomials = [
            np.array([0.17224810, 0.09170660, 0.06416938, 1.00000000]),
            np.array(
                [
                    0.02966941,
                    0.00841010,
                    0.00411771,
                    0.17224810,
                    0.09170660,
                    0.06416938,
                    1.00000000,
                ]
            ),
            np.array(
                [
                    0.00511050,
                    0.00077126,
                    0.00026423,
                    0.02966941,
                    0.00841010,
                    0.00411771,
                    0.17224810,
                    0.09170660,
                    0.06416938,
                    1.00000000,
                ]
            ),
            np.array(
                [
                    0.00088027,
                    0.00007073,
                    0.00001696,
                    0.00511050,
                    0.00077126,
                    0.00026423,
                    0.02966941,
                    0.00841010,
                    0.00411771,
                    0.17224810,
                    0.09170660,
                    0.06416938,
                    1.00000000,
                ]
            ),
        ]

        for i in range(4):
            np.testing.assert_allclose(
                polynomial_expansion_Vandermonde(RGB, i + 1),
                polynomials[i],
                atol=TOLERANCE_ABSOLUTE_TESTS,
            )

    @ignore_numpy_errors
    def test_nan_polynomial_expansion_Vandermonde(self) -> None:
        """
        Test :func:`colour.characterisation.correction.vandermonde.\
polynomial_expansion_Vandermonde` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        polynomial_expansion_Vandermonde(cases)


class TestMatrixColourCorrectionVandermonde:
    """
    Define :func:`colour.characterisation.correction.vandermonde.\
matrix_colour_correction_Vandermonde` definition unit tests methods.
    """

    def test_matrix_colour_correction_Vandermonde(self) -> None:
        """
        Test :func:`colour.characterisation.correction.vandermonde.\
matrix_colour_correction_Vandermonde` definition.
        """

        np.testing.assert_allclose(
            matrix_colour_correction_Vandermonde(MATRIX_TEST, MATRIX_REFERENCE),
            np.array(
                [
                    [0.66770040, 0.02514036, 0.12745797, 0.02485425],
                    [0.03155494, 0.66896825, 0.12187874, 0.03043460],
                    [-0.14502258, 0.07716975, 0.87841836, 0.06666049],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            matrix_colour_correction_Vandermonde(
                MATRIX_TEST, MATRIX_REFERENCE, degree=3
            ),
            np.array(
                [
                    [
                        -0.04328223,
                        -1.87886146,
                        1.83369170,
                        -0.10798116,
                        1.06608177,
                        -0.87495813,
                        0.75525839,
                        -0.08558123,
                        0.15919076,
                        0.02404598,
                    ],
                    [
                        0.00998152,
                        0.44525275,
                        -0.53192490,
                        0.00904507,
                        -0.41034458,
                        0.36173334,
                        0.02904178,
                        0.78362950,
                        0.07894900,
                        0.01986479,
                    ],
                    [
                        -1.66921744,
                        3.62954420,
                        -2.96789849,
                        2.31451409,
                        -3.10767297,
                        1.85975390,
                        -0.98795093,
                        0.85962796,
                        0.63591240,
                        0.07302317,
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
    def test_nan_matrix_colour_correction_Vandermonde(
        self,
    ) -> None:  # pragma: no cover
        """
        Test :func:`colour.characterisation.correction.vandermonde.\
        matrix_colour_correction_Vandermonde` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            with contextlib.suppress(LinAlgError):
                matrix_colour_correction_Vandermonde(
                    np.vstack([case, case, case]),
                    np.transpose(np.vstack([case, case, case])),
                )


class TestApplyMatrixColourCorrectionVandermonde:
    """
    Define :func:`colour.characterisation.correction.vandermonde.\
apply_matrix_colour_correction_Vandermonde` definition unit tests methods.
    """

    def test_apply_matrix_colour_correction_Vandermonde(self) -> None:
        """
        Test :func:`colour.characterisation.correction.vandermonde.\
apply_matrix_colour_correction_Vandermonde` definition.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])

        np.testing.assert_allclose(
            apply_matrix_colour_correction_Vandermonde(
                RGB,
                np.array(
                    [
                        [0.66770040, 0.02514036, 0.12745797, 0.02485425],
                        [0.03155494, 0.66896825, 0.12187874, 0.03043460],
                        [-0.14502258, 0.07716975, 0.87841836, 0.06666049],
                    ]
                ),
            ),
            np.array([0.15034881, 0.10503956, 0.10512517]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_apply_matrix_colour_correction_Vandermonde(self) -> None:
        """
        Test :func:`colour.characterisation.correction.vandermonde.\
apply_matrix_colour_correction_Vandermonde` definition n-dimensional support.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])
        CCM = np.array(
            [
                [0.66770040, 0.02514036, 0.12745797, 0.02485425],
                [0.03155494, 0.66896825, 0.12187874, 0.03043460],
                [-0.14502258, 0.07716975, 0.87841836, 0.06666049],
            ]
        )
        RGB_c = apply_matrix_colour_correction_Vandermonde(RGB, CCM)

        RGB = np.tile(RGB, (6, 1))
        RGB_c = np.tile(RGB_c, (6, 1))
        np.testing.assert_allclose(
            apply_matrix_colour_correction_Vandermonde(RGB, CCM),
            RGB_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB = np.reshape(RGB, (2, 3, 3))
        RGB_c = np.reshape(RGB_c, (2, 3, 3))
        np.testing.assert_allclose(
            apply_matrix_colour_correction_Vandermonde(RGB, CCM),
            RGB_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    @pytest.mark.skipif(
        platform.system() in ("Darwin", "Linux"),
        reason="Hangs on macOS and Linux",
    )
    def test_nan_apply_matrix_colour_correction_Vandermonde(
        self,
    ) -> None:  # pragma: no cover
        """
        Test :func:`colour.characterisation.correction.vandermonde.\
apply_matrix_colour_correction_Vandermonde` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            with contextlib.suppress(LinAlgError):
                apply_matrix_colour_correction_Vandermonde(
                    case,
                    np.dstack([case, case, case, case]),
                )


class TestColourCorrectionVandermonde:
    """
    Define :func:`colour.characterisation.correction.vandermonde.\
colour_correction_Vandermonde` definition unit tests methods.
    """

    def test_colour_correction_Vandermonde(self) -> None:
        """
        Test :func:`colour.characterisation.correction.vandermonde.\
colour_correction_Vandermonde` definition.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])

        np.testing.assert_allclose(
            colour_correction_Vandermonde(RGB, MATRIX_TEST, MATRIX_REFERENCE),
            np.array([0.15034881, 0.10503956, 0.10512517]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            colour_correction_Vandermonde(RGB, MATRIX_TEST, MATRIX_REFERENCE, degree=3),
            np.array([0.15747814, 0.10035799, 0.06616709]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_colour_correction_Vandermonde(self) -> None:
        """
        Test :func:`colour.characterisation.correction.vandermonde.\
colour_correction_Vandermonde` definition n-dimensional support.
        """

        RGB = np.array([0.17224810, 0.09170660, 0.06416938])
        RGB_c = colour_correction_Vandermonde(RGB, MATRIX_TEST, MATRIX_REFERENCE)

        RGB = np.tile(RGB, (6, 1))
        RGB_c = np.tile(RGB_c, (6, 1))
        np.testing.assert_allclose(
            colour_correction_Vandermonde(RGB, MATRIX_TEST, MATRIX_REFERENCE),
            RGB_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        RGB = np.reshape(RGB, (2, 3, 3))
        RGB_c = np.reshape(RGB_c, (2, 3, 3))
        np.testing.assert_allclose(
            colour_correction_Vandermonde(RGB, MATRIX_TEST, MATRIX_REFERENCE),
            RGB_c,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    @pytest.mark.skipif(
        platform.system() in ("Darwin", "Linux"),
        reason="Hangs on macOS and Linux",
    )
    def test_nan_colour_correction_Vandermonde(self) -> None:  # pragma: no cover
        """
        Test :func:`colour.characterisation.correction.vandermonde.\
colour_correction_Vandermonde` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            with contextlib.suppress(LinAlgError):
                colour_correction_Vandermonde(
                    case,
                    np.vstack([case, case, case]),
                    np.transpose(np.vstack([case, case, case])),
                )
