# !/usr/bin/env python
"""Define the unit tests for the :mod:`colour.models.rgb.derivation` module."""

import contextlib
import re
from itertools import product

import numpy as np
from numpy.linalg import LinAlgError

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.models import (
    RGB_luminance,
    RGB_luminance_equation,
    chromatically_adapted_primaries,
    normalised_primary_matrix,
    primaries_whitepoint,
)
from colour.models.rgb.derivation import xy_to_z
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "Testxy_to_z",
    "TestNormalisedPrimaryMatrix",
    "TestChromaticallyAdaptedPrimaries",
    "TestPrimariesWhitepoint",
    "TestRGBLuminanceEquation",
    "TestRGBLuminance",
]


class Testxy_to_z:
    """
    Define :func:`colour.models.rgb.derivation.xy_to_z` definition unit
    tests methods.
    """

    def test_xy_to_z(self):
        """Test :func:`colour.models.rgb.derivation.xy_to_z` definition."""

        np.testing.assert_allclose(
            xy_to_z(np.array([0.2500, 0.2500])),
            0.50000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_z(np.array([0.0001, -0.0770])),
            1.07690000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            xy_to_z(np.array([0.0000, 1.0000])),
            0.00000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_xy_to_z(self):
        """
        Test :func:`colour.models.rgb.derivation.xy_to_z` definition
        n-dimensional arrays support.
        """

        xy = np.array([0.25, 0.25])
        z = xy_to_z(xy)

        xy = np.tile(xy, (6, 1))
        z = np.tile(
            z,
            6,
        )
        np.testing.assert_allclose(xy_to_z(xy), z, atol=TOLERANCE_ABSOLUTE_TESTS)

        xy = np.reshape(xy, (2, 3, 2))
        z = np.reshape(z, (2, 3))
        np.testing.assert_allclose(xy_to_z(xy), z, atol=TOLERANCE_ABSOLUTE_TESTS)

    @ignore_numpy_errors
    def test_nan_xy_to_z(self):
        """
        Test :func:`colour.models.rgb.derivation.xy_to_z` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        xy_to_z(cases)


class TestNormalisedPrimaryMatrix:
    """
    Define :func:`colour.models.rgb.derivation.normalised_primary_matrix`
    definition unit tests methods.
    """

    def test_normalised_primary_matrix(self):
        """
        Test :func:`colour.models.rgb.derivation.normalised_primary_matrix`
        definition.
        """

        np.testing.assert_allclose(
            normalised_primary_matrix(
                np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]),
                np.array([0.32168, 0.33767]),
            ),
            np.array(
                [
                    [0.95255240, 0.00000000, 0.00009368],
                    [0.34396645, 0.72816610, -0.07213255],
                    [0.00000000, 0.00000000, 1.00882518],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            normalised_primary_matrix(
                np.array([0.640, 0.330, 0.300, 0.600, 0.150, 0.060]),
                np.array([0.3127, 0.3290]),
            ),
            np.array(
                [
                    [0.41239080, 0.35758434, 0.18048079],
                    [0.21263901, 0.71516868, 0.07219232],
                    [0.01933082, 0.11919478, 0.95053215],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_normalised_primary_matrix(self):
        """
        Test :func:`colour.models.rgb.derivation.normalised_primary_matrix`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            P = np.array(np.vstack([case, case, case]))
            W = case
            with contextlib.suppress(LinAlgError):
                normalised_primary_matrix(P, W)


class TestChromaticallyAdaptedPrimaries:
    """
    Define :func:`colour.models.rgb.derivation.\
chromatically_adapted_primaries` definition unit tests methods.
    """

    def test_chromatically_adapted_primaries(self):
        """
        Test :func:`colour.models.rgb.derivation.\
chromatically_adapted_primaries` definition.
        """

        np.testing.assert_allclose(
            chromatically_adapted_primaries(
                np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]),
                np.array([0.32168, 0.33767]),
                np.array([0.34570, 0.35850]),
            ),
            np.array(
                [
                    [0.73431182, 0.26694964],
                    [0.02211963, 0.98038009],
                    [-0.05880375, -0.12573056],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            chromatically_adapted_primaries(
                np.array([0.640, 0.330, 0.300, 0.600, 0.150, 0.060]),
                np.array([0.31270, 0.32900]),
                np.array([0.34570, 0.35850]),
            ),
            np.array(
                [
                    [0.64922534, 0.33062196],
                    [0.32425276, 0.60237128],
                    [0.15236177, 0.06118676],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            chromatically_adapted_primaries(
                np.array([0.640, 0.330, 0.300, 0.600, 0.150, 0.060]),
                np.array([0.31270, 0.32900]),
                np.array([0.34570, 0.35850]),
                "Bradford",
            ),
            np.array(
                [
                    [0.64844144, 0.33085331],
                    [0.32119518, 0.59784434],
                    [0.15589322, 0.06604921],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_chromatically_adapted_primaries(self):
        """
        Test :func:`colour.models.rgb.derivation.\
chromatically_adapted_primaries` definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=2))))
        for case in cases:
            P = np.array(np.vstack([case, case, case]))
            W = case
            chromatically_adapted_primaries(P, W, W)


class TestPrimariesWhitepoint:
    """
    Define :func:`colour.models.rgb.derivation.primaries_whitepoint`
    definition unit tests methods.
    """

    def test_primaries_whitepoint(self):
        """
        Test :func:`colour.models.rgb.derivation.primaries_whitepoint`
        definition.
        """

        P, W = primaries_whitepoint(
            np.array(
                [
                    [0.95255240, 0.00000000, 0.00009368],
                    [0.34396645, 0.72816610, -0.07213255],
                    [0.00000000, 0.00000000, 1.00882518],
                ]
            )
        )
        np.testing.assert_allclose(
            P,
            np.array(
                [
                    [0.73470, 0.26530],
                    [0.00000, 1.00000],
                    [0.00010, -0.07700],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            W, np.array([0.32168, 0.33767]), atol=TOLERANCE_ABSOLUTE_TESTS
        )

        P, W = primaries_whitepoint(
            np.array(
                [
                    [0.41240000, 0.35760000, 0.18050000],
                    [0.21260000, 0.71520000, 0.07220000],
                    [0.01930000, 0.11920000, 0.95050000],
                ]
            )
        )
        np.testing.assert_allclose(
            P,
            np.array(
                [
                    [0.64007450, 0.32997051],
                    [0.30000000, 0.60000000],
                    [0.15001662, 0.06000665],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            W,
            np.array([0.31271591, 0.32900148]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    @ignore_numpy_errors
    def test_nan_primaries_whitepoint(self):
        """
        Test :func:`colour.models.rgb.derivation.primaries_whitepoint`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            M = np.array(np.vstack([case, case, case]))
            primaries_whitepoint(M)


class TestRGBLuminanceEquation:
    """
    Define :func:`colour.models.rgb.derivation.RGB_luminance_equation`
    definition unit tests methods.
    """

    def test_RGB_luminance_equation(self):
        """
        Test :func:`colour.models.rgb.derivation.RGB_luminance_equation`
        definition.
        """

        assert isinstance(
            RGB_luminance_equation(
                np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]),
                np.array([0.32168, 0.33767]),
            ),
            str,
        )

        # TODO: Simplify that monster.
        pattern = (
            "Y\\s?=\\s?[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?."
            "\\(R\\)\\s?[+-]\\s?[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?."
            "\\(G\\)\\s?[+-]\\s?[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?."
            "\\(B\\)"
        )
        P = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
        assert re.match(
            pattern,
            RGB_luminance_equation(P, np.array([0.32168, 0.33767])),
        )


class TestRGBLuminance:
    """
    Define :func:`colour.models.rgb.derivation.RGB_luminance` definition
    unit tests methods.
    """

    def test_RGB_luminance(self):
        """
        Test :func:`colour.models.rgb.derivation.RGB_luminance`
        definition.
        """

        np.testing.assert_allclose(
            RGB_luminance(
                np.array([0.18, 0.18, 0.18]),
                np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]),
                np.array([0.32168, 0.33767]),
            ),
            0.18000000,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            RGB_luminance(
                np.array([0.21959402, 0.06986677, 0.04703877]),
                np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]),
                np.array([0.32168, 0.33767]),
            ),
            0.123014562384318,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            RGB_luminance(
                np.array([0.45620519, 0.03081071, 0.04091952]),
                np.array([0.6400, 0.3300, 0.3000, 0.6000, 0.1500, 0.0600]),
                np.array([0.31270, 0.32900]),
            ),
            0.121995947729870,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_RGB_luminance(self):
        """
        Test :func:`colour.models.rgb.derivation.RGB_luminance` definition
        n_dimensional arrays support.
        """

        RGB = (np.array([0.18, 0.18, 0.18]),)
        P = (np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]),)
        W = np.array([0.32168, 0.33767])
        Y = RGB_luminance(RGB, P, W)

        RGB = np.tile(RGB, (6, 1))
        Y = np.tile(Y, 6)
        np.testing.assert_allclose(
            RGB_luminance(RGB, P, W), Y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        RGB = np.reshape(RGB, (2, 3, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_allclose(
            RGB_luminance(RGB, P, W), Y, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_RGB_luminance(self):
        """
        Test :func:`colour.models.rgb.derivation.RGB_luminance`
        definition nan support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        for case in cases:
            RGB = case
            P = np.array(np.vstack([case[0:2], case[0:2], case[0:2]]))
            W = case[0:2]
            with contextlib.suppress(LinAlgError):
                RGB_luminance(RGB, P, W)
