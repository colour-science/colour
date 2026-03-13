"""Define the unit tests for the :mod:`colour.notation.munsell.value` module."""

from __future__ import annotations

import numpy as np

from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.notation import (
    munsell_value_ASTMD1535,
    munsell_value_Ladd1955,
    munsell_value_McCamy1987,
    munsell_value_Moon1943,
    munsell_value_Munsell1933,
    munsell_value_Priest1920,
    munsell_value_Saunderson1944,
)
from colour.utilities import (
    domain_range_scale,
    ignore_numpy_errors,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestMunsellValuePriest1920",
    "TestMunsellValueMunsell1933",
    "TestMunsellValueMoon1943",
    "TestMunsellValueSaunderson1944",
    "TestMunsellValueLadd1955",
    "TestMunsellValueMcCamy1992",
    "TestMunsellValueASTMD1535",
]


class TestMunsellValuePriest1920:
    """
    Define :func:`colour.notation.munsell.munsell_value_Priest1920` definition
    unit tests methods.
    """

    def test_munsell_value_Priest1920(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Priest1920`
        definition.
        """

        np.testing.assert_allclose(
            munsell_value_Priest1920(12.23634268),
            3.498048410185314,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_Priest1920(22.89399987),
            4.7847674833788947,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_Priest1920(6.29022535),
            2.5080321668591092,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_munsell_value_Priest1920(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Priest1920`
        definition n-dimensional arrays support.
        """

        Y = 12.23634268
        V = munsell_value_Priest1920(Y)

        V = np.tile(V, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_allclose(
            munsell_value_Priest1920(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_allclose(
            munsell_value_Priest1920(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_allclose(
            munsell_value_Priest1920(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_munsell_value_Priest1920(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Priest1920`
        definition domain and range scale support.
        """

        Y = 12.23634268
        V = munsell_value_Priest1920(Y)

        d_r = (("reference", 1, 1), ("1", 0.01, 0.1), ("100", 1, 10))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    munsell_value_Priest1920(Y * factor_a),
                    V * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_munsell_value_Priest1920(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Priest1920`
        definition nan support.
        """

        munsell_value_Priest1920(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestMunsellValueMunsell1933:
    """
    Define :func:`colour.notation.munsell.munsell_value_Munsell1933`
    definition unit tests methods.
    """

    def test_munsell_value_Munsell1933(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Munsell1933`
        definition.
        """

        np.testing.assert_allclose(
            munsell_value_Munsell1933(12.23634268),
            4.1627702416858083,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_Munsell1933(22.89399987),
            5.5914543020790592,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_Munsell1933(6.29022535),
            3.0141971134091761,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_munsell_value_Munsell1933(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Munsell1933`
        definition n-dimensional arrays support.
        """

        Y = 12.23634268
        V = munsell_value_Munsell1933(Y)

        V = np.tile(V, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_allclose(
            munsell_value_Munsell1933(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_allclose(
            munsell_value_Munsell1933(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_allclose(
            munsell_value_Munsell1933(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_munsell_value_Munsell1933(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Munsell1933`
        definition domain and range scale support.
        """

        Y = 12.23634268
        V = munsell_value_Munsell1933(Y)

        d_r = (("reference", 1, 1), ("1", 0.01, 0.1), ("100", 1, 10))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    munsell_value_Munsell1933(Y * factor_a),
                    V * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_munsell_value_Munsell1933(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Munsell1933`
        definition nan support.
        """

        munsell_value_Munsell1933(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestMunsellValueMoon1943:
    """
    Define :func:`colour.notation.munsell.munsell_value_Moon1943` definition
    unit tests methods.
    """

    def test_munsell_value_Moon1943(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Moon1943`
        definition.
        """

        np.testing.assert_allclose(
            munsell_value_Moon1943(12.23634268),
            4.0688120634976421,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_Moon1943(22.89399987),
            5.3133627855494412,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_Moon1943(6.29022535),
            3.0645015037679695,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_munsell_value_Moon1943(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Moon1943`
        definition n-dimensional arrays support.
        """

        Y = 12.23634268
        V = munsell_value_Moon1943(Y)

        V = np.tile(V, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_allclose(
            munsell_value_Moon1943(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_allclose(
            munsell_value_Moon1943(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_allclose(
            munsell_value_Moon1943(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_munsell_value_Moon1943(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Moon1943`
        definition domain and range scale support.
        """

        Y = 12.23634268
        V = munsell_value_Moon1943(Y)

        d_r = (("reference", 1, 1), ("1", 0.01, 0.1), ("100", 1, 10))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    munsell_value_Moon1943(Y * factor_a),
                    V * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_munsell_value_Moon1943(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Moon1943`
        definition nan support.
        """

        munsell_value_Moon1943(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestMunsellValueSaunderson1944:
    """
    Define :func:`colour.notation.munsell.munsell_value_Saunderson1944`
    definition unit tests methods.
    """

    def test_munsell_value_Saunderson1944(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Saunderson1944`
        definition.
        """

        np.testing.assert_allclose(
            munsell_value_Saunderson1944(12.23634268),
            4.0444736723175119,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_Saunderson1944(22.89399987),
            5.3783324022305923,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_Saunderson1944(6.29022535),
            2.9089633927316823,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_munsell_value_Saunderson1944(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Saunderson1944`
        definition n-dimensional arrays support.
        """

        Y = 12.23634268
        V = munsell_value_Saunderson1944(Y)

        V = np.tile(V, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_allclose(
            munsell_value_Saunderson1944(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_allclose(
            munsell_value_Saunderson1944(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_allclose(
            munsell_value_Saunderson1944(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_munsell_value_Saunderson1944(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Saunderson1944`
        definition domain and range scale support.
        """

        Y = 12.23634268
        V = munsell_value_Saunderson1944(Y)

        d_r = (("reference", 1, 1), ("1", 0.01, 0.1), ("100", 1, 10))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    munsell_value_Saunderson1944(Y * factor_a),
                    V * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_munsell_value_Saunderson1944(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Saunderson1944`
        definition nan support.
        """

        munsell_value_Saunderson1944(
            np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan])
        )


class TestMunsellValueLadd1955:
    """
    Define :func:`colour.notation.munsell.munsell_value_Ladd1955` definition
    unit tests methods.
    """

    def test_munsell_value_Ladd1955(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Ladd1955`
        definition.
        """

        np.testing.assert_allclose(
            munsell_value_Ladd1955(12.23634268),
            4.0511633044287088,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_Ladd1955(22.89399987),
            5.3718647913936772,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_Ladd1955(6.29022535),
            2.9198269939751613,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_munsell_value_Ladd1955(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Ladd1955`
        definition n-dimensional arrays support.
        """

        Y = 12.23634268
        V = munsell_value_Ladd1955(Y)

        V = np.tile(V, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_allclose(
            munsell_value_Ladd1955(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_allclose(
            munsell_value_Ladd1955(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_allclose(
            munsell_value_Ladd1955(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_munsell_value_Ladd1955(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Ladd1955`
        definition domain and range scale support.
        """

        Y = 12.23634268
        V = munsell_value_Ladd1955(Y)

        d_r = (("reference", 1, 1), ("1", 0.01, 0.1), ("100", 1, 10))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    munsell_value_Ladd1955(Y * factor_a),
                    V * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_munsell_value_Ladd1955(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_Ladd1955`
        definition nan support.
        """

        munsell_value_Ladd1955(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestMunsellValueMcCamy1992:
    """
    Define :func:`colour.notation.munsell.munsell_value_McCamy1987` definition
    unit tests methods.
    """

    def test_munsell_value_McCamy1987(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_McCamy1987`
        definition.
        """

        np.testing.assert_allclose(
            munsell_value_McCamy1987(12.23634268),
            4.081434853194113,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_McCamy1987(22.89399987),
            5.394083970919982,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_McCamy1987(6.29022535),
            2.9750160800320096,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_munsell_value_McCamy1987(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_McCamy1987`
        definition n-dimensional arrays support.
        """

        Y = 12.23634268
        V = munsell_value_McCamy1987(Y)

        V = np.tile(V, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_allclose(
            munsell_value_McCamy1987(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_allclose(
            munsell_value_McCamy1987(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_allclose(
            munsell_value_McCamy1987(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_munsell_value_McCamy1987(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_McCamy1987`
        definition domain and range scale support.
        """

        Y = 12.23634268
        V = munsell_value_McCamy1987(Y)

        d_r = (("reference", 1, 1), ("1", 0.01, 0.1), ("100", 1, 10))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    munsell_value_McCamy1987(Y * factor_a),
                    V * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_munsell_value_McCamy1987(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_McCamy1987`
        definition nan support.
        """

        munsell_value_McCamy1987(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))


class TestMunsellValueASTMD1535:
    """
    Define :func:`colour.notation.munsell.munsell_value_ASTMD1535`
    definition unit tests methods.
    """

    def test_munsell_value_ASTMD1535(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_ASTMD1535`
        definition.
        """

        np.testing.assert_allclose(
            munsell_value_ASTMD1535(12.23634268),
            4.0824437076525664,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_ASTMD1535(22.89399987),
            5.3913268228155395,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            munsell_value_ASTMD1535(6.29022535),
            2.9761930839606454,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_munsell_value_ASTMD1535(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_ASTMD1535`
        definition n-dimensional arrays support.
        """

        Y = 12.23634268
        V = munsell_value_ASTMD1535(Y)

        V = np.tile(V, 6)
        Y = np.tile(Y, 6)
        np.testing.assert_allclose(
            munsell_value_ASTMD1535(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3))
        Y = np.reshape(Y, (2, 3))
        np.testing.assert_allclose(
            munsell_value_ASTMD1535(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        V = np.reshape(V, (2, 3, 1))
        Y = np.reshape(Y, (2, 3, 1))
        np.testing.assert_allclose(
            munsell_value_ASTMD1535(Y), V, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    def test_domain_range_scale_munsell_value_ASTMD1535(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_ASTMD1535`
        definition domain and range scale support.
        """

        Y = 12.23634268
        V = munsell_value_ASTMD1535(Y)

        d_r = (("reference", 1, 1), ("1", 0.01, 0.1), ("100", 1, 10))
        for scale, factor_a, factor_b in d_r:
            with domain_range_scale(scale):
                np.testing.assert_allclose(
                    munsell_value_ASTMD1535(Y * factor_a),
                    V * factor_b,
                    atol=TOLERANCE_ABSOLUTE_TESTS,
                )

    @ignore_numpy_errors
    def test_nan_munsell_value_ASTMD1535(self) -> None:
        """
        Test :func:`colour.notation.munsell.munsell_value_ASTMD1535`
        definition nan support.
        """

        munsell_value_ASTMD1535(np.array([-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]))
