"""Define the unit tests for the :mod:`colour.algebra.common` module."""

from itertools import product

import numpy as np
import pytest

from colour.algebra import (
    eigen_decomposition,
    euclidean_distance,
    get_sdiv_mode,
    is_identity,
    is_spow_enabled,
    linear_conversion,
    linstep_function,
    manhattan_distance,
    normalise_maximum,
    normalise_vector,
    sdiv,
    sdiv_mode,
    set_sdiv_mode,
    set_spow_enable,
    smoothstep_function,
    spow,
    spow_enable,
    vecmul,
)
from colour.constants import TOLERANCE_ABSOLUTE_TESTS
from colour.utilities import ignore_numpy_errors

__author__ = "Colour Developers"
__copyright__ = "Copyright 2013 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "TestGetSdivMode",
    "TestSetSdivMode",
    "TestSdivMode",
    "TestSdiv",
    "TestIsSpowEnabled",
    "TestSetSpowEnabled",
    "TestSpowEnable",
    "TestSpow",
    "TestSmoothstepFunction",
    "TestNormaliseVector",
    "TestNormaliseMaximum",
    "TestVectorDot",
    "TestEuclideanDistance",
    "TestManhattanDistance",
    "TestLinearConversion",
    "TestLinstepFunction",
    "TestIsIdentity",
    "TestEigenDecomposition",
]


class TestGetSdivMode:
    """
    Define :func:`colour.algebra.common.get_sdiv_mode` definition unit tests
    methods.
    """

    def test_get_sdiv_mode(self):
        """Test :func:`colour.algebra.common.get_sdiv_mode` definition."""

        with sdiv_mode("Numpy"):
            assert get_sdiv_mode() == "numpy"

        with sdiv_mode("Ignore"):
            assert get_sdiv_mode() == "ignore"

        with sdiv_mode("Warning"):
            assert get_sdiv_mode() == "warning"

        with sdiv_mode("Raise"):
            assert get_sdiv_mode() == "raise"

        with sdiv_mode("Ignore Zero Conversion"):
            assert get_sdiv_mode() == "ignore zero conversion"

        with sdiv_mode("Warning Zero Conversion"):
            assert get_sdiv_mode() == "warning zero conversion"

        with sdiv_mode("Ignore Limit Conversion"):
            assert get_sdiv_mode() == "ignore limit conversion"

        with sdiv_mode("Warning Limit Conversion"):
            assert get_sdiv_mode() == "warning limit conversion"


class TestSetSdivMode:
    """
    Define :func:`colour.algebra.common.set_sdiv_mode` definition unit tests
    methods.
    """

    def test_set_sdiv_mode(self):
        """Test :func:`colour.algebra.common.set_sdiv_mode` definition."""

        with sdiv_mode(get_sdiv_mode()):
            set_sdiv_mode("Numpy")
            assert get_sdiv_mode() == "numpy"

            set_sdiv_mode("Ignore")
            assert get_sdiv_mode() == "ignore"

            set_sdiv_mode("Warning")
            assert get_sdiv_mode() == "warning"

            set_sdiv_mode("Raise")
            assert get_sdiv_mode() == "raise"

            set_sdiv_mode("Ignore Zero Conversion")
            assert get_sdiv_mode() == "ignore zero conversion"

            set_sdiv_mode("Warning Zero Conversion")
            assert get_sdiv_mode() == "warning zero conversion"

            set_sdiv_mode("Ignore Limit Conversion")
            assert get_sdiv_mode() == "ignore limit conversion"

            set_sdiv_mode("Warning Limit Conversion")
            assert get_sdiv_mode() == "warning limit conversion"


class TestSdivMode:
    """
    Define :func:`colour.algebra.common.sdiv_mode` definition unit
    tests methods.
    """

    def test_sdiv_mode(self):
        """Test :func:`colour.algebra.common.sdiv_mode` definition."""

        with sdiv_mode("Raise"):
            assert get_sdiv_mode() == "raise"

        with sdiv_mode("Ignore Zero Conversion"):
            assert get_sdiv_mode() == "ignore zero conversion"

        @sdiv_mode("Raise")
        def fn_a():
            """:func:`sdiv_mode` unit tests :func:`fn_a` definition."""

            assert get_sdiv_mode() == "raise"

        fn_a()

        @sdiv_mode("Ignore Zero Conversion")
        def fn_b():
            """:func:`sdiv_mode` unit tests :func:`fn_b` definition."""

            assert get_sdiv_mode() == "ignore zero conversion"

        fn_b()


class TestSdiv:
    """
    Define :func:`colour.algebra.common.sdiv` definition unit
    tests methods.
    """

    def test_sdiv(self):
        """Test :func:`colour.algebra.common.sdiv` definition."""

        a = np.array([0, 1, 2])
        b = np.array([2, 1, 0])

        with sdiv_mode("Numpy"):
            pytest.warns(RuntimeWarning, sdiv, a, b)

        with sdiv_mode("Ignore"):
            np.testing.assert_equal(sdiv(a, b), np.array([0, 1, np.inf]))

        with sdiv_mode("Warning"):
            pytest.warns(RuntimeWarning, sdiv, a, b)
            np.testing.assert_equal(sdiv(a, b), np.array([0, 1, np.inf]))

        with sdiv_mode("Raise"):
            pytest.raises(FloatingPointError, sdiv, a, b)

        with sdiv_mode("Ignore Zero Conversion"):
            np.testing.assert_equal(sdiv(a, b), np.array([0, 1, 0]))

        with sdiv_mode("Warning Zero Conversion"):
            pytest.warns(RuntimeWarning, sdiv, a, b)
            np.testing.assert_equal(sdiv(a, b), np.array([0, 1, 0]))

        with sdiv_mode("Ignore Limit Conversion"):
            np.testing.assert_equal(sdiv(a, b), np.nan_to_num(np.array([0, 1, np.inf])))

        with sdiv_mode("Warning Limit Conversion"):
            pytest.warns(RuntimeWarning, sdiv, a, b)
            np.testing.assert_equal(sdiv(a, b), np.nan_to_num(np.array([0, 1, np.inf])))


class TestIsSpowEnabled:
    """
    Define :func:`colour.algebra.common.is_spow_enabled` definition unit
    tests methods.
    """

    def test_is_spow_enabled(self):
        """Test :func:`colour.algebra.common.is_spow_enabled` definition."""

        with spow_enable(True):
            assert is_spow_enabled()

        with spow_enable(False):
            assert not is_spow_enabled()


class TestSetSpowEnabled:
    """
    Define :func:`colour.algebra.common.set_spow_enable` definition unit
    tests methods.
    """

    def test_set_spow_enable(self):
        """Test :func:`colour.algebra.common.set_spow_enable` definition."""

        with spow_enable(is_spow_enabled()):
            set_spow_enable(True)
            assert is_spow_enabled()

        with spow_enable(is_spow_enabled()):
            set_spow_enable(False)
            assert not is_spow_enabled()


class TestSpowEnable:
    """
    Define :func:`colour.algebra.common.spow_enable` definition unit
    tests methods.
    """

    def test_spow_enable(self):
        """Test :func:`colour.algebra.common.spow_enable` definition."""

        with spow_enable(True):
            assert is_spow_enabled()

        with spow_enable(False):
            assert not is_spow_enabled()

        @spow_enable(True)
        def fn_a():
            """:func:`spow_enable` unit tests :func:`fn_a` definition."""

            assert is_spow_enabled()

        fn_a()

        @spow_enable(False)
        def fn_b():
            """:func:`spow_enable` unit tests :func:`fn_b` definition."""

            assert not is_spow_enabled()

        fn_b()


class TestSpow:
    """
    Define :func:`colour.algebra.common.spow` definition unit
    tests methods.
    """

    def test_spow(self):
        """Test :func:`colour.algebra.common.spow` definition."""

        assert spow(2, 2) == 4.0

        assert spow(-2, 2) == -4.0

        np.testing.assert_allclose(
            spow([2, -2, -2, 0], [2, 2, 0.15, 0]),
            np.array([4.00000000, -4.00000000, -1.10956947, 0.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        with spow_enable(True):
            np.testing.assert_allclose(
                spow(-2, 0.15), -1.10956947, atol=TOLERANCE_ABSOLUTE_TESTS
            )

        with spow_enable(False):
            np.testing.assert_equal(spow(-2, 0.15), np.nan)


class TestNormaliseVector:
    """
    Define :func:`colour.algebra.common.normalise_vector` definition unit
    tests methods.
    """

    def test_normalise_vector(self):
        """Test :func:`colour.algebra.common.normalise_vector` definition."""

        np.testing.assert_allclose(
            normalise_vector(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([0.84197033, 0.49722560, 0.20941026]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            normalise_vector(np.array([0.14222010, 0.23042768, 0.10495772])),
            np.array([0.48971705, 0.79344877, 0.36140872]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            normalise_vector(np.array([0.07818780, 0.06157201, 0.28099326])),
            np.array([0.26229003, 0.20655044, 0.94262445]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestNormaliseMaximum:
    """
    Define :func:`colour.algebra.common.normalise_maximum` definition unit
    tests methods.
    """

    def test_normalise_maximum(self):
        """Test :func:`colour.algebra.common.normalise_maximum` definition."""

        np.testing.assert_allclose(
            normalise_maximum(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([1.00000000, 0.59055003, 0.24871454]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            normalise_maximum(
                np.array(
                    [
                        [0.20654008, 0.12197225, 0.05136952],
                        [0.14222010, 0.23042768, 0.10495772],
                        [0.07818780, 0.06157201, 0.28099326],
                    ]
                )
            ),
            np.array(
                [
                    [0.73503571, 0.43407536, 0.18281406],
                    [0.50613349, 0.82004700, 0.37352398],
                    [0.27825507, 0.21912273, 1.00000000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            normalise_maximum(
                np.array(
                    [
                        [0.20654008, 0.12197225, 0.05136952],
                        [0.14222010, 0.23042768, 0.10495772],
                        [0.07818780, 0.06157201, 0.28099326],
                    ]
                ),
                axis=-1,
            ),
            np.array(
                [
                    [1.00000000, 0.59055003, 0.24871454],
                    [0.61720059, 1.00000000, 0.45549094],
                    [0.27825507, 0.21912273, 1.00000000],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            normalise_maximum(
                np.array([0.20654008, 0.12197225, 0.05136952]), factor=10
            ),
            np.array([10.00000000, 5.90550028, 2.48714535]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            normalise_maximum(np.array([-0.11518475, -0.10080000, 0.05089373])),
            np.array([0.00000000, 0.00000000, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            normalise_maximum(
                np.array([-0.20654008, -0.12197225, 0.05136952]), clip=False
            ),
            np.array([-4.02067374, -2.37440899, 1.00000000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestVectorDot:
    """
    Define :func:`colour.algebra.common.vecmul` definition unit tests
    methods.
    """

    def test_vecmul(self):
        """Test :func:`colour.algebra.common.vecmul` definition."""

        m = np.array(
            [
                [0.7328, 0.4296, -0.1624],
                [-0.7036, 1.6975, 0.0061],
                [0.0030, 0.0136, 0.9834],
            ]
        )
        m = np.reshape(np.tile(m, (6, 1)), (6, 3, 3))

        v = np.array([0.20654008, 0.12197225, 0.05136952])
        v = np.tile(v, (6, 1))

        np.testing.assert_allclose(
            vecmul(m, v),
            np.array(
                [
                    [0.19540944, 0.06203965, 0.05279523],
                    [0.19540944, 0.06203965, 0.05279523],
                    [0.19540944, 0.06203965, 0.05279523],
                    [0.19540944, 0.06203965, 0.05279523],
                    [0.19540944, 0.06203965, 0.05279523],
                    [0.19540944, 0.06203965, 0.05279523],
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestEuclideanDistance:
    """
    Define :func:`colour.algebra.common.euclidean_distance` definition unit
    tests methods.
    """

    def test_euclidean_distance(self):
        """Test :func:`colour.algebra.common.euclidean_distance` definition."""

        np.testing.assert_allclose(
            euclidean_distance(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 426.67945353, 72.39590835]),
            ),
            451.71330197,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            euclidean_distance(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193]),
            ),
            52.64986116,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            euclidean_distance(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716]),
            ),
            346.06489172,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_euclidean_distance(self):
        """
        Test :func:`colour.algebra.common.euclidean_distance` definition
        n-dimensional arrays support.
        """

        a = np.array([100.00000000, 21.57210357, 272.22819350])
        b = np.array([100.00000000, 426.67945353, 72.39590835])
        distance = euclidean_distance(a, b)

        a = np.tile(a, (6, 1))
        b = np.tile(b, (6, 1))
        distance = np.tile(distance, 6)
        np.testing.assert_allclose(
            euclidean_distance(a, b), distance, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a = np.reshape(a, (2, 3, 3))
        b = np.reshape(b, (2, 3, 3))
        distance = np.reshape(distance, (2, 3))
        np.testing.assert_allclose(
            euclidean_distance(a, b), distance, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_euclidean_distance(self):
        """
        Test :func:`colour.algebra.common.euclidean_distance` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        euclidean_distance(cases, cases)


class TestManhattanDistance:
    """
    Define :func:`colour.algebra.common.manhattan_distance` definition unit
    tests methods.
    """

    def test_manhattan_distance(self):
        """Test :func:`colour.algebra.common.manhattan_distance` definition."""

        np.testing.assert_allclose(
            manhattan_distance(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 426.67945353, 72.39590835]),
            ),
            604.93963510999993,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            manhattan_distance(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 74.05216981, 276.45318193]),
            ),
            56.705054670000052,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            manhattan_distance(
                np.array([100.00000000, 21.57210357, 272.22819350]),
                np.array([100.00000000, 8.32281957, -73.58297716]),
            ),
            359.06045465999995,
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

    def test_n_dimensional_manhattan_distance(self):
        """
        Test :func:`colour.algebra.common.manhattan_distance` definition
        n-dimensional arrays support.
        """

        a = np.array([100.00000000, 21.57210357, 272.22819350])
        b = np.array([100.00000000, 426.67945353, 72.39590835])
        distance = manhattan_distance(a, b)

        a = np.tile(a, (6, 1))
        b = np.tile(b, (6, 1))
        distance = np.tile(distance, 6)
        np.testing.assert_allclose(
            manhattan_distance(a, b), distance, atol=TOLERANCE_ABSOLUTE_TESTS
        )

        a = np.reshape(a, (2, 3, 3))
        b = np.reshape(b, (2, 3, 3))
        distance = np.reshape(distance, (2, 3))
        np.testing.assert_allclose(
            manhattan_distance(a, b), distance, atol=TOLERANCE_ABSOLUTE_TESTS
        )

    @ignore_numpy_errors
    def test_nan_manhattan_distance(self):
        """
        Test :func:`colour.algebra.common.manhattan_distance` definition nan
        support.
        """

        cases = [-1.0, 0.0, 1.0, -np.inf, np.inf, np.nan]
        cases = np.array(list(set(product(cases, repeat=3))))
        manhattan_distance(cases, cases)


class TestLinearConversion:
    """
    Define :func:`colour.algebra.common.linear_conversion` definition unit
    tests methods.
    """

    def test_linear_conversion(self):
        """Test :func:`colour.algebra.common.linear_conversion` definition."""

        np.testing.assert_allclose(
            linear_conversion(
                np.linspace(0, 1, 10), np.array([0, 1]), np.array([1, np.pi])
            ),
            np.array(
                [
                    1.00000000,
                    1.23795474,
                    1.47590948,
                    1.71386422,
                    1.95181896,
                    2.18977370,
                    2.42772844,
                    2.66568318,
                    2.90363791,
                    3.14159265,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestLinstepFunction:
    """
    Define :func:`colour.algebra.common.linstep_function` definition unit
    tests methods.
    """

    def test_linstep_function(self):
        """Test :func:`colour.algebra.common.linstep_function` definition."""

        np.testing.assert_allclose(
            linstep_function(
                np.linspace(0, 1, 10),
                np.linspace(0, 1, 10),
                np.linspace(0, 2, 10),
            ),
            np.array(
                [
                    0.00000000,
                    0.12345679,
                    0.27160494,
                    0.44444444,
                    0.64197531,
                    0.86419753,
                    1.11111111,
                    1.38271605,
                    1.67901235,
                    2.00000000,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )

        np.testing.assert_allclose(
            linstep_function(
                np.linspace(0, 2, 10),
                np.linspace(0.25, 0.5, 10),
                np.linspace(0.5, 0.75, 10),
                clip=True,
            ),
            np.array(
                [
                    0.25000000,
                    0.33333333,
                    0.41666667,
                    0.50000000,
                    0.58333333,
                    0.63888889,
                    0.66666667,
                    0.69444444,
                    0.72222222,
                    0.75000000,
                ]
            ),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestSmoothstepFunction:
    """
    Define :func:`colour.algebra.common.smoothstep_function` definition unit
    tests methods.
    """

    def test_smoothstep_function(self):
        """Test :func:`colour.algebra.common.smoothstep_function` definition."""

        assert smoothstep_function(0.5) == 0.5
        assert smoothstep_function(0.25) == 0.15625
        assert smoothstep_function(0.75) == 0.84375

        x = np.linspace(-2, 2, 5)
        np.testing.assert_allclose(
            smoothstep_function(x),
            np.array([28.00000, 5.00000, 0.00000, 1.00000, -4.00000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )
        np.testing.assert_allclose(
            smoothstep_function(x, -2, 2, clip=True),
            np.array([0.00000, 0.15625, 0.50000, 0.84375, 1.00000]),
            atol=TOLERANCE_ABSOLUTE_TESTS,
        )


class TestIsIdentity:
    """
    Define :func:`colour.algebra.common.is_identity` definition unit tests
    methods.
    """

    def test_is_identity(self):
        """Test :func:`colour.algebra.common.is_identity` definition."""

        assert is_identity(np.reshape(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]), (3, 3)))

        assert not is_identity(
            np.reshape(np.array([1, 2, 0, 0, 1, 0, 0, 0, 1]), (3, 3))
        )

        assert is_identity(np.reshape(np.array([1, 0, 0, 1]), (2, 2)))

        assert not is_identity(np.reshape(np.array([1, 2, 0, 1]), (2, 2)))


class TestEigenDecomposition:
    """
    Define :func:`colour.algebra.common.eigen_decomposition` definition unit
    tests methods.
    """

    def test_is_identity(self):
        """Test :func:`colour.algebra.common.eigen_decomposition` definition."""

        a = np.diag([1, 2, 3])

        w, v = eigen_decomposition(a)
        np.testing.assert_equal(w, np.array([3.0, 2.0, 1.0]))
        np.testing.assert_equal(
            v, np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        )

        w, v = eigen_decomposition(a, 1)
        np.testing.assert_equal(w, np.array([3.0]))
        np.testing.assert_equal(v, np.array([[0.0], [0.0], [1.0]]))

        w, v = eigen_decomposition(a, descending_order=False)
        np.testing.assert_equal(w, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_equal(
            v, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )

        w, v = eigen_decomposition(a, covariance_matrix=True)
        np.testing.assert_equal(w, np.array([9.0, 4.0, 1.0]))
        np.testing.assert_equal(
            v, np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        )

        w, v = eigen_decomposition(a, descending_order=False, covariance_matrix=True)
        np.testing.assert_equal(w, np.array([1.0, 4.0, 9.0]))
        np.testing.assert_equal(
            v, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )
