# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.algebra.common` module.
"""

import numpy as np
import unittest

from colour.algebra import (
    is_spow_enabled,
    set_spow_enable,
    spow_enable,
    spow,
    smoothstep_function,
    normalise_maximum,
    vector_dot,
    matrix_dot,
    linear_conversion,
    linstep_function,
    is_identity,
)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestIsSpowEnabled',
    'TestSetSpowEnabled',
    'TestSpowEnable',
    'TestSpow',
    'TestSmoothstepFunction',
    'TestNormaliseMaximum',
    'TestVectorDot',
    'TestMatrixDot',
    'TestLinearConversion',
    'TestLinstepFunction',
    'TestIsIdentity',
]


class TestIsSpowEnabled(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.is_spow_enabled` definition unit
    tests methods.
    """

    def test_is_spow_enabled(self):
        """
        Tests :func:`colour.algebra.common.is_spow_enabled` definition.
        """

        with spow_enable(True):
            self.assertTrue(is_spow_enabled())

        with spow_enable(False):
            self.assertFalse(is_spow_enabled())


class TestSetSpowEnabled(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.set_spow_enable` definition unit
    tests methods.
    """

    def test_set_spow_enable(self):
        """
        Tests :func:`colour.algebra.common.set_spow_enable` definition.
        """

        with spow_enable(is_spow_enabled()):
            set_spow_enable(True)
            self.assertTrue(is_spow_enabled())

        with spow_enable(is_spow_enabled()):
            set_spow_enable(False)
            self.assertFalse(is_spow_enabled())


class TestSpowEnable(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.spow_enable` definition unit
    tests methods.
    """

    def test_spow_enable(self):
        """
        Tests :func:`colour.algebra.common.spow_enable` definition.
        """

        with spow_enable(True):
            self.assertTrue(is_spow_enabled())

        with spow_enable(False):
            self.assertFalse(is_spow_enabled())

        @spow_enable(True)
        def fn_a():
            """
            :func:`spow_enable` unit tests :func:`fn_a` definition.
            """

            self.assertTrue(is_spow_enabled())

        fn_a()

        @spow_enable(False)
        def fn_b():
            """
            :func:`spow_enable` unit tests :func:`fn_b` definition.
            """

            self.assertFalse(is_spow_enabled())

        fn_b()


class TestSpow(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.spow` definition unit
    tests methods.
    """

    def test_spow(self):
        """
        Tests :func:`colour.algebra.common.spow` definition.
        """

        self.assertEqual(spow(2, 2), 4.0)

        self.assertEqual(spow(-2, 2), -4.0)

        np.testing.assert_almost_equal(
            spow([2, -2, -2, 0], [2, 2, 0.15, 0]),
            np.array([4.00000000, -4.00000000, -1.10956947, 0.00000000]),
            decimal=7)

        with spow_enable(True):
            np.testing.assert_almost_equal(
                spow(-2, 0.15), -1.10956947, decimal=7)

        with spow_enable(False):
            np.testing.assert_equal(spow(-2, 0.15), np.nan)


class TestNormaliseMaximum(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.normalise_maximum` definition unit
    tests methods.
    """

    def test_normalise_maximum(self):
        """
        Tests :func:`colour.utilities.array.normalise_maximum` definition.
        """

        np.testing.assert_almost_equal(
            normalise_maximum(np.array([0.20654008, 0.12197225, 0.05136952])),
            np.array([1.00000000, 0.59055003, 0.24871454]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise_maximum(
                np.array([
                    [0.20654008, 0.12197225, 0.05136952],
                    [0.14222010, 0.23042768, 0.10495772],
                    [0.07818780, 0.06157201, 0.28099326],
                ])),
            np.array([
                [0.73503571, 0.43407536, 0.18281406],
                [0.50613349, 0.82004700, 0.37352398],
                [0.27825507, 0.21912273, 1.00000000],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise_maximum(
                np.array([
                    [0.20654008, 0.12197225, 0.05136952],
                    [0.14222010, 0.23042768, 0.10495772],
                    [0.07818780, 0.06157201, 0.28099326],
                ]),
                axis=-1),
            np.array([
                [1.00000000, 0.59055003, 0.24871454],
                [0.61720059, 1.00000000, 0.45549094],
                [0.27825507, 0.21912273, 1.00000000],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise_maximum(
                np.array([0.20654008, 0.12197225, 0.05136952]), factor=10),
            np.array([10.00000000, 5.90550028, 2.48714535]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise_maximum(
                np.array([-0.11518475, -0.10080000, 0.05089373])),
            np.array([0.00000000, 0.00000000, 1.00000000]),
            decimal=7)

        np.testing.assert_almost_equal(
            normalise_maximum(
                np.array([-0.20654008, -0.12197225, 0.05136952]), clip=False),
            np.array([-4.02067374, -2.37440899, 1.00000000]),
            decimal=7)


class TestVectorDot(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.vector_dot` definition unit tests
    methods.
    """

    def test_vector_dot(self):
        """
        Tests :func:`colour.utilities.array.vector_dot` definition.
        """

        m = np.array([
            [0.7328, 0.4296, -0.1624],
            [-0.7036, 1.6975, 0.0061],
            [0.0030, 0.0136, 0.9834],
        ])
        m = np.reshape(np.tile(m, (6, 1)), (6, 3, 3))

        v = np.array([0.20654008, 0.12197225, 0.05136952])
        v = np.tile(v, (6, 1))

        np.testing.assert_almost_equal(
            vector_dot(m, v),
            np.array([
                [0.19540944, 0.06203965, 0.05279523],
                [0.19540944, 0.06203965, 0.05279523],
                [0.19540944, 0.06203965, 0.05279523],
                [0.19540944, 0.06203965, 0.05279523],
                [0.19540944, 0.06203965, 0.05279523],
                [0.19540944, 0.06203965, 0.05279523],
            ]),
            decimal=7)


class TestMatrixDot(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.matrix_dot` definition unit tests
    methods.
    """

    def test_matrix_dot(self):
        """
        Tests :func:`colour.utilities.array.matrix_dot` definition.
        """

        a = np.array([
            [0.7328, 0.4296, -0.1624],
            [-0.7036, 1.6975, 0.0061],
            [0.0030, 0.0136, 0.9834],
        ])
        a = np.reshape(np.tile(a, (6, 1)), (6, 3, 3))

        b = a

        np.testing.assert_almost_equal(
            matrix_dot(a, b),
            np.array(
                [[[0.23424208, 1.04184824, -0.27609032],
                  [-1.70994078, 2.57932265, 0.13061813],
                  [-0.00442036, 0.03774904, 0.96667132]],
                 [[0.23424208, 1.04184824, -0.27609032],
                  [-1.70994078, 2.57932265, 0.13061813],
                  [-0.00442036, 0.03774904, 0.96667132]],
                 [[0.23424208, 1.04184824, -0.27609032],
                  [-1.70994078, 2.57932265, 0.13061813],
                  [-0.00442036, 0.03774904, 0.96667132]],
                 [[0.23424208, 1.04184824, -0.27609032],
                  [-1.70994078, 2.57932265, 0.13061813],
                  [-0.00442036, 0.03774904, 0.96667132]],
                 [[0.23424208, 1.04184824, -0.27609032],
                  [-1.70994078, 2.57932265, 0.13061813],
                  [-0.00442036, 0.03774904, 0.96667132]],
                 [[0.23424208, 1.04184824, -0.27609032],
                  [-1.70994078, 2.57932265, 0.13061813],
                  [-0.00442036, 0.03774904, 0.96667132]]]
            ),
            decimal=7)  # yapf: disable


class TestLinearConversion(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.linear_conversion` definition unit
    tests methods.
    """

    def test_linear_conversion(self):
        """
        Tests :func:`colour.utilities.array.linear_conversion` definition.
        """

        np.testing.assert_almost_equal(
            linear_conversion(
                np.linspace(0, 1, 10), np.array([0, 1]), np.array([1, np.pi])),
            np.array([
                1.00000000, 1.23795474, 1.47590948, 1.71386422, 1.95181896,
                2.18977370, 2.42772844, 2.66568318, 2.90363791, 3.14159265
            ]),
            decimal=8)


class TestLinstepFunction(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.linstep_function` definition unit
    tests methods.
    """

    def test_linstep_function(self):
        """
        Tests :func:`colour.utilities.array.linstep_function` definition.
        """

        np.testing.assert_almost_equal(
            linstep_function(
                np.linspace(0, 1, 10),
                np.linspace(0, 1, 10),
                np.linspace(0, 2, 10),
            ),
            np.array([
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
            ]),
            decimal=8)

        np.testing.assert_almost_equal(
            linstep_function(
                np.linspace(0, 2, 10),
                np.linspace(0.25, 0.5, 10),
                np.linspace(0.5, 0.75, 10),
                clip=True),
            np.array([
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
            ]),
            decimal=8)


class TestSmoothstepFunction(unittest.TestCase):
    """
    Defines :func:`colour.algebra.common.smoothstep_function` definition unit
    tests methods.
    """

    def test_smoothstep_function(self):
        """
        Tests :func:`colour.algebra.common.smoothstep_function` definition.
        """

        self.assertEqual(smoothstep_function(0.5), 0.5)
        self.assertEqual(smoothstep_function(0.25), 0.15625)
        self.assertEqual(smoothstep_function(0.75), 0.84375)

        x = np.linspace(-2, 2, 5)
        np.testing.assert_almost_equal(
            smoothstep_function(x),
            np.array([28.00000, 5.00000, 0.00000, 1.00000, -4.00000]))
        np.testing.assert_almost_equal(
            smoothstep_function(x, -2, 2, clip=True),
            np.array([0.00000, 0.15625, 0.50000, 0.84375, 1.00000]))


class TestIsIdentity(unittest.TestCase):
    """
    Defines :func:`colour.algebra.matrix.is_identity` definition unit tests
    methods.
    """

    def test_is_identity(self):
        """
        Tests :func:`colour.algebra.matrix.is_identity` definition.
        """

        self.assertTrue(
            is_identity(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape([3, 3])))

        self.assertFalse(
            is_identity(np.array([1, 2, 0, 0, 1, 0, 0, 0, 1]).reshape([3, 3])))

        self.assertTrue(is_identity(np.array([1, 0, 0, 1]).reshape([2, 2])))

        self.assertFalse(is_identity(np.array([1, 2, 0, 1]).reshape([2, 2])))


if __name__ == '__main__':
    unittest.main()
