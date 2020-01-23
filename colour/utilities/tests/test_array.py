# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.utilities.array` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
from collections import namedtuple

from colour.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE
from colour.utilities import (
    as_array, as_int_array, as_float_array, as_numeric, as_int, as_float,
    as_namedtuple, closest_indexes, closest, normalise_maximum, interval,
    is_uniform, in_array, tstack, tsplit, row_as_diagonal, dot_vector,
    dot_matrix, orient, centroid, linear_conversion, lerp, fill_nan,
    ndarray_write)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestAsArray', 'TestAsIntArray', 'TestAsFloatArray', 'TestAsNumeric',
    'TestAsInt', 'TestAsFloat', 'TestAsNametuple', 'TestClosestIndexes',
    'TestClosest', 'TestNormaliseMaximum', 'TestInterval', 'TestIsUniform',
    'TestInArray', 'TestTstack', 'TestTsplit', 'TestRowAsDiagonal',
    'TestDotVector', 'TestDotMatrix', 'TestOrient', 'TestCentroid',
    'TestLinearConversion', 'TestLerp', 'TestFillNan', 'TestNdarrayWrite'
]


class TestAsArray(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.as_array` definition unit tests
    methods.
    """

    def test_as_array(self):
        """
        Tests :func:`colour.utilities.array.as_array` definition.
        """

        np.testing.assert_equal(as_array([1, 2, 3]), np.array([1, 2, 3]))

        self.assertEqual(
            as_array([1, 2, 3], DEFAULT_FLOAT_DTYPE).dtype,
            DEFAULT_FLOAT_DTYPE)

        self.assertEqual(
            as_array([1, 2, 3], DEFAULT_INT_DTYPE).dtype, DEFAULT_INT_DTYPE)


class TestAsIntArray(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.as_int_array` definition unit tests
    methods.
    """

    def test_as_int_array(self):
        """
        Tests :func:`colour.utilities.array.as_int_array` definition.
        """

        np.testing.assert_equal(
            as_int_array([1.0, 2.0, 3.0]), np.array([1, 2, 3]))

        self.assertEqual(as_int_array([1, 2, 3]).dtype, DEFAULT_INT_DTYPE)


class TestAsFloatArray(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.as_float_array` definition unit tests
    methods.
    """

    def test_as_float_array(self):
        """
        Tests :func:`colour.utilities.array.as_float_array` definition.
        """

        np.testing.assert_equal(as_float_array([1, 2, 3]), np.array([1, 2, 3]))

        self.assertEqual(as_float_array([1, 2, 3]).dtype, DEFAULT_FLOAT_DTYPE)


class TestAsNumeric(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.as_numeric` definition unit tests
    methods.
    """

    def test_as_numeric(self):
        """
        Tests :func:`colour.utilities.array.as_numeric` definition.
        """

        self.assertEqual(as_numeric(1), 1.0)

        self.assertEqual(as_numeric(np.array([1])), 1.0)

        np.testing.assert_almost_equal(
            as_numeric(np.array([1, 2, 3])), np.array([1.0, 2.0, 3.0]))

        self.assertIsInstance(as_numeric(1), DEFAULT_FLOAT_DTYPE)

        self.assertIsInstance(as_numeric(1, int), int)

        self.assertListEqual(as_numeric(['John', 'Doe']), ['John', 'Doe'])

        self.assertEqual(as_numeric('John Doe'), 'John Doe')


class TestAsInt(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.as_int` definition unit tests
    methods.
    """

    def test_as_int(self):
        """
        Tests :func:`colour.utilities.array.as_int` definition.
        """

        self.assertEqual(as_int(1), 1)

        self.assertEqual(as_int(np.array([1])), 1)

        np.testing.assert_almost_equal(
            as_int(np.array([1.0, 2.0, 3.0])), np.array([1, 2, 3]))

        self.assertEqual(
            as_int(np.array([1.0, 2.0, 3.0])).dtype, DEFAULT_INT_DTYPE)

        self.assertIsInstance(as_int(1), int)


class TestAsFloat(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.as_float` definition unit tests
    methods.
    """

    def test_as_float(self):
        """
        Tests :func:`colour.utilities.array.as_float` definition.
        """

        self.assertEqual(as_float(1), 1.0)

        self.assertEqual(as_float(np.array([1])), 1.0)

        np.testing.assert_almost_equal(
            as_float(np.array([1, 2, 3])), np.array([1.0, 2.0, 3.0]))

        self.assertEqual(
            as_float(np.array([1, 2, 3])).dtype, DEFAULT_FLOAT_DTYPE)

        self.assertIsInstance(as_float(1), DEFAULT_FLOAT_DTYPE)


class TestAsNametuple(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.as_namedtuple` definition unit tests
    methods.
    """

    def test_as_namedtuple(self):
        """
        Tests :func:`colour.utilities.array.as_namedtuple` definition.
        """

        NamedTuple = namedtuple('NamedTuple', 'a b c')

        a_a = np.ones(3)
        a_b = np.ones(3) + 1
        a_c = np.ones(3) + 2

        named_tuple = NamedTuple(a_a, a_b, a_c)

        self.assertEqual(named_tuple, as_namedtuple(named_tuple, NamedTuple))

        self.assertEqual(
            named_tuple,
            as_namedtuple({
                'a': a_a,
                'b': a_b,
                'c': a_c
            }, NamedTuple))

        self.assertEqual(named_tuple, as_namedtuple([a_a, a_b, a_c],
                                                    NamedTuple))

        a_r = np.array(
            [tuple(a) for a in np.transpose((a_a, a_b, a_c)).tolist()],
            dtype=[(str('a'), str('f8')),
                   (str('b'), str('f8')),
                   (str('c'), str('f8'))])  # yapf: disable
        np.testing.assert_array_equal(
            np.array(named_tuple), np.array(as_namedtuple(a_r, NamedTuple)))


class TestClosestIndexes(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.closest_indexes` definition unit
    tests methods.
    """

    def test_closest_indexes(self):
        """
        Tests :func:`colour.utilities.array.closest_indexes` definition.
        """

        a = np.array([
            24.31357115,
            63.62396289,
            55.71528816,
            62.70988028,
            46.84480573,
            25.40026416,
        ])

        self.assertEqual(closest_indexes(a, 63.05), 3)

        self.assertEqual(closest_indexes(a, 51.15), 4)

        self.assertEqual(closest_indexes(a, 24.90), 5)

        np.testing.assert_array_equal(
            closest_indexes(a, np.array([63.05, 51.15, 24.90])),
            np.array([3, 4, 5]))


class TestClosest(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.closest` definition unit tests
    methods.
    """

    def test_closest(self):
        """
        Tests :func:`colour.utilities.array.closest` definition.
        """

        a = np.array([
            24.31357115,
            63.62396289,
            55.71528816,
            62.70988028,
            46.84480573,
            25.40026416,
        ])

        self.assertEqual(closest(a, 63.05), 62.70988028)

        self.assertEqual(closest(a, 51.15), 46.84480573)

        self.assertEqual(closest(a, 24.90), 25.40026416)

        np.testing.assert_almost_equal(
            closest(a, np.array([63.05, 51.15, 24.90])),
            np.array([62.70988028, 46.84480573, 25.40026416]),
            decimal=7)


class TestNormaliseMaximum(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.normalise_maximum` definition units
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


class TestInterval(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.interval` definition unit tests
    methods.
    """

    def test_interval(self):
        """
        Tests :func:`colour.utilities.array.interval` definition.
        """

        np.testing.assert_almost_equal(
            interval(range(0, 10, 2)), np.array([2]))

        np.testing.assert_almost_equal(
            interval(range(0, 10, 2), False), np.array([2, 2, 2, 2]))

        np.testing.assert_almost_equal(
            interval([1, 2, 3, 4, 6, 6.5]), np.array([0.5, 1.0, 2.0]))

        np.testing.assert_almost_equal(
            interval([1, 2, 3, 4, 6, 6.5], False),
            np.array([1.0, 1.0, 1.0, 2.0, 0.5]))


class TestIsUniform(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.is_uniform` definition unit tests
    methods.
    """

    def test_is_uniform(self):
        """
        Tests :func:`colour.utilities.array.is_uniform` definition.
        """

        self.assertTrue(is_uniform(range(0, 10, 2)))

        self.assertFalse(is_uniform([1, 2, 3, 4, 6]))


class TestInArray(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.in_array` definition unit tests
    methods.
    """

    def test_in_array(self):
        """
        Tests :func:`colour.utilities.array.in_array` definition.
        """

        self.assertTrue(
            np.array_equal(
                in_array(np.array([0.50, 0.60]), np.linspace(0, 10, 101)),
                np.array([True, True])))

        self.assertFalse(
            np.array_equal(
                in_array(np.array([0.50, 0.61]), np.linspace(0, 10, 101)),
                np.array([True, True])))

        self.assertTrue(
            np.array_equal(
                in_array(np.array([[0.50], [0.60]]), np.linspace(0, 10, 101)),
                np.array([[True], [True]])))

    def test_n_dimensional_in_array(self):
        """
        Tests :func:`colour.utilities.array.in_array` definition n-dimensional
        support.
        """

        np.testing.assert_almost_equal(
            in_array(np.array([0.50, 0.60]), np.linspace(0, 10, 101)).shape,
            np.array([2]))

        np.testing.assert_almost_equal(
            in_array(np.array([[0.50, 0.60]]), np.linspace(0, 10, 101)).shape,
            np.array([1, 2]))

        np.testing.assert_almost_equal(
            in_array(np.array([[0.50], [0.60]]), np.linspace(0, 10,
                                                             101)).shape,
            np.array([2, 1]))


class TestTstack(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.tstack` definition unit tests
    methods.
    """

    def test_tstack(self):
        """
        Tests :func:`colour.utilities.array.tstack` definition.
        """

        a = 0
        np.testing.assert_almost_equal(tstack([a, a, a]), np.array([0, 0, 0]))

        a = np.arange(0, 6)
        np.testing.assert_almost_equal(
            tstack([a, a, a]),
            np.array([
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
            ]))

        a = np.reshape(a, (1, 6))
        np.testing.assert_almost_equal(
            tstack([a, a, a]),
            np.array([[
                [0, 0, 0],
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
            ]]))

        a = np.reshape(a, (1, 2, 3))
        np.testing.assert_almost_equal(
            tstack([a, a, a]),
            np.array([[
                [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
                [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
            ]]))


class TestTsplit(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.tsplit` definition unit tests
    methods.
    """

    def test_tsplit(self):
        """
        Tests :func:`colour.utilities.array.tsplit` definition.
        """

        a = np.array([0, 0, 0])
        np.testing.assert_almost_equal(tsplit(a), np.array([0, 0, 0]))
        a = np.array([
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ])
        np.testing.assert_almost_equal(
            tsplit(a),
            np.array([
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
            ]))

        a = np.array([
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
        ])
        np.testing.assert_almost_equal(
            tsplit(a),
            np.array([
                [[0, 1, 2, 3, 4, 5]],
                [[0, 1, 2, 3, 4, 5]],
                [[0, 1, 2, 3, 4, 5]],
            ]))

        a = np.array([[
            [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            [[3, 3, 3], [4, 4, 4], [5, 5, 5]],
        ]])
        np.testing.assert_almost_equal(
            tsplit(a),
            np.array([
                [[[0, 1, 2], [3, 4, 5]]],
                [[[0, 1, 2], [3, 4, 5]]],
                [[[0, 1, 2], [3, 4, 5]]],
            ]))


class TestRowAsDiagonal(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.row_as_diagonal` definition unit
    tests methods.
    """

    def test_row_as_diagonal(self):
        """
        Tests :func:`colour.utilities.array.row_as_diagonal` definition.
        """

        np.testing.assert_almost_equal(
            row_as_diagonal(np.array(
                [[0.25891593, 0.07299478, 0.36586996],
                 [0.30851087, 0.37131459, 0.16274825],
                 [0.71061831, 0.67718718, 0.09562581],
                 [0.71588836, 0.76772047, 0.15476079],
                 [0.92985142, 0.22263399, 0.88027331]])
            ),
            np.array(
                [[[0.25891593, 0.00000000, 0.00000000],
                  [0.00000000, 0.07299478, 0.00000000],
                  [0.00000000, 0.00000000, 0.36586996]],
                 [[0.30851087, 0.00000000, 0.00000000],
                  [0.00000000, 0.37131459, 0.00000000],
                  [0.00000000, 0.00000000, 0.16274825]],
                 [[0.71061831, 0.00000000, 0.00000000],
                  [0.00000000, 0.67718718, 0.00000000],
                  [0.00000000, 0.00000000, 0.09562581]],
                 [[0.71588836, 0.00000000, 0.00000000],
                  [0.00000000, 0.76772047, 0.00000000],
                  [0.00000000, 0.00000000, 0.15476079]],
                 [[0.92985142, 0.00000000, 0.00000000],
                  [0.00000000, 0.22263399, 0.00000000],
                  [0.00000000, 0.00000000, 0.88027331]]]
            )
        )  # yapf: disable


class TestDotVector(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.dot_vector` definition unit tests
    methods.
    """

    def test_dot_vector(self):
        """
        Tests :func:`colour.utilities.array.dot_vector` definition.
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
            dot_vector(m, v),
            np.array([
                [0.19540944, 0.06203965, 0.05279523],
                [0.19540944, 0.06203965, 0.05279523],
                [0.19540944, 0.06203965, 0.05279523],
                [0.19540944, 0.06203965, 0.05279523],
                [0.19540944, 0.06203965, 0.05279523],
                [0.19540944, 0.06203965, 0.05279523],
            ]),
            decimal=7)


class TestDotMatrix(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.dot_matrix` definition unit tests
    methods.
    """

    def test_dot_matrix(self):
        """
        Tests :func:`colour.utilities.array.dot_matrix` definition.
        """

        a = np.array([
            [0.7328, 0.4296, -0.1624],
            [-0.7036, 1.6975, 0.0061],
            [0.0030, 0.0136, 0.9834],
        ])
        a = np.reshape(np.tile(a, (6, 1)), (6, 3, 3))

        b = a

        np.testing.assert_almost_equal(
            dot_matrix(a, b),
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


class TestOrient(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.orient` definition unit tests
    methods.
    """

    def test_orient(self):
        """
        Tests :func:`colour.utilities.array.orient` definition.
        """

        a = np.tile(np.arange(5), (5, 1))

        np.testing.assert_almost_equal(orient(a, 'Null'), a, decimal=7)

        np.testing.assert_almost_equal(
            orient(a, 'Flip'),
            np.array([
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            orient(a, 'Flop'),
            np.array([
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            orient(a, '90 CW'),
            np.array([
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            orient(a, '90 CCW'),
            np.array([
                [4, 4, 4, 4, 4],
                [3, 3, 3, 3, 3],
                [2, 2, 2, 2, 2],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
            ]),
            decimal=7)

        np.testing.assert_almost_equal(
            orient(a, '180'),
            np.array([
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
                [4, 3, 2, 1, 0],
            ]),
            decimal=7)


class TestCentroid(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.centroid` definition unit tests
    methods.
    """

    def test_centroid(self):
        """
        Tests :func:`colour.utilities.array.centroid` definition.
        """

        a = np.arange(5)
        np.testing.assert_array_equal(centroid(a), np.array([3]))

        a = np.tile(a, (5, 1))
        np.testing.assert_array_equal(centroid(a), np.array([2, 3]))

        a = np.tile(np.linspace(0, 1, 10), (10, 1))
        np.testing.assert_array_equal(centroid(a), np.array([4, 6]))

        a = tstack([a, a, a])
        np.testing.assert_array_equal(centroid(a), np.array([4, 6, 1]))


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


class TestLerp(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.lerp` definition unit
    tests methods.
    """

    def test_lerp(self):
        """
        Tests :func:`colour.utilities.array.lerp` definition.
        """

        np.testing.assert_almost_equal(
            lerp(
                np.linspace(0, 1, 10),
                np.linspace(0, 2, 10),
                np.linspace(0, 1, 10),
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


class TestFillNan(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.fill_nan` definition unit tests
    methods.
    """

    def test_fill_nan(self):
        """
        Tests :func:`colour.utilities.array.fill_nan` definition.
        """

        a = np.array([0.1, 0.2, np.nan, 0.4, 0.5])
        np.testing.assert_almost_equal(
            fill_nan(a), np.array([0.1, 0.2, 0.3, 0.4, 0.5]), decimal=7)

        np.testing.assert_almost_equal(
            fill_nan(a, method='Constant', default=8.0),
            np.array([0.1, 0.2, 8.0, 0.4, 0.5]),
            decimal=7)


class TestNdarrayWrite(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.ndarray_write` definition unit tests
    methods.
    """

    def test_ndarray_write(self):
        """
        Tests :func:`colour.utilities.array.ndarray_write` definition.
        """

        a = np.linspace(0, 1, 10)
        a.setflags(write=False)

        with self.assertRaises(ValueError):
            a += 1

        with ndarray_write(a):
            a += 1


if __name__ == '__main__':
    unittest.main()
