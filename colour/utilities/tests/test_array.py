# -*- coding: utf-8 -*-
"""
Defines the unit tests for the :mod:`colour.utilities.array` module.
"""

import numpy as np
import unittest
from collections import namedtuple

from colour.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_INT_DTYPE
from colour.utilities import (
    as_array, as_int_array, as_float_array, as_numeric, as_int, as_float,
    set_float_precision, set_int_precision, as_namedtuple, closest_indexes,
    closest, interval, is_uniform, in_array, tstack, tsplit, row_as_diagonal,
    orient, centroid, fill_nan, ndarray_write, zeros, ones, full,
    index_along_last_axis)
from colour.utilities import is_networkx_installed

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2021 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = [
    'TestAsArray', 'TestAsIntArray', 'TestAsFloatArray', 'TestAsNumeric',
    'TestAsInt', 'TestAsFloat', 'TestSetFloatPrecision', 'TestSetIntPrecision',
    'TestAsNametuple', 'TestClosestIndexes', 'TestClosest', 'TestInterval',
    'TestIsUniform', 'TestInArray', 'TestTstack', 'TestTsplit',
    'TestRowAsDiagonal', 'TestOrient', 'TestCentroid', 'TestFillNan',
    'TestNdarrayWrite', 'TestZeros', 'TestOnes', 'TestFull',
    'TestIndexAlongLastAxis'
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

        np.testing.assert_equal(
            as_array(dict(zip('abc', [1, 2, 3])).values()), np.array([1, 2,
                                                                      3]))


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


class TestSetFloatPrecision(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.set_float_precision` definition unit
    tests methods.
    """

    def test_set_float_precision(self):
        """
        Tests :func:`colour.utilities.array.set_float_precision` definition.
        """

        self.assertEqual(as_float_array(np.ones(3)).dtype, np.float64)

        set_float_precision(np.float16)

        self.assertEqual(as_float_array(np.ones(3)).dtype, np.float16)

        set_float_precision(np.float64)

        self.assertEqual(as_float_array(np.ones(3)).dtype, np.float64)

    def test_set_float_precision_enforcement(self):
        """
        Tests whether :func:`colour.utilities.array.set_float_precision` effect
        is applied through most of *Colour* public API.
        """

        if not is_networkx_installed():  # pragma: no cover
            return

        from colour.appearance import (CAM_Specification_CAM16,
                                       CAM_Specification_CIECAM02)
        from colour.graph.conversion import (CONVERSION_SPECIFICATIONS_DATA,
                                             convert)

        dtype = np.float32
        set_float_precision(dtype)

        for source, target, _callable in CONVERSION_SPECIFICATIONS_DATA:
            if target in ('Hexadecimal', 'Munsell Colour'):
                continue

            # Spectral distributions are instantiated with float64 data and
            # spectral up-sampling optimization fails.
            if ('Spectral Distribution' in (source, target) or
                    target == 'Complementary Wavelength' or
                    target == 'Dominant Wavelength'):
                continue

            a = np.array([(0.25, 0.5, 0.25), (0.25, 0.5, 0.25)])

            if source == 'CAM16':
                a = CAM_Specification_CAM16(J=0.25, M=0.5, h=0.25)

            if source == 'CIECAM02':
                a = CAM_Specification_CIECAM02(J=0.25, M=0.5, h=0.25)

            if source == 'CMYK':
                a = np.array([(0.25, 0.5, 0.25, 0.5), (0.25, 0.5, 0.25, 0.5)])

            if source == 'Hexadecimal':
                a = np.array(['#FFFFFF', '#FFFFFF'])

            if source == 'Munsell Colour':
                a = ['4.2YR 8.1/5.3', '4.2YR 8.1/5.3']

            if source == 'Wavelength':
                a = 555

            if source.endswith(' xy') or source.endswith(' uv'):
                a = np.array([(0.25, 0.5), (0.25, 0.5)])

            def dtype_getter(x):
                """
                dtype getter callable.
                """

                for specification in ('ATD95', 'CIECAM02', 'CAM16', 'Hunt',
                                      'LLAB', 'Nayatani95', 'RLAB'):
                    if target.endswith(specification):
                        return x[0].dtype

                return x.dtype

            self.assertEqual(dtype_getter(convert(a, source, target)), dtype)

    def tearDown(self):
        """
        After tests actions.
        """

        set_float_precision(np.float64)


class TestSetIntPrecision(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.set_int_precision` definition unit
    tests methods.
    """

    def test_set_int_precision(self):
        """
        Tests :func:`colour.utilities.array.set_int_precision` definition.
        """

        self.assertEqual(as_int_array(np.ones(3)).dtype, np.int64)

        set_int_precision(np.int32)

        self.assertEqual(as_int_array(np.ones(3)).dtype, np.int32)

        set_int_precision(np.int64)

        self.assertEqual(as_int_array(np.ones(3)).dtype, np.int64)

    def tearDown(self):
        """
        After tests actions.
        """

        set_int_precision(np.int64)


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


class TestZeros(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.zeros` definition unit tests
    methods.
    """

    def test_zeros(self):
        """
        Tests :func:`colour.utilities.array.zeros` definition.
        """

        np.testing.assert_equal(zeros(3), np.zeros(3))


class TestOnes(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.ones` definition unit tests
    methods.
    """

    def test_ones(self):
        """
        Tests :func:`colour.utilities.array.ones` definition.
        """

        np.testing.assert_equal(ones(3), np.ones(3))


class TestFull(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.full` definition unit tests
    methods.
    """

    def test_full(self):
        """
        Tests :func:`colour.utilities.array.full` definition.
        """

        np.testing.assert_equal(full(3, 0.5), np.full(3, 0.5))


class TestIndexAlongLastAxis(unittest.TestCase):
    """
    Defines :func:`colour.utilities.array.index_along_last_axis` definition
    unit tests methods.
    """

    def test_index_along_last_axis(self):
        """
        Tests :func:`colour.utilities.array.index_along_last_axis` definition.
        """
        a = np.array([[[[0.51090627, 0.86191718, 0.8687926],
                        [0.82738158, 0.80587656, 0.28285687]],
                       [[0.84085977, 0.03851814, 0.06057988],
                        [0.94659267, 0.79308353, 0.30870888]]],
                      [[[0.50758436, 0.24066455, 0.20199051],
                        [0.4507304, 0.84189245, 0.81160878]],
                       [[0.75421871, 0.88187494, 0.01612045],
                        [0.38777511, 0.58905552, 0.32970469]]],
                      [[[0.99285824, 0.738076, 0.0716432],
                        [0.35847844, 0.0367514, 0.18586322]],
                       [[0.72674561, 0.0822759, 0.9771182],
                        [0.90644279, 0.09689787, 0.93483977]]]])

        indexes = np.array([[[0, 1], [0, 1]], [[2, 1], [2, 1]], [[2, 1],
                                                                 [2, 0]]])

        np.testing.assert_equal(
            index_along_last_axis(a, indexes),
            np.array([[[0.51090627, 0.80587656], [0.84085977, 0.79308353]],
                      [[0.20199051, 0.84189245], [0.01612045, 0.58905552]],
                      [[0.0716432, 0.0367514], [0.9771182, 0.90644279]]]))

    def test_compare_with_argmin_argmax(self):
        """
        Tests :func:`colour.utilities.array.index_along_last_axis` definition
        by comparison with :func:`argmin` and :func:`argmax`.
        """

        a = np.random.random((2, 3, 4, 5, 6, 7))

        np.testing.assert_equal(
            index_along_last_axis(a, np.argmin(a, axis=-1)), np.min(
                a, axis=-1))

        np.testing.assert_equal(
            index_along_last_axis(a, np.argmax(a, axis=-1)), np.max(
                a, axis=-1))

    def test_exceptions(self):
        """
        Tests :func:`colour.utilities.array.index_along_last_axis` definition
        handling of invalid inputs.
        """

        a = as_float_array([[11, 12], [21, 22]])

        # Bad shape
        with self.assertRaises(ValueError):
            indexes = np.array([0])
            index_along_last_axis(a, indexes)

        # Indexes out of range
        with self.assertRaises(IndexError):
            indexes = np.array([123, 456])
            index_along_last_axis(a, indexes)

        # Non-integer indexes
        with self.assertRaises(IndexError):
            indexes = np.array([0., 0.])
            index_along_last_axis(a, indexes)


if __name__ == '__main__':
    unittest.main()
