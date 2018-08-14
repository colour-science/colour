# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.io.luts.lut` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import textwrap
import unittest

from colour.algebra import random_triplet_generator, spow
from colour.io.luts.lut import AbstractLUT
from colour.io.luts import (AbstractLUTSequenceOperator, LUT1D, LUT2D, LUT3D,
                            LUTSequence)
from colour.models import function_gamma
from colour.utilities import tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'RESOURCES_DIRECTORY', 'TestAbstractLUT', 'TestLUT', 'TestLUT1D',
    'TestLUT2D', 'TestLUT3D', 'TestAbstractLUTSequenceOperator',
    'TestLUTSequence'
]

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'resources')

RANDOM_TRIPLETS = np.reshape(
    list(random_triplet_generator(8, random_state=np.random.RandomState(4))),
    (4, 2, 3))


class TestAbstractLUT(unittest.TestCase):
    """
    Defines :class:`colour.io.luts.lut.AbstractLUT` class unit tests methods.
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('table', 'name', 'dimensions', 'domain', 'size',
                               'comments')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(AbstractLUT))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', '__repr__', '__eq__', '__ne__',
                            '__add__', '__iadd__', '__sub__', '__isub__',
                            '__mul__', '__imul__', '__div__', '__idiv__',
                            '__pow__', '__ipow__', 'arithmetical_operation',
                            'linear_table', 'apply', 'copy')

        for method in required_methods:
            self.assertIn(method, dir(AbstractLUT))


class TestLUT(unittest.TestCase):
    """
    Defines :class:`colour.io.luts.lut.LUT1D`,
    :class:`colour.io.luts.lut.LUT2D` and
    :class:`colour.io.luts.lut.LUT3D` classes common unit tests methods.
    """

    def __init__(self, *args):
        """
        Create an instance of the class.

        Other Parameters
        ----------------
        \*args : list, optional
            Arguments.
        """

        super(TestLUT, self).__init__(*args)

        self._LUT_factory = None

        self._table_1 = None
        self._table_2 = None
        self._domain_1 = None
        self._domain_2 = None
        self._dimensions = None
        self._str = None
        self._repr = None
        self._applied_1 = None
        self._applied_2 = None

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('linear_table', )

        for method in required_methods:
            self.assertIn(method, dir(LUT1D))

    def test__init__(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.__init__`,
        :class:`colour.io.luts.lut.LUT2D.__init__` and
        :class:`colour.io.luts.lut.LUT3D.__init__` methods.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT = self._LUT_factory(self._table_1)

        np.testing.assert_almost_equal(LUT.table, self._table_1, decimal=7)

        self.assertEqual(str(id(LUT)), LUT.name)

        np.testing.assert_array_equal(LUT.domain, self._domain_1)

        self.assertEqual(LUT.dimensions, self._dimensions)

    def test_table(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.table`,
        :class:`colour.io.luts.lut.LUT2D.table` and
        :class:`colour.io.luts.lut.LUT3D.table` properties.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT = self._LUT_factory()

        np.testing.assert_array_equal(LUT.table, LUT.linear_table())

        table_1 = self._table_1 * 0.8 + 0.1
        LUT.table = table_1
        np.testing.assert_almost_equal(LUT.table, table_1, decimal=7)

    def test_name(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.name`,
        :class:`colour.io.luts.lut.LUT2D.name` and
        :class:`colour.io.luts.lut.LUT3D.name` properties.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT = self._LUT_factory(self._table_1)

        self.assertEqual(LUT.name, str(id(LUT)))

        # pylint: disable=E1102
        LUT = self._LUT_factory()

        self.assertEqual(LUT.name, 'Unity {0}'.format(self._table_1.shape[0]))

    def test_domain(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.domain`,
        :class:`colour.io.luts.lut.LUT2D.domain` and
        :class:`colour.io.luts.lut.LUT3D.domain` properties.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT = self._LUT_factory()

        np.testing.assert_array_equal(LUT.domain, self._domain_1)

        domain = self._domain_1 * 0.8 + 0.1
        LUT.domain = domain
        np.testing.assert_array_equal(LUT.domain, domain)

    def test_size(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.size`,
        :class:`colour.io.luts.lut.LUT2D.size` and
        :class:`colour.io.luts.lut.LUT3D.size` properties.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT = self._LUT_factory()

        self.assertEqual(LUT.size, LUT.table.shape[0])

    def test_dimensions(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.dimensions`,
        :class:`colour.io.luts.lut.LUT2D.dimensions` and
        :class:`colour.io.luts.lut.LUT3D.dimensions` properties.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT = self._LUT_factory()

        self.assertEqual(LUT.dimensions, self._dimensions)

    def test_comments(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.comments`,
        :class:`colour.io.luts.lut.LUT2D.comments` and
        :class:`colour.io.luts.lut.LUT3D.comments` properties.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT = self._LUT_factory()
        self.assertListEqual(LUT.comments, [])

        comments = ['A first comment.', 'A second comment.']
        # pylint: disable=E1102
        LUT = self._LUT_factory(comments=comments)

        self.assertListEqual(LUT.comments, comments)

    def test__str__(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.__str__`,
        :class:`colour.io.luts.lut.LUT2D.__str__` and
        :class:`colour.io.luts.lut.LUT3D.__str__` methods.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT = self._LUT_factory(name='Nemo')

        self.assertEqual(str(LUT), self._str)

    def test__repr__(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.__repr__`,
        :class:`colour.io.luts.lut.LUT2D.__repr__` and
        :class:`colour.io.luts.lut.LUT3D.__repr__` methods.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT = self._LUT_factory(name='Nemo')

        # The default LUT representation is too large to be embedded, given
        # that :class:`colour.io.luts.lut.LUT3D.__str__` method is defined by
        # :class:`colour.io.luts.lut.AbstractLUT.__str__` method, the two other
        # tests should reasonably cover this case.
        if self._dimensions == 3:
            return

        self.assertEqual(repr(LUT), self._repr)

    def test__eq__(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.__eq__`,
        :class:`colour.io.luts.lut.LUT2D.__eq__` and
        :class:`colour.io.luts.lut.LUT3D.__eq__` methods.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT_1 = self._LUT_factory()
        # pylint: disable=E1102
        LUT_2 = self._LUT_factory()

        self.assertEqual(LUT_1, LUT_2)

    def test__ne__(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.__ne__`,
        :class:`colour.io.luts.lut.LUT2D.__ne__` and
        :class:`colour.io.luts.lut.LUT3D.__ne__` methods.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT_1 = self._LUT_factory()
        # pylint: disable=E1102
        LUT_2 = self._LUT_factory()

        LUT_2 += 0.1
        self.assertNotEqual(LUT_1, LUT_2)

        # pylint: disable=E1102
        LUT_2 = self._LUT_factory()
        LUT_2.domain = self._domain_1 * 0.8 + 0.1
        self.assertNotEqual(LUT_1, LUT_2)

    def test_arithmetical_operation(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.arithmetical_operation`,
        :class:`colour.io.luts.lut.LUT2D.arithmetical_operation` and
        :class:`colour.io.luts.lut.LUT3D.arithmetical_operation` methods.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT_1 = self._LUT_factory()
        # pylint: disable=E1102
        LUT_2 = self._LUT_factory()

        np.testing.assert_almost_equal(
            LUT_1.arithmetical_operation(10, '+', False).table,
            self._table_1 + 10,
            decimal=7)

        np.testing.assert_almost_equal(
            LUT_1.arithmetical_operation(10, '-', False).table,
            self._table_1 - 10,
            decimal=7)

        np.testing.assert_almost_equal(
            LUT_1.arithmetical_operation(10, '*', False).table,
            self._table_1 * 10,
            decimal=7)

        np.testing.assert_almost_equal(
            LUT_1.arithmetical_operation(10, '/', False).table,
            self._table_1 / 10,
            decimal=7)

        np.testing.assert_almost_equal(
            LUT_1.arithmetical_operation(10, '**', False).table,
            self._table_1 ** 10,
            decimal=7)

        np.testing.assert_almost_equal(
            LUT_2.arithmetical_operation(10, '+', True).table,
            self._table_1 + 10,
            decimal=7)

        np.testing.assert_almost_equal(
            LUT_2.arithmetical_operation(10, '-', True).table,
            self._table_1,
            decimal=7)

        np.testing.assert_almost_equal(
            LUT_2.arithmetical_operation(10, '*', True).table,
            self._table_1 * 10,
            decimal=7)

        np.testing.assert_almost_equal(
            LUT_2.arithmetical_operation(10, '/', True).table,
            self._table_1,
            decimal=7)

        np.testing.assert_almost_equal(
            LUT_2.arithmetical_operation(10, '**', True).table,
            self._table_1 ** 10,
            decimal=7)

        # pylint: disable=E1102
        LUT_2 = self._LUT_factory()

        np.testing.assert_almost_equal(
            LUT_2.arithmetical_operation(self._table_1, '+', False).table,
            LUT_2.table + self._table_1,
            decimal=7)

        np.testing.assert_almost_equal(
            LUT_2.arithmetical_operation(LUT_2, '+', False).table,
            LUT_2.table + LUT_2.table,
            decimal=7)

    def test_linear_table(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.linear_table`,
        :class:`colour.io.luts.lut.LUT2D.linear_table` and
        :class:`colour.io.luts.lut.LUT3D.linear_table` methods.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT_1 = self._LUT_factory()

        np.testing.assert_almost_equal(
            LUT_1.linear_table(), self._table_1, decimal=7)

    def test_apply(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.apply`,
        :class:`colour.io.luts.lut.LUT2D.apply` and
        :class:`colour.io.luts.lut.LUT3D.apply` methods.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT_1 = self._LUT_factory(self._table_2)

        np.testing.assert_almost_equal(
            LUT_1.apply(RANDOM_TRIPLETS), self._applied_1, decimal=7)

        # pylint: disable=E1102
        LUT_2 = self._LUT_factory(domain=self._domain_2)
        LUT_2.table = spow(LUT_2.table, 1 / 2.2)

        np.testing.assert_almost_equal(
            LUT_2.apply(RANDOM_TRIPLETS), self._applied_2, decimal=7)

    def test_copy(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.copy`,
        :class:`colour.io.luts.lut.LUT2D.copy` and
        :class:`colour.io.luts.lut.LUT3D.copy` methods.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT_1 = self._LUT_factory()

        self.assertIsNot(LUT_1, LUT_1.copy())
        self.assertEqual(LUT_1, LUT_1.copy())


class TestLUT1D(TestLUT):
    """
    Defines :class:`colour.io.luts.lut.LUT1D` class unit tests methods.
    """

    def __init__(self, *args):
        """
        Create an instance of the class.

        Other Parameters
        ----------------
        \*args : list, optional
            Arguments.
        """

        super(TestLUT1D, self).__init__(*args)

        self._LUT_factory = LUT1D

        self._table_1 = np.linspace(0, 1, 10)
        self._table_2 = self._table_1 ** (1 / 2.2)
        self._domain_1 = np.array([0, 1])
        self._domain_2 = np.array([-0.1, 1.5])
        self._dimensions = 1
        self._str = textwrap.dedent("""
            LUT1D - Nemo
            ------------

            Dimensions : 1
            Domain     : [0 1]
            Size       : (10,)""")[1:]
        self._repr = textwrap.dedent("""
    LUT1D([ 0.        ,  0.11111111,  0.22222222,  0.33333333,  0.44444444,
            0.55555556,  0.66666667,  0.77777778,  0.88888889,  1.        ],
          name='Nemo',
          domain=[0, 1])""")[1:]
        self._applied_1 = np.array([
            [[0.98453144, 0.76000720, 0.98718436],
             [0.85784314, 0.84855994, 0.49723089]],
            [[0.98886872, 0.02065388, 0.53304051],
             [0.68433298, 0.89285575, 0.47463489]],
            [[0.93486051, 0.99221212, 0.43308440],
             [0.79040970, 0.02978976, 0.64753760]],
            [[0.14639477, 0.97966294, 0.68536703],
             [0.97606176, 0.89633381, 0.93651642]],
        ])
        self._applied_2 = np.array([
            [[0.98486877, 0.75787807, 0.98736681],
             [0.85682563, 0.84736915, 0.4879954]],
            [[0.98895283, 0.04585089, 0.53461565],
             [0.68473291, 0.89255862, 0.46473837]],
            [[0.93403795, 0.99210103, 0.42197234],
             [0.79047033, 0.05614915, 0.64540281]],
            [[0.18759013, 0.97981413, 0.68561444],
             [0.97606266, 0.89639002, 0.9356489]],
        ])


class TestLUT2D(TestLUT):
    """
    Defines :class:`colour.io.luts.lut.LUT2D` class unit tests methods.
    """

    def __init__(self, *args):
        """
        Create an instance of the class.

        Other Parameters
        ----------------
        \*args : list, optional
            Arguments.
        """

        super(TestLUT2D, self).__init__(*args)

        self._LUT_factory = LUT2D

        samples = np.linspace(0, 1, 10)
        self._table_1 = tstack([samples, samples, samples])
        self._table_2 = spow(self._table_1, 1 / 2.2)
        self._domain_1 = np.array([[0, 0, 0], [1, 1, 1]])
        self._domain_2 = np.array([[-0.1, -0.1, -0.1], [1.5, 1.5, 1.5]])
        self._dimensions = 2
        self._str = textwrap.dedent("""
            LUT2D - Nemo
            ------------

            Dimensions : 2
            Domain     : [[0 0 0]
                          [1 1 1]]
            Size       : (10, 3)""")[1:]
        self._repr = textwrap.dedent("""
            LUT2D([[ 0.        ,  0.        ,  0.        ],
                   [ 0.11111111,  0.11111111,  0.11111111],
                   [ 0.22222222,  0.22222222,  0.22222222],
                   [ 0.33333333,  0.33333333,  0.33333333],
                   [ 0.44444444,  0.44444444,  0.44444444],
                   [ 0.55555556,  0.55555556,  0.55555556],
                   [ 0.66666667,  0.66666667,  0.66666667],
                   [ 0.77777778,  0.77777778,  0.77777778],
                   [ 0.88888889,  0.88888889,  0.88888889],
                   [ 1.        ,  1.        ,  1.        ]],
                  name='Nemo',
                  domain=[[0, 0, 0],
                          [1, 1, 1]])""")[1:]
        self._applied_1 = np.array([
            [[0.98453144, 0.76000720, 0.98718436],
             [0.85784314, 0.84855994, 0.49723089]],
            [[0.98886872, 0.02065388, 0.53304051],
             [0.68433298, 0.89285575, 0.47463489]],
            [[0.93486051, 0.99221212, 0.43308440],
             [0.79040970, 0.02978976, 0.64753760]],
            [[0.14639477, 0.97966294, 0.68536703],
             [0.97606176, 0.89633381, 0.93651642]],
        ])
        self._applied_2 = np.array([
            [[0.98486877, 0.75787807, 0.98736681],
             [0.85682563, 0.84736915, 0.4879954]],
            [[0.98895283, 0.04585089, 0.53461565],
             [0.68473291, 0.89255862, 0.46473837]],
            [[0.93403795, 0.99210103, 0.42197234],
             [0.79047033, 0.05614915, 0.64540281]],
            [[0.18759013, 0.97981413, 0.68561444],
             [0.97606266, 0.89639002, 0.9356489]],
        ])


class TestLUT3D(TestLUT):
    """
    Defines :class:`colour.io.luts.lut.LUT3D` class unit tests methods.
    """

    def __init__(self, *args):
        """
        Create an instance of the class.

        Other Parameters
        ----------------
        \*args : list, optional
            Arguments.
        """

        super(TestLUT3D, self).__init__(*args)

        self._LUT_factory = LUT3D

        size = 33
        domain = np.array([[0, 0, 0], [1, 1, 1]])
        R, G, B = tsplit(domain)
        samples = [np.linspace(a[0], a[1], size) for a in (B, G, R)]
        table_1 = np.meshgrid(*samples, indexing='ij')
        table_1 = np.transpose(table_1).reshape((size, size, size, 3))
        self._table_1 = np.flip(table_1, -1)
        self._table_2 = spow(self._table_1, 1 / 2.2)
        self._domain_1 = domain
        self._domain_2 = np.array([[-0.1, -0.1, -0.1], [1.5, 1.5, 1.5]])
        self._dimensions = 3
        self._str = textwrap.dedent("""
            LUT3D - Nemo
            ------------

            Dimensions : 3
            Domain     : [[0 0 0]
                          [1 1 1]]
            Size       : (33, 33, 33, 3)""")[1:]
        self._repr = 'Undefined'
        self._applied_1 = np.array([
            [[0.98486974, 0.76022687, 0.98747624],
             [0.85844632, 0.84903362, 0.49827272]],
            [[0.98912224, 0.04125691, 0.53531556],
             [0.68479344, 0.89287549, 0.47829965]],
            [[0.93518100, 0.99238949, 0.43911364],
             [0.79116284, 0.05950617, 0.64907649]],
            [[0.23859990, 0.98002765, 0.68577990],
             [0.97644600, 0.89645863, 0.93680839]],
        ])
        self._applied_2 = np.array([
            [[0.98480377, 0.76026078, 0.98740998],
             [0.85836205, 0.84905549, 0.49768493]],
            [[0.98906469, 0.03192703, 0.53526504],
             [0.68458573, 0.89277461, 0.47842591]],
            [[0.93514331, 0.99234923, 0.4385062],
             [0.79115345, 0.04604939, 0.64892459]],
            [[0.22629886, 0.98002097, 0.68556852],
             [0.97646946, 0.89639137, 0.9367549]],
        ])


class TestAbstractLUTSequenceOperator(unittest.TestCase):
    """
    Defines :class:`colour.io.luts.lut.AbstractLUTSequenceOperator` class unit
    tests methods.
    """

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('apply', )

        for method in required_methods:
            self.assertIn(method, dir(AbstractLUTSequenceOperator))


class TestLUTSequence(unittest.TestCase):
    """
    Defines :class:`colour.io.luts.lut.LUTSequence` class unit tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._LUT_1 = LUT1D(LUT1D.linear_table(16) + 0.125, 'Nemo 1D')
        self._LUT_2 = LUT3D(LUT3D.linear_table(16) ** (1 / 2.2), 'Nemo 3D')
        self._LUT_3 = LUT2D(LUT2D.linear_table(16) * 0.750, 'Nemo 2D')
        self._LUT_sequence = LUTSequence(self._LUT_1, self._LUT_2, self._LUT_3)

        samples = np.linspace(0, 1, 5)

        self._RGB = tstack([samples, samples, samples])

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('sequence', )

        for attribute in required_attributes:
            self.assertIn(attribute, dir(LUTSequence))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__getitem__', '__setitem__', '__delitem__',
                            '__len__', '__str__', '__repr__', '__eq__',
                            '__ne__', 'insert', 'apply', 'copy')

        for method in required_methods:
            self.assertIn(method, dir(LUTSequence))

    def test__init__(self):
        """
        Tests :class:`colour.io.luts.lut.LUTSequence.__init__` method.
        """

        self.assertEqual(
            LUTSequence(self._LUT_1, self._LUT_2, self._LUT_3),
            self._LUT_sequence)

    def test__getitem__(self):
        """
        Tests :class:`colour.io.luts.lut.LUTSequence.__getitem__` method.
        """

        self.assertEqual(self._LUT_sequence[0], self._LUT_1)
        self.assertEqual(self._LUT_sequence[1], self._LUT_2)
        self.assertEqual(self._LUT_sequence[2], self._LUT_3)

    def test__setitem__(self):
        """
        Tests :class:`colour.io.luts.lut.LUTSequence.__setitem__` method.
        """

        LUT_sequence = self._LUT_sequence.copy()
        LUT_sequence[0] = self._LUT_3
        LUT_sequence[1] = self._LUT_1
        LUT_sequence[2] = self._LUT_2

        self.assertEqual(LUT_sequence[1], self._LUT_1)
        self.assertEqual(LUT_sequence[2], self._LUT_2)
        self.assertEqual(LUT_sequence[0], self._LUT_3)

    def test__delitem__(self):
        """
        Tests :class:`colour.io.luts.lut.LUTSequence.__delitem__` method.
        """

        LUT_sequence = self._LUT_sequence.copy()

        del LUT_sequence[0]
        del LUT_sequence[0]

        self.assertEqual(LUT_sequence[0], self._LUT_3)

    def test__len__(self):
        """
        Tests :class:`colour.io.luts.lut.LUTSequence.__len__` method.
        """

        self.assertEqual(len(self._LUT_sequence), 3)

    def test__str__(self):
        """
        Tests :class:`colour.io.luts.lut.LUTSequence.__str__` method.
        """

        self.assertEqual(
            str(self._LUT_sequence),
            textwrap.dedent("""
            LUT Sequence
            ------------

            Overview

                LUT1D ---> LUT3D ---> LUT2D

            Operations

                LUT1D - Nemo 1D
                ---------------

                Dimensions : 1
                Domain     : [0 1]
                Size       : (16,)

                LUT3D - Nemo 3D
                ---------------

                Dimensions : 3
                Domain     : [[0 0 0]
                              [1 1 1]]
                Size       : (16, 16, 16, 3)

                LUT2D - Nemo 2D
                ---------------

                Dimensions : 2
                Domain     : [[0 0 0]
                              [1 1 1]]
                Size       : (16, 3)""")[1:])

    def test__repr__(self):
        """
        Tests :class:`colour.io.luts.lut.LUTSequence.__repr__` method.
        """

        LUT_sequence = self._LUT_sequence.copy()
        LUT_sequence[1].table = LUT3D.linear_table(5)

        self.assertEqual(
            repr(LUT_sequence),
            textwrap.dedent("""
            LUTSequence(
                LUT1D([ 0.125     ,  0.19166667,  0.25833333,  0.325     ,  \
0.39166667,
                        0.45833333,  0.525     ,  0.59166667,  0.65833333,  \
0.725     ,
                        0.79166667,  0.85833333,  0.925     ,  0.99166667,  \
1.05833333,
                        1.125     ],
                      name='Nemo 1D',
                      domain=[0, 1]),
                LUT3D([[[[ 0.  ,  0.  ,  0.  ],
                         [ 0.  ,  0.  ,  0.25],
                         [ 0.  ,  0.  ,  0.5 ],
                         [ 0.  ,  0.  ,  0.75],
                         [ 0.  ,  0.  ,  1.  ]],

                        [[ 0.  ,  0.25,  0.  ],
                         [ 0.  ,  0.25,  0.25],
                         [ 0.  ,  0.25,  0.5 ],
                         [ 0.  ,  0.25,  0.75],
                         [ 0.  ,  0.25,  1.  ]],

                        [[ 0.  ,  0.5 ,  0.  ],
                         [ 0.  ,  0.5 ,  0.25],
                         [ 0.  ,  0.5 ,  0.5 ],
                         [ 0.  ,  0.5 ,  0.75],
                         [ 0.  ,  0.5 ,  1.  ]],

                        [[ 0.  ,  0.75,  0.  ],
                         [ 0.  ,  0.75,  0.25],
                         [ 0.  ,  0.75,  0.5 ],
                         [ 0.  ,  0.75,  0.75],
                         [ 0.  ,  0.75,  1.  ]],

                        [[ 0.  ,  1.  ,  0.  ],
                         [ 0.  ,  1.  ,  0.25],
                         [ 0.  ,  1.  ,  0.5 ],
                         [ 0.  ,  1.  ,  0.75],
                         [ 0.  ,  1.  ,  1.  ]]],

                       [[[ 0.25,  0.  ,  0.  ],
                         [ 0.25,  0.  ,  0.25],
                         [ 0.25,  0.  ,  0.5 ],
                         [ 0.25,  0.  ,  0.75],
                         [ 0.25,  0.  ,  1.  ]],

                        [[ 0.25,  0.25,  0.  ],
                         [ 0.25,  0.25,  0.25],
                         [ 0.25,  0.25,  0.5 ],
                         [ 0.25,  0.25,  0.75],
                         [ 0.25,  0.25,  1.  ]],

                        [[ 0.25,  0.5 ,  0.  ],
                         [ 0.25,  0.5 ,  0.25],
                         [ 0.25,  0.5 ,  0.5 ],
                         [ 0.25,  0.5 ,  0.75],
                         [ 0.25,  0.5 ,  1.  ]],

                        [[ 0.25,  0.75,  0.  ],
                         [ 0.25,  0.75,  0.25],
                         [ 0.25,  0.75,  0.5 ],
                         [ 0.25,  0.75,  0.75],
                         [ 0.25,  0.75,  1.  ]],

                        [[ 0.25,  1.  ,  0.  ],
                         [ 0.25,  1.  ,  0.25],
                         [ 0.25,  1.  ,  0.5 ],
                         [ 0.25,  1.  ,  0.75],
                         [ 0.25,  1.  ,  1.  ]]],

                       [[[ 0.5 ,  0.  ,  0.  ],
                         [ 0.5 ,  0.  ,  0.25],
                         [ 0.5 ,  0.  ,  0.5 ],
                         [ 0.5 ,  0.  ,  0.75],
                         [ 0.5 ,  0.  ,  1.  ]],

                        [[ 0.5 ,  0.25,  0.  ],
                         [ 0.5 ,  0.25,  0.25],
                         [ 0.5 ,  0.25,  0.5 ],
                         [ 0.5 ,  0.25,  0.75],
                         [ 0.5 ,  0.25,  1.  ]],

                        [[ 0.5 ,  0.5 ,  0.  ],
                         [ 0.5 ,  0.5 ,  0.25],
                         [ 0.5 ,  0.5 ,  0.5 ],
                         [ 0.5 ,  0.5 ,  0.75],
                         [ 0.5 ,  0.5 ,  1.  ]],

                        [[ 0.5 ,  0.75,  0.  ],
                         [ 0.5 ,  0.75,  0.25],
                         [ 0.5 ,  0.75,  0.5 ],
                         [ 0.5 ,  0.75,  0.75],
                         [ 0.5 ,  0.75,  1.  ]],

                        [[ 0.5 ,  1.  ,  0.  ],
                         [ 0.5 ,  1.  ,  0.25],
                         [ 0.5 ,  1.  ,  0.5 ],
                         [ 0.5 ,  1.  ,  0.75],
                         [ 0.5 ,  1.  ,  1.  ]]],

                       [[[ 0.75,  0.  ,  0.  ],
                         [ 0.75,  0.  ,  0.25],
                         [ 0.75,  0.  ,  0.5 ],
                         [ 0.75,  0.  ,  0.75],
                         [ 0.75,  0.  ,  1.  ]],

                        [[ 0.75,  0.25,  0.  ],
                         [ 0.75,  0.25,  0.25],
                         [ 0.75,  0.25,  0.5 ],
                         [ 0.75,  0.25,  0.75],
                         [ 0.75,  0.25,  1.  ]],

                        [[ 0.75,  0.5 ,  0.  ],
                         [ 0.75,  0.5 ,  0.25],
                         [ 0.75,  0.5 ,  0.5 ],
                         [ 0.75,  0.5 ,  0.75],
                         [ 0.75,  0.5 ,  1.  ]],

                        [[ 0.75,  0.75,  0.  ],
                         [ 0.75,  0.75,  0.25],
                         [ 0.75,  0.75,  0.5 ],
                         [ 0.75,  0.75,  0.75],
                         [ 0.75,  0.75,  1.  ]],

                        [[ 0.75,  1.  ,  0.  ],
                         [ 0.75,  1.  ,  0.25],
                         [ 0.75,  1.  ,  0.5 ],
                         [ 0.75,  1.  ,  0.75],
                         [ 0.75,  1.  ,  1.  ]]],

                       [[[ 1.  ,  0.  ,  0.  ],
                         [ 1.  ,  0.  ,  0.25],
                         [ 1.  ,  0.  ,  0.5 ],
                         [ 1.  ,  0.  ,  0.75],
                         [ 1.  ,  0.  ,  1.  ]],

                        [[ 1.  ,  0.25,  0.  ],
                         [ 1.  ,  0.25,  0.25],
                         [ 1.  ,  0.25,  0.5 ],
                         [ 1.  ,  0.25,  0.75],
                         [ 1.  ,  0.25,  1.  ]],

                        [[ 1.  ,  0.5 ,  0.  ],
                         [ 1.  ,  0.5 ,  0.25],
                         [ 1.  ,  0.5 ,  0.5 ],
                         [ 1.  ,  0.5 ,  0.75],
                         [ 1.  ,  0.5 ,  1.  ]],

                        [[ 1.  ,  0.75,  0.  ],
                         [ 1.  ,  0.75,  0.25],
                         [ 1.  ,  0.75,  0.5 ],
                         [ 1.  ,  0.75,  0.75],
                         [ 1.  ,  0.75,  1.  ]],

                        [[ 1.  ,  1.  ,  0.  ],
                         [ 1.  ,  1.  ,  0.25],
                         [ 1.  ,  1.  ,  0.5 ],
                         [ 1.  ,  1.  ,  0.75],
                         [ 1.  ,  1.  ,  1.  ]]]],
                      name='Nemo 3D',
                      domain=[[0, 0, 0],
                              [1, 1, 1]]),
                LUT2D([[ 0.  ,  0.  ,  0.  ],
                       [ 0.05,  0.05,  0.05],
                       [ 0.1 ,  0.1 ,  0.1 ],
                       [ 0.15,  0.15,  0.15],
                       [ 0.2 ,  0.2 ,  0.2 ],
                       [ 0.25,  0.25,  0.25],
                       [ 0.3 ,  0.3 ,  0.3 ],
                       [ 0.35,  0.35,  0.35],
                       [ 0.4 ,  0.4 ,  0.4 ],
                       [ 0.45,  0.45,  0.45],
                       [ 0.5 ,  0.5 ,  0.5 ],
                       [ 0.55,  0.55,  0.55],
                       [ 0.6 ,  0.6 ,  0.6 ],
                       [ 0.65,  0.65,  0.65],
                       [ 0.7 ,  0.7 ,  0.7 ],
                       [ 0.75,  0.75,  0.75]],
                      name='Nemo 2D',
                      domain=[[0, 0, 0],
                              [1, 1, 1]])
            )""" [1:]))

    def test__eq__(self):
        """
        Tests :class:`colour.io.luts.lut.LUTSequence.__eq__` method.
        """

        self.assertEqual(self._LUT_sequence,
                         LUTSequence(self._LUT_1, self._LUT_2, self._LUT_3))

    def test__neq__(self):
        """
        Tests :class:`colour.io.luts.lut.LUTSequence.__neq__` method.
        """

        self.assertNotEqual(self._LUT_sequence,
                            LUTSequence(self._LUT_1,
                                        self._LUT_2.copy() * 0.75,
                                        self._LUT_3))

    def test_insert(self):
        """
        Tests :class:`colour.io.luts.lut.LUTSequence.insert` method.
        """

        LUT_sequence = self._LUT_sequence.copy()

        LUT_sequence.insert(1, self._LUT_2.copy())

        self.assertEqual(LUT_sequence,
                         LUTSequence(
                             self._LUT_1,
                             self._LUT_2,
                             self._LUT_2,
                             self._LUT_3,
                         ))

    def test_apply(self):
        """
        Tests :class:`colour.io.luts.lut.LUTSequence.apply` method.
        """

        class GammaOperator(AbstractLUTSequenceOperator):
            """
            Gamma operator for unit tests.

            Parameters
            ----------
            gamma : numeric or array_like
                Gamma value.
            """

            def __init__(self, gamma=1.0):
                self._gamma = gamma

            def apply(self, RGB, *args):
                """
                Applies the *LUT* sequence operator to given *RGB* colourspace
                array.

                Parameters
                ----------
                RGB : array_like
                    *RGB* colourspace array to apply the *LUT* sequence
                    operator onto.

                Returns
                -------
                ndarray
                    Processed *RGB* colourspace array.
                """

                return function_gamma(RGB, self._gamma)

        LUT_sequence = self._LUT_sequence.copy()
        LUT_sequence.insert(1, GammaOperator(1 / 2.2))
        samples = np.linspace(0, 1, 5)
        RGB = tstack([samples, samples, samples])

        np.testing.assert_almost_equal(
            LUT_sequence.apply(RGB),
            np.array([
                [0.48779047, 0.48779047, 0.48779047],
                [0.61222338, 0.61222338, 0.61222338],
                [0.68053686, 0.68053686, 0.68053686],
                [0.72954547, 0.72954547, 0.72954547],
                [0.75000000, 0.75000000, 0.75000000],
            ]))


if __name__ == '__main__':
    unittest.main()
