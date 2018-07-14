# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.io.luts.lut` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import os
import textwrap
import unittest

from colour.algebra import random_triplet_generator
from colour.io.luts.lut import AbstractLUT
from colour.io.luts import LUT1D, LUT2D, LUT3D
from colour.utilities import tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2018 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'RESOURCES_DIRECTORY', 'TestAbstractLUT', 'TestLUT', 'TestLUT1D',
    'TestLUT2D', 'TestLUT3D'
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

        np.testing.assert_array_almost_equal(
            LUT.table, self._table_1, decimal=7)

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
        np.testing.assert_array_almost_equal(LUT.table, table_1, decimal=7)

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

        np.testing.assert_array_almost_equal(
            LUT_1.arithmetical_operation(10, '+', False).table,
            self._table_1 + 10,
            decimal=7)

        np.testing.assert_array_almost_equal(
            LUT_1.arithmetical_operation(10, '-', False).table,
            self._table_1 - 10,
            decimal=7)

        np.testing.assert_array_almost_equal(
            LUT_1.arithmetical_operation(10, '*', False).table,
            self._table_1 * 10,
            decimal=7)

        np.testing.assert_array_almost_equal(
            LUT_1.arithmetical_operation(10, '/', False).table,
            self._table_1 / 10,
            decimal=7)

        np.testing.assert_array_almost_equal(
            LUT_1.arithmetical_operation(10, '**', False).table,
            self._table_1 ** 10,
            decimal=7)

        np.testing.assert_array_almost_equal(
            LUT_2.arithmetical_operation(10, '+', True).table,
            self._table_1 + 10,
            decimal=7)

        np.testing.assert_array_almost_equal(
            LUT_2.arithmetical_operation(10, '-', True).table,
            self._table_1,
            decimal=7)

        np.testing.assert_array_almost_equal(
            LUT_2.arithmetical_operation(10, '*', True).table,
            self._table_1 * 10,
            decimal=7)

        np.testing.assert_array_almost_equal(
            LUT_2.arithmetical_operation(10, '/', True).table,
            self._table_1,
            decimal=7)

        np.testing.assert_array_almost_equal(
            LUT_2.arithmetical_operation(10, '**', True).table,
            self._table_1 ** 10,
            decimal=7)

        # pylint: disable=E1102
        LUT_2 = self._LUT_factory()

        np.testing.assert_array_almost_equal(
            LUT_2.arithmetical_operation(self._table_1, '+', False).table,
            LUT_2.table + self._table_1,
            decimal=7)

        np.testing.assert_array_almost_equal(
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

        np.testing.assert_array_almost_equal(
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

        np.testing.assert_array_almost_equal(
            LUT_1.apply(RANDOM_TRIPLETS), self._applied_1, decimal=7)

        # pylint: disable=E1102
        LUT_2 = self._LUT_factory(domain=self._domain_2)
        LUT_2.table = np.sign(LUT_2.table) * np.abs(LUT_2.table) ** (1 / 2.2)

        np.testing.assert_array_almost_equal(
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
    Defines :func:`colour.io.luts.lut.LUT1D` class unit tests methods.
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
    Defines :func:`colour.io.luts.lut.LUT2D` class unit tests methods.
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
        self._table_2 = self._table_1 ** (1 / 2.2)
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
    Defines :func:`colour.io.luts.lut.LUT3D` class unit tests methods.
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
        self._table_2 = self._table_1 ** (1 / 2.2)
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


if __name__ == '__main__':
    unittest.main()
