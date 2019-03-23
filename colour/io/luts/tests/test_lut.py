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
from colour.io.luts import (AbstractLUTSequenceOperator, LUT1D, LUT3x1D, LUT3D,
                            LUTSequence, LUT_to_LUT)
from colour.models import gamma_function
from colour.utilities import tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'RESOURCES_DIRECTORY', 'TestAbstractLUT', 'TestLUT', 'TestLUT1D',
    'TestLUT3x1D', 'TestLUT3D', 'TestAbstractLUTSequenceOperator',
    'TestLUTSequence'
]

RESOURCES_DIRECTORY = os.path.join(os.path.dirname(__file__), 'resources')

RANDOM_TRIPLETS = np.reshape(
    random_triplet_generator(8, random_state=np.random.RandomState(4)),
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
                            'is_domain_explicit', 'linear_table', 'apply',
                            'copy', 'as_LUT')

        for method in required_methods:
            self.assertIn(method, dir(AbstractLUT))


class TestLUT(unittest.TestCase):
    """
    Defines :class:`colour.io.luts.lut.LUT1D`,
    :class:`colour.io.luts.lut.LUT3x1D` and
    :class:`colour.io.luts.lut.LUT3D` classes common unit tests methods.
    """

    def __init__(self, *args):
        """
        Create an instance of the class.

        Other Parameters
        ----------------
        \\*args : list, optional
            Arguments.
        """

        super(TestLUT, self).__init__(*args)

        self._LUT_factory = None

        self._domain_1 = None
        self._domain_2 = None
        self._domain_3 = None
        self._table_1 = None
        self._table_2 = None
        self._table_3 = None
        self._table_1_kwargs = None
        self._table_2_kwargs = None
        self._table_3_kwargs = None
        self._dimensions = None
        self._str = None
        self._repr = None
        self._applied_1 = None
        self._applied_2 = None
        self._applied_3 = None

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('is_domain_explicit', 'linear_table', 'apply',
                            'as_LUT')

        for method in required_methods:
            self.assertIn(method, dir(LUT1D))

    def test__init__(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.__init__`,
        :class:`colour.io.luts.lut.LUT3x1D.__init__` and
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

        self.assertIsInstance(
            self._LUT_factory(self._table_3, domain=self._domain_3),
            self._LUT_factory)

    def test_table(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.table`,
        :class:`colour.io.luts.lut.LUT3x1D.table` and
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
        :class:`colour.io.luts.lut.LUT3x1D.name` and
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
        :class:`colour.io.luts.lut.LUT3x1D.domain` and
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
        :class:`colour.io.luts.lut.LUT3x1D.size` and
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
        :class:`colour.io.luts.lut.LUT3x1D.dimensions` and
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
        :class:`colour.io.luts.lut.LUT3x1D.comments` and
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
        :class:`colour.io.luts.lut.LUT3x1D.__str__` and
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
        :class:`colour.io.luts.lut.LUT3x1D.__repr__` and
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
        :class:`colour.io.luts.lut.LUT3x1D.__eq__` and
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
        :class:`colour.io.luts.lut.LUT3x1D.__ne__` and
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

    def test_is_domain_explicit(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.is_domain_explicit`,
        :class:`colour.io.luts.lut.LUT3x1D.is_domain_explicit` and
        :class:`colour.io.luts.lut.LUT3D.is_domain_explicit` methods.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        self.assertFalse(self._LUT_factory().is_domain_explicit())

        # pylint: disable=E1102
        self.assertTrue(
            self._LUT_factory(self._table_3,
                              domain=self._domain_3).is_domain_explicit())

    def test_arithmetical_operation(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.arithmetical_operation`,
        :class:`colour.io.luts.lut.LUT3x1D.arithmetical_operation` and
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
        :class:`colour.io.luts.lut.LUT3x1D.linear_table` and
        :class:`colour.io.luts.lut.LUT3D.linear_table` methods.
        """

        if self._LUT_factory is None:
            return

        # pylint: disable=E1102
        LUT_1 = self._LUT_factory()

        np.testing.assert_almost_equal(
            LUT_1.linear_table(), self._table_1, decimal=7)

        np.testing.assert_almost_equal(
            spow(
                self._LUT_factory.linear_table(**self._table_3_kwargs),
                1 / 2.6),
            self._table_3,
            decimal=7)

    def test_apply(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.apply`,
        :class:`colour.io.luts.lut.LUT3x1D.apply` and
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

        # pylint: disable=E1102
        LUT_3 = self._LUT_factory(self._table_3, domain=self._domain_3)

        np.testing.assert_almost_equal(
            LUT_3.apply(RANDOM_TRIPLETS), self._applied_3, decimal=7)

    def test_copy(self):
        """
        Tests :class:`colour.io.luts.lut.LUT1D.copy`,
        :class:`colour.io.luts.lut.LUT3x1D.copy` and
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
        \\*args : list, optional
            Arguments.
        """

        super(TestLUT1D, self).__init__(*args)

        self._LUT_factory = LUT1D

        self._domain_1 = np.array([0, 1])
        self._domain_2 = np.array([-0.1, 1.5])
        self._domain_3 = np.linspace(-0.1, 1.5, 10)
        self._table_1 = np.linspace(0, 1, 10)
        self._table_2 = self._table_1 ** (1 / 2.2)
        self._table_3 = spow(np.linspace(-0.1, 1.5, 10), (1 / 2.6))
        self._table_1_kwargs = {'size': 10, 'domain': self._domain_1}
        self._table_2_kwargs = {'size': 10, 'domain': self._domain_2}
        self._table_3_kwargs = {'size': 10, 'domain': self._domain_3}
        self._dimensions = 1
        self._str = textwrap.dedent("""
            LUT1D - Nemo
            ------------

            Dimensions : 1
            Domain     : [ 0.  1.]
            Size       : (10,)""")[1:]
        self._repr = textwrap.dedent("""
    LUT1D([ 0.        ,  0.11111111,  0.22222222,  0.33333333,  0.44444444,
            0.55555556,  0.66666667,  0.77777778,  0.88888889,  1.        ],
          name='Nemo',
          domain=[ 0.,  1.])""")[1:]
        self._applied_1 = np.array([
            [[0.98453144, 0.53304051, 0.02978976],
             [0.76000720, 0.68433298, 0.64753760]],
            [[0.98718436, 0.89285575, 0.14639477],
             [0.85784314, 0.47463489, 0.97966294]],
            [[0.84855994, 0.93486051, 0.68536703],
             [0.49723089, 0.99221212, 0.97606176]],
            [[0.98886872, 0.43308440, 0.89633381],
             [0.02065388, 0.79040970, 0.93651642]],
        ])
        self._applied_2 = np.array([
            [[0.98486877, 0.53461565, 0.05614915],
             [0.75787807, 0.68473291, 0.64540281]],
            [[0.98736681, 0.89255862, 0.18759013],
             [0.85682563, 0.46473837, 0.97981413]],
            [[0.84736915, 0.93403795, 0.68561444],
             [0.48799540, 0.99210103, 0.97606266]],
            [[0.98895283, 0.42197234, 0.89639002],
             [0.04585089, 0.79047033, 0.93564890]],
        ])

        self._applied_3 = np.array([
            [[0.98718085, 0.58856660, 0.06995805],
             [0.79062078, 0.72580416, 0.68991332]],
            [[0.98928698, 0.90826591, 0.22565356],
             [0.87725399, 0.52099138, 0.98286533]],
            [[0.86904691, 0.94376678, 0.72658532],
             [0.54348223, 0.99327846, 0.97966110]],
            [[0.99062417, 0.47963425, 0.91159110],
             [0.05775947, 0.81950198, 0.94514273]],
        ])


class TestLUT3x1D(TestLUT):
    """
    Defines :class:`colour.io.luts.lut.LUT3x1D` class unit tests methods.
    """

    def __init__(self, *args):
        """
        Create an instance of the class.

        Other Parameters
        ----------------
        \\*args : list, optional
            Arguments.
        """

        super(TestLUT3x1D, self).__init__(*args)

        self._LUT_factory = LUT3x1D

        samples_1 = np.linspace(0, 1, 10)
        samples_2 = np.linspace(-0.1, 1.5, 15)
        samples_3 = np.linspace(-0.1, 3.0, 20)
        self._domain_1 = np.array([[0, 0, 0], [1, 1, 1]])
        self._domain_2 = np.array([[0, -0.1, -0.2], [1, 1.5, 3.0]])
        self._domain_3 = tstack([
            np.hstack([samples_1, np.full(10, np.nan)]),
            np.hstack([samples_2, np.full(5, np.nan)]),
            samples_3,
        ])
        self._table_1 = tstack([samples_1, samples_1, samples_1])
        self._table_2 = self._table_1 ** (1 / 2.2)
        self._table_3 = spow(
            tstack([
                np.hstack([samples_1, np.full(10, np.nan)]),
                np.hstack([samples_2, np.full(5, np.nan)]),
                samples_3,
            ]), 1 / 2.6)
        self._table_1_kwargs = {'size': 10, 'domain': self._domain_1}
        self._table_2_kwargs = {'size': 10, 'domain': self._domain_2}
        self._table_3_kwargs = {
            'size': np.array([10, 15, 20]),
            'domain': self._domain_3
        }
        self._dimensions = 2
        self._str = textwrap.dedent("""
            LUT3x1D - Nemo
            --------------

            Dimensions : 2
            Domain     : [[ 0.  0.  0.]
                          [ 1.  1.  1.]]
            Size       : (10, 3)""")[1:]
        self._repr = textwrap.dedent("""
            LUT3x1D([[ 0.        ,  0.        ,  0.        ],
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
                    domain=[[ 0.,  0.,  0.],
                            [ 1.,  1.,  1.]])""")[1:]
        self._applied_1 = np.array([
            [[0.98453144, 0.53304051, 0.02978976],
             [0.76000720, 0.68433298, 0.64753760]],
            [[0.98718436, 0.89285575, 0.14639477],
             [0.85784314, 0.47463489, 0.97966294]],
            [[0.84855994, 0.93486051, 0.68536703],
             [0.49723089, 0.99221212, 0.97606176]],
            [[0.98886872, 0.43308440, 0.89633381],
             [0.02065388, 0.79040970, 0.93651642]],
        ])
        self._applied_2 = np.array([
            [[0.98453144, 0.53461565, 0.05393585],
             [0.76000720, 0.68473291, 0.62923633]],
            [[0.98718436, 0.89255862, 0.14399599],
             [0.85784314, 0.46473837, 0.97713337]],
            [[0.84855994, 0.93403795, 0.67216031],
             [0.49723089, 0.99210103, 0.97371216]],
            [[0.98886872, 0.42197234, 0.89183123],
             [0.02065388, 0.79047033, 0.93681229]],
        ])
        self._applied_3 = np.array([
            [[0.98685765, 0.58844468, 0.09393531],
             [0.79274650, 0.72453018, 0.69347904]],
            [[0.98911162, 0.90807837, 0.25736920],
             [0.87825083, 0.53046097, 0.98225775]],
            [[0.87021380, 0.94442819, 0.72448386],
             [0.55350090, 0.99318691, 0.97922787]],
            [[0.99054268, 0.49317779, 0.91055390],
             [0.02408419, 0.81991814, 0.94597809]],
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
        \\*args : list, optional
            Arguments.
        """

        super(TestLUT3D, self).__init__(*args)

        self._LUT_factory = LUT3D

        samples_1 = np.linspace(0, 1, 10)
        samples_2 = np.linspace(-0.1, 1.5, 15)
        samples_3 = np.linspace(-0.1, 3.0, 20)
        self._domain_1 = np.array([[0, 0, 0], [1, 1, 1]])
        self._domain_2 = np.array([[0, -0.1, -0.2], [1, 1.5, 3.0]])
        self._domain_3 = tstack([
            np.hstack([samples_1, np.full(10, np.nan)]),
            np.hstack([samples_2, np.full(5, np.nan)]),
            samples_3,
        ])
        self._table_1 = np.meshgrid(
            *[
                np.linspace(axes[0], axes[1], 33)
                for axes in reversed(tsplit(self._domain_1))
            ],
            indexing='ij')
        self._table_1 = np.flip(
            np.transpose(self._table_1).reshape([33, 33, 33, 3]), -1)
        self._table_2 = self._table_1 ** (1 / 2.2)
        self._table_3 = np.meshgrid(
            *[
                axes[:(~np.isnan(axes)).cumsum().argmax() + 1]
                for axes in reversed(tsplit(self._domain_3))
            ],
            indexing='ij')
        self._table_3 = spow(
            np.flip(np.transpose(self._table_3).reshape([10, 15, 20, 3]), -1),
            1 / 2.6)
        self._table_1_kwargs = {'size': 33, 'domain': self._domain_1}
        self._table_2_kwargs = {'size': 33, 'domain': self._domain_2}
        self._table_3_kwargs = {
            'size': np.array([10, 15, 20]),
            'domain': self._domain_3
        }
        self._dimensions = 3
        self._str = textwrap.dedent("""
            LUT3D - Nemo
            ------------

            Dimensions : 3
            Domain     : [[ 0.  0.  0.]
                          [ 1.  1.  1.]]
            Size       : (33, 33, 33, 3)""")[1:]
        self._repr = None
        self._applied_1 = np.array([
            [[0.98486974, 0.53531556, 0.05950617],
             [0.76022687, 0.68479344, 0.64907649]],
            [[0.98747624, 0.89287549, 0.23859990],
             [0.85844632, 0.47829965, 0.98002765]],
            [[0.84903362, 0.93518100, 0.68577990],
             [0.49827272, 0.99238949, 0.97644600]],
            [[0.98912224, 0.43911364, 0.89645863],
             [0.04125691, 0.79116284, 0.93680839]],
        ])

        self._applied_2 = np.array([
            [[0.98486974, 0.53526504, 0.03155191],
             [0.76022687, 0.68458573, 0.64850011]],
            [[0.98747624, 0.89277461, 0.15505443],
             [0.85844632, 0.47842591, 0.97972986]],
            [[0.84903362, 0.93514331, 0.68479574],
             [0.49827272, 0.99234923, 0.97614054]],
            [[0.98912224, 0.43850620, 0.89625878],
             [0.04125691, 0.79115345, 0.93648599]],
        ])

        self._applied_3 = np.array([
            [[0.98685765, 0.58844468, 0.09393531],
             [0.79274650, 0.72453018, 0.69347904]],
            [[0.98911162, 0.90807837, 0.25736920],
             [0.87825083, 0.53046097, 0.98225775]],
            [[0.87021380, 0.94442819, 0.72448386],
             [0.55350090, 0.99318691, 0.97922787]],
            [[0.99054268, 0.49317779, 0.91055390],
             [0.02408419, 0.81991814, 0.94597809]],
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
        self._LUT_3 = LUT3x1D(LUT3x1D.linear_table(16) * 0.750, 'Nemo 3x1D')
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

                LUT1D ---> LUT3D ---> LUT3x1D

            Operations

                LUT1D - Nemo 1D
                ---------------

                Dimensions : 1
                Domain     : [ 0.  1.]
                Size       : (16,)

                LUT3D - Nemo 3D
                ---------------

                Dimensions : 3
                Domain     : [[ 0.  0.  0.]
                              [ 1.  1.  1.]]
                Size       : (16, 16, 16, 3)

                LUT3x1D - Nemo 3x1D
                -------------------

                Dimensions : 2
                Domain     : [[ 0.  0.  0.]
                              [ 1.  1.  1.]]
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
                      domain=[ 0.,  1.]),
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
                      domain=[[ 0.,  0.,  0.],
                              [ 1.,  1.,  1.]]),
                LUT3x1D([[ 0.  ,  0.  ,  0.  ],
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
                        name='Nemo 3x1D',
                        domain=[[ 0.,  0.,  0.],
                                [ 1.,  1.,  1.]])
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

        self.assertNotEqual(
            self._LUT_sequence,
            LUTSequence(self._LUT_1,
                        self._LUT_2.copy() * 0.75, self._LUT_3))

    def test_insert(self):
        """
        Tests :class:`colour.io.luts.lut.LUTSequence.insert` method.
        """

        LUT_sequence = self._LUT_sequence.copy()

        LUT_sequence.insert(1, self._LUT_2.copy())

        self.assertEqual(
            LUT_sequence,
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

                return gamma_function(RGB, self._gamma)

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


class TestLUT_to_LUT(unittest.TestCase):
    """
    Defines :func:`colour.io.luts.lut.LUT_to_LUT` definition unit tests
    methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._domain = np.array([[0.0, -0.1, -0.2], [1.0, 1.5, 3.0]])

        self._LUT_1 = LUT1D(LUT1D.linear_table(16) ** (1 / 2.2))
        self._LUT_2 = LUT3x1D(
            LUT3x1D.linear_table(16) ** (1 / 2.2) * (1.0, 0.75, 0.5),
            domain=self._domain)
        self._LUT_3 = LUT3D(
            LUT3D.linear_table(16) ** (1 / 2.2), domain=self._domain)

    def test_LUT_to_LUT(self):
        """
        Tests :func:`colour.io.luts.lut.LUT_to_LUT` definition.
        """

        # "LUT" 1D to "LUT" 1D.
        LUT = LUT_to_LUT(self._LUT_1, LUT1D)

        self.assertEqual(LUT, self._LUT_1)

        # "LUT" 1D to "LUT" 3x1D.
        LUT = LUT_to_LUT(self._LUT_1, LUT3x1D)
        table = LUT1D.linear_table(16) ** (1 / 2.2)

        self.assertEqual(LUT, LUT3x1D(tstack([table, table, table])))

        # "LUT" 1D to "LUT" 3D.
        self.assertRaises(ValueError, lambda: LUT_to_LUT(self._LUT_1, LUT3D))

        LUT = LUT_to_LUT(self._LUT_1, LUT3D, force_conversion=True, size=5)

        np.testing.assert_almost_equal(
            LUT.table,
            np.array([[[
                [0.00000000, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.53156948],
                [0.00000000, 0.00000000, 0.72933741],
                [0.00000000, 0.00000000, 0.87726669],
                [0.00000000, 0.00000000, 1.00000000],
            ], [
                [0.00000000, 0.53156948, 0.00000000],
                [0.00000000, 0.53156948, 0.53156948],
                [0.00000000, 0.53156948, 0.72933741],
                [0.00000000, 0.53156948, 0.87726669],
                [0.00000000, 0.53156948, 1.00000000],
            ], [
                [0.00000000, 0.72933741, 0.00000000],
                [0.00000000, 0.72933741, 0.53156948],
                [0.00000000, 0.72933741, 0.72933741],
                [0.00000000, 0.72933741, 0.87726669],
                [0.00000000, 0.72933741, 1.00000000],
            ], [
                [0.00000000, 0.87726669, 0.00000000],
                [0.00000000, 0.87726669, 0.53156948],
                [0.00000000, 0.87726669, 0.72933741],
                [0.00000000, 0.87726669, 0.87726669],
                [0.00000000, 0.87726669, 1.00000000],
            ], [
                [0.00000000, 1.00000000, 0.00000000],
                [0.00000000, 1.00000000, 0.53156948],
                [0.00000000, 1.00000000, 0.72933741],
                [0.00000000, 1.00000000, 0.87726669],
                [0.00000000, 1.00000000, 1.00000000],
            ]], [[
                [0.53156948, 0.00000000, 0.00000000],
                [0.53156948, 0.00000000, 0.53156948],
                [0.53156948, 0.00000000, 0.72933741],
                [0.53156948, 0.00000000, 0.87726669],
                [0.53156948, 0.00000000, 1.00000000],
            ], [
                [0.53156948, 0.53156948, 0.00000000],
                [0.53156948, 0.53156948, 0.53156948],
                [0.53156948, 0.53156948, 0.72933741],
                [0.53156948, 0.53156948, 0.87726669],
                [0.53156948, 0.53156948, 1.00000000],
            ], [
                [0.53156948, 0.72933741, 0.00000000],
                [0.53156948, 0.72933741, 0.53156948],
                [0.53156948, 0.72933741, 0.72933741],
                [0.53156948, 0.72933741, 0.87726669],
                [0.53156948, 0.72933741, 1.00000000],
            ], [
                [0.53156948, 0.87726669, 0.00000000],
                [0.53156948, 0.87726669, 0.53156948],
                [0.53156948, 0.87726669, 0.72933741],
                [0.53156948, 0.87726669, 0.87726669],
                [0.53156948, 0.87726669, 1.00000000],
            ], [
                [0.53156948, 1.00000000, 0.00000000],
                [0.53156948, 1.00000000, 0.53156948],
                [0.53156948, 1.00000000, 0.72933741],
                [0.53156948, 1.00000000, 0.87726669],
                [0.53156948, 1.00000000, 1.00000000],
            ]], [[
                [0.72933741, 0.00000000, 0.00000000],
                [0.72933741, 0.00000000, 0.53156948],
                [0.72933741, 0.00000000, 0.72933741],
                [0.72933741, 0.00000000, 0.87726669],
                [0.72933741, 0.00000000, 1.00000000],
            ], [
                [0.72933741, 0.53156948, 0.00000000],
                [0.72933741, 0.53156948, 0.53156948],
                [0.72933741, 0.53156948, 0.72933741],
                [0.72933741, 0.53156948, 0.87726669],
                [0.72933741, 0.53156948, 1.00000000],
            ], [
                [0.72933741, 0.72933741, 0.00000000],
                [0.72933741, 0.72933741, 0.53156948],
                [0.72933741, 0.72933741, 0.72933741],
                [0.72933741, 0.72933741, 0.87726669],
                [0.72933741, 0.72933741, 1.00000000],
            ], [
                [0.72933741, 0.87726669, 0.00000000],
                [0.72933741, 0.87726669, 0.53156948],
                [0.72933741, 0.87726669, 0.72933741],
                [0.72933741, 0.87726669, 0.87726669],
                [0.72933741, 0.87726669, 1.00000000],
            ], [
                [0.72933741, 1.00000000, 0.00000000],
                [0.72933741, 1.00000000, 0.53156948],
                [0.72933741, 1.00000000, 0.72933741],
                [0.72933741, 1.00000000, 0.87726669],
                [0.72933741, 1.00000000, 1.00000000],
            ]], [[
                [0.87726669, 0.00000000, 0.00000000],
                [0.87726669, 0.00000000, 0.53156948],
                [0.87726669, 0.00000000, 0.72933741],
                [0.87726669, 0.00000000, 0.87726669],
                [0.87726669, 0.00000000, 1.00000000],
            ], [
                [0.87726669, 0.53156948, 0.00000000],
                [0.87726669, 0.53156948, 0.53156948],
                [0.87726669, 0.53156948, 0.72933741],
                [0.87726669, 0.53156948, 0.87726669],
                [0.87726669, 0.53156948, 1.00000000],
            ], [
                [0.87726669, 0.72933741, 0.00000000],
                [0.87726669, 0.72933741, 0.53156948],
                [0.87726669, 0.72933741, 0.72933741],
                [0.87726669, 0.72933741, 0.87726669],
                [0.87726669, 0.72933741, 1.00000000],
            ], [
                [0.87726669, 0.87726669, 0.00000000],
                [0.87726669, 0.87726669, 0.53156948],
                [0.87726669, 0.87726669, 0.72933741],
                [0.87726669, 0.87726669, 0.87726669],
                [0.87726669, 0.87726669, 1.00000000],
            ], [
                [0.87726669, 1.00000000, 0.00000000],
                [0.87726669, 1.00000000, 0.53156948],
                [0.87726669, 1.00000000, 0.72933741],
                [0.87726669, 1.00000000, 0.87726669],
                [0.87726669, 1.00000000, 1.00000000],
            ]], [[
                [1.00000000, 0.00000000, 0.00000000],
                [1.00000000, 0.00000000, 0.53156948],
                [1.00000000, 0.00000000, 0.72933741],
                [1.00000000, 0.00000000, 0.87726669],
                [1.00000000, 0.00000000, 1.00000000],
            ], [
                [1.00000000, 0.53156948, 0.00000000],
                [1.00000000, 0.53156948, 0.53156948],
                [1.00000000, 0.53156948, 0.72933741],
                [1.00000000, 0.53156948, 0.87726669],
                [1.00000000, 0.53156948, 1.00000000],
            ], [
                [1.00000000, 0.72933741, 0.00000000],
                [1.00000000, 0.72933741, 0.53156948],
                [1.00000000, 0.72933741, 0.72933741],
                [1.00000000, 0.72933741, 0.87726669],
                [1.00000000, 0.72933741, 1.00000000],
            ], [
                [1.00000000, 0.87726669, 0.00000000],
                [1.00000000, 0.87726669, 0.53156948],
                [1.00000000, 0.87726669, 0.72933741],
                [1.00000000, 0.87726669, 0.87726669],
                [1.00000000, 0.87726669, 1.00000000],
            ], [
                [1.00000000, 1.00000000, 0.00000000],
                [1.00000000, 1.00000000, 0.53156948],
                [1.00000000, 1.00000000, 0.72933741],
                [1.00000000, 1.00000000, 0.87726669],
                [1.00000000, 1.00000000, 1.00000000],
            ]]]),
            decimal=7)

        # "LUT" 3x1D to "LUT" 1D.
        self.assertRaises(ValueError, lambda: LUT_to_LUT(self._LUT_2, LUT1D))

        channel_weights = np.array([1.0, 0.0, 0.0])
        LUT = LUT_to_LUT(
            self._LUT_2,
            LUT1D,
            force_conversion=True,
            channel_weights=channel_weights)

        domain = np.array(
            [np.max(self._domain[0, ...]),
             np.min(self._domain[1, ...])])

        np.testing.assert_array_equal(LUT.domain, domain)

        channel_weights = np.array([1 / 3, 1 / 3, 1 / 3])
        LUT = LUT_to_LUT(
            self._LUT_2,
            LUT1D,
            force_conversion=True,
            channel_weights=channel_weights)

        self.assertEqual(
            LUT,
            LUT1D(
                np.sum(self._LUT_2.table * channel_weights, axis=-1),
                domain=domain))

        # "LUT" 3x1D to "LUT" 3x1D.
        LUT = LUT_to_LUT(self._LUT_2, LUT3x1D)

        self.assertEqual(LUT, self._LUT_2)

        # "LUT" 3x1D to "LUT" 3D.
        self.assertRaises(ValueError, lambda: LUT_to_LUT(self._LUT_2, LUT3D))

        LUT = LUT_to_LUT(self._LUT_2, LUT3D, force_conversion=True, size=5)

        np.testing.assert_almost_equal(
            LUT.table,
            np.array([[[
                [0.00000000, 0.00000000, 0.00000000],
                [0.00000000, 0.00000000, 0.26578474],
                [0.00000000, 0.00000000, 0.36466870],
                [0.00000000, 0.00000000, 0.43863334],
                [0.00000000, 0.00000000, 0.50000000],
            ], [
                [0.00000000, 0.39867711, 0.00000000],
                [0.00000000, 0.39867711, 0.26578474],
                [0.00000000, 0.39867711, 0.36466870],
                [0.00000000, 0.39867711, 0.43863334],
                [0.00000000, 0.39867711, 0.50000000],
            ], [
                [0.00000000, 0.54700305, 0.00000000],
                [0.00000000, 0.54700305, 0.26578474],
                [0.00000000, 0.54700305, 0.36466870],
                [0.00000000, 0.54700305, 0.43863334],
                [0.00000000, 0.54700305, 0.50000000],
            ], [
                [0.00000000, 0.65795001, 0.00000000],
                [0.00000000, 0.65795001, 0.26578474],
                [0.00000000, 0.65795001, 0.36466870],
                [0.00000000, 0.65795001, 0.43863334],
                [0.00000000, 0.65795001, 0.50000000],
            ], [
                [0.00000000, 0.75000000, 0.00000000],
                [0.00000000, 0.75000000, 0.26578474],
                [0.00000000, 0.75000000, 0.36466870],
                [0.00000000, 0.75000000, 0.43863334],
                [0.00000000, 0.75000000, 0.50000000],
            ]], [[
                [0.53156948, 0.00000000, 0.00000000],
                [0.53156948, 0.00000000, 0.26578474],
                [0.53156948, 0.00000000, 0.36466870],
                [0.53156948, 0.00000000, 0.43863334],
                [0.53156948, 0.00000000, 0.50000000],
            ], [
                [0.53156948, 0.39867711, 0.00000000],
                [0.53156948, 0.39867711, 0.26578474],
                [0.53156948, 0.39867711, 0.36466870],
                [0.53156948, 0.39867711, 0.43863334],
                [0.53156948, 0.39867711, 0.50000000],
            ], [
                [0.53156948, 0.54700305, 0.00000000],
                [0.53156948, 0.54700305, 0.26578474],
                [0.53156948, 0.54700305, 0.36466870],
                [0.53156948, 0.54700305, 0.43863334],
                [0.53156948, 0.54700305, 0.50000000],
            ], [
                [0.53156948, 0.65795001, 0.00000000],
                [0.53156948, 0.65795001, 0.26578474],
                [0.53156948, 0.65795001, 0.36466870],
                [0.53156948, 0.65795001, 0.43863334],
                [0.53156948, 0.65795001, 0.50000000],
            ], [
                [0.53156948, 0.75000000, 0.00000000],
                [0.53156948, 0.75000000, 0.26578474],
                [0.53156948, 0.75000000, 0.36466870],
                [0.53156948, 0.75000000, 0.43863334],
                [0.53156948, 0.75000000, 0.50000000],
            ]], [[
                [0.72933741, 0.00000000, 0.00000000],
                [0.72933741, 0.00000000, 0.26578474],
                [0.72933741, 0.00000000, 0.36466870],
                [0.72933741, 0.00000000, 0.43863334],
                [0.72933741, 0.00000000, 0.50000000],
            ], [
                [0.72933741, 0.39867711, 0.00000000],
                [0.72933741, 0.39867711, 0.26578474],
                [0.72933741, 0.39867711, 0.36466870],
                [0.72933741, 0.39867711, 0.43863334],
                [0.72933741, 0.39867711, 0.50000000],
            ], [
                [0.72933741, 0.54700305, 0.00000000],
                [0.72933741, 0.54700305, 0.26578474],
                [0.72933741, 0.54700305, 0.36466870],
                [0.72933741, 0.54700305, 0.43863334],
                [0.72933741, 0.54700305, 0.50000000],
            ], [
                [0.72933741, 0.65795001, 0.00000000],
                [0.72933741, 0.65795001, 0.26578474],
                [0.72933741, 0.65795001, 0.36466870],
                [0.72933741, 0.65795001, 0.43863334],
                [0.72933741, 0.65795001, 0.50000000],
            ], [
                [0.72933741, 0.75000000, 0.00000000],
                [0.72933741, 0.75000000, 0.26578474],
                [0.72933741, 0.75000000, 0.36466870],
                [0.72933741, 0.75000000, 0.43863334],
                [0.72933741, 0.75000000, 0.50000000],
            ]], [[
                [0.87726669, 0.00000000, 0.00000000],
                [0.87726669, 0.00000000, 0.26578474],
                [0.87726669, 0.00000000, 0.36466870],
                [0.87726669, 0.00000000, 0.43863334],
                [0.87726669, 0.00000000, 0.50000000],
            ], [
                [0.87726669, 0.39867711, 0.00000000],
                [0.87726669, 0.39867711, 0.26578474],
                [0.87726669, 0.39867711, 0.36466870],
                [0.87726669, 0.39867711, 0.43863334],
                [0.87726669, 0.39867711, 0.50000000],
            ], [
                [0.87726669, 0.54700305, 0.00000000],
                [0.87726669, 0.54700305, 0.26578474],
                [0.87726669, 0.54700305, 0.36466870],
                [0.87726669, 0.54700305, 0.43863334],
                [0.87726669, 0.54700305, 0.50000000],
            ], [
                [0.87726669, 0.65795001, 0.00000000],
                [0.87726669, 0.65795001, 0.26578474],
                [0.87726669, 0.65795001, 0.36466870],
                [0.87726669, 0.65795001, 0.43863334],
                [0.87726669, 0.65795001, 0.50000000],
            ], [
                [0.87726669, 0.75000000, 0.00000000],
                [0.87726669, 0.75000000, 0.26578474],
                [0.87726669, 0.75000000, 0.36466870],
                [0.87726669, 0.75000000, 0.43863334],
                [0.87726669, 0.75000000, 0.50000000],
            ]], [[
                [1.00000000, 0.00000000, 0.00000000],
                [1.00000000, 0.00000000, 0.26578474],
                [1.00000000, 0.00000000, 0.36466870],
                [1.00000000, 0.00000000, 0.43863334],
                [1.00000000, 0.00000000, 0.50000000],
            ], [
                [1.00000000, 0.39867711, 0.00000000],
                [1.00000000, 0.39867711, 0.26578474],
                [1.00000000, 0.39867711, 0.36466870],
                [1.00000000, 0.39867711, 0.43863334],
                [1.00000000, 0.39867711, 0.50000000],
            ], [
                [1.00000000, 0.54700305, 0.00000000],
                [1.00000000, 0.54700305, 0.26578474],
                [1.00000000, 0.54700305, 0.36466870],
                [1.00000000, 0.54700305, 0.43863334],
                [1.00000000, 0.54700305, 0.50000000],
            ], [
                [1.00000000, 0.65795001, 0.00000000],
                [1.00000000, 0.65795001, 0.26578474],
                [1.00000000, 0.65795001, 0.36466870],
                [1.00000000, 0.65795001, 0.43863334],
                [1.00000000, 0.65795001, 0.50000000],
            ], [
                [1.00000000, 0.75000000, 0.00000000],
                [1.00000000, 0.75000000, 0.26578474],
                [1.00000000, 0.75000000, 0.36466870],
                [1.00000000, 0.75000000, 0.43863334],
                [1.00000000, 0.75000000, 0.50000000],
            ]]]),
            decimal=7)

        # "LUT" 3D to "LUT" 1D.
        self.assertRaises(ValueError, lambda: LUT_to_LUT(self._LUT_3, LUT1D))

        channel_weights = np.array([1.0, 0.0, 0.0])
        LUT = LUT_to_LUT(
            self._LUT_3,
            LUT1D,
            force_conversion=True,
            size=16,
            channel_weights=channel_weights)

        np.testing.assert_almost_equal(
            LUT.table,
            np.array([
                0.00000000, 0.29202031, 0.40017033, 0.48115651, 0.54837380,
                0.60691337, 0.65935329, 0.70721023, 0.75146458, 0.79279273,
                0.83168433, 0.86850710, 0.90354543, 0.93702451, 0.96912624,
                1.00000000
            ]))

        channel_weights = np.array([1 / 3, 1 / 3, 1 / 3])
        LUT = LUT_to_LUT(
            self._LUT_3,
            LUT1D,
            force_conversion=True,
            size=16,
            channel_weights=channel_weights)

        np.testing.assert_almost_equal(
            LUT.table,
            np.array([
                0.18251270, 0.32131094, 0.38946008, 0.44459323, 0.49143103,
                0.53320310, 0.57131488, 0.60668991, 0.63936815, 0.67021168,
                0.69974159, 0.72733338, 0.75390361, 0.77972044, 0.80398593,
                0.82762659
            ]))

        domain = np.array(
            [np.max(self._domain[0, ...]),
             np.min(self._domain[1, ...])])

        np.testing.assert_array_equal(LUT.domain, domain)

        # "LUT" 3D to "LUT" 3x1D.
        self.assertRaises(ValueError, lambda: LUT_to_LUT(self._LUT_3, LUT3x1D))

        LUT = LUT_to_LUT(self._LUT_3, LUT3x1D, force_conversion=True, size=16)

        np.testing.assert_almost_equal(
            LUT.table,
            np.array([
                [0.00000000, 0.00000000, 0.00000000],
                [0.29202031, 0.29202031, 0.29202031],
                [0.40017033, 0.40017033, 0.40017033],
                [0.48115651, 0.48115651, 0.48115651],
                [0.54837380, 0.54837380, 0.54837380],
                [0.60691337, 0.60691337, 0.60691337],
                [0.65935329, 0.65935329, 0.65935329],
                [0.70721023, 0.70721023, 0.70721023],
                [0.75146458, 0.75146458, 0.75146458],
                [0.79279273, 0.79279273, 0.79279273],
                [0.83168433, 0.83168433, 0.83168433],
                [0.86850710, 0.86850710, 0.86850710],
                [0.90354543, 0.90354543, 0.90354543],
                [0.93702451, 0.93702451, 0.93702451],
                [0.96912624, 0.96912624, 0.96912624],
                [1.00000000, 1.00000000, 1.00000000],
            ]))

        # "LUT" 3D to "LUT" 3D.
        LUT = LUT_to_LUT(self._LUT_3, LUT3D)

        self.assertEqual(LUT, self._LUT_3)


if __name__ == '__main__':
    unittest.main()
