# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.continuous.multi_signals` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import operator
import unittest
import re
import textwrap
from six import string_types

from colour.algebra import (CubicSplineInterpolator, Extrapolator,
                            KernelInterpolator)
from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.continuous import MultiSignals, Signal
from colour.utilities import is_pandas_installed, tsplit, tstack

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2020 - Colour Developers'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-developers@colour-science.org'
__status__ = 'Production'

__all__ = ['TestMultiSignals']


class TestMultiSignals(unittest.TestCase):
    """
    Defines :class:`colour.continuous.multi_signals.MultiSignals` class unit
    tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._range_1 = np.linspace(10, 100, 10)
        self._range_2 = tstack([self._range_1] * 3) + np.array([0, 10, 20])
        self._domain_1 = np.arange(0, 10, 1)
        self._domain_2 = np.arange(100, 1100, 100)

        self._multi_signals = MultiSignals(self._range_2)

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('dtype', 'domain', 'range', 'interpolator',
                               'interpolator_args', 'extrapolator',
                               'extrapolator_args', 'function', 'signals',
                               'labels', 'signal_type')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(MultiSignals))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', '__repr__', '__hash__', '__getitem__',
                            '__setitem__', '__contains__', '__eq__', '__ne__',
                            'arithmetical_operation',
                            'multi_signals_unpack_data', 'fill_nan',
                            'domain_distance', 'to_dataframe')

        for method in required_methods:
            self.assertIn(method, dir(MultiSignals))

    def test_dtype(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.dtype`
        property.
        """

        self.assertEqual(self._multi_signals.dtype, None)

        multi_signals = self._multi_signals.copy()
        multi_signals.dtype = DEFAULT_FLOAT_DTYPE
        self.assertEqual(multi_signals.dtype, DEFAULT_FLOAT_DTYPE)

    def test_domain(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.domain`
        property.
        """

        multi_signals = self._multi_signals.copy()

        np.testing.assert_almost_equal(
            multi_signals[np.array([0, 1, 2])],
            np.array([[10.0, 20.0, 30.0], [20.0, 30.0, 40.0],
                      [30.0, 40.0, 50.0]]),
            decimal=7)

        multi_signals.domain = self._domain_1 * 10

        np.testing.assert_array_equal(multi_signals.domain,
                                      self._domain_1 * 10)

        np.testing.assert_almost_equal(
            multi_signals[np.array([0, 1, 2]) * 10],
            np.array([[10.0, 20.0, 30.0], [20.0, 30.0, 40.0],
                      [30.0, 40.0, 50.0]]),
            decimal=7)

        # TODO: Use "assertWarns" when dropping Python 2.7.
        domain = np.linspace(0, 1, 10)
        domain[0] = -np.inf
        multi_signals.domain = domain

    def test_range(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.range`
        property.
        """

        multi_signals = self._multi_signals.copy()

        np.testing.assert_almost_equal(
            multi_signals[np.array([0, 1, 2])],
            np.array([[10.0, 20.0, 30.0], [20.0, 30.0, 40.0],
                      [30.0, 40.0, 50.0]]),
            decimal=7)

        multi_signals.range = self._range_1 * 10

        np.testing.assert_array_equal(multi_signals.range,
                                      tstack([self._range_1] * 3) * 10)

        np.testing.assert_almost_equal(
            multi_signals[np.array([0, 1, 2])],
            np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0],
                      [30.0, 30.0, 30.0]]) * 10,
            decimal=7)

        multi_signals.range = self._range_2 * 10

        np.testing.assert_array_equal(multi_signals.range, self._range_2 * 10)

        np.testing.assert_almost_equal(
            multi_signals[np.array([0, 1, 2])],
            np.array([[10.0, 20.0, 30.0], [20.0, 30.0, 40.0],
                      [30.0, 40.0, 50.0]]) * 10,
            decimal=7)

    def test_interpolator(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.interpolator`
        property.
        """

        multi_signals = self._multi_signals.copy()

        np.testing.assert_almost_equal(
            multi_signals[np.linspace(0, 5, 5)],
            np.array([[10.00000000, 20.00000000,
                       30.00000000], [22.83489024, 32.80460562, 42.77432100],
                      [34.80044921, 44.74343470,
                       54.68642018], [47.55353925, 57.52325463, 67.49297001],
                      [60.00000000, 70.00000000, 80.00000000]]),
            decimal=7)

        multi_signals.interpolator = CubicSplineInterpolator

        np.testing.assert_almost_equal(
            multi_signals[np.linspace(0, 5, 5)],
            np.array([[10.00000000, 20.00000000,
                       30.00000000], [22.50000000, 32.50000000, 42.50000000],
                      [35.00000000, 45.00000000,
                       55.00000000], [47.50000000, 57.50000000, 67.50000000],
                      [60.00000000, 70.00000000, 80.00000000]]),
            decimal=7)

    def test_interpolator_args(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.\
interpolator_args` property.
        """

        multi_signals = self._multi_signals.copy()

        np.testing.assert_almost_equal(
            multi_signals[np.linspace(0, 5, 5)],
            np.array([[10.00000000, 20.00000000,
                       30.00000000], [22.83489024, 32.80460562, 42.77432100],
                      [34.80044921, 44.74343470,
                       54.68642018], [47.55353925, 57.52325463, 67.49297001],
                      [60.00000000, 70.00000000, 80.00000000]]),
            decimal=7)

        multi_signals.interpolator_args = {
            'window': 1,
            'kernel_args': {
                'a': 1
            }
        }

        np.testing.assert_almost_equal(
            multi_signals[np.linspace(0, 5, 5)],
            np.array([[10.00000000, 20.00000000,
                       30.00000000], [18.91328761, 27.91961505, 36.92594248],
                      [28.36993142, 36.47562611,
                       44.58132080], [44.13100443, 53.13733187, 62.14365930],
                      [60.00000000, 70.00000000, 80.00000000]]),
            decimal=7)

    def test_extrapolator(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.extrapolator`
        property.
        """

        self.assertIsInstance(self._multi_signals.extrapolator(), Extrapolator)

    def test_extrapolator_args(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.\
extrapolator_args` property.
        """

        multi_signals = self._multi_signals.copy()

        assert np.all(np.isnan(multi_signals[np.array([-1000, 1000])]))

        multi_signals.extrapolator_args = {
            'method': 'Linear',
        }

        np.testing.assert_almost_equal(
            multi_signals[np.array([-1000, 1000])],
            np.array([[-9990.0, -9980.0, -9970.0], [10010.0, 10020.0,
                                                    10030.0]]),
            decimal=7)

    def test_function(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.function`
        property.
        """

        assert hasattr(self._multi_signals.function, '__call__')

    def test_raise_exception_function(self):
        """
        Tests :func:`colour.continuous.signal.multi_signals.MultiSignals`
        property raised exception.
        """

        self.assertRaises((RuntimeError, TypeError),
                          MultiSignals().function, 0)

    def test_signals(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.signals`
        property.
        """

        multi_signals = self._multi_signals.copy()

        multi_signals.signals = self._range_1
        np.testing.assert_array_equal(multi_signals.domain, self._domain_1)
        np.testing.assert_array_equal(multi_signals.range,
                                      self._range_1[:, np.newaxis])

    def test_labels(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.labels`
        property.
        """

        self.assertListEqual(self._multi_signals.labels, [0, 1, 2])

        multi_signals = self._multi_signals.copy()

        multi_signals.labels = ['a', 'b', 'c']

        self.assertListEqual(multi_signals.labels, ['a', 'b', 'c'])

    def test_signal_type(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.signal_type`
        property.
        """

        multi_signals = MultiSignals(signal_type=Signal)

        self.assertEqual(multi_signals.signal_type, Signal)

    def test__init__(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.__init__`
        method.
        """

        multi_signals = MultiSignals(self._range_1)
        np.testing.assert_array_equal(multi_signals.domain, self._domain_1)
        np.testing.assert_array_equal(multi_signals.range,
                                      self._range_1[:, np.newaxis])

        multi_signals = MultiSignals(self._range_1, self._domain_2)
        np.testing.assert_array_equal(multi_signals.domain, self._domain_2)
        np.testing.assert_array_equal(multi_signals.range,
                                      self._range_1[:, np.newaxis])

        multi_signals = MultiSignals(self._range_2, self._domain_2)
        np.testing.assert_array_equal(multi_signals.domain, self._domain_2)
        np.testing.assert_array_equal(multi_signals.range, self._range_2)

        multi_signals = MultiSignals(dict(zip(self._domain_2, self._range_2)))
        np.testing.assert_array_equal(multi_signals.domain, self._domain_2)
        np.testing.assert_array_equal(multi_signals.range, self._range_2)

        multi_signals = MultiSignals(multi_signals)
        np.testing.assert_array_equal(multi_signals.domain, self._domain_2)
        np.testing.assert_array_equal(multi_signals.range, self._range_2)

        class NotSignal(Signal):
            """
            Not :class:`Signal` class.
            """

            pass

        multi_signals = MultiSignals(self._range_1, signal_type=NotSignal)
        self.assertIsInstance(multi_signals.signals[0], NotSignal)
        np.testing.assert_array_equal(multi_signals.domain, self._domain_1)
        np.testing.assert_array_equal(multi_signals.range,
                                      self._range_1[:, np.newaxis])

        if is_pandas_installed():
            from pandas import DataFrame, Series

            multi_signals = MultiSignals(
                Series(dict(zip(self._domain_2, self._range_1))))
            np.testing.assert_array_equal(multi_signals.domain, self._domain_2)
            np.testing.assert_array_equal(multi_signals.range,
                                          self._range_1[:, np.newaxis])

            data = dict(zip(['a', 'b', 'c'], tsplit(self._range_2)))
            multi_signals = MultiSignals(DataFrame(data, self._domain_2))
            np.testing.assert_array_equal(multi_signals.domain, self._domain_2)
            np.testing.assert_array_equal(multi_signals.range, self._range_2)

    def test__hash__(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.__hash__`
        method.
        """

        self.assertIsInstance(hash(self._multi_signals), int)

    def test__str__(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.__str__`
        method.
        """

        self.assertEqual(
            str(self._multi_signals),
            textwrap.dedent("""
                [[   0.   10.   20.   30.]
                 [   1.   20.   30.   40.]
                 [   2.   30.   40.   50.]
                 [   3.   40.   50.   60.]
                 [   4.   50.   60.   70.]
                 [   5.   60.   70.   80.]
                 [   6.   70.   80.   90.]
                 [   7.   80.   90.  100.]
                 [   8.   90.  100.  110.]
                 [   9.  100.  110.  120.]]""")[1:])

        self.assertIsInstance(str(MultiSignals()), string_types)

    def test__repr__(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.__repr__`
        method.
        """

        self.assertEqual(
            re.sub(r'extrapolator_args={.*}', 'extrapolator_args={...}',
                   repr(self._multi_signals)),
            textwrap.dedent("""
                MultiSignals([[   0.,   10.,   20.,   30.],
                              [   1.,   20.,   30.,   40.],
                              [   2.,   30.,   40.,   50.],
                              [   3.,   40.,   50.,   60.],
                              [   4.,   50.,   60.,   70.],
                              [   5.,   60.,   70.,   80.],
                              [   6.,   70.,   80.,   90.],
                              [   7.,   80.,   90.,  100.],
                              [   8.,   90.,  100.,  110.],
                              [   9.,  100.,  110.,  120.]],
                             labels=[0, 1, 2],
                             interpolator=KernelInterpolator,
                             interpolator_args={},
                             extrapolator=Extrapolator,
                             extrapolator_args={...})""")[1:])

        self.assertIsInstance(repr(MultiSignals()), string_types)

    def test__getitem__(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.__getitem__`
        method.
        """

        np.testing.assert_almost_equal(
            self._multi_signals[0], np.array([10.0, 20.0, 30.0]), decimal=7)

        np.testing.assert_almost_equal(
            self._multi_signals[np.array([0, 1, 2])],
            np.array([[10.0, 20.0, 30.0], [20.0, 30.0, 40.0],
                      [30.0, 40.0, 50.0]]),
            decimal=7)

        np.testing.assert_almost_equal(
            self._multi_signals[np.linspace(0, 5, 5)],
            np.array([[10.00000000, 20.00000000,
                       30.00000000], [22.83489024, 32.80460562, 42.77432100],
                      [34.80044921, 44.74343470,
                       54.68642018], [47.55353925, 57.52325463, 67.49297001],
                      [60.00000000, 70.00000000, 80.00000000]]),
            decimal=7)

        assert np.all(np.isnan(self._multi_signals[np.array([-1000, 1000])]))

        multi_signals = self._multi_signals.copy()
        multi_signals.extrapolator_args = {
            'method': 'Linear',
        }
        np.testing.assert_array_equal(
            multi_signals[np.array([-1000, 1000])],
            np.array([[-9990.0, -9980.0, -9970.0], [10010.0, 10020.0,
                                                    10030.0]]))

        multi_signals.extrapolator_args = {
            'method': 'Constant',
            'left': 0,
            'right': 1
        }
        np.testing.assert_array_equal(
            multi_signals[np.array([-1000, 1000])],
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    def test_raise_exception__getitem__(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.__getitem__`
        method raised exception.
        """

        self.assertRaises(RuntimeError, operator.getitem, MultiSignals(), 0)

    def test__setitem__(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.__setitem__`
        method.
        """

        multi_signals = self._multi_signals.copy()

        multi_signals[0] = 20
        np.testing.assert_almost_equal(
            multi_signals[0], np.array([20.0, 20.0, 20.0]), decimal=7)

        multi_signals[np.array([0, 1, 2])] = 30
        np.testing.assert_almost_equal(
            multi_signals[np.array([0, 1, 2])],
            np.array([[30.0, 30.0, 30.0], [30.0, 30.0, 30.0],
                      [30.0, 30.0, 30.0]]),
            decimal=7)

        multi_signals[0:3] = 40
        np.testing.assert_almost_equal(
            multi_signals[0:3],
            np.array([[40.0, 40.0, 40.0], [40.0, 40.0, 40.0],
                      [40.0, 40.0, 40.0]]),
            decimal=7)

        multi_signals[np.linspace(0, 5, 5)] = 50
        np.testing.assert_almost_equal(
            multi_signals.domain,
            np.array([
                0.00, 1.00, 1.25, 2.00, 2.50, 3.00, 3.75, 4.00, 5.00, 6.00,
                7.00, 8.00, 9.00
            ]),
            decimal=7)
        np.testing.assert_almost_equal(
            multi_signals.range,
            np.array([
                [50.0, 50.0, 50.0],
                [40.0, 40.0, 40.0],
                [50.0, 50.0, 50.0],
                [40.0, 40.0, 40.0],
                [50.0, 50.0, 50.0],
                [40.0, 50.0, 60.0],
                [50.0, 50.0, 50.0],
                [50.0, 60.0, 70.0],
                [50.0, 50.0, 50.0],
                [70.0, 80.0, 90.0],
                [80.0, 90.0, 100.0],
                [90.0, 100.0, 110.0],
                [100.0, 110.0, 120.0],
            ]),
            decimal=7)

        multi_signals[np.array([0, 1, 2])] = np.array([10, 20, 30])
        np.testing.assert_almost_equal(
            multi_signals.range,
            np.array([
                [10.0, 20.0, 30.0],
                [10.0, 20.0, 30.0],
                [50.0, 50.0, 50.0],
                [10.0, 20.0, 30.0],
                [50.0, 50.0, 50.0],
                [40.0, 50.0, 60.0],
                [50.0, 50.0, 50.0],
                [50.0, 60.0, 70.0],
                [50.0, 50.0, 50.0],
                [70.0, 80.0, 90.0],
                [80.0, 90.0, 100.0],
                [90.0, 100.0, 110.0],
                [100.0, 110.0, 120.0],
            ]),
            decimal=7)

        multi_signals[np.array([0, 1, 2])] = np.arange(1, 10, 1).reshape(3, 3)
        np.testing.assert_almost_equal(
            multi_signals.range,
            np.array([
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [50.0, 50.0, 50.0],
                [7.0, 8.0, 9.0],
                [50.0, 50.0, 50.0],
                [40.0, 50.0, 60.0],
                [50.0, 50.0, 50.0],
                [50.0, 60.0, 70.0],
                [50.0, 50.0, 50.0],
                [70.0, 80.0, 90.0],
                [80.0, 90.0, 100.0],
                [90.0, 100.0, 110.0],
                [100.0, 110.0, 120.0],
            ]),
            decimal=7)

    def test__contains__(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.__contains__`
        method.
        """

        self.assertIn(0, self._multi_signals)
        self.assertIn(0.5, self._multi_signals)
        self.assertNotIn(1000, self._multi_signals)

    def test_raise_exception__contains__(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.__contains__`
        method raised exception.
        """

        self.assertRaises(RuntimeError, operator.contains, MultiSignals(), 0)

    def test__len__(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.__len__`
        method.
        """

        self.assertEqual(len(self._multi_signals), 10)

    def test__eq__(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.__eq__`
        method.
        """

        signal_1 = self._multi_signals.copy()
        signal_2 = self._multi_signals.copy()

        self.assertEqual(signal_1, signal_2)

    def test__ne__(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.__ne__`
        method.
        """

        multi_signals_1 = self._multi_signals.copy()
        multi_signals_2 = self._multi_signals.copy()

        multi_signals_2[0] = 20
        self.assertNotEqual(multi_signals_1, multi_signals_2)

        multi_signals_2[0] = np.array([10, 20, 30])
        self.assertEqual(multi_signals_1, multi_signals_2)

        multi_signals_2.interpolator = CubicSplineInterpolator
        self.assertNotEqual(multi_signals_1, multi_signals_2)

        multi_signals_2.interpolator = KernelInterpolator
        self.assertEqual(multi_signals_1, multi_signals_2)

        multi_signals_2.interpolator_args = {'window': 1}
        self.assertNotEqual(multi_signals_1, multi_signals_2)

        multi_signals_2.interpolator_args = {}
        self.assertEqual(multi_signals_1, multi_signals_2)

        class NotExtrapolator(Extrapolator):
            """
            Not :class:`Extrapolator` class.
            """

            pass

        multi_signals_2.extrapolator = NotExtrapolator
        self.assertNotEqual(multi_signals_1, multi_signals_2)

        multi_signals_2.extrapolator = Extrapolator
        self.assertEqual(multi_signals_1, multi_signals_2)

        multi_signals_2.extrapolator_args = {}
        self.assertNotEqual(multi_signals_1, multi_signals_2)

        multi_signals_2.extrapolator_args = {
            'method': 'Constant',
            'left': np.nan,
            'right': np.nan
        }
        self.assertEqual(multi_signals_1, multi_signals_2)

    def test_arithmetical_operation(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.\
arithmetical_operation` method.
        """

        np.testing.assert_almost_equal(
            self._multi_signals.arithmetical_operation(10, '+', False).range,
            self._range_2 + 10,
            decimal=7)

        np.testing.assert_almost_equal(
            self._multi_signals.arithmetical_operation(10, '-', False).range,
            self._range_2 - 10,
            decimal=7)

        np.testing.assert_almost_equal(
            self._multi_signals.arithmetical_operation(10, '*', False).range,
            self._range_2 * 10,
            decimal=7)

        np.testing.assert_almost_equal(
            self._multi_signals.arithmetical_operation(10, '/', False).range,
            self._range_2 / 10,
            decimal=7)

        np.testing.assert_almost_equal(
            self._multi_signals.arithmetical_operation(10, '**', False).range,
            self._range_2 ** 10,
            decimal=7)

        np.testing.assert_almost_equal(
            (self._multi_signals + 10).range, self._range_2 + 10, decimal=7)

        np.testing.assert_almost_equal(
            (self._multi_signals - 10).range, self._range_2 - 10, decimal=7)

        np.testing.assert_almost_equal(
            (self._multi_signals * 10).range, self._range_2 * 10, decimal=7)

        np.testing.assert_almost_equal(
            (self._multi_signals / 10).range, self._range_2 / 10, decimal=7)

        np.testing.assert_almost_equal(
            (self._multi_signals ** 10).range, self._range_2 ** 10, decimal=7)

        multi_signals = self._multi_signals.copy()

        np.testing.assert_almost_equal(
            multi_signals.arithmetical_operation(10, '+', True).range,
            self._range_2 + 10,
            decimal=7)

        np.testing.assert_almost_equal(
            multi_signals.arithmetical_operation(10, '-', True).range,
            self._range_2,
            decimal=7)

        np.testing.assert_almost_equal(
            multi_signals.arithmetical_operation(10, '*', True).range,
            self._range_2 * 10,
            decimal=7)

        np.testing.assert_almost_equal(
            multi_signals.arithmetical_operation(10, '/', True).range,
            self._range_2,
            decimal=7)

        np.testing.assert_almost_equal(
            multi_signals.arithmetical_operation(10, '**', True).range,
            self._range_2 ** 10,
            decimal=7)

        multi_signals = self._multi_signals.copy()
        np.testing.assert_almost_equal(
            multi_signals.arithmetical_operation(self._range_2, '+',
                                                 False).range,
            self._range_2 + self._range_2,
            decimal=7)

        np.testing.assert_almost_equal(
            multi_signals.arithmetical_operation(multi_signals, '+',
                                                 False).range,
            self._range_2 + self._range_2,
            decimal=7)

    def test_is_uniform(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.is_uniform`
        method.
        """

        self.assertTrue(self._multi_signals.is_uniform())

        multi_signals = self._multi_signals.copy()
        multi_signals[0.5] = 1.0
        self.assertFalse(multi_signals.is_uniform())

    def test_copy(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.copy` method.
        """

        self.assertIsNot(self._multi_signals, self._multi_signals.copy())
        self.assertEqual(self._multi_signals, self._multi_signals.copy())

    def test_multi_signals_unpack_data(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.\
multi_signals_unpack_data` method.
        """

        signals = MultiSignals.multi_signals_unpack_data(self._range_1)
        self.assertListEqual(list(signals.keys()), [0])
        np.testing.assert_array_equal(signals[0].domain, self._domain_1)
        np.testing.assert_array_equal(signals[0].range, self._range_1)

        signals = MultiSignals.multi_signals_unpack_data(
            self._range_1, self._domain_2)
        self.assertListEqual(list(signals.keys()), [0])
        np.testing.assert_array_equal(signals[0].domain, self._domain_2)
        np.testing.assert_array_equal(signals[0].range, self._range_1)

        signals = MultiSignals.multi_signals_unpack_data(
            self._range_2, self._domain_2)
        self.assertListEqual(list(signals.keys()), [0, 1, 2])
        np.testing.assert_array_equal(signals[0].range, self._range_1)
        np.testing.assert_array_equal(signals[1].range, self._range_1 + 10)
        np.testing.assert_array_equal(signals[2].range, self._range_1 + 20)

        signals = MultiSignals.multi_signals_unpack_data(
            dict(zip(self._domain_2, self._range_2)))
        self.assertListEqual(list(signals.keys()), [0, 1, 2])
        np.testing.assert_array_equal(signals[0].range, self._range_1)
        np.testing.assert_array_equal(signals[1].range, self._range_1 + 10)
        np.testing.assert_array_equal(signals[2].range, self._range_1 + 20)

        signals = MultiSignals.multi_signals_unpack_data(
            MultiSignals.multi_signals_unpack_data(
                dict(zip(self._domain_2, self._range_2))))
        self.assertListEqual(list(signals.keys()), [0, 1, 2])
        np.testing.assert_array_equal(signals[0].range, self._range_1)
        np.testing.assert_array_equal(signals[1].range, self._range_1 + 10)
        np.testing.assert_array_equal(signals[2].range, self._range_1 + 20)

        if is_pandas_installed():
            from pandas import DataFrame, Series

            signals = MultiSignals.multi_signals_unpack_data(
                Series(dict(zip(self._domain_1, self._range_1))))
            self.assertListEqual(list(signals.keys()), [0])
            np.testing.assert_array_equal(signals[0].domain, self._domain_1)
            np.testing.assert_array_equal(signals[0].range, self._range_1)

            data = dict(zip(['a', 'b', 'c'], tsplit(self._range_2)))
            signals = MultiSignals.multi_signals_unpack_data(
                DataFrame(data, self._domain_1))
            self.assertListEqual(list(signals.keys()), ['a', 'b', 'c'])
            np.testing.assert_array_equal(signals['a'].range, self._range_1)
            np.testing.assert_array_equal(signals['b'].range,
                                          self._range_1 + 10)
            np.testing.assert_array_equal(signals['c'].range,
                                          self._range_1 + 20)

    def test_fill_nan(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.fill_nan`
        method.
        """

        multi_signals = self._multi_signals.copy()

        multi_signals[3:7] = np.nan

        np.testing.assert_almost_equal(
            multi_signals.fill_nan().range,
            np.array([[10.0, 20.0, 30.0], [20.0, 30.0, 40.0], [
                30.0, 40.0, 50.0
            ], [40.0, 50.0, 60.0], [50.0, 60.0, 70.0], [60.0, 70.0, 80.0],
                      [70.0, 80.0, 90.0], [80.0, 90.0, 100.0],
                      [90.0, 100.0, 110.0], [100.0, 110.0, 120.0]]),
            decimal=7)

        multi_signals[3:7] = np.nan

        np.testing.assert_almost_equal(
            multi_signals.fill_nan(method='Constant').range,
            np.array([[10.0, 20.0, 30.0], [20.0, 30.0, 40.0],
                      [30.0, 40.0, 50.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [80.0, 90.0, 100.0],
                      [90.0, 100.0, 110.0], [100.0, 110.0, 120.0]]),
            decimal=7)

    def test_domain_distance(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.\
    domain_distance` method.
        """

        self.assertAlmostEqual(
            self._multi_signals.domain_distance(0.5), 0.5, places=7)

        np.testing.assert_almost_equal(
            self._multi_signals.domain_distance(np.linspace(0, 9, 10) + 0.5),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            decimal=7)

    def test_to_dataframe(self):
        """
        Tests :func:`colour.continuous.multi_signals.MultiSignals.to_dataframe`
        method.
        """

        if is_pandas_installed():
            from pandas import DataFrame

            data = dict(zip(['a', 'b', 'c'], tsplit(self._range_2)))
            assert MultiSignals(
                self._range_2, self._domain_2,
                labels=['a', 'b', 'c']).to_dataframe().equals(
                    DataFrame(data, self._domain_2))


if __name__ == '__main__':
    unittest.main()
