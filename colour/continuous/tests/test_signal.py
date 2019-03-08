# -*- coding: utf-8 -*-
"""
Defines unit tests for :mod:`colour.continuous.signal` module.
"""

from __future__ import division, unicode_literals

import numpy as np
import unittest
import re
import textwrap

from colour.algebra import (CubicSplineInterpolator, Extrapolator,
                            KernelInterpolator)
from colour.continuous import Signal
from colour.utilities import is_pandas_installed

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestSignal']


class TestSignal(unittest.TestCase):
    """
    Defines :class:`colour.continuous.signal.Signal` class unit tests methods.
    """

    def setUp(self):
        """
        Initialises common tests attributes.
        """

        self._range = np.linspace(10, 100, 10)
        self._domain = np.arange(100, 1100, 100)

        self._signal = Signal(self._range)

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('dtype', 'domain', 'range', 'interpolator',
                               'interpolator_args', 'extrapolator',
                               'extrapolator_args', 'function')

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Signal))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__str__', '__repr__', '__hash__', '__getitem__',
                            '__setitem__', '__contains__', '__eq__', '__ne__',
                            'arithmetical_operation', 'signal_unpack_data',
                            'fill_nan', 'domain_distance', 'to_series')

        for method in required_methods:
            self.assertIn(method, dir(Signal))

    def test_domain(self):
        """
        Tests :func:`colour.continuous.signal.Signal.domain` property.
        """

        signal = self._signal.copy()

        np.testing.assert_almost_equal(
            signal[np.array([0, 1, 2])],
            np.array([10.0, 20.0, 30.0]),
            decimal=7)

        signal.domain = np.arange(0, 10, 1) * 10

        np.testing.assert_array_equal(signal.domain, np.arange(0, 10, 1) * 10)

        np.testing.assert_almost_equal(
            signal[np.array([0, 1, 2]) * 10],
            np.array([10.0, 20.0, 30.0]),
            decimal=7)

    def test_range(self):
        """
        Tests :func:`colour.continuous.signal.Signal.range` property.
        """

        signal = self._signal.copy()

        np.testing.assert_almost_equal(
            signal[np.array([0, 1, 2])],
            np.array([10.0, 20.0, 30.0]),
            decimal=7)

        signal.range = self._range * 10

        np.testing.assert_array_equal(signal.range, self._range * 10)

        np.testing.assert_almost_equal(
            signal[np.array([0, 1, 2])],
            np.array([10.0, 20.0, 30.0]) * 10,
            decimal=7)

    def test_interpolator(self):
        """
        Tests :func:`colour.continuous.signal.Signal.interpolator` property.
        """

        signal = self._signal.copy()

        np.testing.assert_almost_equal(
            signal[np.linspace(0, 5, 5)],
            np.array([
                10.00000000, 22.83489024, 34.80044921, 47.55353925, 60.00000000
            ]),
            decimal=7)

        signal.interpolator = CubicSplineInterpolator

        np.testing.assert_almost_equal(
            signal[np.linspace(0, 5, 5)],
            np.array([10.0, 22.5, 35.0, 47.5, 60.0]),
            decimal=7)

    def test_interpolator_args(self):
        """
        Tests :func:`colour.continuous.signal.Signal.interpolator_args`
        property.
        """

        signal = self._signal.copy()

        np.testing.assert_almost_equal(
            signal[np.linspace(0, 5, 5)],
            np.array([
                10.00000000, 22.83489024, 34.80044921, 47.55353925, 60.00000000
            ]),
            decimal=7)

        signal.interpolator_args = {'window': 1, 'kernel_args': {'a': 1}}

        np.testing.assert_almost_equal(
            signal[np.linspace(0, 5, 5)],
            np.array([
                10.00000000, 18.91328761, 28.36993142, 44.13100443, 60.00000000
            ]),
            decimal=7)

    def test_extrapolator(self):
        """
        Tests :func:`colour.continuous.signal.Signal.extrapolator` property.
        """

        self.assertIsInstance(self._signal.extrapolator(), Extrapolator)

    def test_extrapolator_args(self):
        """
        Tests :func:`colour.continuous.signal.Signal.extrapolator_args`
        property.
        """

        signal = self._signal.copy()

        assert np.all(np.isnan(signal[np.array([-1000, 1000])]))

        signal.extrapolator_args = {
            'method': 'Linear',
        }

        np.testing.assert_almost_equal(
            signal[np.array([-1000, 1000])],
            np.array([-9990.0, 10010.0]),
            decimal=7)

    def test_function(self):
        """
        Tests :func:`colour.continuous.signal.Signal.function` property.
        """

        assert hasattr(self._signal.function, '__call__')

    def test__init__(self):
        """
        Tests :func:`colour.continuous.signal.Signal.__init__` method.
        """

        signal = Signal(self._range)
        np.testing.assert_array_equal(signal.domain, np.arange(0, 10, 1))
        np.testing.assert_array_equal(signal.range, self._range)

        signal = Signal(self._range, self._domain)
        np.testing.assert_array_equal(signal.domain, self._domain)
        np.testing.assert_array_equal(signal.range, self._range)

        signal = Signal(dict(zip(self._domain, self._range)))
        np.testing.assert_array_equal(signal.domain, self._domain)
        np.testing.assert_array_equal(signal.range, self._range)

        signal = Signal(signal)
        np.testing.assert_array_equal(signal.domain, self._domain)
        np.testing.assert_array_equal(signal.range, self._range)

        if is_pandas_installed():
            from pandas import Series

            signal = Signal(Series(dict(zip(self._domain, self._range))))
            np.testing.assert_array_equal(signal.domain, self._domain)
            np.testing.assert_array_equal(signal.range, self._range)

    def test__str__(self):
        """
        Tests :func:`colour.continuous.signal.Signal.__str__` method.
        """

        self.assertEqual(
            str(self._signal),
            textwrap.dedent("""
                [[   0.   10.]
                 [   1.   20.]
                 [   2.   30.]
                 [   3.   40.]
                 [   4.   50.]
                 [   5.   60.]
                 [   6.   70.]
                 [   7.   80.]
                 [   8.   90.]
                 [   9.  100.]]""")[1:])

    def test__repr__(self):
        """
        Tests :func:`colour.continuous.signal.Signal.__repr__` method.
        """

        self.assertEqual(
            re.sub(r'extrapolator_args={.*}', 'extrapolator_args={...}',
                   repr(self._signal)),
            textwrap.dedent("""
                Signal([[   0.,   10.],
                        [   1.,   20.],
                        [   2.,   30.],
                        [   3.,   40.],
                        [   4.,   50.],
                        [   5.,   60.],
                        [   6.,   70.],
                        [   7.,   80.],
                        [   8.,   90.],
                        [   9.,  100.]],
                       interpolator=KernelInterpolator,
                       interpolator_args={},
                       extrapolator=Extrapolator,
                       extrapolator_args={...})""")[1:])

    def test__getitem__(self):
        """
        Tests :func:`colour.continuous.signal.Signal.__getitem__` method.
        """

        self.assertEqual(self._signal[0], 10.0)

        np.testing.assert_almost_equal(
            self._signal[np.array([0, 1, 2])],
            np.array([10.0, 20.0, 30.0]),
            decimal=7)

        np.testing.assert_almost_equal(
            self._signal[np.linspace(0, 5, 5)],
            np.array([
                10.00000000, 22.83489024, 34.80044921, 47.55353925, 60.00000000
            ]),
            decimal=7)

        assert np.all(np.isnan(self._signal[np.array([-1000, 1000])]))

        signal = self._signal.copy()
        signal.extrapolator_args = {
            'method': 'Linear',
        }
        np.testing.assert_array_equal(signal[np.array([-1000, 1000])],
                                      np.array([-9990.0, 10010.0]))

        signal.extrapolator_args = {
            'method': 'Constant',
            'left': 0,
            'right': 1
        }
        np.testing.assert_array_equal(signal[np.array([-1000, 1000])],
                                      np.array([0.0, 1.0]))

    def test__setitem__(self):
        """
        Tests :func:`colour.continuous.signal.Signal.__setitem__` method.
        """

        signal = self._signal.copy()

        signal[0] = 20
        np.testing.assert_almost_equal(
            signal.range,
            np.array(
                [20.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]))

        signal[np.array([0, 1, 2])] = 30
        np.testing.assert_almost_equal(
            signal.range,
            np.array(
                [30.0, 30.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]),
            decimal=7)

        signal[0:3] = 40
        np.testing.assert_almost_equal(
            signal.range,
            np.array(
                [40.0, 40.0, 40.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]),
            decimal=7)

        signal[np.linspace(0, 5, 5)] = 50
        np.testing.assert_almost_equal(
            signal.domain,
            np.array([
                0.00, 1.00, 1.25, 2.00, 2.50, 3.00, 3.75, 4.00, 5.00, 6.00,
                7.00, 8.00, 9.00
            ]),
            decimal=7)
        np.testing.assert_almost_equal(
            signal.range,
            np.array([
                50.0, 40.0, 50.0, 40.0, 50.0, 40.0, 50.0, 50.0, 50.0, 70.0,
                80.0, 90.0, 100.0
            ]),
            decimal=7)

        signal[np.array([0, 1, 2])] = np.array([10, 20, 30])
        np.testing.assert_almost_equal(
            signal.range,
            np.array([
                10.0, 20.0, 50.0, 30.0, 50.0, 40.0, 50.0, 50.0, 50.0, 70.0,
                80.0, 90.0, 100.0
            ]),
            decimal=7)

    def test__contains__(self):
        """
        Tests :func:`colour.continuous.signal.Signal.__contains__` method.
        """

        self.assertIn(0, self._signal)
        self.assertIn(0.5, self._signal)
        self.assertNotIn(1000, self._signal)

    def test__len__(self):
        """
        Tests :func:`colour.continuous.signal.Signal.__len__` method.
        """

        self.assertEqual(len(self._signal), 10)

    def test__eq__(self):
        """
        Tests :func:`colour.continuous.signal.Signal.__eq__` method.
        """

        signal_1 = self._signal.copy()
        signal_2 = self._signal.copy()

        self.assertEqual(signal_1, signal_2)

    def test__ne__(self):
        """
        Tests :func:`colour.continuous.signal.Signal.__ne__` method.
        """

        signal_1 = self._signal.copy()
        signal_2 = self._signal.copy()

        signal_2[0] = 20
        self.assertNotEqual(signal_1, signal_2)

        signal_2[0] = 10
        self.assertEqual(signal_1, signal_2)

        signal_2.interpolator = CubicSplineInterpolator
        self.assertNotEqual(signal_1, signal_2)

        signal_2.interpolator = KernelInterpolator
        self.assertEqual(signal_1, signal_2)

        signal_2.interpolator_args = {'window': 1}
        self.assertNotEqual(signal_1, signal_2)

        signal_2.interpolator_args = {}
        self.assertEqual(signal_1, signal_2)

        class NotExtrapolator(Extrapolator):
            """
            Not :class:`Extrapolator` class.
            """

            pass

        signal_2.extrapolator = NotExtrapolator
        self.assertNotEqual(signal_1, signal_2)

        signal_2.extrapolator = Extrapolator
        self.assertEqual(signal_1, signal_2)

        signal_2.extrapolator_args = {}
        self.assertNotEqual(signal_1, signal_2)

        signal_2.extrapolator_args = {
            'method': 'Constant',
            'left': np.nan,
            'right': np.nan
        }
        self.assertEqual(signal_1, signal_2)

    def test_arithmetical_operation(self):
        """
        Tests :func:`colour.continuous.signal.Signal.arithmetical_operation`
        method.
        """

        np.testing.assert_almost_equal(
            self._signal.arithmetical_operation(10, '+', False).range,
            self._range + 10,
            decimal=7)

        np.testing.assert_almost_equal(
            self._signal.arithmetical_operation(10, '-', False).range,
            self._range - 10,
            decimal=7)

        np.testing.assert_almost_equal(
            self._signal.arithmetical_operation(10, '*', False).range,
            self._range * 10,
            decimal=7)

        np.testing.assert_almost_equal(
            self._signal.arithmetical_operation(10, '/', False).range,
            self._range / 10,
            decimal=7)

        np.testing.assert_almost_equal(
            self._signal.arithmetical_operation(10, '**', False).range,
            self._range ** 10,
            decimal=7)

        signal = self._signal.copy()

        np.testing.assert_almost_equal(
            signal.arithmetical_operation(10, '+', True).range,
            self._range + 10,
            decimal=7)

        np.testing.assert_almost_equal(
            signal.arithmetical_operation(10, '-', True).range,
            self._range,
            decimal=7)

        np.testing.assert_almost_equal(
            signal.arithmetical_operation(10, '*', True).range,
            self._range * 10,
            decimal=7)

        np.testing.assert_almost_equal(
            signal.arithmetical_operation(10, '/', True).range,
            self._range,
            decimal=7)

        np.testing.assert_almost_equal(
            signal.arithmetical_operation(10, '**', True).range,
            self._range ** 10,
            decimal=7)

        signal = self._signal.copy()

        np.testing.assert_almost_equal(
            signal.arithmetical_operation(self._range, '+', False).range,
            signal.range + self._range,
            decimal=7)

        np.testing.assert_almost_equal(
            signal.arithmetical_operation(signal, '+', False).range,
            signal.range + signal._range,
            decimal=7)

    def test_is_uniform(self):
        """
        Tests :func:`colour.continuous.signal.Signal.is_uniform` method.
        """

        self.assertTrue(self._signal.is_uniform())

        signal = self._signal.copy()
        signal[0.5] = 1.0
        self.assertFalse(signal.is_uniform())

    def test_copy(self):
        """
        Tests :func:`colour.continuous.signal.Signal.copy` method.
        """

        self.assertIsNot(self._signal, self._signal.copy())
        self.assertEqual(self._signal, self._signal.copy())

    def test_signal_unpack_data(self):
        """
        Tests :func:`colour.continuous.signal.Signal.signal_unpack_data`
        method.
        """

        domain, range_ = Signal.signal_unpack_data(self._range)
        np.testing.assert_array_equal(range_, self._range)
        np.testing.assert_array_equal(domain, np.arange(0, 10, 1))

        domain, range_ = Signal.signal_unpack_data(self._range, self._domain)
        np.testing.assert_array_equal(range_, self._range)
        np.testing.assert_array_equal(domain, self._domain)

        domain, range_ = Signal.signal_unpack_data(
            dict(zip(self._domain, self._range)))
        np.testing.assert_array_equal(range_, self._range)
        np.testing.assert_array_equal(domain, self._domain)

        domain, range_ = Signal.signal_unpack_data(
            Signal(self._range, self._domain))
        np.testing.assert_array_equal(range_, self._range)
        np.testing.assert_array_equal(domain, self._domain)

        if is_pandas_installed():
            from pandas import Series

            domain, range_ = Signal.signal_unpack_data(
                Series(dict(zip(self._domain, self._range))))
            np.testing.assert_array_equal(range_, self._range)
            np.testing.assert_array_equal(domain, self._domain)

    def test_fill_nan(self):
        """
        Tests :func:`colour.continuous.signal.Signal.fill_nan` method.
        """

        signal = self._signal.copy()

        signal[3:7] = np.nan

        np.testing.assert_almost_equal(
            signal.fill_nan().range,
            np.array(
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]),
            decimal=7)

        signal[3:7] = np.nan

        np.testing.assert_almost_equal(
            signal.fill_nan(method='Constant').range,
            np.array([10.0, 20.0, 30.0, 0.0, 0.0, 0.0, 0.0, 80.0, 90.0,
                      100.0]),
            decimal=7)

    def test_domain_distance(self):
        """
        Tests :func:`colour.continuous.signal.Signal.domain_distance` method.
        """

        self.assertAlmostEqual(
            self._signal.domain_distance(0.5), 0.5, places=7)

        np.testing.assert_almost_equal(
            self._signal.domain_distance(np.linspace(0, 9, 10) + 0.5),
            np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
            decimal=7)

    def test_to_series(self):
        """
        Tests :func:`colour.continuous.signal.Signal.to_series` method.
        """

        if is_pandas_installed():
            from pandas import Series

            self.assertEqual(
                Signal(self._range, self._domain).to_series().all(),
                Series(dict(zip(self._domain, self._range))).all())


if __name__ == '__main__':
    unittest.main()
