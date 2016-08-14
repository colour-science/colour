#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines unit tests for :mod:`colour.utilities.iterators` module.
"""

from __future__ import division, unicode_literals

import unittest

from colour.utilities import Peekable

__author__ = 'Rob Ruana, Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['TestPeekable']


class TestPeekable(unittest.TestCase):
    """
    Defines :class:`colour.utilities.iterators.Peekable` class units tests
    methods.

    References
    ----------
    .. [1]  Ruana, R. (n.d.). pockets.tests.test_iterators. Retrieved
            August 14, 2016, from
            https://github.com/RobRuana/pockets/\
blob/master/tests/test_iterators.py
    """

    def test_required_attributes(self):
        """
        Tests presence of required attributes.
        """

        required_attributes = ('sentinel',)

        for attribute in required_attributes:
            self.assertIn(attribute, dir(Peekable))

    def test_required_methods(self):
        """
        Tests presence of required methods.
        """

        required_methods = ('__iter__',
                            '__next__',
                            'exhausted',
                            'next',
                            'peek')

        for method in required_methods:
            self.assertIn(method, dir(Peekable))

    def _assert_equal_twice(self, expected, callable_, *args):
        """
        Asserts twice that given expected value is equal to given callable
        return value.

        Parameters
        ----------
        expected : object
            Expected value.
        callable_ : callable
            Callable.
        \*args : list, optional
            Arguments to call `callable_` with.

        Raises
        ------
        AssertionError
            If assertions fails.
        """

        self.assertEqual(expected, callable_(*args))
        self.assertEqual(expected, callable_(*args))

    def _assert_true_twice(self, callable_, *args):
        """
         Asserts given given callable return value is `True` twice.

         Parameters
         ----------
         callable_ : callable
             Callable.
         \*args : list, optional
             Arguments to call `callable_` with.

         Raises
         ------
         AssertionError
             If assertions fails.
         """

        self.assertTrue(callable_(*args))
        self.assertTrue(callable_(*args))

    def _assert_false_twice(self, callable_, *args):
        """
        Asserts given given callable return value is `False` twice.

        Parameters
        ----------
        callable_ : callable
            Callable.
        \*args : list, optional
            Arguments to call `callable_` with.

        Raises
        ------
        AssertionError
            If assertions fails.
        """

        self.assertFalse(callable_(*args))
        self.assertFalse(callable_(*args))

    def _assert_next(self, iterator, expected, is_last):
        """
        Asserts that given iterator next item is equal to given expected value.

        Parameters
        ----------
        iterator : iterator
            Iterator.
        expected : object
            Expected value.
        is_last : bool
            Expected value is last item of the iterator.

        Raises
        ------
        AssertionError
            If assertions fails.
        """

        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice(expected, iterator.peek)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice(expected, iterator.peek)
        self._assert_false_twice(iterator.exhausted)
        self.assertEqual(expected, next(iterator))

        if is_last:
            self._assert_true_twice(iterator.exhausted)
            self._assert_raises_twice(StopIteration, iterator.next)
        else:
            self._assert_false_twice(iterator.exhausted)

    def _assert_raises_twice(self, expected, callable_, *args):
        """
        Asserts twice that given callable will raise an exception.

        Parameters
        ----------
        expected : object
            Expected value.
        callable_ : callable
            Callable.
        \*args : list, optional
            Arguments to call `callable_` with.

        Raises
        ------
        AssertionError
            If assertions fails.
        """

        self.assertRaises(expected, callable_, *args)
        self.assertRaises(expected, callable_, *args)

    def test__init__(self):
        """
        Tests :meth:`colour.utilities.iterators.Peekable.__init__` method.
        """

        a = iter([1, 2, -1])
        sentinel = -1
        self.assertRaises(TypeError, Peekable, a, sentinel)

        def get_next():
            return next(a)

        iterator = Peekable(get_next, sentinel)
        self.assertEqual(iterator.sentinel, sentinel)
        self._assert_next(iterator, 1, is_last=False)
        self._assert_next(iterator, 2, is_last=True)

    def test__iter__(self):
        """
        Tests :meth:`colour.utilities.iterators.Peekable.__iter__` method.
        """

        a = [1, 2, 3]
        iterator = Peekable(a)
        self.assertTrue(iterator is iterator.__iter__())

        a = []
        b = [i for i in Peekable(a)]
        self.assertEqual([], b)

        a = [1]
        b = [i for i in Peekable(a)]
        self.assertEqual([1], b)

        a = [1, 2]
        b = [i for i in Peekable(a)]
        self.assertEqual([1, 2], b)

        a = [1, 2, 3]
        b = [i for i in Peekable(a)]
        self.assertEqual([1, 2, 3], b)

    def test_next_default(self):
        """
        Tests :meth:`colour.utilities.iterators.Peekable.next` method when
        retrieving values using definition signature.
        """

        a = []
        iterator = Peekable(a)
        self._assert_true_twice(iterator.exhausted)
        self._assert_raises_twice(StopIteration, iterator.next)
        self._assert_true_twice(iterator.exhausted)

        a = [1]
        iterator = Peekable(a)
        self.assertEqual(1, iterator.__next__())

        a = [1]
        iterator = Peekable(a)
        self._assert_next(iterator, 1, is_last=True)

        a = [1, 2]
        iterator = Peekable(a)
        self._assert_next(iterator, 1, is_last=False)
        self._assert_next(iterator, 2, is_last=True)

        a = [1, 2, 3]
        iterator = Peekable(a)
        self._assert_next(iterator, 1, is_last=False)
        self._assert_next(iterator, 2, is_last=False)
        self._assert_next(iterator, 3, is_last=True)

    def test_next_zero(self):
        """
        Tests :meth:`colour.utilities.iterators.Peekable.next` method when
        retrieving zero value.
        """

        a = []
        iterator = Peekable(a)
        self._assert_true_twice(iterator.exhausted)
        self._assert_raises_twice(StopIteration, iterator.next, 0)

        a = [1]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([], iterator.next, 0)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([], iterator.next, 0)

        a = [1, 2]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([], iterator.next, 0)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([], iterator.next, 0)

    def test_next_one(self):
        """
        Tests :meth:`colour.utilities.iterators.Peekable.next` method when
        retrieving one value.
        """

        a = []
        iterator = Peekable(a)
        self._assert_true_twice(iterator.exhausted)
        self._assert_raises_twice(StopIteration, iterator.next, 1)

        a = [1]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self.assertEqual([1], iterator.next(1))
        self._assert_true_twice(iterator.exhausted)
        self._assert_raises_twice(StopIteration, iterator.next, 1)

        a = [1, 2]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self.assertEqual([1], iterator.next(1))
        self._assert_false_twice(iterator.exhausted)
        self.assertEqual([2], iterator.next(1))
        self._assert_true_twice(iterator.exhausted)
        self._assert_raises_twice(StopIteration, iterator.next, 1)

    def test_next_multiple(self):
        """
        Tests :meth:`colour.utilities.iterators.Peekable.next` method when
        retrieving multiple values.
        """

        a = []
        iterator = Peekable(a)
        self._assert_true_twice(iterator.exhausted)
        self._assert_raises_twice(StopIteration, iterator.next, 2)

        a = [1]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self._assert_raises_twice(StopIteration, iterator.next, 2)
        self._assert_false_twice(iterator.exhausted)

        a = [1, 2]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self.assertEqual([1, 2], iterator.next(2))
        self._assert_true_twice(iterator.exhausted)

        a = [1, 2, 3]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self.assertEqual([1, 2], iterator.next(2))
        self._assert_false_twice(iterator.exhausted)
        self._assert_raises_twice(StopIteration, iterator.next, 2)
        self._assert_false_twice(iterator.exhausted)

        a = [1, 2, 3, 4]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self.assertEqual([1, 2], iterator.next(2))
        self._assert_false_twice(iterator.exhausted)
        self.assertEqual([3, 4], iterator.next(2))
        self._assert_true_twice(iterator.exhausted)
        self._assert_raises_twice(StopIteration, iterator.next, 2)
        self._assert_true_twice(iterator.exhausted)

    def test_peek_default(self):
        """
        Tests :meth:`colour.utilities.iterators.Peekable.peek` method when
        peeking values using definition signature.
        """

        a = []
        iterator = Peekable(a)
        self._assert_true_twice(iterator.exhausted)
        self._assert_equal_twice(iterator.sentinel, iterator.peek)
        self._assert_true_twice(iterator.exhausted)

        a = [1]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice(1, iterator.peek)
        self.assertEqual(1, next(iterator))
        self._assert_true_twice(iterator.exhausted)
        self._assert_equal_twice(iterator.sentinel, iterator.peek)
        self._assert_true_twice(iterator.exhausted)

        a = [1, 2]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice(1, iterator.peek)
        self.assertEqual(1, next(iterator))
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice(2, iterator.peek)
        self.assertEqual(2, next(iterator))
        self._assert_true_twice(iterator.exhausted)
        self._assert_equal_twice(iterator.sentinel, iterator.peek)
        self._assert_true_twice(iterator.exhausted)

    def test_peek_zero(self):
        """
        Tests :meth:`colour.utilities.iterators.Peekable.peek` method when
        peeking zero value.
        """

        a = []
        iterator = Peekable(a)
        self._assert_true_twice(iterator.exhausted)
        self._assert_equal_twice([], iterator.peek, 0)

        a = [1]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([], iterator.peek, 0)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([], iterator.peek, 0)

        a = [1, 2]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([], iterator.peek, 0)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([], iterator.peek, 0)

    def test_peek_one(self):
        """
        Tests :meth:`colour.utilities.iterators.Peekable.peek` method when
        peeking one value.
        """

        a = []
        iterator = Peekable(a)
        self._assert_true_twice(iterator.exhausted)
        self._assert_equal_twice([iterator.sentinel], iterator.peek, 1)
        self._assert_true_twice(iterator.exhausted)

        a = [1]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([1], iterator.peek, 1)
        self.assertEqual(1, next(iterator))
        self._assert_true_twice(iterator.exhausted)
        self._assert_equal_twice([iterator.sentinel], iterator.peek, 1)
        self._assert_true_twice(iterator.exhausted)

        a = [1, 2]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([1], iterator.peek, 1)
        self.assertEqual(1, next(iterator))
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([2], iterator.peek, 1)
        self.assertEqual(2, next(iterator))
        self._assert_true_twice(iterator.exhausted)
        self._assert_equal_twice([iterator.sentinel], iterator.peek, 1)
        self._assert_true_twice(iterator.exhausted)

    def test_peek_multiple(self):

        """
        Tests :meth:`colour.utilities.iterators.Peekable.peek` method when
        peeking multiple values.
        """

        a = []
        iterator = Peekable(a)
        self._assert_true_twice(iterator.exhausted)
        self._assert_equal_twice(
            [iterator.sentinel, iterator.sentinel], iterator.peek, 2)
        self._assert_true_twice(iterator.exhausted)

        a = [1]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([1, iterator.sentinel], iterator.peek, 2)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice(
            [1, iterator.sentinel, iterator.sentinel], iterator.peek, 3)
        self._assert_false_twice(iterator.exhausted)

        a = [1, 2]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([1, 2], iterator.peek, 2)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([1, 2, iterator.sentinel], iterator.peek, 3)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice(
            [1, 2, iterator.sentinel, iterator.sentinel], iterator.peek, 4)
        self._assert_false_twice(iterator.exhausted)

        a = [1, 2, 3]
        iterator = Peekable(a)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([1, 2], iterator.peek, 2)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([1, 2, 3], iterator.peek, 3)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice(
            [1, 2, 3, iterator.sentinel], iterator.peek, 4)
        self._assert_false_twice(iterator.exhausted)
        self.assertEqual(1, next(iterator))
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([2, 3], iterator.peek, 2)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice([2, 3, iterator.sentinel], iterator.peek, 3)
        self._assert_false_twice(iterator.exhausted)
        self._assert_equal_twice(
            [2, 3, iterator.sentinel, iterator.sentinel], iterator.peek, 4)
        self._assert_false_twice(iterator.exhausted)


if __name__ == '__main__':
    unittest.main()
