#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Iterators
=========

Defines various iterators convenience classes:

-   :class:`Peekable`: An iterator object that supports peeking ahead.
"""

from __future__ import division, unicode_literals

import collections

__author__ = 'Rob Ruana, Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Peekable']


class Peekable(object):
    """
    Implements an iterator object that supports peeking ahead.

    Arguments
    ---------
    a : iterable
        `a` is interpreted very differently depending on the presence of
        `sentinel`. If `sentinel` is not given, then `a` must be a
        collection object which supports either the iteration protocol or
        the sequence protocol. If `sentinel` is given, then `a` must be a
        callable object.
    sentinel : object, optional
        The value used to indicate the iterator is exhausted.
        If `sentinel` was not given when the `Peekable` was instantiated,
        then it will be set to a new object instance: ``object()``. If
        given, the iterator will call `a` with no arguments for each call
        to its `next` method; if the value returned is equal to `sentinel`,
        :exc:`StopIteration` will be raised, otherwise the value will be
        returned.

    Attributes
    ----------
    sentinel

    Methods
    -------
    __iter__
    __next__
    exhausted
    next
    peek

    References
    ----------
    .. [1]  Ruana, R. (n.d.). pockets.iterators. Retrieved August 14, 2016,
            from https://github.com/RobRuana/pockets\
/blob/master/pockets/iterators.py

    Examples
    --------
    >>> iterator = Peekable([0, 1, 2, 3, 4])
    >>> iterator.peek()
    0
    >>> iterator.next()
    0
    >>> iterator.peek(3)
    [1, 2, 3]
    """

    def __init__(self, *args):
        self._iterable = iter(*args)
        self._cache = collections.deque()
        self._sentinel = args[1] if len(args) > 1 else object()

    @property
    def sentinel(self):
        """
        Property for **self.sentinel** attribute.

        Returns
        -------
        object
            The value used to indicate the iterator is exhausted.
        """

        return self._sentinel

    @sentinel.setter
    def sentinel(self, value):
        """
        Setter for **self.sentinel** attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        self._sentinel = value

    def __iter__(self):
        """
        Notes
        -----
        -   Reimplements the :meth:`object.__iter__` method.
        """

        return self

    def __next__(self, n=None):
        """
        Notes
        -----
        -   Reimplements the :meth:`object.__next__` method.
        """

        return getattr(self, 'next')(n)

    def _fill_cache(self, n):
        """
        Caches `n` items. If `n` is 0 or None, then 1 item is cached.
        """

        if not n:
            n = 1
        try:
            while len(self._cache) < n:
                self._cache.append(next(self._iterable))
        except StopIteration:
            while len(self._cache) < n:
                self._cache.append(self.sentinel)

    def exhausted(self):
        """
        Determines if iterator is exhausted.

        Returns
        -------
        bool
            True if iterator has more items, False otherwise.

        Notes
        -----
        -   `has_next` never raises :exc:`StopIteration`.
        """

        return self.peek() == self._sentinel

    def next(self, n=None):
        """
        Gets the next item or `n` items of the iterator.

        Arguments
        ---------
        n : int, optional
            The number of items to retrieve. Defaults to `None`.

        Returns
        -------
        object or list
            The next item or `n` items of the iterator.
            If `n` is None, the item itself is returned. If `n` is an int,
            the items will be returned in a list. If `n` is 0, an empty
            list is returned.

        Raises
        ------
        StopIteration
            If the iterator is exhausted, even if `n` is 0.

        Examples
        --------
        >>> iterator = Peekable([0, 1, 2, 3, 4])
        >>> iterator.next()
        0
        >>> iterator.next(0)
        []
        >>> iterator.next(1)
        [1]
        >>> iterator.next(2)
        [2, 3]
        """

        self._fill_cache(n)
        if not n:
            if self._cache[0] == self.sentinel:
                raise StopIteration
            if n is None:
                result = self._cache.popleft()
            else:
                result = []
        else:
            if self._cache[n - 1] == self.sentinel:
                raise StopIteration
            result = [self._cache.popleft() for _i in range(n)]
        return result

    def peek(self, n=None):
        """
        Previews the next item or `n` items of the iterator.

        The iterator is not advanced when peek is called.

        Arguments
        ---------
        n : int, optional
            The number of items to retrieve. Defaults to `None`.

        Returns
        -------
        object or list
            The next item or `n` items of the iterator.
            If `n` is None, the item itself is returned. If `n` is an int,
            the items will be returned in a list. If `n` is 0, an empty
            list is returned. If the iterator is exhausted,
            `peek_iter.sentinel` is returned, or placed as the last item in
            the returned list.

        Notes
        -----
        -   `peek` never raises :exc:`StopIteration`.

        Examples
        --------
        >>> interator = Peekable([0, 1, 2, 3, 4])
        >>> interator.sentinel = -1
        >>> interator.peek()
        0
        >>> interator.peek(0)
        []
        >>> interator.peek(1)
        [0]
        >>> interator.peek(2)
        [0, 1]
        >>> interator.peek(6)
        [0, 1, 2, 3, 4, -1]
        """

        self._fill_cache(n)
        if n is None:
            result = self._cache[0]
        else:
            result = [self._cache[i] for i in range(n)]
        return result
