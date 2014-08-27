#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decorators
==========

Defines various utility decorators.

"""

from __future__ import division, unicode_literals

import functools

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2008 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['memoize']


def memoize(cache=None):
    """
    Implements method / definition memoization.

    Any method / definition decorated will get its return value cached and
    restored whenever called with the same arguments.

    Parameters
    ----------
    cache : dict
        Alternate cache.

    Returns
    -------
    object
        Callable object.

    References
    ----------
    .. [1]  https://github.com/KelSolaar/Foundations/blob/develop/foundations/decorators.py  # noqa
    """

    if cache is None:
        cache = {}

    def memoize_decorator(callable):
        """
        Implements method / definition memoization.

        Parameters
        ----------
        object : object
            Callable object to decorate.

        Returns
        -------
        object
            Object.
        """

        @functools.wraps(callable)
        def memoize_wrapper(*args, **kwargs):
            """
            Implements method / definition memoization.

            Parameters
            ----------
            \*args : \*
                Arguments.
            \*\*kwargs : \*\*
                Keywords arguments.

            Returns
            -------
            object
                Callable object.
            """

            if kwargs:
                key = args, frozenset(kwargs.items())
            else:
                key = args

            if key not in cache:
                cache[key] = callable(*args, **kwargs)

            return cache[key]

        return memoize_wrapper

    return memoize_decorator
