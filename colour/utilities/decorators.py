# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**decorators.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package generic decorators objects.

**Others:**

"""

from __future__ import unicode_literals

import functools

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2008 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["memoize"]


def memoize(cache=None):
    """
    | Implements method / definition memoization.
    | Any method / definition decorated will get its return value cached and
    restored whenever called with the same arguments.

    :param cache: Alternate cache.
    :type cache: dict
    :return: Object.
    :rtype: object

    References:

    -  https://github.com/KelSolaar/Foundations/blob/develop/foundations/decorators.py
    """

    if cache is None:
        cache = {}

    def memoize_decorator(object):
        """
        Implements method / definition memoization.

        :param object: Object to decorate.
        :type object: object
        :return: Object.
        :rtype: object
        """

        @functools.wraps(object)
        def memoize_wrapper(*args, **kwargs):
            """
            Implements method / definition memoization.

            :param \*args: Arguments.
            :type \*args: \*
            :param \*\*kwargs: Keywords arguments.
            :type \*\*kwargs: \*\*
            :return: Object.
            :rtype: object
            """

            if kwargs:
                key = args, frozenset(kwargs.iteritems())
            else:
                key = args

            if key not in cache:
                cache[key] = object(*args, **kwargs)

            return cache[key]

        return memoize_wrapper

    return memoize_decorator
