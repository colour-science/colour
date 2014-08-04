# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**common.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package algebra common utilities objects
    that don't fall in any specific category.

**Others:**

"""

from __future__ import unicode_literals

import numpy as np

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["FLOATING_POINT_NUMBER_PATTERN",
           "EVEN_INTEGER_THRESHOLD",
           "get_steps",
           "get_closest",
           "to_ndarray",
           "is_uniform",
           "is_iterable",
           "is_number",
           "is_even_integer"]

FLOATING_POINT_NUMBER_PATTERN = "[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"

EVEN_INTEGER_THRESHOLD = 0.001


def get_steps(distribution):
    """
    Returns the steps of given distribution.

    :param distribution: Distribution to retrieve the steps.
    :type distribution: tuple or list or Array or Matrix
    :return: steps.
    :rtype: tuple
    """

    return tuple(set([distribution[i + 1] - distribution[i] for i in
                      range(len(distribution) - 1)]))


def get_closest(y, x):
    """
    Returns closest *y* variable element to reference *x* variable.

    Usage::

        >>> y = np.array([24.31357115, 63.62396289, 55.71528816, 62.70988028, 46.84480573, 25.40026416])
        >>> get_closest(63, y)
        62.70988028

    :param y: Variable to search for the closest element.
    :type y: array_like
    :param x: Reference variable.
    :type x: int or float
    :return: Closest *y* variable element.
    :rtype: int or float
    """

    return y[(np.abs(np.array(y) - x)).argmin()]


def to_ndarray(x):
    """
    Converts given *x* variable to ndarray.

    Usage::

        >>> to_ndarray(1)
        [1]

    :param x: Variable to convert.
    :type x: object
    :return: *x* variable converted to ndarray.
    :rtype: ndarray
    """

    return np.array(x) if is_iterable(x) else np.array((x,))


def is_uniform(distribution):
    """
    Returns if given distribution is uniform.

    :param distribution: Distribution to check for uniformity.
    :type distribution: tuple or list or Array or Matrix
    :return: Is uniform.
    :rtype: bool
    """

    return True if len(get_steps(distribution)) == 1 else False


def is_iterable(x):
    """
    Returns if given *x* variable is iterable.

    Usage::

        >>> is_iterable([1, 2, 3])
        True
        >>> is_iterable(1)
        False

    :param x: Variable to check the iterability.
    :type x: object
    :return: *x* variable iterability.
    :rtype: bool
    """

    try:
        for _ in x:
            break
        return True
    except TypeError:
        return False


def is_number(x):
    """
    Returns if given *x* variable is a number.

    Usage::

        >>> is_number(1)
        True
        >>> is_number((1,))
        False

    :param x: Variable to check.
    :type x: object
    :return: Is *x* variable a number.
    :rtype: bool
    """

    return isinstance(x, (int, long, float, complex))


def is_even_integer(x):
    """
    Returns if given *x* variable is an even integer.

    Usage::

        >>> is_even_integer(1)
        True
        >>> is_even_integer(1.01)
        False

    :param x: Variable to check.
    :type x: object
    :return: Is *x* variable an even integer.
    :rtype: bool
    """

    return abs(x - round(x)) <= EVEN_INTEGER_THRESHOLD