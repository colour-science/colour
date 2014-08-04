# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**regression.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package regression helper objects.

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

__all__ = ["linear_regression"]


def linear_regression(y, x=None, additional_statistics=False):
    """
    Performs the statistics calculation about the ideal trend line from given data using the *least-squares* method.
    The equation of the line is *y = b + mx* or *y = b + m1x1 + m1x2 + ... + mnxn* where the dependent *y* value
    is a function of the independent *x* values.

    Usage::

        >>> y = np.array([1,2,1,3,2,3,3,4,4,3])
        >>> x1 = np.array([40, 45, 38, 50, 48, 55, 53, 55, 58, 40])
        >>> x2 = np.array([25, 20, 30, 30, 28, 30, 34, 36, 32, 34])
        >>> linear_regression(y)
        [ 0.29090909  1.        ]
        >>> linear_regression(y, x1)
        [ 0.12251941 -3.30543572]
        >>> linear_regression(y, zip(x1, x2))
        [ 0.09980023  0.08762575 -4.83038079]
        >>> linear_regression(y, zip(x1, x2), additional_statistics=True)
        (array([ 0.09980023,  0.08762575, -4.83038079]), array([ 2.13762499]))

    :param y: Dependent and already known *y* variable values used to curve fit an ideal trend line.
    :type y: ndarray
    :param x: Independent *x* variable(s) values corresponding with *y* variable.
    :type x: ndarray
    :param additional_statistics: Output additional regression statistics, by default only the *b* variable and *m* coefficients are returned.
    :type additional_statistics: ndarray
    :return: Regression statistics.
    :rtype: ndarray ({{mn, mn-1, ..., b}, {sum_of_squares_residual}})

    References:

    -  http://en.wikipedia.org/wiki/Simple_linear_regression (Last accessed 24 May 2014)
    """

    if x is None:
        x = np.arange(1, len(y) + 1)
    else:
        if len(x) != len(y):
            raise ValueError("'y' and 'x' have incompatible dimensions!")

    x = np.vstack([np.array(x).T, np.ones(len(x))]).T
    result = np.linalg.lstsq(x, y)

    if additional_statistics:
        return result[0:2]
    else:
        return result[0]