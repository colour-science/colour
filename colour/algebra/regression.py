# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression Analysis
===================

Defines various objects to perform statistical regression analysis:

-   :func:`linear_regression`: Implements multiple linear regression.

References
----------

.. [1]  http://en.wikipedia.org/wiki/Regression_analysis
"""

from __future__ import unicode_literals

import numpy as np

from colour.algebra import to_ndarray

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["linear_regression"]


def linear_regression(y, x=None, additional_statistics=False):
    """
    Performs the statistics calculation about the ideal trend line from given
    data using the *least-squares* method.

    The equation of the line is :math:`y=b+mx` or
    :math:`y=b+m1x1+m1x2+...+mnxn` where the dependent variable :math:`y` value
    is a function of the independent variable :math:`x` values.

    Parameters
    ----------

    y : array_like
        Dependent and already known :math:`y` variable values used to curve
        fit an ideal trend line.
    x : array_like
        Independent :math:`x` variable(s) values corresponding with :math:`y`
        variable.
    additional_statistics : ndarray
        Output additional regression statistics, by default only the :math:`b`
        variable and :math:`m` coefficients are returned.

    Returns
    -------

    ndarray ({{mn, mn-1, ..., b}, {sum_of_squares_residual}})
        Regression statistics.

    References
    ----------

    .. [2]  http://en.wikipedia.org/wiki/Simple_linear_regression
            (Last accessed 24 May 2014)

    Examples
    --------

    Linear regression with the dependent and already known :math:`y` variable:

    >>> y = np.array([1,2,1,3,2,3,3,4,4,3])
    >>> colour.linear_regression(y)
    [ 0.29090909  1.        ]

    Linear regression with the dependent :math:`y` variable and independent
    :math:`x` variable:

    >>> x1 = np.array([40, 45, 38, 50, 48, 55, 53, 55, 58, 40])
    >>> colour.linear_regression(y, x1)
    [ 0.12251941 -3.30543572]

    Multiple linear regression with the dependent :math:`y` variable and
    multiple independent :math:`x_i` variables:

    >>> x2 = np.array([25, 20, 30, 30, 28, 30, 34, 36, 32, 34])
    >>> colour.linear_regression(y, zip(x1, x2))
    [ 0.09980023  0.08762575 -4.83038079]

    Multiple linear regression with additional statistics:

    >>> colour.linear_regression(y, zip(x1, x2), additional_statistics=True)
    (array([ 0.09980023,  0.08762575, -4.83038079]), array([ 2.13762499]))
    """

    y = to_ndarray(y)

    if x is None:
        x = np.arange(1, len(y) + 1)
    else:
        x = to_ndarray(x)
        if len(x) != len(y):
            raise ValueError(
                "'y' and 'x' variables have incompatible dimensions!")

    x = np.vstack([np.array(x).T, np.ones(len(x))]).T
    result = np.linalg.lstsq(x, y)

    if additional_statistics:
        return result[0:2]
    else:
        return result[0]