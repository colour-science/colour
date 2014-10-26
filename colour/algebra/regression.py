#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Regression Analysis
===================

Defines various objects to perform statistical regression analysis:

-   :func:`linear_regression`: Implements multiple linear regression.

References
----------
.. [1]  Wikipedia. (n.d.). Regression analysis. Retrieved May 24, 2014, from
        http://en.wikipedia.org/wiki/Regression_analysis
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.algebra import as_array

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2014 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['linear_regression']


def linear_regression(y, x=None, additional_statistics=False):
    """
    Performs the statistics computation about the ideal trend line from given
    data using the *least-squares* method.

    The equation of the line is :math:`y=b+mx` or
    :math:`y=b+m1x1+m1x2+...+mnxn` where the dependent variable :math:`y` value
    is a function of the independent variable :math:`x` values.

    Parameters
    ----------
    y : array_like
        Dependent and already known :math:`y` variable values used to curve
        fit an ideal trend line.
    x : array_like, optional
        Independent :math:`x` variable(s) values corresponding with :math:`y`
        variable.
    additional_statistics : ndarray
        Output additional regression statistics, by default only the :math:`b`
        variable and :math:`m` coefficients are returned.

    Returns
    -------
    ndarray, ({{mn, mn-1, ..., b}, {sum_of_squares_residual}})
        Regression statistics.

    Raises
    ------
    ValueError
        If :math:`y` and :math:`x` variables have incompatible dimensions.

    References
    ----------
    .. [2]  Wikipedia. (n.d.). Simple linear regression. Retrieved May 24,
            2014, from http://en.wikipedia.org/wiki/Simple_linear_regression

    Examples
    --------
    Linear regression with the dependent and already known :math:`y` variable:

    >>> y = np.array([1, 2, 1, 3, 2, 3, 3, 4, 4, 3])
    >>> linear_regression(y)  # doctest: +ELLIPSIS
    array([ 0.2909090...,  1.        ])

    Linear regression with the dependent :math:`y` variable and independent
    :math:`x` variable:

    >>> x1 = np.array([40, 45, 38, 50, 48, 55, 53, 55, 58, 40])
    >>> linear_regression(y, x1)  # doctest: +ELLIPSIS
    array([ 0.1225194..., -3.3054357...])

    Multiple linear regression with the dependent :math:`y` variable and
    multiple independent :math:`x_i` variables:

    >>> x2 = np.array([25, 20, 30, 30, 28, 30, 34, 36, 32, 34])
    >>> linear_regression(y, tuple(zip(x1, x2)))  # doctest: +ELLIPSIS
    array([ 0.0998002...,  0.0876257..., -4.8303807...])

    Multiple linear regression with additional statistics:

    >>> linear_regression(y, tuple(zip(x1, x2)), True)  # doctest: +ELLIPSIS
    (array([ 0.0998002...,  0.0876257..., -4.8303807...]), array([ 2.1376249...]))
    """

    y = as_array(y)

    if x is None:
        x = np.arange(1, len(y) + 1)
    else:
        x = as_array(x)
        if len(x) != len(y):
            raise ValueError(
                '"y" and "x" variables have incompatible dimensions!')

    x = np.vstack([np.array(x).T, np.ones(len(x))]).T
    result = np.linalg.lstsq(x, y)

    if additional_statistics:
        return result[0:2]
    else:
        return result[0]
