#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extrapolation
=============

Defines classes for extrapolating variables:

-   :class:`Extrapolator`: 1-D function extrapolation.
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.utilities import as_numeric, is_numeric

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013 - 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Extrapolator']


class Extrapolator(object):
    """
    Extrapolates the 1-D function of given interpolator.

    The :class:`Extrapolator` class acts as a wrapper around a given *Colour*
    or *scipy* interpolator class instance with compatible signature.
    Two extrapolation methods are available:

    -   *Linear*: Linearly extrapolates given points using the slope defined by
        the interpolator boundaries (xi[0], xi[1]) if x < xi[0] and
        (xi[-1], xi[-2]) if x > xi[-1].
    -   *Constant*: Extrapolates given points by assigning the interpolator
        boundaries values xi[0] if x < xi[0] and xi[-1] if x > xi[-1].

    Specifying the *left* and *right* arguments takes precedence on the chosen
    extrapolation method and will assign the respective *left* and *right*
    values to the given points.

    Parameters
    ----------
    interpolator : object
        Interpolator object.
    method : unicode, optional
        **{'Linear', 'Constant'}**,
        Extrapolation method.
    left : numeric, optional
        Value to return for x < xi[0].
    right : numeric, optional
        Value to return for x > xi[-1].

    Methods
    -------
    __class__

    Notes
    -----
    The interpolator must define *x* and *y* attributes.

    References
    ----------
    .. [1]  sastanin. (n.d.). How to make scipy.interpolate give an
            extrapolated result beyond the input range? Retrieved August 08,
            2014, from http://stackoverflow.com/a/2745496/931625

    Examples
    --------
    Extrapolating a single numeric variable:

    >>> from colour.algebra import LinearInterpolator
    >>> x = np.array([3, 4, 5])
    >>> y = np.array([1, 2, 3])
    >>> interpolator = LinearInterpolator(x, y)
    >>> extrapolator = Extrapolator(interpolator)
    >>> extrapolator(1)
    -1.0

    Extrapolating an *array_like* variable:

    >>> extrapolator(np.array([6, 7 , 8]))
    array([ 4.,  5.,  6.])

    Using the *Constant* extrapolation method:

    >>> x = np.array([3, 4, 5])
    >>> y = np.array([1, 2, 3])
    >>> interpolator = LinearInterpolator(x, y)
    >>> extrapolator = Extrapolator(interpolator, method='Constant')
    >>> extrapolator(np.array([0.1, 0.2, 8, 9]))
    array([ 1.,  1.,  3.,  3.])

    Using defined *left* boundary and *Constant* extrapolation method:

    >>> x = np.array([3, 4, 5])
    >>> y = np.array([1, 2, 3])
    >>> interpolator = LinearInterpolator(x, y)
    >>> extrapolator = Extrapolator(interpolator, method='Constant', left=0)
    >>> extrapolator(np.array([0.1, 0.2, 8, 9]))
    array([ 0.,  0.,  3.,  3.])
    """

    def __init__(self,
                 interpolator=None,
                 method='Linear',
                 left=None,
                 right=None):

        self.__interpolator = None
        self.interpolator = interpolator
        self.__method = None
        self.method = method
        self.__right = None
        self.right = right
        self.__left = None
        self.left = left

    @property
    def interpolator(self):
        """
        Property for **self.__interpolator** private attribute.

        Returns
        -------
        object
            self.__interpolator
        """

        return self.__interpolator

    @interpolator.setter
    def interpolator(self, value):
        """
        Setter for **self.__interpolator** private attribute.

        Parameters
        ----------
        value : object
            Attribute value.
        """

        if value is not None:
            assert hasattr(value, 'x'), (
                '"{0}" interpolator has no "x" attribute!'.format(value))
            assert hasattr(value, 'y'), (
                '"{0}" interpolator has no "y" attribute!'.format(value))

        self.__interpolator = value

    @property
    def method(self):
        """
        Property for **self.__method** private attribute.

        Returns
        -------
        unicode
            self.__method
        """

        return self.__method

    @method.setter
    def method(self, value):
        """
        Setter for **self.__method** private attribute.

        Parameters
        ----------
        value : unicode
            Attribute value.
        """

        if value is not None:
            assert isinstance(value, basestring), (  # noqa
                ('"{0}" attribute: "{1}" is not a '
                 '"basestring" instance!').format('method', value))
            value = value.lower()

        self.__method = value

    @property
    def left(self):
        """
        Property for **self.__left** private attribute.

        Returns
        -------
        numeric
            self.__left
        """

        return self.__left

    @left.setter
    def left(self, value):
        """
        Setter for **self.__left** private attribute.

        Parameters
        ----------
        value : numeric
            Attribute value.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" is not a "numeric"!').format(
                'left', value)
        self.__left = value

    @property
    def right(self):
        """
        Property for **self.__right** private attribute.

        Returns
        -------
        numeric
            self.__right
        """

        return self.__right

    @right.setter
    def right(self, value):
        """
        Setter for **self.__right** private attribute.

        Parameters
        ----------
        value : numeric
            Attribute value.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" is not a "numeric"!').format(
                'right', value)
        self.__right = value

    def __call__(self, x):
        """
        Evaluates the Extrapolator at given point(s).

        Parameters
        ----------
        x : numeric or array_like
            Point(s) to evaluate the Extrapolator at.

        Returns
        -------
        float or ndarray
            Extrapolated points value(s).
        """

        x = np.atleast_1d(x).astype(np.float_)

        xe = as_numeric(self.__evaluate(x))

        return xe

    def __evaluate(self, x):
        """
        Performs the extrapolating evaluation at given points.

        Parameters
        ----------
        x : ndarray
            Points to evaluate the Extrapolator at.

        Returns
        -------
        ndarray
            Extrapolated points values.
        """

        xi = self.__interpolator.x
        yi = self.__interpolator.y

        y = np.empty_like(x)

        if self.__method == 'linear':
            y[x < xi[0]] = (yi[0] + (x[x < xi[0]] - xi[0]) *
                            (yi[1] - yi[0]) / (xi[1] - xi[0]))
            y[x > xi[-1]] = (yi[-1] + (x[x > xi[-1]] - xi[-1]) *
                             (yi[-1] - yi[-2]) / (xi[-1] - xi[-2]))
        elif self.__method == 'constant':
            y[x < xi[0]] = yi[0]
            y[x > xi[-1]] = yi[-1]

        if self.__left is not None:
            y[x < xi[0]] = self.__left
        if self.__right is not None:
            y[x > xi[-1]] = self.__right

        in_range = np.logical_and(x >= xi[0], x <= xi[-1])
        y[in_range] = self.__interpolator(x[in_range])

        return y
