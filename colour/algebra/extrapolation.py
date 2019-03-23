# -*- coding: utf-8 -*-
"""
Extrapolation
=============

Defines classes for extrapolating variables:

-   :class:`colour.Extrapolator`: 1-D function extrapolation.

References
----------
-   :cite:`Sastanina` : sastanin. (n.d.). How to make scipy.interpolate give
    an extrapolated result beyond the input range? Retrieved August 8, 2014,
    from http://stackoverflow.com/a/2745496/931625
-   :cite:`Westland2012i` : Westland, S., Ripamonti, C., & Cheung, V. (2012).
    Extrapolation Methods. In Computational Colour Science Using MATLAB
    (2nd ed., p. 38). ISBN:978-0-470-66569-5
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.constants import DEFAULT_FLOAT_DTYPE
from colour.utilities import as_float, is_numeric, is_string

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2019 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['Extrapolator']


class Extrapolator(object):
    """
    Extrapolates the 1-D function of given interpolator.

    The :class:`colour.Extrapolator` class acts as a wrapper around a given
    *Colour* or *scipy* interpolator class instance with compatible signature.
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
    dtype : type
        Data type used for internal conversions.

    Methods
    -------
    __class__

    Notes
    -----
    -   The interpolator must define *x* and *y* attributes.

    References
    ----------
    :cite:`Sastanina`, :cite:`Westland2012i`

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
                 right=None,
                 dtype=DEFAULT_FLOAT_DTYPE):

        self._interpolator = None
        self.interpolator = interpolator
        self._method = None
        self.method = method
        self._right = None
        self.right = right
        self._left = None
        self.left = left

        self._dtype = dtype

    @property
    def interpolator(self):
        """
        Getter and setter property for the *Colour* or *scipy* interpolator
        class instance.

        Parameters
        ----------
        value : callable
            Value to set the *Colour* or *scipy* interpolator class instance
            with.

        Returns
        -------
        callable
            *Colour* or *scipy* interpolator class instance.
        """

        return self._interpolator

    @interpolator.setter
    def interpolator(self, value):
        """
        Setter for the **self.interpolator** property.
        """

        if value is not None:
            assert hasattr(value, 'x'), (
                '"{0}" interpolator has no "x" attribute!'.format(value))
            assert hasattr(value, 'y'), (
                '"{0}" interpolator has no "y" attribute!'.format(value))

        self._interpolator = value

    @property
    def method(self):
        """
        Getter and setter property for the extrapolation method.

        Parameters
        ----------
        value : unicode
            Value to set the extrapolation method. with.

        Returns
        -------
        unicode
            Extrapolation method.
        """

        return self._method

    @method.setter
    def method(self, value):
        """
        Setter for the **self.method** property.
        """

        if value is not None:
            assert is_string(value), (
                ('"{0}" attribute: "{1}" is not a "string" like object!'
                 ).format('method', value))
            value = value.lower()

        self._method = value

    @property
    def left(self):
        """
        Getter and setter property for left value to return for x < xi[0].

        Parameters
        ----------
        value : numeric
            Left value to return for x < xi[0].

        Returns
        -------
        numeric
            Left value to return for x < xi[0].
        """

        return self._left

    @left.setter
    def left(self, value):
        """
        Setter for the **self.left** property.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" is not a "numeric"!').format(
                    'left', value)
        self._left = value

    @property
    def right(self):
        """
        Getter and setter property for right value to return for x > xi[-1].

        Parameters
        ----------
        value : numeric
            Right value to return for x > xi[-1].

        Returns
        -------
        numeric
            Right value to return for x > xi[-1].
        """

        return self._right

    @right.setter
    def right(self, value):
        """
        Setter for the **self.right** property.
        """

        if value is not None:
            assert is_numeric(value), (
                '"{0}" attribute: "{1}" is not a "numeric"!').format(
                    'right', value)
        self._right = value

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

        x = np.atleast_1d(x).astype(self._dtype)

        xe = as_float(self._evaluate(x))

        return xe

    def _evaluate(self, x):
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

        xi = self._interpolator.x
        yi = self._interpolator.y

        y = np.empty_like(x)

        if self._method == 'linear':
            y[x < xi[0]] = (yi[0] + (x[x < xi[0]] - xi[0]) * (yi[1] - yi[0]) /
                            (xi[1] - xi[0]))
            y[x > xi[-1]] = (yi[-1] + (x[x > xi[-1]] - xi[-1]) *
                             (yi[-1] - yi[-2]) / (xi[-1] - xi[-2]))
        elif self._method == 'constant':
            y[x < xi[0]] = yi[0]
            y[x > xi[-1]] = yi[-1]

        if self._left is not None:
            y[x < xi[0]] = self._left
        if self._right is not None:
            y[x > xi[-1]] = self._right

        in_range = np.logical_and(x >= xi[0], x <= xi[-1])
        y[in_range] = self._interpolator(x[in_range])

        return y
