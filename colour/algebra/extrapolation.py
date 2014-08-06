# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extrapolation
=============

Defines classes for extrapolating variables:

-   :class:`Extrapolator1d`: 1-D function extrapolation.
"""

from __future__ import unicode_literals

import numpy as np

from colour.algebra import is_number, to_ndarray

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["Extrapolator1d"]


class Extrapolator1d(object):
    """
    Extrapolates the 1-D function of given interpolator.

    The Extrapolator1d acts as a wrapper around a given *Colour* or *scipy*
    interpolator class instance with compatible signature. Two extrapolation
    methods are available:

    -   *Linear*: Linearly extrapolates given points using the slope defined by
        the interpolator boundaries (xi[0], xi[1]) if x < xi[0] and
        (xi[-1], xi[-2]) if x > xi[-1].
    -   *Constant*: Extrapolates given points by assigning the interpolator
        boundaries values xi[0] if x < xi[0] and xi[-1] if x > xi[-1].

    Specifying the *left* and *right* arguments takes precedence on the chosen
    extrapolation method and will assign the respective *left* and *right*
    values to the given points.

    Notes
    -----

    The interpolator must define *x* and *y* attributes.

    References
    ----------

    .. [1]  http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range

    Examples
    --------

    Extrapolating a single float variable:

    >>> x = np.array([3, 4, 5])
    >>> y = np.array([1, 2, 3])
    >>> interpolator = colour.LinearInterpolator1d(x, y)
    >>> Extrapolator1d = colour.Extrapolator1d(interpolator)
    >>> Extrapolator1d(1)
    -1

    Extrapolating an *array_like* variable:

    >>> Extrapolator1d(np.array([6, 7 , 8]))
    array([4, 5, 6])

    Using the *Constant* extrapolation method:

    >>> x = np.array([3, 4, 5])
    >>> y = np.array([1, 2, 3])
    >>> interpolator = colour.LinearInterpolator1d(x, y)
    >>> Extrapolator1d = colour.Extrapolator1d(interpolator, method="Constant")
    >>> Extrapolator1d(np.array([0.1, 0.2, 8., 9.]))
    array([ 3.,  3.,  5.,  5.])

    Using defined *left* boundary and *Constant* extrapolation method:

    >>> x = np.array([3, 4, 5])
    >>> y = np.array([1, 2, 3])
    >>> interpolator = colour.LinearInterpolator1d(x, y)
    >>> Extrapolator1d = colour.Extrapolator1d(interpolator, method="Constant", left=0)
    >>> Extrapolator1d(np.array([0.1, 0.2, 8., 9.]))
    array([ 0.,  0.,  5.,  5.])
    """

    def __init__(self,
                 interpolator=None,
                 method="Linear",
                 left=None,
                 right=None):
        """
        Parameters
        ----------

        interpolator : object
            Interpolator object.
        method : unicode, optional
            ("Linear", "Constant")
            Extrapolation method.
        left : int or float, optional
            Value to return for x < xi[0].
        right : int or float, optional
            Value to return for x > xi[-1].
        """

        # --- Setting class attributes. ---
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
            assert hasattr(value, "x"), \
                "'{0}' attribute has no 'x' attribute!".format(
                    "interpolator", value)
            assert hasattr(value, "y"), \
                "'{0}' attribute has no 'y' attribute!".format(
                    "interpolator", value)

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
            assert type(value) in (str, unicode), \
                "'{0}' attribute: '{1}' type is not 'str' or 'unicode'!".format(
                    "method", value)
        self.__method = value

    @property
    def left(self):
        """
        Property for **self.__left** private attribute.

        Returns
        -------

        int or long or float or complex
            self.__left
        """

        return self.__left

    @left.setter
    def left(self, value):
        """
        Setter for **self.__left** private attribute.

        Parameters
        ----------

        value : int or long or float or complex
            Attribute value.
        """

        if value is not None:
            assert is_number(value), "'{0}' attribute: '{1}' type is not \
'int', 'long', 'float' or 'complex'!".format("left", value)
        self.__left = value

    @property
    def right(self):
        """
        Property for **self.__right** private attribute.

        Returns
        -------

        int or long or float or complex
            self.__right
        """

        return self.__right

    @right.setter
    def right(self, value):
        """
        Setter for **self.__right** private attribute.

        Parameters
        ----------

        value : int or long or float or complex
            Attribute value.
        """

        if value is not None:
            assert is_number(value), "'{0}' attribute: '{1}' type is not \
'int', 'long', 'float' or 'complex'!".format("right", value)
        self.__right = value

    def __call__(self, x):
        """
        Evaluates the Extrapolator1d at given point(s).

        Parameters
        ----------

        x : float or array_like
            Point(s) to evaluate the Extrapolator1d at.

        Returns
        -------

        float or ndarray
            Extrapolated points value(s).
        """

        xe = self.__evaluate(to_ndarray(x))

        if is_number(x):
            return type(x)(xe)
        else:
            return xe

    def __evaluate(self, x):
        """
        Performs the extrapolating evaluation at given points.

        Parameters
        ----------

        x : ndarray
            Points to evaluate the Extrapolator1d at.

        Returns
        -------

        ndarray
            Extrapolated points values.
        """

        xi = self.__interpolator.x
        yi = self.__interpolator.y

        y = np.empty_like(x)

        if self.__method == "Linear":
            y[x < xi[0]] = (yi[0] + (x[x < xi[0]] - xi[0]) *
                            (yi[1] - yi[0]) / (xi[1] - xi[0]))
            y[x > xi[-1]] = (yi[-1] + (x[x > xi[-1]] - xi[-1]) *
                             (yi[-1] - yi[-2]) / (xi[-1] - xi[-2]))
        elif self.__method == "Constant":
            y[x < xi[0]] = xi[0]
            y[x > xi[-1]] = xi[-1]

        if self.__left is not None:
            y[x < xi[0]] = self.__left
        if self.__right is not None:
            y[x > xi[-1]] = self.__right

        in_range = np.logical_and(x >= xi[0], x <= xi[-1])
        y[in_range] = self.__interpolator(x[in_range])

        return y