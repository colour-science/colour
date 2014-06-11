#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**interpolation.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package interpolation helper objects.

**Others:**

"""

from __future__ import unicode_literals

import bisect

import numpy

import color.utilities.common
import color.utilities.exceptions
import color.utilities.verbose


__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "SpragueInterpolator"]

LOGGER = color.utilities.verbose.install_logger()


class SpragueInterpolator(object):
    """
    Constructs a fifth-order polynomial that passes through *y* dependent variable.

    The Sprague (1880) method is recommended by the *CIE* for interpolating functions
    having a uniformly spaced independent variable.

    Reference: http://div1.cie.co.at/?i_ca_id=551&pubid=47, **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, Page 33.

    Usage::

        >>> y = numpy.array([5.9200, 9.3700, 10.8135, 4.5100, 69.5900, 27.8007, 86.0500])
        >>> x = numpy.arange(len(y))
        >>> f = SpragueInterpolator(x, y)
        <__main__.SpragueInterpolator object at 0x101845a90>
        >>> f(0.5)
        7.21850256056
    """

    # http://div1.cie.co.at/?i_ca_id=551&pubid=47, Table V
    sprague_c_coefficients = numpy.array([[884, -1960, 3033, -2648, 1080, -180],
                                          [508, -540, 488, -367, 144, -24],
                                          [-24, 144, -367, 488, -540, 508],
                                          [-180, 1080, -2648, 3033, -1960, 884]])
    """
    Defines the coefficients used to generate extra points for bounds interpolation.
    """

    def __init__(self, x=None, y=None):
        """
        Initializes the class.

        :param x: Independent *x* variable values corresponding with *y* variable.
        :type x: ndarray
        :param y: Dependent and already known *y* variable values to interpolate.
        :type y: ndarray
        """

        # --- Setting class attributes. ---
        self.__xp = None
        self.__yp = None

        self.__x = None
        self.x = x
        self.__y = None
        self.y = y

        self.__validate_dimensions()

    @property
    def x(self):
        """
        Property for **self.__x** attribute.

        :return: self.__x.
        :rtype: ndarray or matrix
        """

        return self.__x

    @x.setter
    def x(self, value):
        """
        Setter for **self.__x** attribute.

        :param value: Attribute value.
        :type value: ndarray or matrix
        """

        if value is not None:
            assert type(value) in (numpy.ndarray, numpy.matrix), \
                "'{0}' attribute: '{1}' type is not 'ndarray' or 'matrix'!".format("x", value)

            assert value.ndim == 1, "'x' independent variable array must have exactly one dimension!"

            assert color.utilities.common.is_uniform(value), "'x' independent variable is not uniform!"

            if not issubclass(value.dtype.type, numpy.inexact):
                value = value.astype(numpy.float_)

            steps = color.utilities.common.get_steps(value)[0]

            xp1 = value[0] - steps * 2
            xp2 = value[0] - steps
            xp3 = value[-1] + steps
            xp4 = value[-1] + steps * 2

            self.__xp = numpy.concatenate(((xp1, xp2), value, (xp3, xp4)))

        self.__x = value

    @x.deleter
    def x(self):
        """
        Deleter for **self.__x** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "x"))

    @property
    def y(self):
        """
        Property for **self.__y** attribute.

        :return: self.__y.
        :rtype: ndarray or matrix
        """

        return self.__y

    @y.setter
    def y(self, value):
        """
        Setter for **self.__y** attribute.

        :param value: Attribute value.
        :type value: ndarray or matrix
        """

        if value is not None:
            assert type(value) in (numpy.ndarray, numpy.matrix), \
                "'{0}' attribute: '{1}' type is not 'ndarray' or 'matrix'!".format("y", value)

            assert value.ndim == 1, "'y' dependent variable array must have exactly one dimension!"

            assert len(value) >= 6, "'y' dependent variable values count must be in domain [6:]!".format(len(value))

            if not issubclass(value.dtype.type, numpy.inexact):
                value = value.astype(numpy.float_)

            yp1 = numpy.ravel((self.sprague_c_coefficients[0] * numpy.matrix(value[0:6]).transpose()) / 209.)[0]
            yp2 = numpy.ravel((self.sprague_c_coefficients[1] * numpy.matrix(value[0:6]).transpose()) / 209.)[0]
            yp3 = numpy.ravel((self.sprague_c_coefficients[2] * numpy.matrix(value[-6:]).transpose()) / 209.)[0]
            yp4 = numpy.ravel((self.sprague_c_coefficients[3] * numpy.matrix(value[-6:]).transpose()) / 209.)[0]

            self.__yp = numpy.concatenate(((yp1, yp2), value, (yp3, yp4)))

        self.__y = value

    @y.deleter
    def y(self):
        """
        Deleter for **self.__y** attribute.
        """

        raise color.utilities.exceptions.ProgrammingError(
            "{0} | '{1}' attribute is not deletable!".format(self.__class__.__name__, "y"))

    def __call__(self, x):
        """
        Evaluates the interpolating polynomial at given point(s).

        :param x: Point(s) to evaluate the interpolant at.
        :type x: float or ndarray
        :return: Interpolated value(s).
        :rtype: float or ndarray
        """

        try:
            return numpy.array(map(self.__evaluate, x))
        except TypeError as error:
            return self.__evaluate(x)

    def __evaluate(self, x):
        """
        Performs the interpolating polynomial evaluation at given point.

        :param x: Point to evaluate the interpolant at.
        :type x: float
        :return: Interpolated value.
        :rtype: float
        """

        self.__validate_dimensions()
        self.__validate_interpolation_range(x)

        if x in self.__x:
            return self.__y[numpy.where(self.__x==x)][0]

        i = bisect.bisect(self.__xp, x) - 1
        X = (x - self.__xp[i]) / (self.__xp[i + 1] - self.__xp[i])

        r = self.__yp

        a0p = r[i]
        a1p = (2. * r[i - 2] - 16. * r[i - 1] + 16. * r[i + 1] - 2. * r[i + 2]) / 24.
        a2p = (-r[i - 2] + 16. * r[i - 1] - 30. * r[i] + 16. * r[i + 1] - r[i + 2]) / 24.
        a3p = (-9. * r[i - 2] + 39. * r[i - 1] - 70. * r[i] + 66. * r[i + 1] - 33. * r[i + 2] + 7 * r[i + 3]) / 24.
        a4p = (13. * r[i - 2] - 64. * r[i - 1] + 126. * r[i] - 124. * r[i + 1] + 61. * r[i + 2] - 12 * r[i + 3]) / 24.
        a5p = (-5. * r[i - 2] + 25. * r[i - 1] - 50. * r[i] + 50. * r[i + 1] - 25. * r[i + 2] + 5 * r[i + 3]) / 24.

        y = a0p + a1p * X + a2p * X ** 2 + a3p * X ** 3 + a4p * X ** 4 + a5p * X ** 5

        return y

    def __validate_dimensions(self):
        """
        Validates variables dimensions to be the same.
        """

        if len(self.__x) != len(self.__y):
            raise color.utilities.exceptions.ProgrammingError(
                "'x' independent and 'y' dependent variables have different dimensions: '{0}', '{1}'".format(
                    len(self.__x), len(self.__y)))

    def __validate_interpolation_range(self, x):
        """
        Validates given point to be in interpolation range.
        """

        below_interpolation_range = x < self.__x[0]
        above_interpolation_range = x > self.__x[-1]

        if below_interpolation_range.any():
            raise ValueError("'{0}' is below interpolation range.".format(x))

        if above_interpolation_range.any():
            raise ValueError("'{0}' is above interpolation range.".format(x))
