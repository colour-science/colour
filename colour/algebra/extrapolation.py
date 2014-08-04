# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**extrapolation.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package extrapolation helper objects.

**Others:**

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
    Extrapolates a 1-D function for given interpolator.

    References:

    -  http://stackoverflow.com/questions/2745329/how-to-make-scipy-interpolate-give-an-extrapolated-result-beyond-the-input-range
    """

    def __init__(self, interpolator=None, kind="linear"):
        """
        Initialises the class.

        :param interpolator: Interpolator object.
        :type interpolator: object
        """

        # --- Setting class attributes. ---
        self.__interpolator = None
        self.interpolator = interpolator
        self.__kind = None
        self.kind = kind


    @property
    def interpolator(self):
        """
        Property for **self.__interpolator** attribute.

        :return: self.__interpolator.
        :rtype: object
        """

        return self.__interpolator

    @interpolator.setter
    def interpolator(self, value):
        """
        Setter for **self.__interpolator** attribute.

        :param value: Attribute value.
        :type value: object
        """

        if value is not None:
            assert hasattr(value, "x"), \
                "'{0}' attribute has no 'x' attribute!".format("interpolator",
                                                               value)
            assert hasattr(value, "y"), \
                "'{0}' attribute has no 'y' attribute!".format("interpolator",
                                                               value)

        self.__interpolator = value

    @property
    def kind(self):
        """
        Property for **self.__kind** attribute.

        :return: self.__kind.
        :rtype: dict
        """

        return self.__kind

    @kind.setter
    def kind(self, value):
        """
        Setter for **self.__kind** attribute.

        :param value: Attribute value.
        :type value: dict
        """

        if value is not None:
            assert type(value) in (str, unicode), \
                "'{0}' attribute: '{1}' type is not 'str' or 'unicode'!".format(
                    "kind", value)
        self.__kind = value

    def __call__(self, x):
        """
        Evaluates the extrapolator at given point(s).

        :param x: Point(s) to evaluate the extrapolator at.
        :type x: float or array_like
        :return: Extrapolated value(s).
        :rtype: float or ndarray
        """

        xe = self.__evaluate(to_ndarray(x))

        if is_number(x):
            return type(x)(xe)
        else:
            return xe

    def __evaluate(self, x):
        """
        Performs the extrapolating evaluation at given points.

        :param x: Points to evaluate the extrapolator at.
        :type x: ndarray
        :return: Extrapolated value.
        :rtype: ndarray
        """

        xi = self.__interpolator.x
        yi = self.__interpolator.y

        y = np.empty_like(x)

        y[x < xi[0]] = yi[0] + (x[x < xi[0]] - xi[0]) * \
                               (yi[1] - yi[0]) / (xi[1] - xi[0])
        y[x > xi[-1]] = yi[-1] + (x[x > xi[-1]] - xi[-1]) * \
                                 (yi[-1] - yi[-2]) / (xi[-1] - xi[-2])

        in_range = np.logical_and(x >= xi[0], x <= xi[-1])
        y[in_range] = self.__interpolator(x[in_range])

        return y