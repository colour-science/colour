#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Interpolation
=============

Defines classes for interpolating variables.

-   :class:`LinearInterpolator`: 1-D function linear interpolation.
-   :class:`SpragueInterpolator`: 1-D function fifth-order polynomial
    interpolation using *Sprague (1880)* method.
-   :class:`CubicSplineInterpolator`: 1-D function cubic spline interpolation.
-   :class:`PchipInterpolator`: 1-D function piecewise cube Hermite
    interpolation.
-   :func:`lagrange_coefficients`: Computation of *Lagrange Coefficients*.
"""

from __future__ import division, unicode_literals

import numpy as np
import scipy.interpolate
from six.moves import reduce

from colour.utilities import as_numeric, interval

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = ['LinearInterpolator',
           'SpragueInterpolator',
           'CubicSplineInterpolator',
           'PchipInterpolator',
           'lagrange_coefficients']


class LinearInterpolator(object):
    """
    Linearly interpolates a 1-D function.

    Parameters
    ----------
    x : ndarray
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y : ndarray
        Dependent and already known :math:`y` variable values to
        interpolate.

    Methods
    -------
    __call__

    Notes
    -----
    This class is a wrapper around *numpy.interp* definition.

    See Also
    --------
    SpragueInterpolator

    Examples
    --------
    Interpolating a single numeric variable:

    >>> y = np.array([5.9200,
    ...               9.3700,
    ...               10.8135,
    ...               4.5100,
    ...               69.5900,
    ...               27.8007,
    ...               86.0500])
    >>> x = np.arange(len(y))
    >>> f = LinearInterpolator(x, y)
    >>> # Doctests ellipsis for Python 2.x compatibility.
    >>> f(0.5)  # doctest: +ELLIPSIS
    7.64...

    Interpolating an *array_like* variable:

    >>> f([0.25, 0.75])
    array([ 6.7825,  8.5075])
    """

    def __init__(self, x=None, y=None):
        self.__x = None
        self.x = x
        self.__y = None
        self.y = y

        self._validate_dimensions()

    @property
    def x(self):
        """
        Property for **self.__x** private attribute.

        Returns
        -------
        array_like
            self.__x
        """

        return self.__x

    @x.setter
    def x(self, value):
        """
        Setter for **self.__x** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(np.float_)

            assert value.ndim == 1, (
                '"x" independent variable must have exactly one dimension!')

        self.__x = value

    @property
    def y(self):
        """
        Property for **self.__y** private attribute.

        Returns
        -------
        array_like
            self.__y
        """

        return self.__y

    @y.setter
    def y(self, value):
        """
        Setter for **self.__y** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(np.float_)

            assert value.ndim == 1, (
                '"y" dependent variable must have exactly one dimension!')

        self.__y = value

    def __call__(self, x):
        """
        Evaluates the interpolating polynomial at given point(s).


        Parameters
        ----------
        x : numeric or array_like
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        float or ndarray
            Interpolated value(s).
        """

        x = np.atleast_1d(x).astype(np.float_)

        xi = as_numeric(self._evaluate(x))

        return xi

    def _evaluate(self, x):
        """
        Performs the interpolating polynomial evaluation at given points.

        Parameters
        ----------
        x : ndarray
            Points to evaluate the interpolant at.

        Returns
        -------
        ndarray
            Interpolated points values.
        """

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        return np.interp(x, self.__x, self.__y)

    def _validate_dimensions(self):
        """
        Validates variables dimensions to be the same.
        """

        if len(self.__x) != len(self.__y):
            raise ValueError(
                ('"x" independent and "y" dependent variables have different '
                 'dimensions: "{0}", "{1}"').format(len(self.__x),
                                                    len(self.__y)))

    def _validate_interpolation_range(self, x):
        """
        Validates given point to be in interpolation range.
        """

        below_interpolation_range = x < self.__x[0]
        above_interpolation_range = x > self.__x[-1]

        if below_interpolation_range.any():
            raise ValueError('"{0}" is below interpolation range.'.format(x))

        if above_interpolation_range.any():
            raise ValueError('"{0}" is above interpolation range.'.format(x))


class SpragueInterpolator(object):
    """
    Constructs a fifth-order polynomial that passes through :math:`y` dependent
    variable.

    *Sprague (1880)* method is recommended by the *CIE* for interpolating
    functions having a uniformly spaced independent variable.

    Parameters
    ----------
    x : array_like
        Independent :math:`x` variable values corresponding with :math:`y`
        variable.
    y : array_like
        Dependent and already known :math:`y` variable values to
        interpolate.

    Methods
    -------
    __call__

    See Also
    --------
    LinearInterpolator

    Notes
    -----
    The minimum number :math:`k` of data points required along the
    interpolation axis is :math:`k=6`.

    References
    ----------
    .. [1]  CIE TC 1-38. (2005). 9.2.4 Method of interpolation for uniformly
            spaced independent variable. In CIE 167:2005 Recommended Practice
            for Tabulating Spectral Data for Use in Colour Computations
            (pp. 1–27). ISBN:978-3-901-90641-1
    .. [2]  Westland, S., Ripamonti, C., & Cheung, V. (2012). Interpolation
            Methods. In Computational Colour Science Using MATLAB
            (2nd ed., pp. 29–37). ISBN:978-0-470-66569-5

    Examples
    --------
    Interpolating a single numeric variable:

    >>> y = np.array([5.9200,
    ...               9.3700,
    ...               10.8135,
    ...               4.5100,
    ...               69.5900,
    ...               27.8007,
    ...               86.0500])
    >>> x = np.arange(len(y))
    >>> f = SpragueInterpolator(x, y)
    >>> f(0.5)  # doctest: +ELLIPSIS
    7.2185025...

    Interpolating an *array_like* variable:

    >>> f([0.25, 0.75])  # doctest: +ELLIPSIS
    array([ 6.7295161...,  7.8140625...])
    """

    SPRAGUE_C_COEFFICIENTS = np.array(
        [[884, -1960, 3033, -2648, 1080, -180],
         [508, -540, 488, -367, 144, -24],
         [-24, 144, -367, 488, -540, 508],
         [-180, 1080, -2648, 3033, -1960, 884]])
    """
    Defines the coefficients used to generate extra points for boundaries
    interpolation.

    SPRAGUE_C_COEFFICIENTS : array_like, (4, 6)

    References
    ----------
    .. [3]  CIE TC 1-38. (2005). Table V. Values of the c-coefficients of
            Equ.s 6 and 7. In CIE 167:2005 Recommended Practice for Tabulating
            Spectral Data for Use in Colour Computations (p. 19).
            ISBN:978-3-901-90641-1
    """

    def __init__(self, x=None, y=None):
        self._xp = None
        self._yp = None

        self.__x = None
        self.x = x
        self.__y = None
        self.y = y

        self._validate_dimensions()

    @property
    def x(self):
        """
        Property for **self.__x** private attribute.

        Returns
        -------
        array_like
            self.__x
        """

        return self.__x

    @x.setter
    def x(self, value):
        """
        Setter for **self.__x** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(np.float_)

            assert value.ndim == 1, (
                '"x" independent variable must have exactly one dimension!')

            value_interval = interval(value)[0]

            xp1 = value[0] - value_interval * 2
            xp2 = value[0] - value_interval
            xp3 = value[-1] + value_interval
            xp4 = value[-1] + value_interval * 2

            self._xp = np.concatenate(((xp1, xp2), value, (xp3, xp4)))

        self.__x = value

    @property
    def y(self):
        """
        Property for **self.__y** private attribute.

        Returns
        -------
        array_like
            self.__y
        """

        return self.__y

    @y.setter
    def y(self, value):
        """
        Setter for **self.__y** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        if value is not None:
            value = np.atleast_1d(value).astype(np.float_)

            assert value.ndim == 1, (
                '"y" dependent variable must have exactly one dimension!')

            assert len(value) >= 6, (
                '"y" dependent variable values count must be in domain [6:]!')

            yp1 = np.ravel((np.dot(
                self.SPRAGUE_C_COEFFICIENTS[0],
                np.array(value[0:6]).reshape((6, 1)))) / 209)[0]
            yp2 = np.ravel((np.dot(
                self.SPRAGUE_C_COEFFICIENTS[1],
                np.array(value[0:6]).reshape((6, 1)))) / 209)[0]
            yp3 = np.ravel((np.dot(
                self.SPRAGUE_C_COEFFICIENTS[2],
                np.array(value[-6:]).reshape((6, 1)))) / 209)[0]
            yp4 = np.ravel((np.dot(
                self.SPRAGUE_C_COEFFICIENTS[3],
                np.array(value[-6:]).reshape((6, 1)))) / 209)[0]

            self._yp = np.concatenate(((yp1, yp2), value, (yp3, yp4)))

        self.__y = value

    def __call__(self, x):
        """
        Evaluates the interpolating polynomial at given point(s).

        Parameters
        ----------
        x : numeric or array_like
            Point(s) to evaluate the interpolant at.

        Returns
        -------
        numeric or ndarray
            Interpolated value(s).
        """

        return self._evaluate(x)

    def _evaluate(self, x):
        """
        Performs the interpolating polynomial evaluation at given point.

        Parameters
        ----------
        x : numeric
            Point to evaluate the interpolant at.

        Returns
        -------
        float
            Interpolated point values.
        """

        x = np.asarray(x)

        self._validate_dimensions()
        self._validate_interpolation_range(x)

        i = np.searchsorted(self._xp, x) - 1
        X = (x - self._xp[i]) / (self._xp[i + 1] - self._xp[i])

        r = self._yp

        a0p = r[i]
        a1p = ((2 * r[i - 2] - 16 * r[i - 1] + 16 * r[i + 1] - 2 *
                r[i + 2]) / 24)
        a2p = ((-r[i - 2] + 16 * r[i - 1] - 30 * r[i] + 16 * r[i + 1] -
                r[i + 2]) / 24)
        a3p = ((-9 * r[i - 2] + 39 * r[i - 1] - 70 * r[i] + 66 *
                r[i + 1] - 33 * r[i + 2] + 7 * r[i + 3]) / 24)
        a4p = ((13 * r[i - 2] - 64 * r[i - 1] + 126 * r[i] - 124 *
                r[i + 1] + 61 * r[i + 2] - 12 * r[i + 3]) / 24)
        a5p = ((-5 * r[i - 2] + 25 * r[i - 1] - 50 * r[i] + 50 *
                r[i + 1] - 25 * r[i + 2] + 5 * r[i + 3]) / 24)

        y = (a0p + a1p * X + a2p * X ** 2 + a3p * X ** 3 + a4p * X ** 4 +
             a5p * X ** 5)

        return y

    def _validate_dimensions(self):
        """
        Validates variables dimensions to be the same.
        """

        if len(self.__x) != len(self.__y):
            raise ValueError(
                ('"x" independent and "y" dependent variables have different '
                 'dimensions: "{0}", "{1}"').format(len(self.__x),
                                                    len(self.__y)))

    def _validate_interpolation_range(self, x):
        """
        Validates given point to be in interpolation range.
        """

        below_interpolation_range = x < self.__x[0]
        above_interpolation_range = x > self.__x[-1]

        if below_interpolation_range.any():
            raise ValueError('"{0}" is below interpolation range.'.format(x))

        if above_interpolation_range.any():
            raise ValueError('"{0}" is above interpolation range.'.format(x))


class CubicSplineInterpolator(scipy.interpolate.interp1d):
    """
    Interpolates a 1-D function using cubic spline interpolation.

    Notes
    -----
    This class is a wrapper around *scipy.interpolate.interp1d* class.
    """

    def __init__(self, *args, **kwargs):
        super(CubicSplineInterpolator, self).__init__(
            kind='cubic', *args, **kwargs)


class PchipInterpolator(scipy.interpolate.PchipInterpolator):
    """
    Interpolates a 1-D function using Piecewise Cubic Hermite Interpolating
    Polynomial interpolation.

    Notes
    -----
    This class is a wrapper around *scipy.interpolate.PchipInterpolator*
    class.
    """

    def __init__(self, x=None, y=None, *args, **kwargs):
        super(PchipInterpolator, self).__init__(x, y, *args, **kwargs)

        self.__y = y

    @property
    def y(self):
        """
        Property for **self.__y** private attribute.

        Returns
        -------
        array_like
            self.__y
        """

        return self.__y

    @y.setter
    def y(self, value):
        """
        Setter for **self.__y** private attribute.

        Parameters
        ----------
        value : array_like
            Attribute value.
        """

        raise AttributeError('"{0}" attribute is read only!'.format('y'))


def lagrange_coefficients(r, n=4):
    """
    Computes the *Lagrange Coefficients* at given point :math:`r` for degree
    :math:`n`.

    Parameters
    ----------
    r : numeric
        Point to get the *Lagrange Coefficients* at.
    n : int, optional
        Degree of the *Lagrange Coefficients* being calculated.

    Returns
    -------
    ndarray

    References
    ----------
    .. [4]  Fairman, H. S. (1985). The calculation of weight factors for
            tristimulus integration. Color Research & Application, 10(4),
            199–203. doi:10.1002/col.5080100407
    .. [5]  Wikipedia. (n.d.). Lagrange polynomial - Definition. Retrieved
            January 20, 2016, from
            https://en.wikipedia.org/wiki/Lagrange_polynomial#Definition

    Examples
    --------
    >>> lagrange_coefficients(0.1)
    array([ 0.8265,  0.2755, -0.1305,  0.0285])
    """

    r_i = np.arange(n)
    L_n = []
    for j in range(len(r_i)):
        basis = [(r - r_i[i]) / (r_i[j] - r_i[i])
                 for i in range(len(r_i)) if i != j]
        L_n.append(reduce(lambda x, y: x * y, basis))  # noqa

    return np.array(L_n)
