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

import math
import numpy

import color.exceptions
import color.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER",
           "SPRAGUE_C_COEFFICIENTS",
           "sprague_interpolation"]

LOGGER = color.verbose.install_logger()

# http://div1.cie.co.at/?i_ca_id=551&pubid=47, Table V
SPRAGUE_C_COEFFICIENTS = numpy.array([[884, -1960, 3033, -2648, 1080, -180],
                                      [508, -540, 488, -367, 144, -24],
                                      [-24, 144, -367, 488, -540, 508],
                                      [-180, 1080, -2648, 3033, -1960, 884]])


def sprague_interpolation(y, sampling_rate=10):
    """
    Constructs a fifth-order polynomial that passes through *y* dependent variable with given sampling rate and evaluates
    the polynomial to return the interpolated *y* values.
    The Sprague (1880) method is recommended by the *CIE* for interpolating functions
    having a uniformly spaced independent variable.

    Reference: http://div1.cie.co.at/?i_ca_id=551&pubid=47, **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, Page 33.

    Usage::

        >>> y = [5.9200, 9.3700, 10.8135, 4.5100, 69.5900, 27.8007, 86.0500]
        >>> sprague_interpolation(y, sampling_rate=2)
        [  5.92         7.21850256   9.37        12.23568833  10.8135       1.66284023
           4.51        40.76525391  69.59        51.84316346  27.8007      43.77379185
          86.05      ]

    :param y: Dependent and already known *y* variable values to interpolate.
    :type y: Array
    :param sampling_rate: Sampling rate, sampling_rate=2 doubles *y* sampling rate, sampling_rate=3 triples *y* sampling rate, and so on.
    :type sampling_rate: int
    :return: Interpolated *y* variable.
    :rtype: Array
    """

    if len(y) < 6:
        raise color.exceptions.ProgrammingError(
            "Dependent variable values count must be in domain [6:], current values count: '{0}'.".format(len(y)))

    if sampling_rate < 2 or sampling_rate - math.floor(sampling_rate) > 0:
        raise color.exceptions.ProgrammingError(
            "Invalid sampling_rate value '{0}', must be in domain [2:].".format(sampling_rate))

    p1 = numpy.ravel((SPRAGUE_C_COEFFICIENTS[0] * numpy.matrix(y[0:6]).transpose()) / 209.)[0]
    p2 = numpy.ravel((SPRAGUE_C_COEFFICIENTS[1] * numpy.matrix(y[0:6]).transpose()) / 209.)[0]
    p3 = numpy.ravel((SPRAGUE_C_COEFFICIENTS[2] * numpy.matrix(y[-6:]).transpose()) / 209.)[0]
    p4 = numpy.ravel((SPRAGUE_C_COEFFICIENTS[3] * numpy.matrix(y[-6:]).transpose()) / 209.)[0]

    yp = numpy.concatenate(((p1, p2), y, (p3, p4)))
    X = numpy.linspace(1. / sampling_rate, 1. - 1. / sampling_rate, sampling_rate - 1.)
    interpolated_points = numpy.array([])

    for i in range(2, len(y) + 1):
        a0p = yp[i]
        a1p = (2. * yp[i - 2] - 16. * yp[i - 1] + 16. * yp[i + 1] - 2. * yp[i + 2]) / 24.
        a2p = (-yp[i - 2] + 16. * yp[i - 1] - 30. * yp[i] + 16. * yp[i + 1] - yp[i + 2]) / 24.
        a3p = (-9. * yp[i - 2] + 39. * yp[i - 1] - 70. * yp[i] + 66. * yp[i + 1] - 33. * yp[i + 2] + 7 * yp[i + 3]) / 24.
        a4p = (13. * yp[i - 2] - 64. * yp[i - 1] + 126. * yp[i] - 124. * yp[i + 1] + 61. * yp[i + 2] - 12 * yp[i + 3]) / 24.
        a5p = (-5. * yp[i - 2] + 25. * yp[i - 1] - 50. * yp[i] + 50. * yp[i + 1] - 25. * yp[i + 2] + 5 * yp[i + 3]) / 24.

        y = a0p + a1p * X + a2p * X ** 2 + a3p * X ** 3 + a4p * X ** 4 + a5p * X ** 5

        interpolated_points = numpy.append(interpolated_points, yp[i])
        interpolated_points = numpy.append(interpolated_points, y)

    return numpy.append(interpolated_points, yp[-3])