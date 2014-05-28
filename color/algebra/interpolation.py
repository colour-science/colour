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


def sprague_interpolation(points, samples=10):
    """
    Constructs a fifth-order polynomial that passes through given set of points with given samples count
    and evaluates the polynomial to provide the interpolated points.
    The Sprague (1880) method is recommended by the *CIE* for interpolating functions
    having a uniformly spaced independent variable.

    Reference: http://div1.cie.co.at/?i_ca_id=551&pubid=47, **Stephen Westland, Caterina Ripamonti, Vien Cheung**, *Computational Colour Science Using MATLAB, 2nd Edition*, Page 33.

    Usage::

        >>> points = [5.9200, 9.3700, 10.8135, 4.5100, 69.5900, 27.8007, 86.0500]
        >>> sprague_interpolation(points, samples=2)
        [  5.92         7.21850256   9.37        12.23568833  10.8135       1.66284023
           4.51        40.76525391  69.59        51.84316346  27.8007      43.77379185
          86.05      ]

    :param points: Points to interpolate.
    :type points: Array
    :param samples: Number of sample to generate in-between points.
    :type samples: int
    :return: Interpolated points.
    :rtype: Array
    """

    if len(points) < 6:
        raise color.exceptions.ProgrammingError(
            "Sprague interpolation needs at least '6' points, current points count: '{0}'.".format(len(points)))

    if samples < 2 or samples - math.floor(samples) > 0:
        raise color.exceptions.ProgrammingError("Invalid samples value '{0}', must be in domain [2:].".format(samples))

    p1 = numpy.ravel((SPRAGUE_C_COEFFICIENTS[0] * numpy.matrix(points[0:6]).transpose()) / 209.)[0]
    p2 = numpy.ravel((SPRAGUE_C_COEFFICIENTS[1] * numpy.matrix(points[0:6]).transpose()) / 209.)[0]
    p3 = numpy.ravel((SPRAGUE_C_COEFFICIENTS[2] * numpy.matrix(points[-6:]).transpose()) / 209.)[0]
    p4 = numpy.ravel((SPRAGUE_C_COEFFICIENTS[3] * numpy.matrix(points[-6:]).transpose()) / 209.)[0]

    p = numpy.concatenate(((p1, p2), points, (p3, p4)))
    x = numpy.linspace(1. / samples, 1. - 1. / samples, samples - 1.)
    interpolated_points = numpy.array([])

    for i in range(2, len(points) + 1):
        a0 = p[i]
        a1 = (2. * p[i - 2] - 16. * p[i - 1] + 16. * p[i + 1] - 2. * p[i + 2]) / 24.
        a2 = (-p[i - 2] + 16. * p[i - 1] - 30. * p[i] + 16. * p[i + 1] - p[i + 2]) / 24.
        a3 = (-9. * p[i - 2] + 39. * p[i - 1] - 70. * p[i] + 66. * p[i + 1] - 33. * p[i + 2] + 7 * p[i + 3]) / 24.
        a4 = (13. * p[i - 2] - 64. * p[i - 1] + 126. * p[i] - 124. * p[i + 1] + 61. * p[i + 2] - 12 * p[i + 3]) / 24.
        a5 = (-5. * p[i - 2] + 25. * p[i - 1] - 50. * p[i] + 50. * p[i + 1] - 25. * p[i + 2] + 5 * p[i + 3]) / 24.

        y = a0 + a1 * x + a2 * x ** 2 + a3 * x ** 3 + a4 * x ** 4 + a5 * x ** 5

        interpolated_points = numpy.append(interpolated_points, p[i])
        interpolated_points = numpy.append(interpolated_points, y)

    return numpy.append(interpolated_points, p[-3])