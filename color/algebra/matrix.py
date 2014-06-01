#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**matrix.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Color** package matrix helper objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import color.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["LOGGER", "is_identity", "linear_interpolate_matrices"]

LOGGER = color.verbose.install_logger()


def is_identity(matrix, n=3):
    """
    Returns if given matrix is an identity matrix.

    Usage::

        >>> is_identity(numpy.matrix([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3))
        True
        >>> is_identity(numpy.matrix([1, 2, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3))
        False

    :param matrix: Matrix.
    :type matrix: matrix (N)
    :param n: Matrix dimension.
    :type n: int
    :return: Is identity matrix.
    :rtype: bool
    """

    return numpy.array_equal(numpy.identity(n), matrix)


def linear_interpolate_matrices(a, b, matrix_1, matrix_2, c):
    """
    Interpolates linearly given matrices and given base values using given interpolation value.

    Usage::

        >>> a = 2850
        >>> b = 7500
        >>> matrix_1 = numpy.matrix([0.5309, -0.0229, -0.0336, -0.6241, 1.3265, 0.3337, -0.0817, 0.1215, 0.6664]).reshape((3, 3))
        >>> matrix_2 = numpy.matrix([0.4716, 0.0603, -0.083, -0.7798, 1.5474, 0.248, -0.1496, 0.1937, 0.6651]).reshape((3, 3))
        >>> c = 6500
        >>> linear_interpolate_matrices(a, b, matrix_1, matrix_2, c)
        matrix([[ 0.48435269,  0.04240753, -0.07237634],
            [-0.74631613,  1.49989462,  0.26643011],
            [-0.13499785,  0.17817312,  0.66537957]])

    :param a: A value.
    :type a: float
    :param b: B value.
    :type b: float
    :param matrix_1: Matrix 1.
    :type matrix_1: matrix (N)
    :param matrix_2: Matrix 2.
    :type matrix_2: matrix (N)
    :param c: Interpolation value.
    :type c: float
    :return: Matrix.
    :rtype: matrix (N)
    """

    if a == b:
        return matrix_1

    shape = matrix_1.shape
    length = matrix_1.size
    matrix_1, matrix_2 = numpy.ravel(matrix_1), numpy.ravel(matrix_2)

    # TODO: Investigate numpy implementation issues when c < a or c > b.
    # return numpy.matrix([numpy.interp(c, (a, b), zip(matrix_1, matrix_2)[i]) for i in range(length)]).reshape(shape)

    return numpy.matrix(
        [matrix_1[i] + (c - a) * ((matrix_2[i] - matrix_1[i]) / (b - a)) for i in range(length)]).reshape(shape)
