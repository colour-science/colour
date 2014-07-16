# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**derivation.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package *derivation* objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import colour.utilities.verbose

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["xy_to_z",
           "get_normalised_primary_matrix",
           "get_RGB_luminance_equation",
           "get_RGB_luminance"]

LOGGER = colour.utilities.verbose.install_logger()


def xy_to_z(xy):
    """
    Returns the *z* coordinate using given *chromaticity coordinates*.

    References:

    -  `RP 177-1993 SMPTE RECOMMENDED PRACTICE - Television Color Equations: 3.3.2 <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_

    Usage::

        >>> xy_to_z((0.25, 0.25))
        0.5

    :param xy: X, y chromaticity coordinate.
    :type xy: tuple
    :return: Z coordinate.
    :rtype: float
    """

    return 1 - xy[0] - xy[1]


def get_normalised_primary_matrix(primaries, whitepoint):
    """
    Returns the *normalised primary matrix* using given *primaries* and *whitepoint* matrices.

    References:

    -  `RP 177-1993 SMPTE RECOMMENDED PRACTICE - Television Color Equations: 3.3.2 - 3.3.6 <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_

    Usage::

        >>> primaries = numpy.matrix([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]).reshape((3, 2))
        >>> whitepoint = (0.32168, 0.33767)
        >>> get_normalised_primary_matrix(primaries, whitepoint)
        matrix([[  9.52552396e-01,   0.00000000e+00,   9.36786317e-05],
            [  3.43966450e-01,   7.28166097e-01,  -7.21325464e-02],
            [  0.00000000e+00,   0.00000000e+00,   1.00882518e+00]])

    :param primaries: Primaries chromaticity coordinate matrix ( 3 x 2 ).
    :type primaries: matrix
    :param whitepoint: Illuminant / whitepoint chromaticity coordinates.
    :type whitepoint: tuple
    :return: Normalised primary matrix.
    :rtype: float (3x3)
    """

    # Add 'z' coordinates to the primaries and transposing the matrix.
    primaries = numpy.hstack((primaries,
                              numpy.matrix(
                                  map(lambda x: xy_to_z(numpy.ravel(x)), primaries)).reshape(
                                  (3, 1))))
    primaries = numpy.transpose(primaries)

    whitepoint = numpy.matrix(
        [whitepoint[0] / whitepoint[1], 1, xy_to_z(whitepoint) / whitepoint[1]]).reshape(
        (3, 1))

    coefficients = primaries.getI() * whitepoint
    coefficients = numpy.diagflat(coefficients)

    npm = primaries * coefficients

    LOGGER.debug("> Transposed primaries:\n{0}".format(repr(primaries)))
    LOGGER.debug("> Whitepoint:\n{0}".format(repr(whitepoint)))
    LOGGER.debug("> Coefficients:\n{0}".format(repr(coefficients)))
    LOGGER.debug("> Normalised primary matrix':\n{0}".format(repr(npm)))

    return npm


def get_RGB_luminance_equation(primaries, whitepoint):
    """
    Returns the *luminance equation* from given *primaries* and *whitepoint* matrices.

    References:

    -  `RP 177-1993 SMPTE RECOMMENDED PRACTICE - Television Color Equations: 3.3.8 <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_

    Usage::

        >>> primaries = numpy.matrix([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]).reshape((3, 2))
        >>> whitepoint = (0.32168, 0.33767)
        >>> get_luminance_equation(primaries, whitepoint)
        Y = 0.343966449765(R) + 0.728166096613(G) + -0.0721325463786(B)

    :param primaries: Primaries chromaticity coordinate matrix.
    :type primaries: matrix (3x2)
    :param whitepoint: Illuminant / whitepoint chromaticity coordinates.
    :type whitepoint: tuple
    :return: *Luminance* equation.
    :rtype: unicode
    """

    return "Y = {0}(R) + {1}(G) + {2}(B)".format(
        *numpy.ravel(get_normalised_primary_matrix(primaries, whitepoint))[3:6])


def get_RGB_luminance(RGB, primaries, whitepoint):
    """
    Returns the *luminance* of given *RGB* components from given *primaries* and *whitepoint* matrices.

    References:

    -  `RP 177-1993 SMPTE RECOMMENDED PRACTICE - Television Color Equations: 3.3.3 - 3.3.6 <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_

    Usage::

        >>> RGB = numpy.matrix([40.6, 4.2, 67.4]).reshape((3, 1))
        >>> primaries = numpy.matrix([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700]).reshape((3, 2))
        >>> whitepoint = (0.32168, 0.33767)
        >>> get_luminance(primaries, whitepoint)
        12.1616018403

    :param RGB: *RGB* chromaticity coordinate matrix.
    :type RGB: matrix (3x1)
    :param primaries: Primaries chromaticity coordinate matrix.
    :type primaries: matrix (3x2)
    :param whitepoint: Illuminant / whitepoint chromaticity coordinates.
    :type whitepoint: tuple
    :return: *Luminance*.
    :rtype: float
    """

    R, G, B = numpy.ravel(RGB)
    X, Y, Z = numpy.ravel(get_normalised_primary_matrix(primaries, whitepoint))[3:6]

    return X * R + Y * G + Z * B