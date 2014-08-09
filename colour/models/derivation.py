#!/usr/bin/env python
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

import numpy as np

__author__ = "Colour Developers"
__copyright__ = "Copyright (C) 2013 - 2014 - Colour Developers"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Colour Developers"
__email__ = "colour-science@googlegroups.com"
__status__ = "Production"

__all__ = ["xy_to_z",
           "get_normalised_primary_matrix",
           "get_RGB_luminance_equation",
           "get_RGB_luminance"]


def xy_to_z(xy):
    """
    Returns the *z* coordinate using given *chromaticity coordinates*.

    Examples::

        >>> xy_to_z((0.25, 0.25))
        0.5

    :param xy: X, y chromaticity coordinate.
    :type xy: array_like
    :return: Z coordinate.
    :rtype: float

    References:

    -  `RP 177-1993 SMPTE RECOMMENDED PRACTICE - Television Color Equations: \
    3.3.2 <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_
    """

    return 1 - xy[0] - xy[1]


def get_normalised_primary_matrix(primaries, whitepoint):
    """
    Returns the *normalised primary matrix* using given *primaries* and
    *whitepoint* matrices.

    Examples::

        >>> primaries = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
        >>> whitepoint = (0.32168, 0.33767)
        >>> get_normalised_primary_matrix(primaries, whitepoint)
        array([[  9.52552396e-01,   0.00000000e+00,   9.36786317e-05],
               [  3.43966450e-01,   7.28166097e-01,  -7.21325464e-02],
               [  0.00000000e+00,   0.00000000e+00,   1.00882518e+00]])

    :param primaries: Primaries chromaticity coordinate matrix (3, 2).
    :type primaries: array_like
    :param whitepoint: Illuminant / whitepoint chromaticity coordinates.
    :type whitepoint: array_like
    :return: Normalised primary matrix.
    :rtype: ndarray (3, 3)

    References:

    -  `RP 177-1993 SMPTE RECOMMENDED PRACTICE - Television Color Equations: \
    3.3.2 - 3.3.6 <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_
    """

    # Add 'z' coordinates to the primaries and transposing the matrix.
    primaries = primaries.reshape((3, 2))
    z = np.array([xy_to_z(np.ravel(primary)) for primary in primaries])
    primaries = np.hstack((primaries, z.reshape((3, 1))))

    primaries = np.transpose(primaries)

    whitepoint = np.array([
        whitepoint[0] / whitepoint[1],
        1,
        xy_to_z(whitepoint) / whitepoint[1]]).reshape((3, 1))

    coefficients = np.dot(np.linalg.inv(primaries), whitepoint)
    coefficients = np.diagflat(coefficients)

    npm = np.dot(primaries, coefficients)

    return npm


def get_RGB_luminance_equation(primaries, whitepoint):
    """
    Returns the *luminance equation* from given *primaries* and *whitepoint*
    matrices.

    Examples::

        >>> primaries = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
        >>> whitepoint = (0.32168, 0.33767)
        >>> get_RGB_luminance_equation(primaries, whitepoint)
        Y = 0.343966449765(R) + 0.728166096613(G) + -0.0721325463786(B)

    :param primaries: Primaries chromaticity coordinate matrix.
    :type primaries: array_like (3, 2)
    :param whitepoint: Illuminant / whitepoint chromaticity coordinates.
    :type whitepoint: array_like
    :return: *Luminance* equation.
    :rtype: unicode

    References:

    -  `RP 177-1993 SMPTE RECOMMENDED PRACTICE - Television Color Equations: \
    3.3.8 <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_
    """

    return "Y = {0}(R) + {1}(G) + {2}(B)".format(
        *np.ravel(get_normalised_primary_matrix(primaries, whitepoint))[3:6])


def get_RGB_luminance(RGB, primaries, whitepoint):
    """
    Returns the *luminance* of given *RGB* components from given *primaries* and *whitepoint* matrices.

    Examples::

        >>> RGB = np.array([40.6, 4.2, 67.4])
        >>> primaries = np.array([0.73470, 0.26530, 0.00000, 1.00000, 0.00010, -0.07700])
        >>> whitepoint = (0.32168, 0.33767)
        >>> get_RGB_luminance(primaries, whitepoint)
        12.1616018403

    :param RGB: *RGB* chromaticity coordinate matrix.
    :type RGB: array_like (3, 1)
    :param primaries: Primaries chromaticity coordinate matrix.
    :type primaries: array_like (3, 2)
    :param whitepoint: Illuminant / whitepoint chromaticity coordinates.
    :type whitepoint: array_like
    :return: *Luminance*.
    :rtype: float

    References:

    -  `RP 177-1993 SMPTE RECOMMENDED PRACTICE - Television Color Equations: \
    3.3.3 - 3.3.6 <http://car.france3.mars.free.fr/HD/INA-%2026%20jan%2006/SMPTE%20normes%20et%20confs/rp177.pdf>`_
    """

    R, G, B = np.ravel(RGB)
    X, Y, Z = np.ravel(get_normalised_primary_matrix(primaries,
                                                     whitepoint))[3:6]

    return X * R + Y * G + Z * B