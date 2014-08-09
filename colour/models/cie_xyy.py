#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**cie_xyy.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package colour *CIE xyY* colourspace objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["XYZ_to_xyY",
           "xyY_to_XYZ",
           "xy_to_XYZ",
           "XYZ_to_xy"]


def XYZ_to_xyY(XYZ,
               illuminant=ILLUMINANTS.get(
                   "CIE 1931 2 Degree Standard Observer").get("D50")):
    """
    Converts from *CIE XYZ* colourspace to *CIE xyY* colourspace and reference
    *illuminant*.

    Examples::

        >>> XYZ_to_xyY(np.array([0.1180583421, 0.1034, 0.0515089229]))
        array([[ 0.4325]
               [ 0.3788]
               [ 0.1034]])

    :param XYZ: *CIE XYZ* colourspace matrix.
    :type XYZ: array_like (3, 1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: array_like
    :return: *CIE xyY* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: Input *CIE XYZ* colourspace matrix is in domain [0, 1].
    :note: Output *CIE xyY* colourspace matrix is in domain [0, 1].

    References:

    -  http://www.brucelindbloom.com/Eqn_XYZ_to_xyY.html \
    (Last accessed 24 February 2014)
    """

    X, Y, Z = np.ravel(XYZ)

    if X == 0 and Y == 0 and Z == 0:
        return np.array([illuminant[0], illuminant[1], Y]).reshape((3, 1))
    else:
        return np.array([X / (X + Y + Z), Y / (X + Y + Z), Y]).reshape((3, 1))


def xyY_to_XYZ(xyY):
    """
    Converts from *CIE xyY* colourspace to *CIE XYZ* colourspace.

    Examples::

        >>> xyY_to_XYZ(np.array([0.4325, 0.3788, 0.1034]))
        array([[ 0.11805834]
               [ 0.1034    ]
               [ 0.05150892]])

    :param xyY: *CIE xyY* colourspace matrix.
    :type xyY: array_like (3, 1)
    :return: *CIE XYZ* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: Input *CIE xyY* colourspace matrix is in domain [0, 1].
    :note: Output *CIE XYZ* colourspace matrix is in domain [0, 1].

    References:

    -  http://www.brucelindbloom.com/Eqn_xyY_to_XYZ.html \
    (Last accessed 24 February 2014)
    """

    x, y, Y = np.ravel(xyY)

    if y == 0:
        return np.array([0., 0., 0.]).reshape((3, 1))
    else:
        return np.array([x * Y / y, Y, (1. - x - y) * Y / y]).reshape((3, 1))


def xy_to_XYZ(xy):
    """
    Returns the *CIE XYZ* colourspace matrix from given *xy* chromaticity
    coordinates.

    Examples::

        >>> xy_to_XYZ((0.25, 0.25))
        array([[ 1.],
               [ 1.],
               [ 2.]])

    :param xy: *xy* chromaticity coordinate.
    :type xy: array_like
    :return: *CIE XYZ* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: Input *xy* is in domain [0, 1].
    :note: Output *CIE XYZ* colourspace matrix is in domain [0, 1].
    """

    return xyY_to_XYZ(np.array([xy[0], xy[1], 1.]).reshape((3, 1)))


def XYZ_to_xy(XYZ,
              illuminant=ILLUMINANTS.get(
                  "CIE 1931 2 Degree Standard Observer").get("D50")):
    """
    Returns the *xy* chromaticity coordinates from given *CIE XYZ* colourspace
    matrix.

    Examples::

        >>> XYZ_to_xy(np.array([0.97137399, 1., 1.04462134]))
        (0.32207410281368043, 0.33156550013623531)
        >>> XYZ_to_xy((0.97137399, 1., 1.04462134))
        (0.32207410281368043, 0.33156550013623531)

    :param XYZ: *CIE XYZ* colourspace matrix.
    :type XYZ: array_like (3, 1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: array_like
    :return: *xy* chromaticity coordinates.
    :rtype: tuple

    :note: Input *CIE XYZ* colourspace matrix is in domain [0, 1].
    :note: Output *xy* is in domain [0, 1].
    """

    xyY = np.ravel(XYZ_to_xyY(XYZ, illuminant))
    return xyY[0], xyY[1]