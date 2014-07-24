# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**cie_ucs.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package colour *CIE UCS* colourspace objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["XYZ_to_UCS",
           "UCS_to_XYZ",
           "UCS_to_uv",
           "UCS_uv_to_xy"]


def XYZ_to_UCS(XYZ):
    """
    Converts from *CIE XYZ* colourspace to *CIE UCS* colourspace.

    References:

    -  http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ

    Usage::

        >>> XYZ_to_UCS(numpy.array([0.1180583421, 0.1034, 0.0515089229]))
        array([[ 0.07870556]
              [ 0.1034    ]
              [ 0.12182529]])

    :param XYZ: *CIE XYZ* colourspace matrix.
    :type XYZ: array_like (3, 1)
    :return: *CIE UCS* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: *CIE XYZ* is in domain [0, 1].
    :note: *CIE UCS* is in domain [0, 1].
    """

    X, Y, Z = numpy.ravel(XYZ)

    return numpy.array([2. / 3. * X, Y, 1. / 2. * (-X + 3. * Y + Z)]).reshape((3, 1))


def UCS_to_XYZ(UVW):
    """
    Converts from *CIE UCS* colourspace to *CIE XYZ* colourspace.

    References:

    -  http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ

    Usage::

        >>> UCS_to_XYZ(numpy.array([0.07870556, 0.1034, 0.12182529]))
        array([[ 0.11805834]
               [ 0.1034    ]
               [ 0.05150892]])

    :param UVW: *CIE UCS* colourspace matrix.
    :type UVW: array_like (3, 1)
    :return: *CIE XYZ* colourspace matrix.
    :rtype: ndarray (3, 1)

    :note: *CIE UCS* is in domain [0, 1].
    :note: *CIE XYZ* is in domain [0, 1].
    """

    U, V, W = numpy.ravel(UVW)

    return numpy.array([3. / 2. * U, V, 3. / 2. * U - (3. * V) + (2. * W)]).reshape((3, 1))


def UCS_to_uv(UVW):
    """
    Returns the *uv* chromaticity coordinates from given *CIE UCS* colourspace matrix.

    References:

    -  http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ

    Usage::

        >>> UCS_to_uv(numpy.array([0.1180583421, 0.1034, 0.0515089229]))
        (0.43249999995420696, 0.378800000065942)

    :param UVW: *CIE UCS* colourspace matrix.
    :type UVW: array_like (3, 1)
    :return: *uv* chromaticity coordinates.
    :rtype: tuple

    :note: *CIE UCS* is in domain [0, 1].
    :note: *uv* is in domain [0, 1].
    """

    U, V, W = numpy.ravel(UVW)

    return U / (U + V + W), V / (U + V + W)

def UCS_uv_to_xy(uv):
    """
    Returns the *xy* chromaticity coordinates from given *CIE UCS* colourspace *uv* chromaticity coordinates.

    References:

    -  http://en.wikipedia.org/wiki/CIE_1960_color_space#Relation_to_CIEXYZ

    Usage::

        >>> UCS_uv_to_xy((0.43249999995420696, 0.378800000065942))
        (0.7072386352886122, 0.4129510522116816)

    :param uv: *CIE UCS uv* chromaticity coordinate.
    :type uv: array_like
    :return: *xy* chromaticity coordinates.
    :rtype: tuple

    :note: *uv* is in domain [0, 1].
    :note: *xy* is in domain [0, 1].
    """

    return 3. * uv[0] / (2. * uv[0] - 8. * uv[1] + 4.), 2. * uv[1] / (2. * uv[0] - 8. * uv[1] + 4.)