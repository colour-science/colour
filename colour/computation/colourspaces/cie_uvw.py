# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
**cie_uvw.py**

**Platform:**
    Windows, Linux, Mac Os X.

**Description:**
    Defines **Colour** package colour *CIE UVW* colourspace objects.

**Others:**

"""

from __future__ import unicode_literals

import numpy

import colour.computation.colourspaces.cie_ucs
import colour.computation.colourspaces.cie_xyy
import colour.dataset.illuminants.chromaticity_coordinates
import colour.utilities.exceptions

__author__ = "Thomas Mansencal"
__copyright__ = "Copyright (C) 2013 - 2014 - Thomas Mansencal"
__license__ = "GPL V3.0 - http://www.gnu.org/licenses/"
__maintainer__ = "Thomas Mansencal"
__email__ = "thomas.mansencal@gmail.com"
__status__ = "Production"

__all__ = ["XYZ_to_UVW"]


def XYZ_to_UVW(XYZ,
               illuminant=colour.dataset.illuminants.chromaticity_coordinates.ILLUMINANTS.get(
                   "CIE 1931 2 Degree Standard Observer").get("D50")):
    """
    Converts from *CIE XYZ* colourspace to *CIE 1964 U\*V*\W\** colourspace.

    References:

    -  http://en.wikipedia.org/wiki/CIE_1964_color_space

    Usage::

        >>> XYZ_to_UVW(numpy.matrix([11.80583421, 10.34, 5.15089229]).reshape((3, 1)))
        matrix([[  7.87055614]
                [ 10.34      ]
                [ 12.18252904]])

    :param XYZ: *CIE XYZ* matrix.
    :type XYZ: matrix (3x1)
    :param illuminant: Reference *illuminant* chromaticity coordinates.
    :type illuminant: tuple
    :return: *CIE 1964 U\*V*\W\** matrix.
    :rtype: matrix (3x1)
    """

    x, y, Y = numpy.ravel(colour.computation.colourspaces.cie_xyy.XYZ_to_xyY(XYZ, illuminant))
    u, v = numpy.ravel(colour.computation.colourspaces.cie_ucs.UCS_to_uv(
        colour.computation.colourspaces.cie_ucs.XYZ_to_UCS(XYZ)))
    u0, v0 = numpy.ravel(colour.computation.colourspaces.cie_ucs.UCS_to_uv(
        colour.computation.colourspaces.cie_ucs.XYZ_to_UCS(
            colour.computation.colourspaces.cie_xyy.xy_to_XYZ(illuminant))))

    W = 25. * Y ** (1. / 3.) - 17.
    U = 13. * W * (u - u0)
    V = 13. * W * (v - v0)

    return numpy.matrix([U, V, W]).reshape((3, 1))